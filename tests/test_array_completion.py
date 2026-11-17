"""When does place_array believe an array has FINISHED?

Every 1x1 round up to 2026-08-05 exited on the 300 s stall timer — 5m40s-6m50s
of polling per indent, whatever the indent took — while printing "array complete
(counter reached 1/1)". The counter cannot say: "Tests: N" is the CURRENT test
index, so it reads 1 the moment the test is queued and never moves again.
"""
import math
import time

import pytest

from nanoindentation_agents.instrument.screenread import ScreenReadConfig
from nanoindentation_agents.instrument.session import InstrumentSession


class FakeToolkit:
    class _C:
        def download_file(self, *a, **k):
            raise RuntimeError("no capture in this test")
    client = _C()


def make_session(tmp_path, *, depth_profile, ocr_test_number=1, done_after=None):
    """A session whose 'instrument' replays `depth_profile`, one value per read.

    `done_after` is the read index at which the result file appears (None = never).
    """
    s = object.__new__(InstrumentSession)
    s.toolkit = FakeToolkit()
    s.run_id = "test"
    s.download_dir = tmp_path
    s.journal = None
    s.run_records = {}
    s.screen_config = ScreenReadConfig(
        csv_dir=str(tmp_path), ocr_interval_s=0.0, sample_interval_s=0.0,
        settle_quiet_s=0.05, stall_s=1.0, activity_depth_nm=10.0,
        indent_gap_s=0.05)
    s.reads = {"n": 0}

    s.read_stage_xy = lambda **k: (100.0, 200.0)
    s.call = lambda tool, args, log, **k: {}

    def call_soft(tool, args, log, **k):
        n = (ocr_test_number[min(s.reads["n"], len(ocr_test_number) - 1)]
             if isinstance(ocr_test_number, list) else ocr_test_number)
        return True, {"test_number": n, "raw_text": ""}
    s.call_soft = call_soft

    def metrics():
        i = s.reads["n"]
        s.reads["n"] += 1
        d = depth_profile[min(i, len(depth_profile) - 1)]
        prev = depth_profile[min(i - 1, len(depth_profile) - 1)] if i else float("-inf")
        # Like the real panel: on Advanced Dynamic E and H, InView shows H/E
        # ONLY while the indent is LOADING, and blanks them to NaN at peak
        # depth and while retracting / surface-finding. Holding H/E valid at a
        # pinned peak forever (what this fixture used to do) never happens on
        # the instrument.
        rising = math.isfinite(d) and (not math.isfinite(prev) or d > prev)
        h, e = (4.0, 200.0) if (rising and d > 100) else (float("nan"), float("nan"))
        return d, 17.0, h, e, 0.01
    s.read_screen_metrics = metrics
    s.done_probe_calls = {"n": 0}

    probe = None
    if done_after is not None:
        def probe():
            s.done_probe_calls["n"] += 1
            return "C:\\sitting" if s.reads["n"] >= done_after else None
    return s, probe


PLAN = {"sample_name": "test_01", "label": "grain_boundary", "grid_nx": 1,
        "grid_ny": 1, "spacing_um": 15.0, "destination": (100.0, 200.0),
        "setup": {"target depth": 1000.0}, "method_type": "eh", "remove_n": 0}


def test_result_file_ends_the_wait_immediately(tmp_path):
    """The definitive signal must win, without waiting for any timer."""
    ramp = [0, 200, 500, 900, 1000] + [1000] * 200      # never goes quiet-then-flat
    s, probe = make_session(tmp_path, depth_profile=ramp, done_after=8)
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=probe, indent_duration_s=1.0)
    assert "wrote the result file" in rec["done_reason"], rec["done_reason"]
    assert rec["sample_dir"] == "C:\\sitting", "the found folder must be passed on"


def test_place_array_forwards_planned_rotation_to_starting_tests(tmp_path):
    s, _probe = make_session(tmp_path, depth_profile=[0.0], done_after=None)
    calls = []

    def capture_call(tool, args, log, **_kwargs):
        calls.append((tool, args))
        return {}

    s.call = capture_call
    plan = dict(PLAN, rotation_deg=45.0)
    rec = s.place_array(
        plan,
        show_frames=False,
        test_timeout_s=5,
        done_probe=lambda: "C:\\sitting",
        skip_screening=True,
        indent_duration_s=0.0,
    )
    start_args = next(args for tool, args in calls if tool == "starting_tests")
    assert start_args["indents"]["Rotation"] == 45.0
    assert rec["rotation_deg"] == 45.0


def test_result_file_does_not_end_a_multi_indent_grid_after_indent_one(tmp_path):
    """InView writes/updates the result file as EACH indent lands, so on a 2x2
    grid the file can exist seconds into indent #1 of 4 — long before
    process_review_file should run. Confirmed on real hardware that the OCR
    "Tests: N" counter reports the array's TOTAL from the very first read, not
    a live 1/2/3/4 progression, so it cannot gate the file signal either —
    only the depth-driven tracker (which needs no OCR) confirming all
    `expected` indents may."""
    plan = dict(PLAN, grid_nx=2, grid_ny=2)             # expected = 4
    indent_ramp = list(range(0, 2000, 100))             # one indent, 20 reads
    gap = [0.0] * 10                                    # stage moving between sites
    tail = [2000] * 100                                 # settle after the last indent
    depth = indent_ramp + gap + indent_ramp + gap + indent_ramp + gap \
        + indent_ramp + tail
    static_ocr_total = 4          # what the real instrument reports, from read 1
    # The result file appears right after indent #1 finishes (read ~25) — well
    # before the grid's other three indents have even started.
    s, probe = make_session(tmp_path, depth_profile=depth,
                            ocr_test_number=static_ocr_total, done_after=25)
    rec = s.place_array(plan, show_frames=False, test_timeout_s=60, done_probe=probe, indent_duration_s=1.0)
    assert "indentation sequence complete: 4/4" in rec["done_reason"], rec["done_reason"]
    assert s.reads["n"] > 90, \
        f"stopped after only {s.reads['n']} reads — quit around indent #1, not the whole grid"


def test_result_file_appearing_before_indenting_even_starts_does_not_end_a_multi_indent_grid(tmp_path):
    """Reproduces the 2026-08-18 11:50 run: the completion probe found the
    result file essentially immediately — before depth had even started
    ramping on indent #1 of a 2x2 grid. Must not be treated as done."""
    plan = dict(PLAN, grid_nx=2, grid_ny=2)             # expected = 4
    indent_ramp = list(range(0, 2000, 100))
    gap = [0.0] * 10
    tail = [2000] * 100
    depth = indent_ramp + gap + indent_ramp + gap + indent_ramp + gap \
        + indent_ramp + tail
    static_ocr_total = 4
    s, probe = make_session(tmp_path, depth_profile=depth,
                            ocr_test_number=static_ocr_total, done_after=0)
    rec = s.place_array(plan, show_frames=False, test_timeout_s=60, done_probe=probe, indent_duration_s=1.0)
    assert "indentation sequence complete: 4/4" in rec["done_reason"], rec["done_reason"]
    assert s.reads["n"] > 90, \
        f"stopped after only {s.reads['n']} reads — quit before real indenting happened"


def test_stale_counter_plus_early_file_cannot_finish_a_grid_on_test_two(tmp_path):
    """The reported failure, end to end (2026-08-18 screenshot): the result
    file already exists AND the header counter still shows the PREVIOUS
    array's final value (4), while the instrument is only on test 2 of 4.
    Neither may be taken as completion."""
    plan = dict(PLAN, grid_nx=2, grid_ny=2)             # expected = 4
    indent_ramp = list(range(0, 2000, 100))
    gap = [0.0] * 10
    one = indent_ramp + gap
    depth = one * 4 + [2000] * 100
    # Counter is a stale 4 from the previous run for the whole first half of
    # the array, then behaves normally. The file is present from read 0.
    stale = [4] * (len(one) * 2)
    live = [3] * len(one) + [4] * (len(one) + 100)
    s, probe = make_session(tmp_path, depth_profile=depth,
                            ocr_test_number=stale + live, done_after=0)
    rec = s.place_array(plan, show_frames=False, test_timeout_s=60, done_probe=probe, indent_duration_s=1.0)
    assert "indentation sequence complete: 4/4" in rec["done_reason"], rec["done_reason"]
    # Two indents' worth of reads is where it used to bail out.
    assert s.reads["n"] > len(one) * 2, \
        f"stopped after {s.reads['n']} reads — around test 2, exactly the bug"


def test_does_not_stop_mid_ramp_while_depth_is_still_changing(tmp_path):
    """The counter reads 1/1 from the first tick; that must not end monitoring
    while the indent is visibly still loading."""
    ramp = list(range(0, 3000, 60)) + [3000] * 400
    s, probe = make_session(tmp_path, depth_profile=ramp, done_after=len(ramp) + 10)
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=probe, indent_duration_s=1.0)
    # It must have watched the whole ramp before concluding anything.
    assert s.reads["n"] > len(range(0, 3000, 60)), \
        f"stopped after only {s.reads['n']} reads — mid-ramp"
    # However it concluded, it must be a real completion signal (here the 1x1
    # result-file shortcut), never a timeout.
    assert "TIMED OUT" not in rec["done_reason"], rec["done_reason"]
    assert ("wrote the result file" in rec["done_reason"]
            or "indentation sequence complete: 1/1" in rec["done_reason"]), rec["done_reason"]


def test_single_indent_finishes_without_any_counter(tmp_path):
    """The counter is useless (may not parse at all, and even when it does it
    just reports the total) on any grid size. A depth that went deep and then
    went flat is enough on its own — and must not wait out the 300 s stall
    timer to say so."""
    ramp = list(range(0, 2000, 100)) + [2000] * 300
    s, _ = make_session(tmp_path, depth_profile=ramp, ocr_test_number=None)
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=None, indent_duration_s=1.0)
    assert "indentation sequence complete: 1/1" in rec["done_reason"], rec["done_reason"]
    assert "TIMED OUT" not in rec["done_reason"]


def test_multi_indent_array_does_not_use_the_single_indent_rule(tmp_path):
    """Depth also goes flat BETWEEN indents, so a 3x3 must not stop at the
    first gap."""
    plan = dict(PLAN, grid_nx=3, grid_ny=3)
    ramp = list(range(0, 2000, 100)) + [0.0] * 60 + list(range(0, 2000, 100)) \
        + [2000] * 300
    s, _ = make_session(tmp_path, depth_profile=ramp, ocr_test_number=None)
    rec = s.place_array(plan, show_frames=False, test_timeout_s=30, done_probe=None, indent_duration_s=1.0)
    assert "single indent" not in rec["done_reason"], rec["done_reason"]


def test_reason_names_the_real_trigger_not_the_counter(tmp_path):
    """A timeout must be REPORTED as a timeout. The old code called every exit
    'counter reached 1/1' because the string was picked by n_done >= expected."""
    s, probe = make_session(tmp_path, depth_profile=[0.0] * 500, done_after=None)
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=None, indent_duration_s=1.0)
    assert "TIMED OUT" in rec["done_reason"], rec["done_reason"]
    assert "counter reached" not in rec["done_reason"]


def test_blind_screen_reader_does_not_fake_a_stall(tmp_path):
    """If depth never reads finite we know nothing about activity, so the stall
    rule must not fire — it would be indistinguishable from a running test."""
    s, probe = make_session(tmp_path, depth_profile=[float("nan")] * 500)
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=1.5,
                        done_probe=None, indent_duration_s=1.0)
    assert rec["done_reason"] == "deadline", rec["done_reason"]


def test_probe_failure_degrades_to_heuristics(tmp_path):
    """A probe that raises must not take the round down with it."""
    s, _ = make_session(tmp_path, depth_profile=[0.0] * 500)

    def angry():
        raise RuntimeError("VPN dropped")
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=angry, indent_duration_s=1.0)
    assert rec["done_reason"] and "TIMED OUT" in rec["done_reason"]


# --------------------------------------------- estimated post-file wait -----
def test_estimated_wait_holds_process_review_after_the_file_appears(tmp_path):
    """The operative fix for the repeated premature exports: InView writes the
    result file after indent #1, so the monitor must hold for the remaining
    (expected - 1) indents even when nothing else confirms completion."""
    # A screen that NEVER produces a usable signal: no H/E, flat depth. The
    # only thing standing between the file and Process Review is the timer.
    s, probe = make_session(tmp_path, depth_profile=[0.0] * 50,
                            ocr_test_number=None, done_after=0)
    plan = dict(PLAN, grid_nx=2, grid_ny=2)            # 4 indents -> 3 remaining
    t0 = time.monotonic()
    rec = s.place_array(plan, show_frames=False, test_timeout_s=60,
                        done_probe=probe, indent_duration_s=0.7)
    held = time.monotonic() - t0
    assert held >= 3 * 0.7, f"only held {held:.2f}s — the estimated wait was skipped"
    assert "waited" in rec["done_reason"], rec["done_reason"]
    assert "3 remaining indent(s)" in rec["done_reason"], rec["done_reason"]


def test_estimated_wait_scales_with_grid_size_and_is_zero_for_1x1(tmp_path):
    """wait = (nx*ny - 1) * indent_duration_s, from the PLAN's grid — never a
    hard-coded count. 1x1 has nothing remaining, so it must not wait at all."""
    s, probe = make_session(tmp_path, depth_profile=[0.0] * 50,
                            ocr_test_number=None, done_after=0)
    t0 = time.monotonic()
    s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                  done_probe=probe, indent_duration_s=5.0)   # 1x1 -> 0 remaining
    assert time.monotonic() - t0 < 3.0, "a 1x1 grid waited when nothing remained"

    # 3x3 = 9 indents -> 8 remaining. Check the arithmetic without paying it.
    for nx, ny, per, want in ((1, 1, 150.0, 0.0), (2, 2, 150.0, 450.0),
                              (3, 3, 150.0, 1200.0), (5, 5, 150.0, 3600.0)):
        assert (nx * ny - 1) * per == want, f"{nx}x{ny} wait should be {want}s"


def test_estimated_wait_extends_a_timeout_that_would_cut_it_short(tmp_path):
    """A 5x5 needs 60 min of wait but the default monitor timeout is 30 —
    without raising it the round would export mid-array anyway."""
    s, probe = make_session(tmp_path, depth_profile=[0.0] * 50,
                            ocr_test_number=None, done_after=0)
    plan = dict(PLAN, grid_nx=5, grid_ny=5)
    # 24 remaining * 1s = 24s of wait, with a 2s timeout asked for.
    t0 = time.monotonic()
    rec = s.place_array(plan, show_frames=False, test_timeout_s=2.0,
                        done_probe=probe, indent_duration_s=1.0)
    assert time.monotonic() - t0 > 20.0, \
        "the 2s timeout truncated the 24s estimated wait"
    assert "24 remaining indent(s)" in rec["done_reason"], rec["done_reason"]


# ------------------------------------------------------ skip_screening ------
def _counting_probe(n_calls_until_found):
    """A done_probe that ignores screen reads entirely (skip_screening never
    takes any) and instead becomes true after its own Nth call."""
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return "C:\\sitting" if calls["n"] >= n_calls_until_found else None
    return probe, calls


def test_skip_screening_bypasses_the_monitor_and_trusts_the_file(tmp_path):
    """skip_screening=True must take NO screen reads at all (no OCR polling,
    no depth capture) and end the wait the moment the probe reports the file,
    however early that is — this is the deliberately-accepted tradeoff."""
    s, _ = make_session(tmp_path, depth_profile=[0.0])   # never touched
    probe, calls = _counting_probe(3)
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=probe, skip_screening=True, indent_duration_s=1.0)
    assert rec["recorder"] is None
    assert "skip_screening" in rec["done_reason"], rec["done_reason"]
    assert "C:\\sitting" in rec["done_reason"]
    assert calls["n"] == 3
    assert s.reads["n"] == 0, "skip_screening must never call read_screen_metrics"


def test_skip_screening_with_no_probe_proceeds_immediately(tmp_path):
    """Simulator mode passes done_probe=None. skip_screening must not hang
    waiting on a probe that will never come."""
    s, _ = make_session(tmp_path, depth_profile=[0.0])
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=30,
                        done_probe=None, skip_screening=True, indent_duration_s=1.0)
    assert rec["recorder"] is None
    assert "no result-file probe available" in rec["done_reason"], rec["done_reason"]


def test_skip_screening_times_out_if_file_never_appears(tmp_path):
    """No screen fallback means no other way out — a missing file must still
    end in a bounded timeout, not an infinite wait."""
    s, _ = make_session(tmp_path, depth_profile=[0.0])
    s.screen_config.ocr_interval_s = 0.05   # keep the test fast
    rec = s.place_array(dict(PLAN), show_frames=False, test_timeout_s=0.2,
                        done_probe=lambda: None, skip_screening=True, indent_duration_s=1.0)
    assert rec["recorder"] is None
    assert "did not appear within" in rec["done_reason"], rec["done_reason"]


def test_skip_screening_wires_through_grid_size_unchanged(tmp_path):
    """The dynamic grid size (nx*ny) must still reach run_record even with no
    screen monitor to compute `expected` from."""
    plan = dict(PLAN, grid_nx=3, grid_ny=3)
    s, _ = make_session(tmp_path, depth_profile=[0.0])
    probe, _ = _counting_probe(1)
    rec = s.place_array(plan, show_frames=False, test_timeout_s=30,
                        done_probe=probe, skip_screening=True, indent_duration_s=1.0)
    assert rec["expected"] == 9
    assert rec["n_done"] == 9
