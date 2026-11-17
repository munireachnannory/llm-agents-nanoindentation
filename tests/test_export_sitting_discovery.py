"""WHICH SITTING FOLDER did InView put this test in?

InView opens a new sitting for every test, so within a campaign the trailing
number tracks the ROUND: round 1 -> <profile>_<date>_1, round 5 -> ..._5. The
probe window was a hard-coded 1..4, so round 5 was the first round whose .NMD
could not be found at all — it waited out nmd_wait_s and fell back to screen
data while rounds 1-4 exported cleanly. Seen on runs 0805_170826 and
0806_131828, both times on test_05 and never before it.
"""
import pytest

from nanoindentation_agents.instrument.export import ExportConfig, ExportPipeline

DATE = "20260806"
ROOT = r"C:\Users\Public\Documents\Nanomechanics\Profiles\Shared AENI"


def make_pipeline(tmp_path, *, present=(), find_files_deployed=False, **cfg_kw):
    """A pipeline whose 'instrument' owns exactly the folders in `present`."""
    cfg = ExportConfig(session_date=DATE, test_number=1, **cfg_kw)
    p = object.__new__(ExportPipeline)
    p.cfg = cfg
    p.download_dir = tmp_path
    p.open_dir = cfg.session_dir()
    p._sitting_cache = None
    p._find_files_ok = None
    p.probes = []

    def remote_exists(directory, filename, timeout=120):
        p.probes.append((directory, filename))
        return ((directory, filename) in present, 200 if (directory, filename) in present
                else 404)

    def find_files(pattern, directory=None, limit=50, timeout=180):
        if not find_files_deployed:
            return None                       # not deployed on the iMicro PC
        return [{"name": fn, "directory": d, "path": rf"{d}\{fn}"}
                for (d, fn) in present if fn.lower() == pattern.lower()]

    p.remote_exists = remote_exists
    p.find_files = find_files
    return p


def sitting(n, date=DATE, pad=False):
    tail = f"{n:02d}" if pad else f"{n}"
    return rf"{ROOT}\Samples\Practice\Practice_{date}_{tail}"


# --------------------------------------------------------------- the bug
def test_round_five_sitting_is_reachable(tmp_path):
    """The exact 0806_131828 failure: rounds 1-4 exported, round 5 did not."""
    p = make_pipeline(tmp_path, present=[(sitting(5), "test_05.NMD")])
    p._sitting_cache = sitting(4)             # round 4 confirmed here

    assert p.wait_for_nmd("test_05", wait_s=0) == sitting(5)


def test_next_sitting_is_probed_before_the_blind_window(tmp_path):
    """Ordering is load-bearing — the walk stops at the first hit, once per poll."""
    p = make_pipeline(tmp_path, present=[(sitting(5), "test_05.NMD")])
    p._sitting_cache = sitting(4)
    p.wait_for_nmd("test_05", wait_s=0)

    probed = [d for d, _ in p.probes]
    assert probed.index(sitting(5)) <= 3, f"..._5 probed too late: {probed[:6]}"


def test_campaign_longer_than_max_sitting_still_resolves(tmp_path):
    """A 20-round campaign outlives any constant; the lookahead must carry it."""
    p = make_pipeline(tmp_path, present=[(sitting(18), "test_18.NMD")])
    p._sitting_cache = sitting(17)

    assert p.wait_for_nmd("test_18", wait_s=0) == sitting(18)


def test_cold_start_falls_back_to_the_blind_window(tmp_path):
    """Recovery after a kernel restart has no cache — the window must still cover
    a normal campaign (it was 4; the run that broke had MAX_ROUNDS = 10)."""
    p = make_pipeline(tmp_path, present=[(sitting(9), "test_09.NMD")])
    assert p._sitting_cache is None
    assert p.wait_for_nmd("test_09", wait_s=0) == sitting(9)


# ------------------------------------------------- the in-monitor done probe
def test_result_ready_looks_in_the_new_sitting(tmp_path):
    """The definitive completion signal: last round's folder is the WRONG place,
    which is why every round after the first ended on a screen heuristic."""
    p = make_pipeline(tmp_path, present=[(sitting(2), "test_02.NMD")])
    p._sitting_cache = sitting(1)

    assert p.result_ready("test_02") == sitting(2)


def test_result_ready_stays_cheap(tmp_path):
    """It runs every few seconds inside the monitor loop — a handful of probes,
    never the full candidate list."""
    p = make_pipeline(tmp_path, present=[])
    p._sitting_cache = sitting(4)
    p.result_ready("test_05")

    assert len(p.probes) <= 4, f"{len(p.probes)} probes per monitor tick is too many"


# ----------------------------------------------------------- folder parsing
@pytest.mark.parametrize("directory, expected", [
    (sitting(4), 4),
    (sitting(4, pad=True), 4),
    (sitting(12), 12),
    (rf"{ROOT}\Samples\Practice\Practice_{DATE}", None),   # date, NOT sitting 20260806
    (rf"{ROOT}\Samples\Practice", None),
    (None, None),
])
def test_sitting_index(directory, expected):
    assert ExportPipeline._sitting_index(directory) == expected


def test_blind_window_is_bounded(tmp_path):
    """A misparsed folder must not explode the candidate list."""
    p = make_pipeline(tmp_path)
    p._sitting_cache = sitting(97)
    assert len(p.candidate_session_dirs()) < 250


# ------------------------------------------------------------- /find_files
def test_find_files_short_circuits_the_guessing(tmp_path):
    """Where deployed, ask the instrument instead of constructing names."""
    odd = rf"{ROOT}\Samples\Practice\Practice_20260806_31"
    p = make_pipeline(tmp_path, present=[(odd, "test_05.NMD")],
                      find_files_deployed=True)

    assert p.wait_for_nmd("test_05", wait_s=0) == odd
    assert p.probes == [], "should not have needed a single constructed probe"


def test_missing_find_files_is_only_asked_once(tmp_path):
    """It is not deployed on the iMicro PC — a poll loop must not pay every pass."""
    p = make_pipeline(tmp_path, present=[(sitting(5), "test_05.NMD")])
    calls = []
    inner = p.find_files
    p.find_files = lambda *a, **k: (calls.append(1), inner(*a, **k))[1]

    p.discover_sample_dir("test_05")
    p.discover_sample_dir("test_05")
    p.discover_sample_dir("test_05")
    assert len(calls) == 1 and p._find_files_ok is False