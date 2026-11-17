"""LiveScreenRecorder must identify which indent a sample belongs to from the
PHYSICAL signal, not from the on-screen "Tests: N" OCR reading.

Confirmed on the real 2026-08-18 2x2 run: every screen-read line showed
"indent 4/4" from the very first sample onward — the OCR field reports the
array's TOTAL planned count, not a live 1/2/3/4 progression. Naively using it
as the bucket key folded all four physically distinct indents (clearly
different Modulus values on screen) into one, so the run ended reporting
"1/4 indents seen on screen".
"""
from nanoindentation_agents.instrument.screenread import (
    LiveScreenRecorder, ScreenReadConfig, TestSequenceTracker,
    parse_current_test_index)


# The InView screen as it really is (2026-08-18 screenshot): the HEADER shows
# the current test index, while the Sample Definition Panel's "# TESTS" column
# shows the array total. Both contain the word TESTS.
REAL_SCREEN_OCR = """Method: Advanced Dynamic E and H with X and Y_2   Sample: test_01
Tip:IRG2 5th Berkovich 500nm (08-15-26 06:54:17)    Tests: 2
Multi-Sample Test Definition
PROJECT NAME Practice
SAMPLE LIST                 # TESTS
test_01                       4
"""


def test_parses_the_header_index_not_the_panel_total():
    """Must return 2 (currently running), never 4 (the grid total)."""
    assert parse_current_test_index(REAL_SCREEN_OCR) == 2


def test_stale_counter_from_the_previous_array_is_ignored():
    """THE bug behind every premature Process Review: InView still shows the
    PREVIOUS array's final index when a new array is queued, and the old code
    latched it with `if parsed > n_done`, so the very first read pinned the
    counter at 4/4 forever."""
    t = TestSequenceTracker(4)
    for _ in range(6):
        t.observe(4)                      # left over from the last run
    assert not t.armed and not t.reached_end, \
        "a stale counter was accepted as real progress"

    for _ in range(2):
        t.observe(1)                      # this array actually starts
    assert t.armed and t.current == 1
    for v in (2, 2, 3, 3):
        t.observe(v)
    assert t.current == 3 and not t.reached_end, "declared done on test 3 of 4"
    for _ in range(2):
        t.observe(4)
    assert t.reached_end


def test_counter_arms_on_physical_evidence_when_the_reset_is_missed():
    """Polling every 5 s can miss the brief '1'. A real indent ramp is proof
    this array started, and the index must then begin at 1, not at the stale
    value."""
    t = TestSequenceTracker(4)
    for _ in range(4):
        t.observe(4)                      # stale
    t.observe(4, physical_started=True)
    assert t.armed and t.current == 1 and not t.reached_end


def test_single_frame_ocr_misread_does_not_advance_the_index():
    t = TestSequenceTracker(4)
    for _ in range(2):
        t.observe(1)
    t.observe(3)                          # one-frame garbage
    assert t.current == 1, "an unconfirmed OCR reading moved the index"


def test_tracker_scales_to_any_grid_size():
    for n in (1, 4, 9, 25):
        t = TestSequenceTracker(n)
        for _ in range(2):
            t.observe(1)
        for v in range(2, n + 1):
            for _ in range(2):
                t.observe(v)
        assert t.current == n and t.reached_end, f"{n}-indent grid never completed"


def make_recorder(tmp_path, *, grid_nx=2, grid_ny=2):
    plan = {"sample_name": "test_01", "grid_nx": grid_nx, "grid_ny": grid_ny,
            "spacing_um": 15.0, "destination": (100.0, 200.0),
            "setup": {"target depth": 1000.0}}
    cfg = ScreenReadConfig(csv_dir=str(tmp_path))
    return LiveScreenRecorder(plan, cfg, "test")


def _run_indent(rec, static_ocr_total, modulus):
    """One ramp-up-then-withdraw cycle, like one real indent on screen."""
    for d in range(0, 1000, 100):
        hardness = 1.0 if d >= 100 else float("nan")
        e = modulus if d >= 100 else float("nan")
        rec.observe(static_ocr_total, float(d), 0.05, hardness, e)
    for d in (0.0, -1.0, float("nan"), float("nan")):
        rec.observe(static_ocr_total, d, 0.0, float("nan"), float("nan"))


def test_static_ocr_total_does_not_collapse_indents_into_one_bucket(tmp_path):
    """Reproduces the real 2x2 run: OCR always reports 4 (the total), never
    1/2/3/4. Four physically distinct indents (different Modulus, exactly
    like the real log's 31.6 / 26.9 / 6.7 / 33.0 GPa) must still land in
    four separate buckets, not fold into whichever one the OCR happens to
    report."""
    rec = make_recorder(tmp_path)
    static_ocr_total = rec.n_indents          # what the real instrument sent
    moduli = [31.6, 26.9, 6.7, 33.0]
    for e in moduli:
        _run_indent(rec, static_ocr_total, e)

    assert len(rec.records) == 4, f"only {len(rec.records)} distinct indent(s) recorded"
    recorded = [rec.records[k]["Modulus (GPa)"] for k in sorted(rec.records)]
    assert recorded == moduli


def test_advance_is_dynamic_across_grid_sizes(tmp_path):
    """No hardcoded '4' anywhere in the tracking — a 3x3 grid must recognize
    all 9 indents from the same physical-signal logic."""
    rec = make_recorder(tmp_path, grid_nx=3, grid_ny=3)
    static_ocr_total = rec.n_indents           # == 9
    moduli = [10.0 + i for i in range(9)]
    for e in moduli:
        _run_indent(rec, static_ocr_total, e)

    assert len(rec.records) == 9
    assert [rec.records[k]["Modulus (GPa)"] for k in sorted(rec.records)] == moduli


def test_single_indent_grid_is_unaffected(tmp_path):
    """A 1x1 array has nothing to advance past — everything folds into the
    one bucket, same as before this change (there is nowhere else to go)."""
    rec = make_recorder(tmp_path, grid_nx=1, grid_ny=1)
    _run_indent(rec, rec.n_indents, 42.0)

    assert rec.current_index == 1
    assert len(rec.records) == 1
    assert rec.records[1]["Modulus (GPa)"] == 42.0


# The REAL screen trace from the 2026-08-18 10:59 hardware run (a 2x2 array),
# transcribed from the [screen] log lines in order: (depth_nm, H_GPa, E_GPa).
# Note the surface-finding ramps reporting H=E=0.00, and the >400 nm depth
# drops WITHIN a single indent (1568 -> 97.8, and 2206 -> 149.2) that made the
# old depth-drop counter over-count and fire "complete" during indent 2.
_N = float("nan")
REAL_2X2_TRACE = [
    (345.0, 0.00, 0.0), (947.7, 0.00, 0.0), (1568.3, 0.00, 0.0),      # surface find
    (97.8, 4.18, 80.9), (521.8, 0.65, 31.6),                          # INDENT 1
    (1008.0, _N, _N), (_N, _N, _N), (_N, _N, _N),
    (-0.5, _N, _N), (-1.8, _N, _N), (1.1, _N, _N),                    # retracted
    (146.8, _N, _N), (663.1, _N, _N), (1186.1, _N, _N),
    (1709.3, _N, _N), (2206.8, _N, _N),                               # surface find
    (149.2, 1.29, 40.5), (503.4, 0.49, 26.9), (967.4, 0.38, 21.9),    # INDENT 2
    (_N, _N, _N), (_N, _N, _N), (_N, _N, _N),
    (217.0, 0.24, 6.7), (606.4, 0.18, 6.4), (948.3, 0.24, 10.9),      # INDENT 3
    (_N, _N, _N), (_N, _N, _N), (_N, _N, _N),
    (9.2, _N, _N), (550.5, _N, _N), (1090.9, _N, _N),                 # surface find
    (120.2, 1.66, 33.0), (363.1, 1.07, 31.9), (782.1, 0.72, 31.3),    # INDENT 4
    (_N, _N, _N), (_N, _N, _N), (_N, _N, _N), (_N, _N, _N),
]


def test_real_hardware_trace_counts_exactly_four_indents(tmp_path):
    """Replay of the actual 2x2 run. The old depth-drop rule reached the grid
    total during indent 2 and launched Process Review there; the H/E-burst
    rule must count exactly 4, and must NOT report the grid complete before
    indent 4's ramp has been seen."""
    rec = make_recorder(tmp_path)
    t = {"now": 0.0}
    rec.clock = lambda: t["now"]
    last_indent_start = REAL_2X2_TRACE.index((120.2, 1.66, 33.0))
    for i, (d, h, e) in enumerate(REAL_2X2_TRACE):
        rec.observe(rec.n_indents, d, 0.05, h, e)   # OCR always reports the total
        if i < last_indent_start:
            assert not rec.grid_complete, \
                f"grid declared complete at sample {i} — before indent 4 even started"
        assert rec.completed_indents <= 3, \
            f"over-counted to {rec.completed_indents} at sample {i} (indent 4 unfinished)"
    # Indent 4's ramp ended; the tip stays out. Let the gap debounce elapse.
    t["now"] += rec.cfg.indent_gap_s + 1.0
    rec.observe(rec.n_indents, _N, 0.0, _N, _N)

    assert rec.completed_indents == 4, f"counted {rec.completed_indents}"
    assert rec.grid_complete
    # and each indent kept its own distinct modulus, not one merged bucket.
    # Indent 1 averages only 31.6: its 97.8 nm sample is rejected as
    # "depth < 100 nm", exactly as the real log reported it.
    assert len(rec.records) == 4
    assert [rec.records[k]["Modulus (GPa)"] for k in sorted(rec.records)] == \
        [31.6, 29.767, 8.0, 32.067]


def test_surface_find_depth_drops_do_not_advance_the_indent(tmp_path):
    """The specific regression: a >400 nm depth collapse inside one indent's
    surface-finding approach must not count as a finished indent."""
    rec = make_recorder(tmp_path)
    for d in (345.0, 947.7, 1568.3, 97.8):        # the drop 1568 -> 97.8
        rec.observe(rec.n_indents, d, 0.05, 0.00, 0.0)   # H=E=0.00: surface finding
    assert rec.completed_indents == 0, "a surface-find drop was counted as an indent"


def test_does_not_advance_on_noise_before_a_real_indent_happened(tmp_path):
    """A shallow, noisy blip that never reached the acceptance depth must not
    be mistaken for a finished indent."""
    rec = make_recorder(tmp_path)
    static_ocr_total = rec.n_indents
    # Shallow noise: H/E briefly finite at low depth, then a big-looking drop
    # — but depth never reached accept_depth_nm, so this is NOT a real indent.
    rec.observe(static_ocr_total, 5.0, 0.01, 1.0, 20.0)
    rec.observe(static_ocr_total, 0.0, 0.0, float("nan"), float("nan"))
    assert rec.current_index == 1
