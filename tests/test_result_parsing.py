"""Reading an InView Results export, against the real files from run 0805_170826.

Three defects that run exposed, all downstream of a download that WORKED:

1. InView appends Average / Standard Deviation / Coefficient of Variation rows
   to every export. They were being counted as indents — a 1x1 array reported
   "2/1 valid indents" and averaged its one measurement with the Average row.
2. The export's X/Y are in InView's own frame: every file was offset from the
   live stage readback by (+44162.3, +21.2) um. Fed to the GP as stage
   coordinates, they put every observation ~44 mm outside the search area.
3. When the values were simply out of range the log said "no hardness inside
   0.3-100.0 GPa", which reads as a download failure. It was not one.
"""
from pathlib import Path

import pandas as pd
import pytest

from nanoindentation_agents.instrument import results as R

DATA = Path(__file__).resolve().parents[1] / "notebooks/downloaded_data"

# (file, hardness in the ONE real row, live stage position of that array)
REAL = [
    ("test_01_0805_170826_export_Results.CSV", 0.183, (15912.0, 58895.6)),
    ("test_02_0805_170826_export_Results.CSV", 0.434, (15963.3, 58906.5)),
    ("test_03_0805_170826_export_Results.CSV", 0.126, (16043.5, 58810.5)),
]


def _read(name):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"{name} not on disk")
    return R.read_inview_table(p)


@pytest.mark.parametrize("name,hardness,pos", REAL)
def test_one_indent_gives_exactly_one_row(name, hardness, pos):
    """Not three. The Average / StdDev / CoV rows are statistics, not indents."""
    rows = _read(name)
    assert len(rows) == 1, f"got {len(rows)} rows:\n{rows}"
    assert rows["Hardness"].iloc[0] == pytest.approx(hardness, abs=1e-3)


def _cell(text: str):
    """A CSV field as Excel would hold it: a real number where it is one."""
    t = text.strip()
    try:
        return float(t)
    except ValueError:
        return t


def test_workbook_results_sheet_is_the_per_indent_table(inview_workbook):
    """9 indents — not 12. Average / Std / CoV are statistics in Excel too."""
    wb = inview_workbook
    summary, _curves = R.read_inview_workbook(wb.path)
    assert len(summary) == wb.n_indents, f"got {len(summary)} rows:\n{summary}"
    assert list(summary["Test"]) == list(range(1, wb.n_indents + 1))
    assert summary["Hardness"].iloc[2] == pytest.approx(7.348, abs=1e-3)


def test_workbook_hardness_matches_inviews_own_average_row(inview_workbook):
    """Reducing the kept rows must reproduce the Average row InView wrote."""
    wb = inview_workbook
    summary, _ = R.read_inview_workbook(wb.path)
    out = R.average_hardness_modulus(R.normalize_result_columns(summary),
                                     verbose=False)
    assert out["n_rows"] == out["n_valid"] == wb.n_indents
    assert out["hardness_mean"] == pytest.approx(wb.mean_hardness, abs=1e-3)
    assert out["modulus_mean"] == pytest.approx(wb.mean_modulus, abs=1e-2)


def test_workbook_curve_sheets_are_found_one_per_indent(inview_workbook):
    """'Test 1'..'Test 9' are the load-depth curves. The count follows the
    array, so nothing may assume a fixed number of them."""
    wb = inview_workbook
    _summary, curves = R.read_inview_workbook(wb.path)
    assert sorted(curves) == list(range(1, wb.n_indents + 1))
    curve = R.normalize_result_columns(curves[1])
    assert float(curve["Contact Depth (nm)"].max()) == \
        pytest.approx(wb.peak_depth_nm, abs=1.0)
    assert float(curve["Load (mN)"].max()) == pytest.approx(wb.peak_load_mn, abs=0.1)


def test_workbook_curves_reduce_to_one_row_per_indent(inview_workbook):
    """The curve-reduction fallback must work off sheets exactly as it does off
    the _Test<n> files."""
    wb = inview_workbook
    _summary, curves = R.read_inview_workbook(wb.path)
    per_indent = R.curves_to_per_indent(
        {i: R.normalize_result_columns(c) for i, c in curves.items()})
    assert len(per_indent) == wb.n_indents
    ok, why = R.validate_rows(per_indent, expected_indents=wb.n_indents)
    assert ok, why


def test_workbook_validates_as_a_primary_source(inview_workbook):
    wb = inview_workbook
    summary, _ = R.read_inview_workbook(wb.path)
    ok, why = R.validate_rows(R.normalize_result_columns(summary),
                              expected_indents=wb.n_indents)
    assert ok, why
    assert f"{wb.n_indents}/{wb.n_indents} valid" in why


def test_excel_export_reads_the_same_as_the_csv_one(tmp_path):
    """The same indent, exported both ways, must parse to the same table.

    A single-sheet workbook also has to work: `read_inview_table` falls back
    past the missing "Results" sheet rather than returning nothing.
    """
    openpyxl = pytest.importorskip("openpyxl")
    name = REAL[1][0]
    p = DATA / name
    if not p.exists():
        pytest.skip(f"{name} not on disk")

    wb = openpyxl.Workbook()
    ws = wb.active
    for line in p.read_text().splitlines():
        ws.append([_cell(c) for c in line.split(",")])
    xlsx = tmp_path / name.replace(".CSV", ".xlsx")
    wb.save(xlsx)

    from_csv, from_xlsx = R.read_inview_table(p), R.read_inview_table(xlsx)
    assert len(from_xlsx) == len(from_csv) == 1, f"got:\n{from_xlsx}"
    assert list(from_xlsx.columns) == list(from_csv.columns)
    assert from_xlsx["Hardness"].iloc[0] == pytest.approx(REAL[1][1], abs=1e-3)


def test_summary_rows_survive_if_they_are_all_there_is(tmp_path):
    """An export carrying only a summary line must still yield something."""
    import io
    csv = ("Test,Hardness,Modulus\n"
           ",GPa,GPa\n"
           "Average, 3.2, 210.0\n")
    p = tmp_path / "_only_summary_tmp.CSV"
    p.write_text(csv)
    try:
        rows = R.read_inview_table(p)
        assert len(rows) == 1
    finally:
        p.unlink()


def test_average_counts_one_valid_indent_not_two():
    """The 8/05 symptom: "(2/1 valid)" on a single-indent array."""
    rows = R.normalize_result_columns(_read(REAL[1][0]))
    out = R.average_hardness_modulus(rows, verbose=False)
    assert out["n_rows"] == 1 and out["n_valid"] == 1
    assert out["hardness_mean"] == pytest.approx(0.434, abs=1e-3)
    assert out["hardness_std"] == 0.0


@pytest.mark.parametrize("name,hardness,pos", REAL)
def test_export_coordinates_are_not_taken_as_stage_coordinates(name, hardness, pos):
    """They are 44 mm away in InView's frame; the planned grid must win."""
    rows = R.normalize_result_columns(_read(name))
    out = R.ensure_xy(rows, {"position": pos}, 1, 1, 15.0)
    assert abs(float(out["X (um)"].iloc[0]) - pos[0]) < 1.0, \
        f"X came back as {float(out['X (um)'].iloc[0])}, stage says {pos[0]}"
    assert abs(float(out["Y (um)"].iloc[0]) - pos[1]) < 1.0


def test_plausible_coordinates_are_still_trusted():
    """Do not throw away a good export: coordinates near the array stay."""
    rows = pd.DataFrame({"Hardness (GPa)": [3.0, 3.1],
                         "Modulus (GPa)": [200.0, 205.0],
                         "X (um)": [15960.0, 15975.0],
                         "Y (um)": [58900.0, 58900.0]})
    out = R.ensure_xy(rows, {"position": (15967.5, 58900.0)}, 2, 1, 15.0)
    assert float(out["X (um)"].iloc[0]) == 15960.0
    assert float(out["X (um)"].iloc[1]) == 15975.0


def test_ensure_xy_applies_planned_grid_rotation():
    rows = pd.DataFrame({"Hardness (GPa)": [7.0, 7.1]})
    out = R.ensure_xy(rows, {"position": (100.0, 200.0)}, 2, 1, 10.0, 90.0)
    assert out["X (um)"].tolist() == pytest.approx([100.0, 100.0])
    assert out["Y (um)"].tolist() == pytest.approx([195.0, 205.0])


def test_out_of_range_is_reported_as_a_measurement_problem():
    rows = R.normalize_result_columns(_read(REAL[2][0]))      # H = 0.126 GPa
    ok, why = R.validate_rows(rows, expected_indents=1)
    assert not ok
    assert "0.126" in why and "parsed fine" in why


def test_in_range_data_still_validates():
    rows = R.normalize_result_columns(_read(REAL[1][0]))      # H = 0.434 GPa
    ok, why = R.validate_rows(rows, expected_indents=1)
    assert ok, why
    assert "1/1 valid" in why


def test_a_nan_array_does_not_poison_the_improvement_criterion():
    """One failed indent must not permanently block progress evaluation."""
    import math
    from nanoindentation_agents.agents.orchestrator_agent import OrchestratorAgent
    from nanoindentation_agents.objective import parse_objective_fallback

    obj = parse_objective_fallback("maximize hardness", indent_budget=20)
    orch = OrchestratorAgent(obj)
    state = {
        "indent_history": [
            {"test_name": "test_01", "area_name": "a", "position": (0.0, 0.0),
             "n_indents": 1, "hardness_mean": float("nan"),
             "modulus_mean": float("nan"), "valid_indent_count": 0},
            {"test_name": "test_02", "area_name": "b", "position": (30.0, 0.0),
             "n_indents": 1, "hardness_mean": 4.0, "modulus_mean": 200.0,
             "valid_indent_count": 1},
        ],
        "visited_target_keys": ["a", "b"],
        "orchestrator_round": 2,
    }
    prog = orch.evaluate_progress(state)
    assert math.isfinite(prog.improvement_pct), \
        f"improvement came back {prog.improvement_pct}"
