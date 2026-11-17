"""Shared fixtures.

`inview_workbook` reproduces the Excel export cell-for-cell from a real InView
workbook (a 3x3 array, 2026-08-10). The Excel export is ONE WORKBOOK where the
CSV export is a file family:

    Results | Sample Information | Pre-Test Inputs | Sample Summary |
    Post-Test Inputs | Cycles-GroupedByCycle | Cycles-GroupedByTest |
    Test 1 | Test 2 | ... | Test <N>

"Results" is what the CSV family calls _Results — names row, units row, one row
per indent, then Average / Standard Deviation / Coefficient of Variation.
"Test <n>" (WITH a space; the CSV family writes _Test<n> without one) is the
load-depth curve, one sheet PER INDENT, so their number follows the array size
and must never be assumed.
"""
from types import SimpleNamespace

import pytest

# Rows 1-14 of the real Results sheet.
XL_RESULTS = [
    ["Test", "X", "Y", "Pmax", "hmax", "Reporting Depth", "HV", "Hardness",
     "Modulus", "Stiffness^2/Load", "Drift Rate", "Temperature"],
    ["", "µm", "µm", "mN", "nm", "nm", "None", "GPa", "GPa", "GPa", "nm/s", "C"],
    [1, 59515.636, 15081.266, 24.995, 466, 397.2, 903.3, 9.557, 67.1, 639.4, 0.29, 0],
    [2, 59515.636, 15141.273, 24.978, 470, 401.6, 870.0, 9.205, 66.3, 649.3, 0.267, 0],
    [3, 59515.636, 15201.281, 24.965, 505, 436.7, 694.5, 7.348, 58.71, 646.4, 0.202, 0],
    [4, 59455.629, 15201.281, 24.970, 465, 396.6, 903.5, 9.558, 67.28, 642.6, 0.174, 0],
    [5, 59455.629, 15141.273, 24.973, 468, 400.1, 876.0, 9.267, 66.75, 653.0, 0.154, 0],
    [6, 59455.629, 15081.266, 24.977, 465, 396.5, 908.5, 9.611, 67.11, 636.1, 0.145, 0],
    [7, 59395.621, 15081.266, 24.947, 465, 396.8, 899.1, 9.512, 67.37, 647.4, 0.089, 0],
    [8, 59395.621, 15141.273, 24.954, 464, 395.8, 905.7, 9.582, 67.5, 644.9, 0.106, 0],
    [9, 59395.621, 15201.281, 24.948, 465, 396.5, 901.3, 9.536, 67.39, 646.0, 0.092, 0],
    ["Average", 59455.629, 15141.273, 24.9673, 470.1, 401.96, 873.53, 9.2417,
     66.167, 645.01, 0.1687, 0],
    ["Standard Deviation", 48.9959, 48.9959, 0.015, 12.4, 12.41, 64.58, 0.6832,
     2.661, 4.81, 0.0689, 0],
    ["Coefficient of Variation", 0.000824, 0.003236, 0.0006, 0.026, 0.0309,
     0.0739, 0.073928, 0.04022, 0.0075, 0.408271],
]
XL_N_INDENTS = 9
XL_PEAK_DEPTH_NM, XL_PEAK_LOAD_MN = 466.0, 25.0


def _curve_sheet(peak_h: float, peak_e: float, n: int = 60) -> list:
    """A "Test <n>" sheet: names, units, the Surface Index marker, then the
    curve — H/E blank through the surface approach and populated deep, as
    InView writes them."""
    rows = [["Markers", "DEPTH", "LOAD", "TIME", "MODULUS", "HARDNESS"],
            ["", "nm", "mN", "s", "GPa", "GPa"],
            ["Surface Index", 0, 0, 0, "", ""]]
    for i in range(1, n + 1):
        rows.append(["", XL_PEAK_DEPTH_NM * i / n, XL_PEAK_LOAD_MN * (i / n) ** 2,
                     0.01 * i,
                     peak_e if i > n // 2 else "",
                     peak_h if i > n // 2 else ""])
    return rows


def write_inview_workbook(path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    results = wb.create_sheet("Results")
    for row in XL_RESULTS:
        results.append(row)
    for name in ("Sample Information", "Pre-Test Inputs", "Sample Summary",
                 "Post-Test Inputs", "Cycles-GroupedByCycle", "Cycles-GroupedByTest"):
        wb.create_sheet(name).append(["Property", "Value"])
    for r in XL_RESULTS[2:2 + XL_N_INDENTS]:
        ws = wb.create_sheet(f"Test {int(r[0])}")
        for row in _curve_sheet(peak_h=r[7], peak_e=r[8]):
            ws.append(row)
    wb.save(path)


@pytest.fixture
def inview_workbook(tmp_path):
    """The real Excel export, plus the numbers InView itself reports for it."""
    p = tmp_path / "test_01_0810_120000_export.xlsx"
    write_inview_workbook(p)
    return SimpleNamespace(
        path=p, name=p.name, export_name=p.stem, sample="test_01",
        n_indents=XL_N_INDENTS,
        mean_hardness=XL_RESULTS[-3][7],      # InView's own Average row
        mean_modulus=XL_RESULTS[-3][8],
        peak_depth_nm=XL_PEAK_DEPTH_NM, peak_load_mn=XL_PEAK_LOAD_MN,
    )