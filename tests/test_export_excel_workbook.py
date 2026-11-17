"""The Excel export end to end: ONE workbook must yield what the CSV FAMILY does.

The two containers differ in shape, not just extension:

    CSV    <NAME>_Results.CSV + <NAME>_Test1.CSV ... <NAME>_Test<N>.CSV
    Excel  <NAME>.xlsx, sheets Results | ... | Test 1 ... Test <N>

so the pipeline cannot just swap ".CSV" for ".xlsx": in Excel mode there is no
"_Results" file to find, and probing for one finds nothing, which is
indistinguishable from an export that never ran. It must probe for the bare
workbook and read the curves out of its sheets.
"""
import shutil
from pathlib import Path

import pytest

from nanoindentation_agents.instrument.export import ExportConfig, ExportPipeline

from test_export_sitting_discovery import make_pipeline, sitting


def excel_pipeline(tmp_path, workbook, *, extra_present=()):
    """A pipeline whose instrument holds `workbook` in sitting 1, wired so a
    download copies the real file into the download dir."""
    remote = tmp_path / "remote"
    remote.mkdir(exist_ok=True)
    shutil.copy(workbook.path, remote / workbook.name)

    d = sitting(1)
    present = [(d, f"{workbook.sample}.NMD"),          # preflight
               (d, workbook.name)] + list(extra_present)
    p = make_pipeline(tmp_path, present=present, export_format="excel")

    p.calls = []
    p._call_auto_raw = lambda method, timeout_s=None, **kw: (
        p.calls.append((method, kw)), {"status": "ok"})[1]
    p._download_named = lambda directory, filename: Path(
        shutil.copy(remote / filename, tmp_path / filename))
    return p


def run_record(workbook, expected=9):
    return {"sample_name": workbook.sample, "export_name": workbook.export_name,
            "sample_dir": sitting(1), "expected": expected, "method_type": "eh"}


def test_workbook_supplies_summary_and_curves_in_one_download(tmp_path,
                                                              inview_workbook):
    wb = inview_workbook
    p = excel_pipeline(tmp_path, wb)

    out = p.fetch_result_csv(run_record(wb))

    assert out is not None, "the export produced nothing"
    assert len(out["summary_rows"]) == wb.n_indents
    assert sorted(out["curves"]) == list(range(1, wb.n_indents + 1))
    assert out["summary_rows"]["Hardness (GPa)"].mean() == \
        pytest.approx(wb.mean_hardness, abs=1e-3)


def test_the_workbook_is_downloaded_once(tmp_path, inview_workbook):
    """Nine curves, one file: the CSV path's per-curve download must not run."""
    wb = inview_workbook
    p = excel_pipeline(tmp_path, wb)
    downloads = []
    inner = p._download_named
    p._download_named = lambda d, fn: (downloads.append(fn), inner(d, fn))[1]

    p.fetch_result_csv(run_record(wb))

    assert downloads == [wb.name], f"downloaded {downloads}"


def test_review_is_driven_in_excel_mode(tmp_path, inview_workbook):
    """The format reaches process_review_file — otherwise InView writes CSVs
    while this side looks for a workbook."""
    wb = inview_workbook
    p = excel_pipeline(tmp_path, wb)

    p.fetch_result_csv(run_record(wb))

    method, kwargs = p.calls[0]
    assert method == "process_review_file"
    assert kwargs["export_format"] == "excel"


def test_bare_workbook_is_probed_before_the_family_suffixes(tmp_path,
                                                            inview_workbook):
    """<NAME>.xlsx is the FIRST name tried in Excel mode; the _Results/_Test
    names do not exist there and every probe for one is a wasted round trip."""
    wb = inview_workbook
    p = excel_pipeline(tmp_path, wb)

    p.fetch_result_csv(run_record(wb))

    probed = [fn for _d, fn in p.probes if not fn.lower().endswith(".nmd")]
    assert probed[0] == wb.name, f"probe order was {probed[:4]}"


def test_curve_sheet_count_follows_the_array(tmp_path, inview_workbook):
    """`expected` is what the CSV path used to enumerate _Test<n> files. The
    workbook carries its own sheets, so a wrong `expected` must not truncate
    them."""
    wb = inview_workbook
    p = excel_pipeline(tmp_path, wb)

    out = p.fetch_result_csv(run_record(wb, expected=1))

    assert len(out["curves"]) == wb.n_indents


def test_csv_mode_still_looks_for_the_family(tmp_path):
    """The default path is unchanged: no bare-name probe, .CSV suffixes only."""
    cfg = ExportConfig(export_format="csv")
    assert cfg.export_exts == [".CSV"] and not cfg.is_excel
    assert ExportConfig(export_format="excel").export_exts == [".xlsx", ".xls"]


def test_a_workbook_with_no_usable_sheet_falls_back(tmp_path, inview_workbook):
    """An export that produced an unreadable workbook must return None so the
    caller drops to screen data — not raise, and not report empty results."""
    import openpyxl

    wb = inview_workbook
    blank = openpyxl.Workbook()
    blank.save(wb.path)                       # overwrite with an empty workbook
    p = excel_pipeline(tmp_path, wb)

    assert p.fetch_result_csv(run_record(wb)) is None
