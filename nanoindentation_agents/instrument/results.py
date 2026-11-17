"""
Result parsing, validation, and per-indent reduction.

The rule carried over from notebook 5: the downloaded CSV is PREFERRED but
never trusted blindly. A candidate table becomes the primary source only if
it passes the validation ladder; the live screen readings always remain as
the fallback, and the two are cross-checked when both exist.

Facts about InView exports this module encodes (verified on hardware 7/29):
* A CSV export is a FAMILY: <name>_Results.CSV, _SampleSummary.CSV,
  _Test<N>.CSV. An EXCEL export is ONE WORKBOOK, <name>.xlsx, whose "Results"
  and "Test <n>" sheets hold the same tables — see read_inview_workbook.
* _SampleSummary is a per-DEPTH-BIN profile, NOT per-indent — counting its
  rows as indents corrupts budgets and averages, so it is rejected by shape.
* Tables carry a names/units/marker three-row header; the header row is found
  by metric-name content, not position.
* Real exports use BARE column names (HARDNESS, MODULUS), not 'Hardness (GPa)'.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .screenread import ScreenReadConfig

CANONICAL_COLS = ["Sample", "Indent", "X (um)", "Y (um)", "Load (mN)",
                  "Hardness (GPa)", "Modulus (GPa)", "Contact Depth (nm)",
                  "Region (sim truth)"]

_METRIC_TOKENS = ("depth", "load", "hardness", "modulus", "time", "indent",
                  "sample", "test", "x (", "y (")


# --------------------------------------------------------------------- parsing
def _numeric_row(fields: list[str]) -> bool:
    """True if this looks like a DATA row: every non-blank field numeric, >= 2."""
    vals = [f.strip() for f in fields if f.strip() != ""]
    if len(vals) < 2:
        return False
    for v in vals:
        try:
            float(v)
        except ValueError:
            return False
    return True


def _header_line_index(raw: list[str]) -> Optional[int]:
    """Index of the row holding COLUMN NAMES, chosen by metric-name content.

    A "first non-numeric row above the first numeric row" rule picks InView's
    marker row ('Surface Index,0,0,0') — verified wrong on a real _Test1.CSV.
    """
    for i, line in enumerate(raw[:25]):
        low = line.lower()
        if sum(1 for t in _METRIC_TOKENS if t in low) >= 2:
            return i
    return None


# Labels InView puts in the FIRST column of the statistics rows it appends to a
# Results export. These are not indents and must never be counted as ones.
_SUMMARY_ROW_LABELS = frozenset({
    "average", "mean", "standard deviation", "std deviation", "std dev", "stdev",
    "coefficient of variation", "cov", "minimum", "maximum", "median",
})


EXCEL_EXTS = (".xlsx", ".xlsm", ".xls")

# Sheets of the Excel export, verified against a real InView workbook 2026-08-10:
#   Results | Sample Information | Pre-Test Inputs | Sample Summary |
#   Post-Test Inputs | Cycles-GroupedByCycle | Cycles-GroupedByTest |
#   Test 1 | Test 2 | ... | Test <N>
# "Results" is what the CSV family calls _Results; "Test <n>" (WITH a space —
# the CSV family writes _Test<n> without one) is _Test<n>. There is one curve
# sheet per indent, so their number follows the array and is never assumed.
_SUMMARY_SHEET = "results"
_CURVE_SHEET_RX = re.compile(r"^test[\s_-]*(\d+)$", re.I)


def read_inview_table(path, *, sheet=None) -> pd.DataFrame:
    """Read an InView export into a DataFrame, handling its two-row header.
    Blank H/E cells (surface-approach samples) survive as NaN.
    InView's trailing statistics rows are dropped — see `_SUMMARY_ROW_LABELS`.

    CSV and Excel exports are the SAME table in two containers, so both go
    through one ladder; only the reader changes. For a workbook this returns the
    per-indent "Results" sheet — use `read_inview_workbook` to get the curve
    sheets with it, or pass `sheet` for one specific sheet.
    """
    p = Path(path)
    if p.suffix.lower() in EXCEL_EXTS:
        book = _open_workbook(p)
        if sheet is not None:
            return _sheet_table(book, sheet)
        return _summary_sheet(book)[1]
    raw = p.read_text(errors="replace").splitlines()
    return _read_inview_ladder(raw, lambda **kw: pd.read_csv(p, **kw))


def read_inview_workbook(path) -> tuple:
    """Split an InView .xlsx export into (per-indent table, {indent: curve}).

    The Excel export is ONE WORKBOOK where the CSV export is a file family, so
    the curves arrive as sheets rather than as separate downloads. Returns the
    same shapes the CSV path produces, so everything downstream — validation,
    curve reduction, plots — is unchanged.
    """
    book = _open_workbook(Path(path))
    _name, summary = _summary_sheet(book)
    curves = {}
    for sh in book.sheet_names:
        m = _CURVE_SHEET_RX.match(str(sh).strip())
        if not m:
            continue
        tbl = _sheet_table(book, sh)
        if not tbl.empty:
            curves[int(m.group(1))] = tbl
    return summary, curves


def _open_workbook(p: Path):
    try:
        return pd.ExcelFile(p)
    except ImportError as exc:                        # engine, not file, missing
        raise ImportError(
            f"Reading {p.name} needs an Excel engine — pip install openpyxl "
            "(.xlsx/.xlsm) or xlrd (.xls)."
        ) from exc


def _sheet_table(book, sheet) -> pd.DataFrame:
    """One worksheet through the CSV ladder.

    A bare ``pd.read_excel`` is wrong here for exactly the reasons the CSV path
    documents: the names/units/marker block means the header must be FOUND
    rather than assumed to be row 0, and InView's trailing Average / Standard
    Deviation / Coefficient of Variation rows would be counted as indents.
    """
    grid = book.parse(sheet, header=None, dtype=object)
    if grid.empty:
        return pd.DataFrame()
    return _read_inview_ladder(_excel_grid_as_lines(grid),
                               lambda **kw: book.parse(sheet, **kw))


def _summary_sheet(book) -> tuple:
    """(sheet name, per-indent table) — the "Results" sheet when there is one.

    Falls back to the first sheet that parses to something non-empty, so a
    workbook written by a different InView build (or a hand-made one) still
    yields its table instead of nothing.
    """
    for sh in book.sheet_names:
        if str(sh).strip().lower() == _SUMMARY_SHEET:
            tbl = _sheet_table(book, sh)
            if not tbl.empty:
                return sh, tbl
    for sh in book.sheet_names:
        if _CURVE_SHEET_RX.match(str(sh).strip()):
            continue                       # a curve is not a per-indent table
        tbl = _sheet_table(book, sh)
        if not tbl.empty:
            return sh, tbl
    return None, pd.DataFrame()


def _excel_grid_as_lines(grid: pd.DataFrame) -> list[str]:
    """A header-less Excel grid rendered as the comma-joined lines the ladder
    inspects, so header detection sees an Excel export exactly as it sees a CSV."""
    return [",".join("" if pd.isna(v) else str(v).strip() for v in row)
            for row in grid.itertuples(index=False, name=None)]


def _read_inview_ladder(raw: list[str], reader) -> pd.DataFrame:
    """Find the header row in `raw`, re-read through `reader`, and clean up.

    `reader(**kwargs)` must accept pandas' ``skiprows``/``header`` — that is all
    ``read_csv`` and ``ExcelFile.parse`` need to have in common.
    """
    if not raw:
        return pd.DataFrame()

    hdr = _header_line_index(raw)
    if hdr is not None:
        df = reader(skiprows=hdr)
    else:
        widths = Counter(len(l.split(",")) for l in raw if l.strip())
        if not widths:
            return pd.DataFrame()
        width = widths.most_common(1)[0][0]
        first = next((i for i, l in enumerate(raw)
                      if l.strip() and len(l.split(",")) == width
                      and _numeric_row(l.split(","))), None)
        if first is None:
            return reader()
        hrow = first - 1
        if hrow >= 0 and not _numeric_row(raw[hrow].split(",")):
            df = reader(skiprows=hrow)
        else:
            df = reader(skiprows=first, header=None)

    numeric = df.apply(pd.to_numeric, errors="coerce")
    df = df[numeric.notna().any(axis=1)].copy()      # drops the units row

    # InView appends Average / Standard Deviation / Coefficient of Variation
    # rows to EVERY Results export. They are numeric, so the units-row filter
    # above keeps them, and every consumer downstream then counts them AS
    # INDENTS. On 2026-08-05 a 1x1 array reported "2/1 valid indents" and
    # averaged its single measurement together with the Average row restating
    # it; on a real 3x3 the Average row would be silently double-weighted into
    # the mean and fed to the GP as a fourth site.
    # Kept ONLY if nothing else remains, so an export that carries just a
    # summary line still yields something rather than nothing.
    if len(df.columns):
        label = df[df.columns[0]].astype(str).str.strip().str.lower()
        summary = label.isin(_SUMMARY_ROW_LABELS)
        if summary.any() and (~summary).any():
            df = df[~summary].copy()

    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().any():
            df[c] = conv
    return df.reset_index(drop=True)


def normalize_result_columns(rows: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Map InView's headers onto the canonical names the agents consume.
    Existing correct columns are never overwritten."""
    if rows is None or getattr(rows, "empty", True):
        return rows
    lower = {str(c).strip().lower(): c for c in rows.columns}

    def _find(keys, exclude=()):
        for want in keys:
            for lc, orig in lower.items():
                if want in lc and not any(x in lc for x in exclude):
                    return orig
        return None

    wanted = {
        "Hardness (GPa)":     (("hardness",), ()),
        "Modulus (GPa)":      (("modulus", "young"), ("reduced",)),
        "Load (mN)":          (("load",), ("unload", "rate", "cite")),
        "Contact Depth (nm)": (("contact depth", "depth"), ("rate", "cite", "min")),
    }
    out = rows.copy()
    for canon, (keys, excl) in wanted.items():
        if canon in out.columns:
            continue
        src = _find(keys, excl)
        if src is not None:
            out[canon] = pd.to_numeric(out[src], errors="coerce")
    h = out.get("Hardness (GPa)")
    if h is not None and len(h.dropna()) and float(h.dropna().median()) > 200:
        print(f"  [export] WARNING: median hardness {float(h.dropna().median()):.0f} — "
              "this export looks like MPa, not GPa. Check the ReviewData export units.")
    return out


# ---------------------------------------------------------------- statistics
def average_hardness_modulus(rows, *, h_range=(0.3, 100.0), label="", verbose=True) -> dict:
    """Average hardness AND modulus over a per-indent table -> stats dict.
    Only indents whose hardness falls inside `h_range` count. std is population
    (ddof=0) to match the rest of the pipeline."""
    lo, hi = h_range
    blank = {"n_rows": 0, "n_valid": 0,
             "hardness_mean": float("nan"), "hardness_std": float("nan"),
             "hardness_min": float("nan"), "hardness_max": float("nan"),
             "hardness_cv_pct": float("nan"),
             "modulus_mean": float("nan"), "modulus_std": float("nan"),
             "modulus_min": float("nan"), "modulus_max": float("nan"),
             "modulus_cv_pct": float("nan")}
    if rows is None or getattr(rows, "empty", True):
        return blank
    if "Hardness (GPa)" not in rows.columns or "Modulus (GPa)" not in rows.columns:
        return blank

    h = pd.to_numeric(rows["Hardness (GPa)"], errors="coerce")
    e = pd.to_numeric(rows["Modulus (GPa)"], errors="coerce")
    valid = h.notna() & e.notna() & (h > lo) & (h < hi)
    hv, ev = h[valid], e[valid]
    if not len(hv):
        if verbose:
            seen = ", ".join(f"{v:.3f}" for v in h.dropna().head(4)) or "nothing numeric"
            tag = f" [{label}]" if label else ""
            print(f"  average{tag}: 0/{len(rows)} rows counted — the table parsed "
                  f"fine but its hardness ({seen} GPa) is outside {lo}–{hi} GPa. "
                  "This is a MEASUREMENT problem, not a download or parsing one.")
        return {**blank, "n_rows": int(len(rows))}

    def _cv(s):
        m = float(s.mean())
        return float(s.std(ddof=0) / m * 100.0) if m else float("nan")

    out = {
        "n_rows": int(len(rows)), "n_valid": int(valid.sum()),
        "hardness_mean": float(hv.mean()), "hardness_std": float(hv.std(ddof=0)),
        "hardness_min": float(hv.min()), "hardness_max": float(hv.max()),
        "hardness_cv_pct": _cv(hv),
        "modulus_mean": float(ev.mean()), "modulus_std": float(ev.std(ddof=0)),
        "modulus_min": float(ev.min()), "modulus_max": float(ev.max()),
        "modulus_cv_pct": _cv(ev),
    }
    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"  average{tag}: H = {out['hardness_mean']:.3f} ± "
              f"{out['hardness_std']:.3f} GPa (CV {out['hardness_cv_pct']:.1f}%)")
        print(f"  average{tag}: E = {out['modulus_mean']:.1f} ± "
              f"{out['modulus_std']:.1f} GPa (CV {out['modulus_cv_pct']:.1f}%)")
        print(f"  average{tag}: {out['n_valid']}/{out['n_rows']} rows counted "
              f"(hardness inside {lo}–{hi} GPa)")
    return out


# ----------------------------------------------------------- curve reduction
def curve_to_indent(curve, indent_no, *, deep_frac=0.5, h_range=(0.3, 100.0)):
    """Reduce ONE _Test<N> load-depth curve to ONE per-indent record.
    MEDIAN over the deepest `deep_frac` of finite-H/E samples (last points spike)."""
    lo, hi = h_range
    if curve is None or getattr(curve, "empty", True):
        return None
    h = pd.to_numeric(curve.get("Hardness (GPa)"), errors="coerce")
    e = pd.to_numeric(curve.get("Modulus (GPa)"), errors="coerce")
    d = pd.to_numeric(curve.get("Contact Depth (nm)"), errors="coerce")
    ld = pd.to_numeric(curve.get("Load (mN)"), errors="coerce")
    if h is None or e is None or d is None:
        return None
    fin = h.notna() & e.notna() & d.notna() & (h > 0) & (e > 0)
    if not bool(fin.any()):
        return None
    dmax = float(d[fin].max())
    deep = fin & (d >= deep_frac * dmax)
    if not bool(deep.any()):
        deep = fin
    hh, ee = float(h[deep].median()), float(e[deep].median())
    return {
        "Sample": None, "Indent": int(indent_no),
        "X (um)": None, "Y (um)": None,
        "Load (mN)": float(ld[deep].max()) if ld is not None and ld.notna().any()
        else float("nan"),
        "Hardness (GPa)": round(hh, 4), "Modulus (GPa)": round(ee, 3),
        "Contact Depth (nm)": round(dmax, 2),
        "n_curve_samples": int(deep.sum()),
        "validation_status": "valid" if lo < hh < hi else "out_of_range",
        "Region (sim truth)": "result_curve",
    }


def curves_to_per_indent(curves, run_record=None, *, deep_frac=0.5,
                         h_range=(0.3, 100.0)) -> pd.DataFrame:
    """Reduce {indent_no: curve_df} into ONE per-indent DataFrame."""
    rec = (run_record or {}).get("recorder")
    out = []
    for n in sorted(curves):
        row = curve_to_indent(curves[n], n, deep_frac=deep_frac, h_range=h_range)
        if row is None:
            continue
        if run_record:
            row["Sample"] = run_record["sample_name"]
            if rec is not None and 1 <= n <= rec.n_indents:
                _r, _c, gx, gy = rec.grid_position(n)
                row["X (um)"], row["Y (um)"] = gx, gy
        out.append(row)
    cols = CANONICAL_COLS[:8] + ["n_curve_samples", "validation_status",
                                 "Region (sim truth)"]
    cols = list(dict.fromkeys(cols))
    return pd.DataFrame(out, columns=cols)


# ------------------------------------------------------------- validation
def is_summary_table(rows) -> bool:
    """True for InView's _SampleSummary shape: one row per DEPTH BIN.
    normalize_result_columns happily maps 'HARDNESS Mean' -> 'Hardness (GPa)',
    so this table validates and silently wins unless rejected by SHAPE."""
    if rows is None or getattr(rows, "empty", True):
        return False
    cols = [str(c) for c in rows.columns]
    triples = sum(1 for c in cols if c.endswith((" Mean", " StdDev", " Median")))
    return triples >= max(3, len(cols) // 3)


def validate_rows(rows, h_range=(0.3, 100.0), expected_indents=None):
    """Is this table usable as the PRIMARY source? -> (bool, reason)."""
    lo, hi = h_range
    if rows is None or getattr(rows, "empty", True):
        return False, "no rows"
    if expected_indents and len(rows) > max(4, 3 * int(expected_indents)):
        return False, (f"{len(rows)} rows for {int(expected_indents)} planned "
                       "indent(s) -- time-series or multi-test dump, not per-indent")
    if is_summary_table(rows):
        return False, ("per-depth summary table (Mean/StdDev/Median columns) -- "
                       "rows are DEPTH BINS, not indents")
    missing = [c for c in ("Hardness (GPa)", "Modulus (GPa)") if c not in rows.columns]
    if missing:
        return False, f"missing {missing}; columns = {list(rows.columns)[:8]}"
    h = pd.to_numeric(rows["Hardness (GPa)"], errors="coerce")
    valid = (h > lo) & (h < hi)
    if int(valid.sum()) == 0:
        # SAY WHAT WAS ACTUALLY READ. "no hardness inside 0.3-100 GPa" was true
        # but read as "the download failed" — on 2026-08-05 the exports had
        # arrived and parsed perfectly and simply contained H = 0.126 GPa.
        seen = ", ".join(f"{v:.3f}" for v in h.dropna().head(4)) or "nothing numeric"
        return False, (f"the file parsed fine ({len(rows)} row(s), hardness {seen} GPa) "
                       f"but NO value falls inside {lo}–{hi} GPa — the export is good, "
                       "the MEASUREMENTS are out of the accepted band")
    if not np.isfinite(h[valid].mean()):
        return False, "non-finite mean hardness"
    return True, f"{int(valid.sum())}/{len(rows)} valid indents"


def ensure_xy(rows, run_record, nx, ny, sp, rotation_deg: float = 0.0):
    """Guarantee per-indent 'X (um)'/'Y (um)' on whatever table won.

    Without this, a downloaded result CSV (which carries no position columns)
    breaks every plot AND silently collapses the GP to one point per array.
    Order: existing X/Y -> any InView-style position column -> the array
    centre laid out on the plan grid (exact for a 1x1 array).
    """
    if rows is None or getattr(rows, "empty", True):
        return rows
    out = rows.copy()

    def _num(col):
        return pd.to_numeric(out[col], errors="coerce")

    def _find(axis):
        pats = [rf"^{axis} \(um\)$", rf"^{axis} axis position",
                rf"^{axis} position", rf"^{axis}$"]
        low = {str(c).strip().lower(): c for c in out.columns}
        for p in pats:
            for lc, orig in low.items():
                if re.match(p, lc) and not any(k in lc for k in ("std", "median", "mean")):
                    return orig
        return None

    # THE EXPORT'S X/Y ARE NOT NECESSARILY STAGE COORDINATES. InView writes them
    # in its own frame: on 2026-08-05 every Results.CSV came back offset from the
    # live stage readback by a rock-steady (+44162.3, +21.2) um across four
    # samples. Taken at face value they put every measurement ~44 mm outside the
    # search area, which is invisible in the printout (the array's own position
    # is recorded separately and looked right) but blinds the GP completely —
    # the surrogate had no observation anywhere near the sites it was ranking,
    # so EI went flat and every proposal read "predicted objective 0.00+-0.55".
    # Coordinates are therefore only believed when they land near where the
    # stage says the array actually is.
    pos0 = (run_record or {}).get("position") or (None, None)
    span = max(float(sp or 0.0) * max(int(nx), int(ny)), 50.0) + 250.0

    def _plausible(X, Y) -> bool:
        if pos0[0] is None:
            return True                     # nothing to check against
        dx = float(pd.to_numeric(X, errors="coerce").dropna().median()) - float(pos0[0])
        dy = float(pd.to_numeric(Y, errors="coerce").dropna().median()) - float(pos0[1])
        if math.hypot(dx, dy) <= span:
            return True
        print(f"  coords      : IGNORING the export's X/Y — they sit "
              f"({dx:+.0f}, {dy:+.0f}) um from where the stage says this array is, "
              f"so they are in InView's own frame, not stage coordinates. "
              "Falling back to the planned grid.")
        return False

    have = ("X (um)" in out.columns and "Y (um)" in out.columns
            and _num("X (um)").notna().any() and _num("Y (um)").notna().any()
            and _plausible(out["X (um)"], out["Y (um)"]))
    if not have:
        cx, cy = _find("x"), _find("y")
        if cx is not None and cy is not None:
            X, Y = _num(cx), _num(cy)
            if X.notna().any() and Y.notna().any() and _plausible(X, Y):
                out["X (um)"], out["Y (um)"] = X, Y
                have = True
                print(f"  coords      : taken from {cx!r}/{cy!r}")
    if not have:
        pos = (run_record or {}).get("position") or (0.0, 0.0)
        px, py = float(pos[0]), float(pos[1])
        _nx, _ny, _sp = max(int(nx), 1), max(int(ny), 1), float(sp or 0.0)
        theta = math.radians(float(rotation_deg or 0.0))
        ct, st = math.cos(theta), math.sin(theta)
        xy = []
        for i0 in range(len(out)):
            r_, c_ = i0 // _nx, i0 % _nx
            ux = (c_ - (_nx - 1) / 2.0) * _sp
            uy = (r_ - (_ny - 1) / 2.0) * _sp
            xy.append((px + ux * ct - uy * st,
                       py + ux * st + uy * ct))
        out["X (um)"] = [p[0] for p in xy]
        out["Y (um)"] = [p[1] for p in xy]
        print(f"  coords      : filled from the array centre ({px:.1f}, {py:.1f}) + "
              f"{_nx}x{_ny} @ {_sp:g} um — the export carried no per-indent X/Y")
    return out


def combined_rows(executions: list[dict]) -> pd.DataFrame:
    """Stack every execution's rows with GUARANTEED coordinates.
    Returns a TYPED empty frame when there are no rows."""
    frames = []
    for ex in executions or []:
        rows = ex.get("rows")
        if rows is None or getattr(rows, "empty", True):
            continue
        d = ex["indent_data"]
        rr = ex.get("run_record") or {"position": d.get("position")}
        df = ensure_xy(rows, rr, d.get("grid_nx", 1), d.get("grid_ny", 1),
                       d.get("spacing_um", 0.0),
                       d.get("rotation_deg", 0.0)).copy()
        df["region"] = d.get("area_name") or d.get("test_name")
        df["array"] = d.get("test_name")
        if "Indent" not in df.columns:
            df["Indent"] = range(1, len(df) + 1)
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=CANONICAL_COLS[:8] + ["region", "array"])


# ---------------------------------------------------------------- plotting
def plot_load_depth_curves(csv_result, *, title=None, max_curves=25, overlay=True):
    """Plot LOAD vs DEPTH for every downloaded _Test<N> curve."""
    import matplotlib.pyplot as plt

    curves = (csv_result or {}).get("curves") or {}
    if not curves:
        print("  no per-indent curve files were downloaded — nothing to plot.")
        return None
    keys = sorted(curves)[:max_curves]
    name = title or f"Load–depth — {(csv_result or {}).get('sample_name', '')}"

    if overlay:
        fig, ax = plt.subplots(figsize=(7, 5))
        for n in keys:
            c = curves[n]
            d = pd.to_numeric(c.get("Contact Depth (nm)"), errors="coerce")
            ld = pd.to_numeric(c.get("Load (mN)"), errors="coerce")
            m = d.notna() & ld.notna()
            if m.any():
                ax.plot(d[m], ld[m], lw=1.0, alpha=0.85, label=f"indent {n}")
        ax.set_xlabel("Depth (nm)"); ax.set_ylabel("Load (mN)")
        ax.set_title(name); ax.grid(alpha=0.3)
        if len(keys) <= 12:
            ax.legend(fontsize=8, ncol=2)
        plt.tight_layout(); plt.show()
        return fig

    ncol = min(4, len(keys))
    nrow = int(math.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, n in zip(flat, keys):
        c = curves[n]
        d = pd.to_numeric(c.get("Contact Depth (nm)"), errors="coerce")
        ld = pd.to_numeric(c.get("Load (mN)"), errors="coerce")
        m = d.notna() & ld.notna()
        ax.plot(d[m], ld[m], lw=1.0)
        ax.set_title(f"indent {n}", fontsize=9); ax.grid(alpha=0.3)
    for ax in flat[len(keys):]:
        ax.axis("off")
    fig.suptitle(name); fig.supxlabel("Depth (nm)"); fig.supylabel("Load (mN)")
    plt.tight_layout(); plt.show()
    return fig


# ------------------------------------------------------------- finalization
def finalize_indent_data(run_record, csv_result=None, *,
                         screen_config: Optional[ScreenReadConfig] = None,
                         plot_curves: bool = True, prefer: str = "csv") -> dict:
    """Choose the primary source, average H/E, and build ``indent_data``.

    Precedence (each candidate must pass validate_rows to win):
        1. downloaded per-indent table                  -> "result_file_download"
           (_Results.CSV, or the workbook's "Results" sheet)
        2. per-indent H/E reduced from the curves       -> "result_curve_reduction"
           (_Test<N>.CSV files, or the workbook's "Test <n>" sheets)
        3. live screen readings                         -> "screen_read_fallback"

    Tolerates run_records recovered from the journal after a kernel restart
    (recorder is None -> the screen candidate is an empty table).
    """
    scfg = screen_config or ScreenReadConfig()
    rec = run_record.get("recorder")
    nx, ny, sp = run_record["grid_nx"], run_record["grid_ny"], run_record["spacing_um"]
    h_range = scfg.hardness_valid_range

    candidates = []
    if csv_result:
        candidates.append(("result_file_download", csv_result.get("summary_rows"),
                           csv_result.get("summary_name")))
        if csv_result.get("curves"):
            candidates.append(("result_curve_reduction",
                               curves_to_per_indent(csv_result["curves"], run_record,
                                                    h_range=h_range),
                               f"{len(csv_result['curves'])} curve file(s)"))
    screen_rows = rec.to_result_rows() if rec is not None \
        else pd.DataFrame(columns=CANONICAL_COLS)
    screen_path = str(rec.csv_path) if rec is not None \
        else run_record.get("screen_csv", "(recorder lost — kernel restarted)")
    candidates.append(("screen_read_fallback", screen_rows, screen_path))
    if prefer == "screen":
        candidates.reverse()

    rows, data_source, data_path = None, None, None
    print(f"FINALIZE '{run_record['sample_name']}' — choosing the primary source")
    for src, cand, path in candidates:
        good, why = validate_rows(cand, h_range, expected_indents=nx * ny)
        print(f"  {'USE ' if good and rows is None else 'skip'} {src:<24} {why}")
        if good and rows is None:
            rows, data_source, data_path = cand, src, path
    if rows is None:
        rows, data_source, data_path = screen_rows, "screen_read_fallback", screen_path
        print("  >> nothing validated — falling back to the screen table (may be empty).")
    print(f"  PRIMARY     : {data_source}  ({data_path})")

    # ---- cross-check the two independent measurements of the same indents ----
    cross = {}
    csv_summary = (csv_result or {}).get("summary_rows")
    if (csv_summary is not None and not getattr(csv_summary, "empty", True)
            and not screen_rows.empty):
        a = average_hardness_modulus(csv_summary, h_range=h_range, verbose=False)
        b = average_hardness_modulus(screen_rows, h_range=h_range, verbose=False)
        if (np.isfinite(a["hardness_mean"]) and np.isfinite(b["hardness_mean"])
                and b["hardness_mean"]):
            dh = abs(a["hardness_mean"] - b["hardness_mean"]) / b["hardness_mean"] * 100.0
            cross = {"csv_H": a["hardness_mean"], "screen_H": b["hardness_mean"],
                     "H_diff_pct": dh, "csv_n": a["n_valid"], "screen_n": b["n_valid"]}
            print(f"  cross-check : CSV H {a['hardness_mean']:.2f} vs screen H "
                  f"{b['hardness_mean']:.2f} GPa ({dh:.1f}% apart)")
            if dh > 25.0:
                print("  >> WARNING: the two sources disagree by >25%. Check export "
                      "units and the screen metric keys before trusting this round.")

    rotation = float(run_record.get("rotation_deg", 0.0))
    rows = ensure_xy(rows, run_record, nx, ny, sp, rotation)
    stats = average_hardness_modulus(rows, h_range=h_range, label=data_source)

    if plot_curves and csv_result:
        plot_load_depth_curves(
            csv_result, title=f"Load–depth — {run_record.get('label', '')} "
                              f"({run_record['sample_name']})")

    indent_data = {
        "test_name": run_record["sample_name"],
        "sample_name": run_record["sample_name"],
        # n_indents comes from the PLAN, never from the file: budget accounting
        # must not depend on the shape of whatever table the export produced.
        "n_indents": int(nx) * int(ny), "n_result_rows": int(len(rows)),
        "grid_nx": nx, "grid_ny": ny, "spacing_um": sp,
        "rotation_deg": rotation,
        "method_type": run_record.get("method_type", "eh"),
        "hardness_mean": stats["hardness_mean"], "hardness_std": stats["hardness_std"],
        "modulus_mean": stats["modulus_mean"], "modulus_std": stats["modulus_std"],
        "valid_indent_count": int(stats["n_valid"]),
        "data_path": data_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "position": run_record["position"],
        "area_name": run_record.get("label", run_record["sample_name"]),
        "results": rows.to_dict("records"),
        "data_source": data_source, "screen_csv": screen_path,
        "screen_valid_count": len(rec.records) if rec is not None else 0,
        "n_screen_samples": rec.n_samples if rec is not None else 0,
        "needs_recovery": data_source == "screen_read_fallback",
        "monitor_lost_connection": run_record.get("monitor_lost_connection", False),
        "monitor_done_reason": run_record.get("done_reason"),
        "export_name": run_record["export_name"],
        "hardness_stats": stats, "cross_check": cross,
    }
    if indent_data["needs_recovery"]:
        print(f"  NOTE: the real result file was never used. The indents ARE on the "
              f"instrument as '{run_record['sample_name']}' — recover later with "
              "runner.recover_pending() (no new indents).")
    return {"indent_data": indent_data, "rows": rows,
            "image": run_record.get("image"),
            "tool_log": run_record.get("tool_log", []),
            "monitor_status": run_record.get("monitor_status", {}),
            "recorder": rec, "csv_result": csv_result, "run_record": run_record}
