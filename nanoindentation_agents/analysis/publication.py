"""
publication.py — the journal-figure set for one autonomous campaign.

Ten figures, one house style, generated in a single call once the rounds are
finished. Everything is rebuilt from artefacts that survive a kernel restart —
the campaign journal, the downloaded InView CSVs, the saved frames — so a run can
be re-plotted months later without the notebook session that produced it:

    figs = CampaignFigures.from_journal(JOURNAL_DIR, "0806_131828",
                                        download_dir=DOWNLOAD_DIR)
    figs.generate_all(REPO_ROOT / "paper" / "figures")

A figure whose inputs are missing returns None and says so in the manifest. It
never invents data and never half-draws a panel: an empty axis in a paper is a
claim that something was measured and found empty.


COLOUR IS COMPUTED, NOT PICKED
------------------------------
Every categorical hue below passed the six checks (OKLab ΔE×100, Machado-
Oliveira-Fernandes 2009 CVD simulation at severity 1.0) against a WHITE surface,
because these figures are printed:

  categorical, in this fixed order   worst adjacent CVD ΔE = 15.8  (target >= 8)
    #009E73  #0072B2  #D55E00  #CC79A7
    worst normal-vision ΔE = 25.4 (floor 15); contrast 3.06-5.19:1 (floor 3)
    All-pairs (scatter) is a harder test and caps the count: the first THREE
    slots clear it at ΔE 11.0, so no scatter here carries more than three
    categorical series.

  ordinal round ramp (6 discrete steps, Blues 0.45-0.98)
    monotone L, min adjacent ΔL = 0.077 (floor 0.06), lightest step 2.13:1
    (floor 2.0). Rounds are ORDINAL — swapping round 2 and round 5 would change
    the meaning — so they take a one-hue ramp, never categorical hues.

Magnitude uses one hue light->dark with a colourbar; uncertainty uses a DIFFERENT
single hue so a posterior mean and its sigma cannot be mistaken for each other;
signed error uses two opposed hues through a neutral grey. No dual axes anywhere:
two measures of different scale get two panels.
"""
from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- palette
CAT = ("#009E73", "#0072B2", "#D55E00", "#CC79A7")   # validated order — do not reorder
ROUND_STEPS = ("#7fb9da", "#58a1cf", "#3888c1", "#1d6cb1", "#08519c", "#083573")
INK, MUTED, FAINT = "#1a1a1a", "#5c5c5c", "#c9c9c9"
GOOD, BAD = "#009E73", "#D55E00"                     # status only, never a series

SINGLE_COL, ONE_HALF_COL, DOUBLE_COL = 3.35, 5.20, 6.85   # inches (85/132/174 mm)

_H, _E = "Hardness (GPa)", "Modulus (GPa)"
_X, _Y = "X (um)", "Y (um)"
_DEPTH, _LOAD = "Contact Depth (nm)", "Load (mN)"


def _cmap(name: str, lo: float = 0.12, hi: float = 0.95):
    """A single-hue ramp truncated away from white, so the light end stays visible."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colormaps
    base = colormaps[name]
    return LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(lo, hi, 256)))


def _diverging():
    """Two opposed hues through a NEUTRAL grey — the midpoint must read as zero."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_grey_orange", ["#0072B2", "#8fb8d2", "#eeeeec", "#e9b183", "#D55E00"])


# ----------------------------------------------------------------------- style
@dataclass
class PubStyle:
    """Journal house style. One place, so every figure reads as one system."""
    base_pt: float = 8.0
    dpi: int = 600
    family: tuple = ("Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans")

    def rc(self) -> dict:
        p = self.base_pt
        return {
            "font.family": "sans-serif",
            "font.sans-serif": list(self.family),
            "font.size": p,
            "axes.labelsize": p, "axes.titlesize": p + 1,
            "xtick.labelsize": p - 1, "ytick.labelsize": p - 1,
            "legend.fontsize": p - 1, "figure.titlesize": p + 1,
            "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
            "axes.linewidth": 0.6, "axes.titlepad": 4.0, "axes.labelpad": 2.5,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": False, "grid.color": FAINT, "grid.linewidth": 0.4,
            "lines.linewidth": 1.2, "lines.markersize": 3.5,
            "lines.markeredgewidth": 0.6,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 2.8, "ytick.major.size": 2.8,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
            "legend.frameon": False, "legend.handlelength": 1.6,
            "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
            "figure.dpi": 120, "savefig.dpi": self.dpi,
            "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            # TrueType, not Type-3: journals reject Type-3, and editors need the
            # text to stay selectable in Illustrator.
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        }

    @contextmanager
    def applied(self):
        import matplotlib.pyplot as plt
        with plt.rc_context(self.rc()):
            yield


def _panel_label(ax, text: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left")


def _finish(ax, *, xlabel="", ylabel="", title="", ygrid=False) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=FAINT, linewidth=0.4)


def _round_colors(n: int) -> list:
    """Ordinal steps for n series, spread across the FULL validated ramp.

    Taking the first n steps would crowd every series into the light end, where
    the lightness separation that makes the ramp readable is smallest.
    """
    n = max(int(n), 1)
    if n == 1:
        return [ROUND_STEPS[3]]
    if n <= len(ROUND_STEPS):
        idx = np.linspace(0, len(ROUND_STEPS) - 1, n).round().astype(int)
        return [ROUND_STEPS[i] for i in idx]
    cm = _cmap("Blues", 0.35, 0.95)
    return [cm(x) for x in np.linspace(0.0, 1.0, n)]


# ------------------------------------------------------------------ data layer
@dataclass
class CampaignData:
    """Everything the figures read, gathered once from durable artefacts."""
    run_id: str = ""
    objective_text: str = ""
    all_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: list = field(default_factory=list)      # state["indent_history"]
    decisions: list = field(default_factory=list)    # state["orchestrator_decisions"]
    events: list = field(default_factory=list)       # raw journal records
    curves: dict = field(default_factory=dict)       # sample -> {indent: DataFrame}
    frames: dict = field(default_factory=dict)       # sample -> png path
    surrogate: Any = None
    progress: Any = None
    min_improvement_pct: float = float("nan")
    um_per_px: float = float("nan")
    arrival_tol_um: float = float("nan")

    # -------------------------------------------------------------- accessors
    def ev(self, name: str) -> list:
        return [e for e in self.events if e.get("event") == name]

    @property
    def samples(self) -> list:
        """Sample names in execution order — 'Test 1..N' in every figure."""
        seen = [d.get("sample_name") for d in self.history if d.get("sample_name")]
        if seen:
            return list(dict.fromkeys(seen))
        return list(dict.fromkeys(e.get("sample") for e in self.ev("RESULT_RECORDED")))

    def round_of(self, sample: str) -> Optional[int]:
        for e in self.ev("RESULT_RECORDED"):
            if e.get("sample") == sample and e.get("round") is not None:
                return int(e["round"])
        try:
            return self.samples.index(sample) + 1
        except ValueError:
            return None

    def label_of(self, sample: str) -> str:
        i = self.samples.index(sample) + 1 if sample in self.samples else 0
        return f"Test {i}"


class CampaignFigures:
    """Build the paper's figure set from one campaign."""

    def __init__(self, data: CampaignData, style: Optional[PubStyle] = None):
        self.d = data
        self.style = style or PubStyle()
        # Figure 1a, opt-in: set these (or pass them to fig01_overview) to put a
        # real micrograph beside the site map. See fig01_overview's docstring.
        self.micrograph: Optional[Path] = None
        self.micrograph_crop: Optional[tuple] = None

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_runner(cls, runner, *, state=None, surrogate=None, progress=None,
                    style: Optional[PubStyle] = None) -> "CampaignFigures":
        """The live path — everything is already in memory."""
        state = state if state is not None else runner.state
        d = CampaignData(
            run_id=getattr(runner.journal, "run_id", ""),
            objective_text=str(state.get("research_question") or ""),
            all_rows=runner.results_rows(),
            history=list(state.get("indent_history") or []),
            decisions=list(state.get("orchestrator_decisions") or []),
            events=list(runner.journal.events()),
            surrogate=surrogate, progress=progress,
            min_improvement_pct=float(
                getattr(runner.objective.success, "min_improvement_pct", float("nan"))),
            arrival_tol_um=float(getattr(runner.c, "arrival_tol_um", float("nan"))),
        )
        for ex in runner.executions:
            name = ex["indent_data"].get("sample_name")
            cr = (ex.get("csv_result") or {}).get("curves")
            if name and cr:
                d.curves[name] = dict(cr)
        cls._fill_from_disk(d, Path(runner.session.download_dir))
        return cls(d, style)

    @classmethod
    def from_journal(cls, journal_dir, run_id: str, *, download_dir,
                     objective=None, style: Optional[PubStyle] = None) -> "CampaignFigures":
        """The durable path — after a kernel restart, or months later."""
        from ..campaign import CampaignJournal
        from ..instrument import results as R

        j = CampaignJournal(journal_dir, run_id)
        events = list(j.events())
        history = j.recorded_results()
        start = next(iter(j.events("CAMPAIGN_START")), {})
        d = CampaignData(
            run_id=run_id, objective_text=str(start.get("objective") or ""),
            history=history, events=events,
            all_rows=R.combined_rows([
                {"indent_data": h, "rows": pd.DataFrame(h.get("results") or []),
                 "run_record": {"position": h.get("position")}} for h in history]),
            min_improvement_pct=float(
                getattr(getattr(objective, "success", None), "min_improvement_pct",
                        float("nan"))),
        )
        cls._fill_from_disk(d, Path(download_dir))
        # Refit the surrogate from the recorded indents so the posterior panel
        # survives a kernel restart too. Deterministic compile — no LLM, so the
        # objective is reconstructed identically every time this is re-run.
        if objective is None and d.objective_text:
            try:
                from ..objective import compile_objective
                objective = compile_objective(d.objective_text, use_llm=False)
                d.min_improvement_pct = float(objective.success.min_improvement_pct)
            except Exception:
                objective = None
        if objective is not None and d.history:
            # The search area MUST come from the journal. A freshly compiled
            # objective carries the placeholder (0,100,0,75) relative bounds,
            # while recorded sites are absolute stage coordinates near
            # (15400, 58900) — fitting against the placeholder puts the whole
            # posterior in a corner of an axis 60 mm wide.
            area = next(iter(d.ev("SEARCH_AREA")), {})
            bounds = area.get("fov_bounds")
            if bounds and len(bounds) == 4:
                objective.constraints.fov_bounds = tuple(float(b) for b in bounds)
            try:
                from .objective_acquisition import ObjectiveSurrogate
                d.surrogate = ObjectiveSurrogate(objective).fit(d.history)
            except Exception:
                d.surrogate = None
        pf = next(iter(d.ev("PREFLIGHT")), {})
        d.arrival_tol_um = float(pf.get("arrival_tol_um") or float("nan"))
        return cls(d, style)

    @staticmethod
    def _fill_from_disk(d: CampaignData, download_dir: Path) -> None:
        """Curves and frames that the session may no longer hold in memory.

        EVERY glob is scoped to this run's id. Sample names restart at test_01 in
        every campaign and the download folder is shared, so an unscoped
        `test_01_*` matches four other days' curves — a figure that silently
        mixes runs is worse than a figure that is missing.
        """
        from ..instrument import results as R

        area = next(iter(d.ev("SEARCH_AREA")), {})
        d.um_per_px = float(area.get("um_per_px") or float("nan"))
        if not d.run_id:
            return
        for sample in d.samples:
            if sample not in d.curves:
                found = {}
                for p in sorted(download_dir.glob(f"{sample}_{d.run_id}_*_Test*.CSV")):
                    m = re.search(r"_Test(\d+)\.CSV$", p.name, re.I)
                    if not m:
                        continue
                    try:
                        found[int(m.group(1))] = R.normalize_result_columns(
                            R.read_inview_table(p))
                    except Exception:
                        continue
                # An Excel campaign has no _Test<n> files at all: the curves are
                # "Test <n>" SHEETS inside one <sample>_<run>_export workbook,
                # so globbing only for CSVs loses every curve figure.
                for ext in R.EXCEL_EXTS:
                    for p in sorted(download_dir.glob(f"{sample}_{d.run_id}_*{ext}")):
                        try:
                            _summary, sheets = R.read_inview_workbook(p)
                        except Exception:
                            continue
                        for n, tbl in sheets.items():
                            found.setdefault(n, R.normalize_result_columns(tbl))
                if found:
                    d.curves[sample] = found
            adir = download_dir / "arrival"
            hits = sorted(adir.glob(f"{d.run_id}_{sample}_arrive*.png")) \
                if adir.is_dir() else []
            if hits:
                d.frames[sample] = hits[0]

    # ------------------------------------------------------------------ helpers
    def _per_test(self) -> pd.DataFrame:
        """One row per test: round, position, H/E mean and scatter, provenance."""
        rows = []
        for h in self.d.history:
            name = h.get("sample_name")
            pos = h.get("position") or (np.nan, np.nan)
            g = self.d.all_rows[self.d.all_rows["array"] == name] \
                if len(self.d.all_rows) else pd.DataFrame()
            n = int(len(g))
            rows.append({
                "sample": name, "label": self.d.label_of(name),
                "round": self.d.round_of(name),
                "x_um": float(pos[0]), "y_um": float(pos[1]),
                "n": n,
                "H": float(h.get("hardness_mean", np.nan)),
                "H_sd": float(g[_H].std(ddof=1)) if n > 1 else np.nan,
                "E": float(h.get("modulus_mean", np.nan)),
                "E_sd": float(g[_E].std(ddof=1)) if n > 1 else np.nan,
                "source": h.get("data_source", "unknown"),
                "validated": h.get("data_source") == "result_file_download",
            })
        df = pd.DataFrame(rows)
        return df.sort_values("round").reset_index(drop=True) if len(df) else df

    def _fig(self, w: float, h: float, **kw):
        import matplotlib.pyplot as plt
        return plt.subplots(figsize=(w, h), **kw)

    # =====================================================================
    # FIGURE 1 — where the campaign looked, and what it found there
    # =====================================================================
    def fig01_overview(self, *, micrograph=None, crop=None):
        """Sites measured, optionally beside a micrograph.

        `micrograph` is OPT-IN and there is no default, deliberately. The frames
        this pipeline saves are screenshots of the whole InView window — toolbar,
        focus widget, InView's own scale bar and its "Single Engage" banner all
        included. Publishing one as "the sample surface" would put instrument
        chrome in a journal figure, so a micrograph appears only when you pass a
        real one, with `crop=(left, top, right, bottom)` in pixels if the frame
        still needs trimming to the video panel.
        """
        import matplotlib.pyplot as plt
        t = self._per_test()
        if not len(t):
            return None
        micrograph = micrograph or self.micrograph
        crop = crop or self.micrograph_crop
        frame = Path(micrograph) if micrograph else None
        fig, axes = plt.subplots(
            1, 2 if frame else 1, figsize=(DOUBLE_COL if frame else SINGLE_COL, 2.7),
            gridspec_kw={"width_ratios": [1, 1.15]} if frame else None)
        axes = np.atleast_1d(axes)

        if frame:
            ax = axes[0]
            img = plt.imread(frame)
            if crop:
                l, tp, r, b = (int(v) for v in crop)
                img = img[tp:b, l:r]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True), sp.set_linewidth(0.6)
            if np.isfinite(self.d.um_per_px) and self.d.um_per_px > 0:
                w_px = 20.0 / self.d.um_per_px             # 20 um scale bar
                h, w = img.shape[:2]
                x0, y0 = 0.06 * w, 0.94 * h
                ax.plot([x0, x0 + w_px], [y0, y0], color="white", lw=2.4,
                        solid_capstyle="butt")
                ax.text(x0 + w_px / 2, y0 - 0.02 * h, "20 µm", color="white",
                        ha="center", va="bottom", fontsize=self.style.base_pt - 1)
            _panel_label(ax, "a", dx=0.0, dy=1.06)
            ax.set_title("Sample surface", loc="left", pad=7)

        ax = axes[-1]
        sc = ax.scatter(t["x_um"], t["y_um"], c=t["H"], cmap=_cmap("Blues"),
                        s=52, edgecolor=INK, linewidth=0.6, zorder=3)
        for _, r in t.iterrows():
            ax.annotate(f"{int(r['round'])}", (r["x_um"], r["y_um"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=self.style.base_pt - 1.5, color=MUTED)
        ax.plot(t["x_um"], t["y_um"], color=MUTED, lw=0.6, ls=":", zorder=2)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("hardness (GPa)")
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(width=0.6, length=2.5)
        _finish(ax, xlabel="stage x (µm)", ylabel="stage y (µm)",
                title="Measured sites (labelled by round)")
        if frame:
            _panel_label(ax, "b", dx=-0.22, dy=1.06)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 2 — the measured property of every test (two panels, never two axes)
    # =====================================================================
    def fig02_properties_by_test(self):
        import matplotlib.pyplot as plt
        t = self._per_test()
        if not len(t):
            return None
        fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.5), sharex=True)
        best = t["H"].idxmax()
        for ax, key, sd, lab in ((axes[0], "H", "H_sd", "hardness (GPa)"),
                                 (axes[1], "E", "E_sd", "modulus (GPa)")):
            ax.plot(t["round"], t[key], color=MUTED, lw=0.8, zorder=1)
            ax.errorbar(t["round"], t[key], yerr=t[sd], fmt="o", color=CAT[1],
                        ecolor=CAT[1], elinewidth=0.8, capsize=2, capthick=0.8,
                        markeredgecolor="white", zorder=3)
            ax.scatter([t.loc[best, "round"]], [t.loc[best, key]], s=46,
                       facecolor=CAT[0], edgecolor=INK, linewidth=0.7, zorder=4)
            _finish(ax, ylabel=lab, ygrid=True)
        axes[0].annotate("best site", (t.loc[best, "round"], t.loc[best, "H"]),
                         textcoords="offset points", xytext=(6, 4), color=CAT[0],
                         fontsize=self.style.base_pt - 1)
        _panel_label(axes[0], "a"), _panel_label(axes[1], "b")
        axes[1].set_xlabel("campaign round")
        axes[1].set_xticks(t["round"].tolist())
        if t["n"].max() <= 1:
            # Below the axis, not over the data — a note that overlaps a point is
            # a note that hides a measurement.
            axes[1].text(0.0, -0.42, "one indent per site: no within-site scatter",
                         transform=axes[1].transAxes, color=MUTED,
                         fontsize=self.style.base_pt - 1.5)
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 3 — the raw evidence: load-depth curves
    # =====================================================================
    def fig03_load_depth(self):
        import matplotlib.pyplot as plt
        if not self.d.curves:
            return None
        samples = [s for s in self.d.samples if s in self.d.curves]
        cols = _round_colors(len(samples))
        fig, ax = self._fig(SINGLE_COL, 2.6)
        for c, s in zip(cols, samples):
            for k, (idx, df) in enumerate(sorted(self.d.curves[s].items())):
                if _DEPTH not in df or _LOAD not in df:
                    continue
                ax.plot(df[_DEPTH], df[_LOAD], color=c, lw=0.9, alpha=0.95,
                        label=self.d.label_of(s) if k == 0 else None)
        if not ax.lines:
            plt.close(fig)
            return None
        _finish(ax, xlabel="displacement into surface (nm)", ylabel="load (mN)")
        ax.set_xlim(left=0), ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", ncol=1 if len(samples) <= 5 else 2)
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 4 — depth-resolved H and E (the indentation size effect)
    # =====================================================================
    def fig04_depth_profiles(self, depth_min_nm: float = 50.0):
        import matplotlib.pyplot as plt
        if not self.d.curves:
            return None
        samples = [s for s in self.d.samples if s in self.d.curves]
        cols = _round_colors(len(samples))
        fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.5), sharex=True)
        drew = False
        for c, s in zip(cols, samples):
            for k, (_, df) in enumerate(sorted(self.d.curves[s].items())):
                if _DEPTH not in df:
                    continue
                m = df[_DEPTH] >= depth_min_nm
                for ax, col in ((axes[0], _H), (axes[1], _E)):
                    if col in df:
                        ax.plot(df.loc[m, _DEPTH], df.loc[m, col], color=c, lw=0.9,
                                label=self.d.label_of(s) if k == 0 and col == _H else None)
                        drew = True
        if not drew:
            plt.close(fig)
            return None
        _finish(axes[0], ylabel="hardness (GPa)")
        _finish(axes[1], xlabel="displacement into surface (nm)",
                ylabel="modulus (GPa)")
        axes[0].legend(loc="upper right", ncol=2, fontsize=self.style.base_pt - 2)
        _panel_label(axes[0], "a"), _panel_label(axes[1], "b")
        axes[1].set_xlim(left=depth_min_nm)   # the cut is visible in the axis itself
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 5 — what the agent believed: GP posterior and its uncertainty
    # =====================================================================
    def fig05_gp_posterior(self):
        import matplotlib.pyplot as plt
        s = self.d.surrogate
        if s is None or getattr(s, "n_points", 0) == 0:
            return None
        maps = s.posterior_maps(n=70)
        # A posterior drawn over the wrong extent still LOOKS like a result: the
        # sites land in one corner and the map reads as a blank field. If the
        # observations are not inside the plotted area, the search bounds and the
        # data disagree — refuse the panel rather than publish a pretty artefact.
        x0, x1, y0, y1 = maps["extent"]
        span = max(x1 - x0, y1 - y0)
        inside = ((s.X[:, 0] >= x0 - 0.05 * span) & (s.X[:, 0] <= x1 + 0.05 * span)
                  & (s.X[:, 1] >= y0 - 0.05 * span) & (s.X[:, 1] <= y1 + 0.05 * span))
        if not inside.all():
            print("      fig05: sites fall outside the surrogate's search area "
                  f"({int((~inside).sum())}/{len(inside)}) — skipping rather than "
                  "drawing a posterior over the wrong extent.")
            return None
        best = s.best_observed()
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
        for ax, key, lab, cm in (
                (axes[0], "scalar_mean", "objective score (a.u.)", _cmap("Blues")),
                (axes[1], "scalar_sigma", "posterior σ (a.u.)", _cmap("Purples"))):
            im = ax.imshow(maps[key], origin="lower", extent=maps["extent"],
                           aspect="equal", cmap=cm)
            ax.scatter(s.X[:, 0], s.X[:, 1], s=16, c="white", edgecolor=INK,
                       linewidths=0.5, zorder=3)
            if best:
                ax.scatter([best["x_um"]], [best["y_um"]], marker="*", s=90,
                           c=GOOD, edgecolor=INK, linewidths=0.5, zorder=4)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label(lab)
            cb.outline.set_linewidth(0.6)
            cb.ax.tick_params(width=0.6, length=2.5)
            _finish(ax, xlabel="stage x (µm)", ylabel="stage y (µm)")
        axes[0].set_title("Predicted objective", loc="left")
        axes[1].set_title("Model uncertainty", loc="left")
        _panel_label(axes[0], "a"), _panel_label(axes[1], "b")
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 6 — did it learn? improvement over the baseline, round by round
    # =====================================================================
    def fig06_learning_curve(self):
        import matplotlib.pyplot as plt
        rounds = [(int(d["round"]), float(d.get("improvement_pct") or 0.0))
                  for d in self.d.decisions if d.get("round") is not None]
        if not rounds:
            rounds = [(int(e["round"]), float(e.get("improvement_pct") or 0.0))
                      for e in self.d.ev("ROUND_DECISION") if e.get("round") is not None]
        # The terminal decision ("objective_met") is logged as one round past the
        # last measurement. Plotting it adds a round in which nothing was
        # measured and the curve merely repeats itself.
        executed = [int(e["round"]) for e in self.d.ev("ROUND_DONE")
                    if e.get("round") is not None]
        last = max(executed) if executed else None
        rounds = sorted({r: v for r, v in rounds
                         if last is None or r <= last}.items())
        if len(rounds) < 2:
            return None
        x = [r for r, _ in rounds]
        y = np.maximum.accumulate([v for _, v in rounds])   # best-so-far
        fig, ax = self._fig(SINGLE_COL, 2.3)
        ax.plot(x, y, color=CAT[1], marker="o", markeredgecolor="white", zorder=3)
        thr = self.d.min_improvement_pct
        if np.isfinite(thr):
            ax.axhline(thr, color=MUTED, lw=0.8, ls="--", zorder=1)
            ax.text(x[0], thr, f" success threshold {thr:g}%", va="bottom",
                    ha="left", color=MUTED, fontsize=self.style.base_pt - 1.5)
            hit = next((r for r, v in zip(x, y) if v >= thr), None)
            if hit is not None:
                ax.scatter([hit], [dict(zip(x, y))[hit]], s=46, facecolor=GOOD,
                           edgecolor=INK, linewidth=0.7, zorder=4)
                ax.annotate(f"met at round {hit}", (hit, dict(zip(x, y))[hit]),
                            textcoords="offset points", xytext=(4, -10),
                            color=GOOD, fontsize=self.style.base_pt - 1)
        _finish(ax, xlabel="campaign round",
                ylabel="best site vs baseline (%)", ygrid=True)
        ax.set_xticks(x)
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 7 — the acquisition function collapsing as the model saturates
    # =====================================================================
    def fig07_acquisition(self):
        import matplotlib.pyplot as plt
        src = self.d.decisions or self.d.ev("ROUND_DECISION")   # journal carries it too
        pts = [(int(d["round"]), float(d.get("max_ei") or 0.0))
               for d in src
               if d.get("round") is not None and d.get("max_ei") is not None]
        pts = sorted(dict(pts).items())
        if len(pts) < 2:
            return None
        x, y = zip(*pts)
        fig, ax = self._fig(SINGLE_COL, 2.3)
        ax.bar(x, y, width=0.62, color=CAT[1], edgecolor="white", linewidth=0.8)
        _finish(ax, xlabel="campaign round", ylabel="max expected improvement",
                ygrid=True)
        ax.set_xticks(list(x))
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 8 — hardness against modulus, every indent, coloured by round
    # =====================================================================
    def fig08_hardness_vs_modulus(self):
        import matplotlib.pyplot as plt
        df = self.d.all_rows
        if not len(df) or _H not in df or _E not in df:
            return None
        rounds = df["array"].map(lambda s: self.d.round_of(s) or 0)
        fig, ax = self._fig(SINGLE_COL, 2.6)
        sc = ax.scatter(df[_E], df[_H], c=rounds, cmap=_cmap("Blues", 0.30, 0.95),
                        s=26, edgecolor=INK, linewidth=0.4, zorder=3)
        if len(df) >= 3 and df[_E].std() > 0:
            k = np.polyfit(df[_E].astype(float), df[_H].astype(float), 1)
            xs = np.linspace(df[_E].min(), df[_E].max(), 50)
            r = float(np.corrcoef(df[_E].astype(float), df[_H].astype(float))[0, 1])
            ax.plot(xs, np.polyval(k, xs), color=MUTED, lw=0.9, ls="--", zorder=2)
            ax.text(0.03, 0.92, f"r = {r:.2f}", transform=ax.transAxes, color=MUTED)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("campaign round")
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(width=0.6, length=2.5)
        _finish(ax, xlabel="modulus (GPa)", ylabel="hardness (GPa)")
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 9 — can it hit what it aims at? (the autonomy claim, measured)
    # =====================================================================
    def fig09_positioning(self):
        import matplotlib.pyplot as plt
        pos = [(int(e["round"]), float(e.get("error_um") or np.nan))
               for e in self.d.ev("POSITIONED") if e.get("round") is not None]
        arr = [(int(e["round"]), float(e.get("error_um") or np.nan))
               for e in self.d.ev("ARRIVAL_VERIFIED") if e.get("round") is not None]
        unver = sorted({int(e["round"]) for e in self.d.ev("ARRIVAL_UNVERIFIED")
                        if e.get("round") is not None})
        if not pos:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(ONE_HALF_COL, 2.3))
        px, py = zip(*sorted(dict(pos).items()))
        axes[0].bar(px, py, width=0.62, color=CAT[1], edgecolor="white", linewidth=0.8)
        _finish(axes[0], xlabel="round", ylabel="stage error (µm)", ygrid=True)
        axes[0].set_xticks(list(px))
        axes[0].set_title("Commanded vs achieved", loc="left")

        if arr:
            ax, ay = zip(*sorted(dict(arr).items()))
            axes[1].bar(ax, ay, width=0.62, color=CAT[1], edgecolor="white",
                        linewidth=0.8)
            tol = self.d.arrival_tol_um
            if np.isfinite(tol):
                axes[1].axhline(tol, color=MUTED, lw=0.8, ls="--")
                axes[1].text(min(ax), tol, f" tolerance {tol:g} µm", va="bottom",
                             color=MUTED, fontsize=self.style.base_pt - 1.5)
        for r in unver:
            axes[1].bar([r], [0], width=0.62)      # keep the round on the axis
            axes[1].annotate("not\nverified", (r, 0), ha="center", va="bottom",
                             color=BAD, fontsize=self.style.base_pt - 2)
        _finish(axes[1], xlabel="round", ylabel="feature offset (µm)", ygrid=True)
        axes[1].set_title("Image-verified arrival", loc="left")
        _panel_label(axes[0], "a"), _panel_label(axes[1], "b")
        fig.tight_layout()
        return fig

    # =====================================================================
    # FIGURE 10 — an autonomous round, end to end, in wall-clock time
    # =====================================================================
    # Boundaries of the four phases, in the order the journal writes them.
    _PHASE_MARKS = ["ROUND_DECISION", "ROUND_PROPOSED", "POSITIONED",
                    "ARRAY_PLACED", "RESULT_RECORDED"]
    _PHASE_NAMES = ["decide", "position", "indent", "export"]

    def _round_phases(self) -> dict:
        """{round: {phase: seconds}} by walking the journal in sequence.

        Pairing on a `round` field does NOT work: ARRAY_PLACED carries only a
        sample name, so matching on round silently dropped the indent and export
        phases and produced a timeline claiming a round took 1.3 minutes when it
        took ten. Walk the events in order instead and let the round currently in
        progress own whatever follows it.
        """
        marks: dict = {}
        cur = None
        for e in sorted(self.d.events, key=lambda x: x.get("seq", 0)):
            if e.get("round") is not None:
                try:
                    cur = int(e["round"])
                except (TypeError, ValueError):
                    pass
            if cur is None or e.get("event") not in self._PHASE_MARKS:
                continue
            try:
                t = datetime.fromisoformat(e["ts"])
            except Exception:
                continue
            marks.setdefault(cur, {}).setdefault(e["event"], t)   # first wins

        out: dict = {}
        for r, m in sorted(marks.items()):
            spans = {}
            for name, a, b in zip(self._PHASE_NAMES, self._PHASE_MARKS,
                                  self._PHASE_MARKS[1:]):
                if a in m and b in m and m[b] > m[a]:
                    spans[name] = (m[a], (m[b] - m[a]).total_seconds())
            if spans:
                out[r] = spans
        return out

    def fig10_round_timeline(self):
        import matplotlib.pyplot as plt
        by_round = self._round_phases()
        if not by_round:
            return None
        cols = dict(zip(self._PHASE_NAMES, _round_colors(len(self._PHASE_NAMES))))

        fig, ax = self._fig(ONE_HALF_COL, 2.5)
        for i, (r, ph) in enumerate(by_round.items()):
            t0_round = min(t for t, _ in ph.values())
            for name, (t0, dur) in ph.items():
                ax.broken_barh([((t0 - t0_round).total_seconds() / 60.0, dur / 60.0)],
                               (i - 0.32, 0.64), facecolors=cols[name],
                               edgecolor="white", linewidth=0.8)
            total = max((t - t0_round).total_seconds() + d for t, d in ph.values()) / 60.0
            ax.annotate(f"{total:.1f} min", (total, i), xytext=(4, 0),
                        textcoords="offset points", va="center", color=MUTED,
                        fontsize=self.style.base_pt - 1.5)
        ax.set_yticks(range(len(by_round)))
        ax.set_yticklabels([f"round {r}" for r in by_round])
        ax.invert_yaxis()
        ax.margins(x=0.12)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cols[n])
                   for n in self._PHASE_NAMES]
        ax.legend(handles, self._PHASE_NAMES, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, 1.20), fontsize=self.style.base_pt - 1)
        _finish(ax, xlabel="time within round (min)")
        ax.xaxis.grid(True, color=FAINT, linewidth=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------- driver
    FIGURES = [
        ("fig01_overview", "Sample surface and the sites the campaign measured"),
        ("fig02_properties_by_test", "Hardness and modulus of every test"),
        ("fig03_load_depth", "Load-displacement curves"),
        ("fig04_depth_profiles", "Depth-resolved hardness and modulus"),
        ("fig05_gp_posterior", "Gaussian-process posterior and its uncertainty"),
        ("fig06_learning_curve", "Improvement over the survey baseline by round"),
        ("fig07_acquisition", "Expected improvement per round"),
        ("fig08_hardness_vs_modulus", "Hardness against modulus for every indent"),
        ("fig09_positioning", "Positioning and arrival accuracy"),
        ("fig10_round_timeline", "Wall-clock anatomy of an autonomous round"),
    ]

    def generate_all(self, outdir, *, formats: Iterable[str] = ("pdf", "png"),
                     show: bool = False) -> dict:
        """Render every figure. Returns a manifest; writes captions.md alongside."""
        import matplotlib.pyplot as plt

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {"run_id": self.d.run_id,
                          "generated_at": datetime.now().isoformat(timespec="seconds"),
                          "objective": self.d.objective_text,
                          "dpi": self.style.dpi, "figures": []}
        with self.style.applied():
            for i, (name, title) in enumerate(self.FIGURES, start=1):
                entry = {"n": i, "id": name, "title": title, "files": []}
                try:
                    fig = getattr(self, name)()
                except Exception as exc:                # a bad panel must not stop the set
                    entry["skipped"] = f"{type(exc).__name__}: {exc}"
                    manifest["figures"].append(entry)
                    print(f"  [{i:2d}] {name:26s} ERROR — {entry['skipped']}")
                    continue
                if fig is None:
                    entry["skipped"] = "inputs not available in this run"
                    manifest["figures"].append(entry)
                    print(f"  [{i:2d}] {name:26s} skipped — no data")
                    continue
                for ext in formats:
                    p = outdir / f"{name}.{ext}"
                    fig.savefig(p)
                    entry["files"].append(str(p))
                manifest["figures"].append(entry)
                print(f"  [{i:2d}] {name:26s} -> {', '.join(formats)}")
                plt.show() if show else plt.close(fig)

        (outdir / "figures_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        (outdir / "captions.md").write_text(self._captions(manifest), encoding="utf-8")
        made = sum(1 for f in manifest["figures"] if f["files"])
        print(f"\n{made}/{len(self.FIGURES)} figures -> {outdir}")
        return manifest

    def _captions(self, manifest: dict) -> str:
        """Draft captions — numbers filled from the data, prose for you to edit."""
        t = self._per_test()
        n_ind = int(len(self.d.all_rows))
        n_test = int(len(t))
        val = int(t["validated"].sum()) if len(t) else 0
        lines = [f"# Draft captions — run {self.d.run_id}", "",
                 f"_{n_ind} indents across {n_test} test sites; {val}/{n_test} from "
                 f"validated export files._", ""]
        body = {
            "fig01_overview":
                "Autonomous site selection. (a) The sample surface as the agent saw "
                "it; the scale bar is 20 µm. (b) Every measured site in stage "
                "coordinates, numbered by the round that chose it and coloured by "
                "measured hardness; the dotted path shows the order of visits.",
            "fig02_properties_by_test":
                "Measured (a) hardness and (b) modulus at each site, in the order "
                "the campaign visited them. Error bars are one standard deviation "
                "across the indents of that array; the highlighted point is the "
                "hardest site found.",
            "fig03_load_depth":
                "Load-displacement curves for every test, coloured by round. All "
                "tests used the same load function, so differences in the curves "
                "reflect the material, not the protocol.",
            "fig04_depth_profiles":
                "Depth-resolved (a) hardness and (b) modulus from the continuous "
                "stiffness measurement, shown beyond the near-surface region where "
                "tip-area calibration dominates.",
            "fig05_gp_posterior":
                "The agent's belief after the final round. (a) Posterior mean of the "
                "objective score over the search area. (b) Posterior standard "
                "deviation; white circles are measured sites and the star is the "
                "best observed site. Uncertainty collapses where the campaign "
                "measured, which is what drives the next site choice.",
            "fig06_learning_curve":
                "Best-so-far improvement of the objective score over the survey "
                "baseline, by round, against the pre-registered success threshold.",
            "fig07_acquisition":
                "Maximum expected improvement available to the agent at each round. "
                "Its decay is the model saturating: the surface has been explored "
                "well enough that no candidate site promises much.",
            "fig08_hardness_vs_modulus":
                "Hardness against modulus for every individual indent, coloured by "
                "round. The dashed line is a least-squares fit.",
            "fig09_positioning":
                "Closed-loop accuracy. (a) Distance between the commanded and the "
                "achieved stage coordinate. (b) Offset between the chosen feature "
                "and the indenter axis, measured by registering the live frame "
                "against the frame the target was chosen from, against the "
                "tolerance that gates indentation.",
            "fig10_round_timeline":
                "Wall-clock anatomy of each autonomous round, from the "
                "orchestrator's decision to the recorded result, with no human in "
                "the loop.",
        }
        for f in manifest["figures"]:
            lines.append(f"**Figure {f['n']}.** {body.get(f['id'], f['title'])}")
            if f.get("skipped"):
                lines.append(f"> _not generated: {f['skipped']}_")
            lines.append("")
        return "\n".join(lines)
