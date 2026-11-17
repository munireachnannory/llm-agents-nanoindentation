"""
final_analysis_agent.py
=======================
Final Analysis Agent for the objective-driven workflow (notebook 5).

Runs once, after all measurement rounds are complete. It reviews every indent
from every test/array, and produces an expert, paper-ready read-out:

1. a deterministic numerical summary (per-test statistics, cross-iteration
   trajectory, headline metrics) — always computed and authoritative;
2. a plain-English interpretation written as a nanoindentation expert would
   (main findings, overall conclusion, limitations / unusual results, and
   recommended next steps), produced by an LLM (role ``final_analysis``) when
   one is available, otherwise assembled deterministically from the same facts.

The design mirrors ``orchestrator_agent``: the numbers come from code and are
never overridden by the model; the LLM only narrates the numbers it is given.
Tests are reported generically as ``Test 1, Test 2, ...`` (in order of first
appearance) so the report reads the same regardless of internal region labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..registry.models import get_llm_for_role
from .orchestrator_agent import _call_with_timeout

FINAL_ANALYSIS_ROLE = "final_analysis"

# result-table column names (as produced by ui.combined_rows)
_H = "Hardness (GPa)"
_E = "Modulus (GPa)"
_X = "X (um)"
_Y = "Y (um)"

# heuristics for flagging unusual results
_HIGH_CV = 0.15          # within-test scatter above this is "heterogeneous"
_MIN_HE_RATIO = 6.0      # E/H below this is unusual for most crystalline solids
_MAX_HE_RATIO = 60.0     # E/H above this suggests a very compliant / soft response


def _fmt(x: Any, nd: int = 2) -> str:
    """Format a possibly-missing float without ``nan`` noise."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(xf):
        return "n/a"
    return f"{xf:.{nd}f}"


# ---------------------------------------------------------------------------
# Deterministic result containers
# ---------------------------------------------------------------------------
@dataclass
class TestStats:
    """One test/array's numerical summary (labelled generically as 'Test k')."""
    label: str                       # "Test 1"
    region_key: str                  # internal key (grounding only, not shown)
    n: int
    x_um: float
    y_um: float
    h_mean: float
    h_std: float
    h_min: float
    h_max: float
    h_cv: float
    e_mean: float
    e_std: float
    he_ratio: float
    lit_low: Optional[float] = None
    lit_high: Optional[float] = None
    lit_status: str = "no reference"  # within | above | below | no reference

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalAnalysisReport:
    """Full deterministic summary + expert narrative for the whole campaign."""
    objective_text: str
    n_indents: int
    n_tests: int
    test_stats: list[TestStats] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    headline: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    # narrative (expert plain English)
    findings: str = ""
    cross_comparison: str = ""
    conclusion: str = ""
    limitations: str = ""
    next_steps: str = ""
    advisor: str = "rules"           # llm | rules
    llm_comment: str = ""

    # ---------------------------------------------------------------- views
    def stats_table(self) -> pd.DataFrame:
        rows = []
        for t in self.test_stats:
            rows.append({
                "test": t.label,
                "n": t.n,
                "x (µm)": round(t.x_um, 1),
                "y (µm)": round(t.y_um, 1),
                "H mean (GPa)": round(t.h_mean, 2),
                "H std": round(t.h_std, 2) if np.isfinite(t.h_std) else np.nan,
                "E mean (GPa)": round(t.e_mean, 1),
                "E std": round(t.e_std, 1) if np.isfinite(t.e_std) else np.nan,
                "E/H": round(t.he_ratio, 1) if np.isfinite(t.he_ratio) else np.nan,
                "vs literature": t.lit_status,
            })
        return pd.DataFrame(rows)

    def trajectory_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.trajectory)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["test_stats"] = [t.to_dict() for t in self.test_stats]
        return d

    def to_markdown(self) -> str:
        h = self.headline
        parts = [
            "# Final Analysis — Nanoindentation Campaign",
            "",
            f"**Objective:** {self.objective_text}",
            "",
            f"**Scope:** {self.n_indents} indents across {self.n_tests} test location(s).",
            "",
            "## Main findings",
            self.findings or "_(not available)_",
            "",
            "## Comparison across iterations and test locations",
            self.cross_comparison or "_(not available)_",
            "",
            "## Overall conclusion",
            self.conclusion or "_(not available)_",
            "",
            "## Limitations and unusual results",
            self.limitations or "_(none noted)_",
            "",
            "## Recommended next steps",
            self.next_steps or "_(not available)_",
            "",
            "---",
            f"_Hardness range {_fmt(h.get('h_min'))}–{_fmt(h.get('h_max'))} GPa; "
            f"modulus range {_fmt(h.get('e_min'), 1)}–{_fmt(h.get('e_max'), 1)} GPa; "
            f"best Test = {h.get('best_test', 'n/a')}; "
            f"improvement over baseline {_fmt(h.get('improvement_pct'), 1)}%. "
            f"Narrative by: {self.advisor}._",
        ]
        return "\n".join(parts)

    def display(self) -> None:
        """Render in a notebook (falls back to plain print outside IPython)."""
        try:
            from IPython.display import Markdown, display as _disp
            _disp(Markdown(self.to_markdown()))
            print("\nPer-test numerical summary:")
            _disp(self.stats_table())
            if self.trajectory:
                print("\nProgress across iterations:")
                _disp(self.trajectory_table())
        except Exception:
            print(self.to_markdown())
            print("\nPer-test numerical summary:")
            print(self.stats_table().to_string(index=False))
            if self.trajectory:
                print("\nProgress across iterations:")
                print(self.trajectory_table().to_string(index=False))


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class FinalAnalysisAgent:
    """Reviews all completed tests and reports like a nanoindentation expert."""

    ROLE = FINAL_ANALYSIS_ROLE

    def __init__(
        self,
        objective: Any = None,
        cfg: Any = None,
        *,
        use_llm: bool = True,
        llm_timeout_s: float = 90.0,
    ):
        self.objective = objective
        self.cfg = cfg
        self.use_llm = use_llm
        self.llm_timeout_s = llm_timeout_s

    # ----------------------------------------------------------- llm util
    def _llm_chat(self, prompt: str, system: str) -> Optional[str]:
        if not self.use_llm:
            return None

        def _call() -> str:
            llm = get_llm_for_role(self.ROLE, self.cfg)
            return llm.chat(user_message=prompt, system_message=system)

        return _call_with_timeout(_call, self.llm_timeout_s)

    # ------------------------------------------------- deterministic stats
    def _test_stats(
        self,
        all_rows: pd.DataFrame,
        expectations: Optional[dict[str, tuple[float, float]]],
    ) -> list[TestStats]:
        expectations = expectations or {}
        stats: list[TestStats] = []
        # preserve order of first appearance so "Test k" is stable
        regions = list(dict.fromkeys(all_rows["region"]))
        for i, key in enumerate(regions, start=1):
            g = all_rows[all_rows["region"] == key]
            h = g[_H].astype(float)
            e = g[_E].astype(float)
            n = int(len(g))
            h_mean = float(h.mean())
            h_std = float(h.std(ddof=1)) if n > 1 else float("nan")
            e_mean = float(e.mean())
            e_std = float(e.std(ddof=1)) if n > 1 else float("nan")
            h_cv = (h_std / h_mean) if (n > 1 and h_mean) else float("nan")
            he_ratio = (e_mean / h_mean) if h_mean else float("nan")

            lit_low = lit_high = None
            lit_status = "no reference"
            if key in expectations:
                lit_low, lit_high = (float(v) for v in expectations[key])
                if h_mean < lit_low:
                    lit_status = "below"
                elif h_mean > lit_high:
                    lit_status = "above"
                else:
                    lit_status = "within"

            stats.append(TestStats(
                label=f"Test {i}", region_key=str(key), n=n,
                x_um=float(g[_X].mean()), y_um=float(g[_Y].mean()),
                h_mean=h_mean, h_std=h_std, h_min=float(h.min()), h_max=float(h.max()),
                h_cv=h_cv, e_mean=e_mean, e_std=e_std, he_ratio=he_ratio,
                lit_low=lit_low, lit_high=lit_high, lit_status=lit_status,
            ))
        return stats

    def _trajectory(self, state: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-round learning curve from the orchestrator's decision log."""
        if not state:
            return []
        traj = []
        for d in (state.get("orchestrator_decisions") or []):
            traj.append({
                "round": d.get("round"),
                "action": d.get("action"),
                "improvement_%": round(float(d.get("improvement_pct") or 0.0), 1),
                "indents_used": d.get("indents_used"),
                "max_EI": round(float(d.get("max_ei") or 0.0), 4),
            })
        return traj

    def _headline(
        self,
        all_rows: pd.DataFrame,
        stats: list[TestStats],
        progress: Any,
        state: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        h = all_rows[_H].astype(float)
        e = all_rows[_E].astype(float)
        best = max(stats, key=lambda t: t.h_mean) if stats else None
        softest = min(stats, key=lambda t: t.h_mean) if stats else None

        # objective attainment (prefer the passed progress, else state snapshot)
        prog = {}
        if progress is not None:
            prog = progress.to_dict() if hasattr(progress, "to_dict") else dict(progress)
        elif state is not None:
            prog = dict(state.get("objective_progress") or {})

        # H–E correlation across all indents (needs a few points)
        corr = float("nan")
        if len(all_rows) >= 3 and h.std() > 0 and e.std() > 0:
            corr = float(np.corrcoef(h, e)[0, 1])

        lit_refs = [t for t in stats if t.lit_status != "no reference"]
        lit_within = [t for t in lit_refs if t.lit_status == "within"]

        return {
            "h_min": float(h.min()), "h_max": float(h.max()),
            "e_min": float(e.min()), "e_max": float(e.max()),
            "h_spread_pct": (float(h.max() - h.min()) / float(h.mean()) * 100.0)
                            if float(h.mean()) else float("nan"),
            "best_test": best.label if best else "n/a",
            "best_h": best.h_mean if best else float("nan"),
            "softest_test": softest.label if softest else "n/a",
            "softest_h": softest.h_mean if softest else float("nan"),
            "he_corr": corr,
            "improvement_pct": float(prog.get("improvement_pct", float("nan"))),
            "baseline_scalar": float(prog.get("baseline_scalar", float("nan"))),
            "best_scalar": float(prog.get("best_scalar", float("nan"))),
            "min_improvement_required": (
                float(self.objective.success.min_improvement_pct)
                if self.objective is not None else float("nan")),
            "all_criteria_met": bool(prog.get("all_criteria_met", False)),
            "lit_refs": len(lit_refs),
            "lit_within": len(lit_within),
        }

    def _flags(self, stats: list[TestStats], all_rows: pd.DataFrame) -> list[str]:
        flags: list[str] = []
        for t in stats:
            if t.n == 1:
                flags.append(
                    f"{t.label} has a single indent — no within-test scatter is "
                    f"available, so its hardness/modulus cannot be given an uncertainty.")
            elif np.isfinite(t.h_cv) and t.h_cv > _HIGH_CV:
                flags.append(
                    f"{t.label} shows high scatter (CV {t.h_cv*100:.0f}% in hardness), "
                    f"suggesting local heterogeneity, surface roughness, pile-up, or a "
                    f"phase/grain boundary within the array.")
            if np.isfinite(t.he_ratio) and not (_MIN_HE_RATIO <= t.he_ratio <= _MAX_HE_RATIO):
                flags.append(
                    f"{t.label} has an unusual E/H ratio ({t.he_ratio:.1f}); check the "
                    f"area-function calibration, tip condition, or drift for that array.")
            if t.lit_status in ("above", "below"):
                band = f"{_fmt(t.lit_low)}–{_fmt(t.lit_high)} GPa"
                flags.append(
                    f"{t.label} hardness ({_fmt(t.h_mean)} GPa) falls {t.lit_status} the "
                    f"literature range ({band}) for that region type.")
        if (all_rows[_H].astype(float) <= 0).any() or (all_rows[_E].astype(float) <= 0).any():
            flags.append("Some indents report non-positive hardness or modulus — likely "
                         "failed contacts that should be screened before averaging.")
        return flags

    # ---------------------------------------------------------- narrative
    def _facts_block(self, report: FinalAnalysisReport) -> str:
        lines = [
            f"OBJECTIVE: {report.objective_text}",
            f"SCOPE: {report.n_indents} indents across {report.n_tests} test location(s).",
            "",
            "PER-TEST RESULTS (means; std shown when >1 indent):",
        ]
        for t in report.test_stats:
            std = "" if not np.isfinite(t.h_std) else f" ± {t.h_std:.2f}"
            estd = "" if not np.isfinite(t.e_std) else f" ± {t.e_std:.1f}"
            lit = "" if t.lit_status == "no reference" else (
                f"; literature {_fmt(t.lit_low)}-{_fmt(t.lit_high)} GPa -> {t.lit_status}")
            lines.append(
                f"- {t.label} at ({t.x_um:.0f}, {t.y_um:.0f}) um, n={t.n}: "
                f"H={t.h_mean:.2f}{std} GPa, E={t.e_mean:.1f}{estd} GPa, "
                f"E/H={_fmt(t.he_ratio, 1)}{lit}")

        h = report.headline
        lines += [
            "",
            "CROSS-CAMPAIGN METRICS:",
            f"- hardness spans {_fmt(h.get('h_min'))}-{_fmt(h.get('h_max'))} GPa "
            f"({_fmt(h.get('h_spread_pct'), 0)}% spread); "
            f"modulus {_fmt(h.get('e_min'), 1)}-{_fmt(h.get('e_max'), 1)} GPa.",
            f"- hardest: {h.get('best_test')} (H={_fmt(h.get('best_h'))} GPa); "
            f"softest: {h.get('softest_test')} (H={_fmt(h.get('softest_h'))} GPa).",
            f"- H–E correlation across indents: {_fmt(h.get('he_corr'))}.",
            f"- objective improvement over baseline: {_fmt(h.get('improvement_pct'), 1)}% "
            f"(required >= {_fmt(h.get('min_improvement_required'), 1)}%); "
            f"all criteria met: {h.get('all_criteria_met')}.",
            f"- tests with a literature reference: {h.get('lit_within')}/{h.get('lit_refs')} "
            f"within the expected band.",
        ]
        if report.trajectory:
            lines += ["", "ITERATION TRAJECTORY (per round):"]
            for r in report.trajectory:
                lines.append(
                    f"- round {r.get('round')}: action={r.get('action')}, "
                    f"improvement={r.get('improvement_%')}%, "
                    f"indents_used={r.get('indents_used')}, max_EI={r.get('max_EI')}")
        if report.flags:
            lines += ["", "AUTOMATICALLY FLAGGED (verify in narrative):"]
            lines += [f"- {f}" for f in report.flags]
        return "\n".join(lines)

    def _llm_narrative(self, report: FinalAnalysisReport) -> Optional[dict[str, str]]:
        prompt = (
            "You are reviewing a completed autonomous nanoindentation campaign. "
            "Below are the deterministic results computed from the raw indents. Do NOT "
            "invent numbers; interpret only what is given. Refer to locations exactly as "
            "'Test 1', 'Test 2', etc.\n\n"
            f"{self._facts_block(report)}\n\n"
            "Write a concise expert discussion. Return ONLY JSON with these string keys:\n"
            '{"findings": "3-5 sentences on the main mechanical findings and what the '
            'hardness/modulus values imply about the material", '
            '"cross_comparison": "2-4 sentences comparing the tests to each other and '
            'across iterations (did later rounds find harder/stiffer sites; is the trend '
            'converging)", '
            '"conclusion": "2-3 sentences giving the overall conclusion of the study '
            'relative to the objective", '
            '"limitations": "2-4 sentences on limitations and any unusual/suspect results, '
            'grounded in the flagged items and small sample sizes", '
            '"next_steps": "2-4 sentences of concrete recommended next measurements or '
            'analysis (e.g. more indents per site, additional locations, load function, '
            'validation)"}'
        )
        system = (
            "You are a senior instrumented-indentation (nanoindentation) expert who "
            "reports results with the Oliver–Pharr framework. You write precise, sober, "
            "publication-style discussion and never fabricate values. Respond with strict "
            "JSON only."
        )
        response = self._llm_chat(prompt, system=system)
        if not response:
            return None
        try:
            from ..registry.tool_catalog import parse_llm_json
            data = parse_llm_json(response)
            keys = ("findings", "cross_comparison", "conclusion", "limitations", "next_steps")
            out = {k: str(data[k]).strip() for k in keys if isinstance(data.get(k), str)}
            return out or None
        except Exception as exc:  # noqa: BLE001 — any parse issue -> deterministic fallback
            print(f"[final_analysis] narrative JSON unusable ({type(exc).__name__}) — using template")
            return None

    def _fallback_narrative(self, report: FinalAnalysisReport) -> dict[str, str]:
        """Deterministic expert-style narrative assembled from the same facts."""
        h = report.headline
        hard, soft = h.get("best_test", "n/a"), h.get("softest_test", "n/a")
        spread = _fmt(h.get("h_spread_pct"), 0)
        findings = (
            f"Across {report.n_indents} indents at {report.n_tests} location(s), measured "
            f"hardness ranged {_fmt(h.get('h_min'))}–{_fmt(h.get('h_max'))} GPa and reduced "
            f"modulus {_fmt(h.get('e_min'), 1)}–{_fmt(h.get('e_max'), 1)} GPa. {hard} was the "
            f"hardest site (H={_fmt(h.get('best_h'))} GPa) and {soft} the most compliant "
            f"(H={_fmt(h.get('softest_h'))} GPa), a {spread}% spread that indicates the agents "
            f"resolved genuinely distinct mechanical regions rather than measurement noise."
        )
        corr = h.get("he_corr")
        corr_txt = ("hardness and modulus track together across sites, consistent with a "
                    "single stiffening mechanism"
                    if np.isfinite(corr) and corr > 0.5 else
                    "hardness and modulus are only weakly coupled across sites")
        if report.trajectory:
            first = report.trajectory[0].get("improvement_%")
            last = report.trajectory[-1].get("improvement_%")
            traj_txt = (f"Over {len(report.trajectory)} adaptive round(s) the best-site "
                        f"improvement moved from {first}% to {last}%, ")
        else:
            traj_txt = ""
        cross = (f"{traj_txt}and {corr_txt}. "
                 f"{h.get('lit_within')}/{h.get('lit_refs')} tests with a literature reference "
                 f"fell inside the expected band.")
        met = h.get("all_criteria_met")
        conclusion = (
            f"The campaign {'met' if met else 'did not fully meet'} its objective: the best "
            f"site improved on the survey baseline by {_fmt(h.get('improvement_pct'), 1)}% "
            f"(target ≥ {_fmt(h.get('min_improvement_required'), 1)}%). "
            + ("The objective-directed targeting successfully located the sought property extreme."
               if met else
               "Further measurement is needed to satisfy all success criteria.")
        )
        lim_items = report.flags[:5] or [
            "No anomalies were flagged, but sample sizes per site are modest."]
        limitations = (
            "Key caveats: " + " ".join(f"({i+1}) {t}" for i, t in enumerate(lim_items))
        )
        next_steps = (
            f"Recommended next steps: add replicate indents at {hard} and {soft} to tighten "
            f"the statistics; place confirming arrays between the current extremes to map the "
            f"transition; and validate the hardest site with an independent load function or "
            f"microscopy of the residual imprint to rule out pile-up/edge artifacts."
        )
        return {
            "findings": findings, "cross_comparison": cross, "conclusion": conclusion,
            "limitations": limitations, "next_steps": next_steps,
        }

    # -------------------------------------------------------------- run
    def analyze(
        self,
        all_rows: pd.DataFrame,
        state: Optional[dict[str, Any]] = None,
        expectations: Optional[dict[str, tuple[float, float]]] = None,
        progress: Any = None,
    ) -> FinalAnalysisReport:
        """Build the full deterministic + narrative report for the campaign."""
        if all_rows is None or len(all_rows) == 0:
            return FinalAnalysisReport(
                objective_text=self._objective_text(state), n_indents=0, n_tests=0,
                conclusion="No indents were recorded — nothing to analyze.",
                advisor="rules")

        stats = self._test_stats(all_rows, expectations)
        report = FinalAnalysisReport(
            objective_text=self._objective_text(state),
            n_indents=int(len(all_rows)),
            n_tests=len(stats),
            test_stats=stats,
            trajectory=self._trajectory(state),
            headline=self._headline(all_rows, stats, progress, state),
            flags=self._flags(stats, all_rows),
        )

        narrative = self._llm_narrative(report)
        if narrative:
            report.advisor = "llm"
        else:
            narrative = self._fallback_narrative(report)
            report.advisor = "rules"
        report.findings = narrative.get("findings", "")
        report.cross_comparison = narrative.get("cross_comparison", "")
        report.conclusion = narrative.get("conclusion", "")
        report.limitations = narrative.get("limitations", "")
        report.next_steps = narrative.get("next_steps", "")
        return report

    # --------------------------------------------------------------- util
    def _objective_text(self, state: Optional[dict[str, Any]]) -> str:
        if self.objective is not None:
            try:
                return self.objective.raw_text or self.objective.summary_text()[:400]
            except Exception:
                pass
        if state:
            return str(state.get("research_question") or state.get("research_objective") or "")[:400]
        return ""


def run_final_analysis(
    all_rows: pd.DataFrame,
    *,
    objective: Any = None,
    cfg: Any = None,
    state: Optional[dict[str, Any]] = None,
    expectations: Optional[dict[str, tuple[float, float]]] = None,
    progress: Any = None,
    use_llm: bool = True,
) -> FinalAnalysisReport:
    """Convenience one-shot entry point for the notebook."""
    agent = FinalAnalysisAgent(objective, cfg, use_llm=use_llm)
    return agent.analyze(all_rows, state=state, expectations=expectations, progress=progress)
