"""
orchestrator_agent.py
=====================
Main Orchestrator Agent for the objective-driven workflow (notebook 5).

The orchestrator owns the compiled ResearchObjective and:
1. seeds agent state with the objective + objective-derived critic criteria,
2. briefs every sub-agent (per-role directives injected into their prompts),
3. evaluates progress toward the objective after each measurement round,
4. decides the next action from a bounded action space.

Deterministic rules are always computed and remain authoritative; an LLM
(Hugging Face, role "orchestrator") may advise on soft decisions, and every
decision records its provenance in ``advisor``.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from ..objective import ResearchObjective
from ..critics.measurement_critic import _objective_array_scores

BRIEFING_ROLES = (
    "image_analysis", "literature", "hypothesis", "planning",
    "measurement_strategy", "measurement_critic",
)

ORCHESTRATOR_ACTIONS = (
    "continue", "refine_hypothesis", "objective_met", "budget_exhausted", "escalate_to_human",
)
# hard stops the LLM advisor is never allowed to override
_HARD_ACTIONS = ("objective_met", "budget_exhausted")


@dataclass
class ObjectiveProgress:
    """Deterministic snapshot of how far the campaign is from the objective."""
    round_index: int = 0
    indents_used: int = 0
    indent_budget: int = 0
    budget_remaining: int = 0
    arrays_done: int = 0
    locations_visited: int = 0
    best_site: dict[str, Any] = field(default_factory=dict)
    best_scalar: float = 0.0
    baseline_scalar: float = 0.0
    improvement_pct: float = 0.0
    surrogate_mode: str = "untrained"
    max_ei: float = 0.0
    mean_posterior_sigma: float = float("nan")
    criteria_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    all_criteria_met: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def table(self) -> pd.DataFrame:
        rows = [
            {"metric": name, "met": status["met"], "detail": status["detail"]}
            for name, status in self.criteria_status.items()
        ]
        rows.append({
            "metric": "budget",
            "met": self.budget_remaining > 0,
            "detail": f"{self.indents_used}/{self.indent_budget} indents used "
                      f"({self.budget_remaining} remaining), round {self.round_index}",
        })
        rows.append({
            "metric": "surrogate",
            "met": self.surrogate_mode != "untrained",
            "detail": f"{self.surrogate_mode}, max EI {self.max_ei:.4f}",
        })
        return pd.DataFrame(rows)


@dataclass
class OrchestratorDecision:
    """One bounded-action-space routing decision."""
    action: str
    reason: str
    proposal_source: str = "none"          # bo_acquisition | image_targets | none
    n_sites: int = 1
    advisor: str = "rules"                 # rules | llm
    llm_comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _call_with_timeout(fn, seconds: float) -> Any:
    """Run fn() with a wall-clock timeout; None on timeout/error (LLM fallback)."""
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=seconds)
    except (_FuturesTimeout, Exception) as exc:  # noqa: BLE001 — any failure means fallback
        print(f"[orchestrator] LLM call unavailable ({type(exc).__name__}) — using deterministic path")
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


class OrchestratorAgent:
    """Main agent: objective in, briefings + progress evaluations + decisions out."""

    ROLE = "orchestrator"

    def __init__(
        self,
        objective: ResearchObjective,
        cfg: Any = None,
        *,
        use_llm: bool = True,
        llm_timeout_s: float = 60.0,
    ):
        self.objective = objective
        self.cfg = cfg
        self.use_llm = use_llm
        self.llm_timeout_s = llm_timeout_s

    # ------------------------------------------------------------- llm util
    def _llm_chat(self, prompt: str, system: str) -> Optional[str]:
        if not self.use_llm:
            return None

        def _call() -> str:
            from ..registry.models import get_llm_for_role
            llm = get_llm_for_role(self.ROLE, self.cfg)
            return llm.chat(user_message=prompt, system_message=system)

        return _call_with_timeout(_call, self.llm_timeout_s)

    # ------------------------------------------------------ state seeding
    def objective_state_updates(self) -> dict[str, Any]:
        """Initial state updates: broadcast the objective to the whole team."""
        return {
            "research_objective": self.objective.to_dict(),
            "research_question": self.objective.raw_text,
            "experiment_context": self.objective.summary_text(),
            "objective_criteria": self.measurement_criteria(),
        }

    def measurement_criteria(self) -> dict[str, Any]:
        """Objective-derived success criteria consumed by the measurement critic."""
        s = self.objective.success
        return {
            "min_indent_arrays": s.min_confirming_arrays,
            "min_visited_locations": s.min_visited_locations,
            "min_improvement_pct": s.min_improvement_pct,
            "property_weights": self.objective.normalized_weights(),
            "directions": self.objective.directions(),
            "objective_summary": self.objective.summary_text()[:400],
        }

    # ---------------------------------------------------------- briefings
    def _goal_phrase(self) -> str:
        return "; ".join(g.describe() for g in self.objective.property_goals)

    def _fallback_briefings(self) -> dict[str, str]:
        obj, c, s = self.objective, self.objective.constraints, self.objective.success
        material = obj.material_system or "the sample"
        wants_high = any(g.direction == "maximize" for g in obj.property_goals)
        wants_low = any(g.direction == "minimize" for g in obj.property_goals)
        if wants_high and not wants_low:
            regions = ("bright/second-phase precipitates and boundary-hardened zones "
                       "(highest expected hardness/stiffness)")
        elif wants_low and not wants_high:
            regions = "dark matrix regions, pores and compliant zones (lowest expected properties)"
        else:
            regions = "the full range of contrast regions spanning low to high expected properties"
        return {
            "image_analysis": (
                f"Objective: {self._goal_phrase()}. Prioritize {regions}. "
                f"Always include one contrasting reference site so improvement over "
                f"baseline can be quantified."
            ),
            "literature": (
                f"Find quantitative hardness/modulus values for {material}: typical GPa "
                f"ranges per phase/region and the microstructural factors that shift them "
                f"toward the objective ({self._goal_phrase()})."
            ),
            "hypothesis": (
                f"Hypothesize WHICH microstructural regions of {material} best satisfy: "
                f"{self._goal_phrase()}. State expected GPa ranges per region and a testable "
                f"margin (best site >= {s.min_improvement_pct:g}% over the sample baseline)."
            ),
            "planning": (
                f"Plan {c.grid_nx}x{c.grid_ny} arrays at {c.spacing_um:g} um spacing. Total "
                f"budget {c.indent_budget} indents across <= {c.max_rounds} rounds; keep "
                f">= {c.min_site_spacing_um:g} um between array centers."
            ),
            "measurement_strategy": (
                "For each proposed site, choose the minimum sufficient number of indents "
                "and only vary features explicitly delegated to agent control. Skip invalid, "
                "duplicate, or unnecessary sites; explain every selected value briefly."
            ),
            "measurement_critic": (
                f"Judge success against the objective, not generic variation: best site must "
                f"beat the sample baseline by >= {s.min_improvement_pct:g}% on the weighted "
                f"objective ({self._goal_phrase()}), with >= {s.min_confirming_arrays} arrays "
                f"and >= {s.min_visited_locations} locations."
            ),
        }

    def build_briefings(self, state: dict[str, Any]) -> tuple[dict[str, str], str]:
        """Per-role directives; (briefings, provenance 'llm'|'fallback')."""
        fallback = self._fallback_briefings()
        prompt = (
            "Write one short directive (max 60 words) for each agent role of an autonomous "
            "nanoindentation lab so its work aligns with this research objective.\n\n"
            f"OBJECTIVE:\n{self.objective.summary_text()}\n\n"
            f"Roles: {', '.join(BRIEFING_ROLES)}.\n"
            'Return ONLY JSON: {"image_analysis": "...", "literature": "...", '
            '"hypothesis": "...", "planning": "...", "measurement_strategy": "...", '
            '"measurement_critic": "..."}'
        )
        response = self._llm_chat(
            prompt, system="You are the orchestrator of an autonomous materials lab. "
                           "Respond with strict JSON only.")
        if response:
            try:
                from ..registry.tool_catalog import parse_llm_json
                data = parse_llm_json(response)
                briefings = {
                    role: str(data[role]).strip()[:800]
                    for role in BRIEFING_ROLES
                    if isinstance(data.get(role), str) and data[role].strip()
                }
                if briefings:
                    # fallback fills any role the LLM skipped
                    return {**fallback, **briefings}, "llm"
            except Exception as exc:
                print(f"[orchestrator] briefing JSON unusable ({type(exc).__name__}) — using templates")
        return fallback, "fallback"

    def literature_seed(self) -> dict[str, Any]:
        """Pre-seeded search query/keywords (= literature-agent injection point)."""
        material = self.objective.material_system or "thin film"
        props = " and ".join(g.name for g in self.objective.property_goals) or "hardness"
        keywords = ["nanoindentation", material, *[g.name for g in self.objective.property_goals]]
        return {
            "literature_search_query": f"nanoindentation {props} of {material}",
            "literature_search_keywords": keywords[:8],
        }

    def briefing_state_updates(self, state: dict[str, Any]) -> dict[str, Any]:
        briefings, provenance = self.build_briefings(state)
        updates = {
            "orchestrator_briefings": briefings,
            **self.literature_seed(),
        }
        updates["scratchpad"] = list(state.get("scratchpad") or []) + [
            f"orchestrator: briefed {len(briefings)} roles ({provenance})"
        ]
        return updates

    # ------------------------------------------------------------ progress
    def evaluate_progress(
        self,
        state: dict[str, Any],
        surrogate: Any = None,
    ) -> ObjectiveProgress:
        """Deterministic progress metrics + per-criterion status."""
        c, s = self.objective.constraints, self.objective.success
        indent_history = list(state.get("indent_history") or [])
        indents_used = sum(int(d.get("n_indents") or 0) for d in indent_history)
        arrays_done = len(indent_history)
        locations = len(state.get("visited_target_keys") or [])

        # same weighted physical improvement the measurement critic uses.
        # NON-FINITE SCORES ARE DROPPED, NOT MAXED OVER: an array whose result
        # file validated to nothing scores NaN, and `max()` over a list holding
        # NaN returns NaN depending only on where it sits. That NaN then fails
        # every `>=` silently, so ONE bad indent permanently blocks the
        # improvement criterion no matter how good every later array is — which
        # is why 2026-08-05 printed "nan% improvement" in every round. An array
        # with no usable data must count as no evidence, not as poison.
        scores = [s_ for s_ in _objective_array_scores(
            indent_history, self.measurement_criteria())
            if isinstance(s_, (int, float)) and math.isfinite(s_)]
        improvement_pct = max(scores) * 100.0 if scores else 0.0

        summary = surrogate.summary() if surrogate is not None else {}
        progress = ObjectiveProgress(
            round_index=int(state.get("orchestrator_round") or 0),
            indents_used=indents_used,
            indent_budget=c.indent_budget,
            budget_remaining=max(0, c.indent_budget - indents_used),
            arrays_done=arrays_done,
            locations_visited=locations,
            best_site=dict(summary.get("best") or {}),
            best_scalar=float((summary.get("best") or {}).get("scalar_value") or 0.0),
            baseline_scalar=float(summary.get("baseline_scalar") or 0.0),
            improvement_pct=float(improvement_pct),
            surrogate_mode=str(summary.get("mode") or "untrained"),
            max_ei=float(summary.get("max_ei") or 0.0),
            mean_posterior_sigma=float(summary.get("mean_scalar_sigma") or float("nan")),
        )
        progress.criteria_status = {
            "improvement": {
                "met": improvement_pct >= s.min_improvement_pct,
                "detail": f"best array +{improvement_pct:.1f}% vs baseline "
                          f"(needs >= {s.min_improvement_pct:g}%)",
            },
            "arrays": {
                "met": arrays_done >= s.min_confirming_arrays,
                "detail": f"{arrays_done} arrays (needs >= {s.min_confirming_arrays})",
            },
            "locations": {
                "met": locations >= s.min_visited_locations,
                "detail": f"{locations} locations (needs >= {s.min_visited_locations})",
            },
        }
        progress.all_criteria_met = all(v["met"] for v in progress.criteria_status.values())
        return progress

    # ------------------------------------------------------------ decision
    def _minimum_next_indent_cost(self) -> int:
        """Smallest grid currently permitted by the authority contract.

        Existing/disabled strategy configurations retain the objective grid.
        When grid dimensions are delegated, this lets the orchestrator continue
        with a remaining budget that can still fund a valid smaller array; the
        runner performs the final per-plan budget check.
        """
        c = self.objective.constraints
        nx, ny = int(c.grid_nx), int(c.grid_ny)
        settings = getattr(self.cfg, "measurement_strategy", None)
        if settings is None or not settings.enabled:
            return max(1, nx * ny)
        params = settings.grid_control.params
        sx, sy = params.get("grid_nx"), params.get("grid_ny")
        if sx is not None and sx.mode == "agent" and sx.agent_allowed:
            nx = max(1, int(math.ceil(sx.min if sx.min is not None else 1)))
        if sy is not None and sy.mode == "agent" and sy.agent_allowed:
            ny = max(1, int(math.ceil(sy.min if sy.min is not None else 1)))
        return max(1, nx * ny)

    def _rules_decision(
        self,
        state: dict[str, Any],
        progress: ObjectiveProgress,
    ) -> OrchestratorDecision:
        c, s = self.objective.constraints, self.objective.success

        # success first — a met objective is a success even with zero budget left
        if progress.all_criteria_met:
            return OrchestratorDecision(
                "objective_met",
                "All objective criteria satisfied: " + "; ".join(
                    v["detail"] for v in progress.criteria_status.values()))
        next_cost = self._minimum_next_indent_cost()
        if progress.budget_remaining < next_cost:
            return OrchestratorDecision(
                "budget_exhausted",
                f"Only {progress.budget_remaining} indents left — the next "
                f"permitted array needs at least {next_cost}.")
        if progress.round_index >= c.max_rounds:
            return OrchestratorDecision(
                "budget_exhausted",
                f"Round limit reached ({progress.round_index}/{c.max_rounds}).")

        critique_decision = state.get("measurement_critique_decision", "")
        if critique_decision == "hypothesis_invalid":
            already_refined = any(
                d.get("action") == "refine_hypothesis"
                for d in (state.get("orchestrator_decisions") or []))
            if already_refined:
                return OrchestratorDecision(
                    "escalate_to_human",
                    "Measurement critic invalidated the hypothesis again after a "
                    "refinement — operator guidance needed.")
            return OrchestratorDecision(
                "refine_hypothesis",
                "Measurement critic invalidated the hypothesis — the objective "
                "landscape looks different from what was hypothesized.")

        if (progress.arrays_done >= 3
                and progress.improvement_pct < s.min_improvement_pct / 2.0
                and progress.max_ei < s.ei_plateau):
            return OrchestratorDecision(
                "escalate_to_human",
                f"Flat objective landscape: after {progress.arrays_done} arrays the best "
                f"improvement is {progress.improvement_pct:.1f}% and max EI "
                f"{progress.max_ei:.4f} < plateau {s.ei_plateau:g}. Suggest widening the "
                f"search area or revising the objective.")

        source = ("image_targets"
                  if progress.arrays_done < 2 or progress.surrogate_mode == "untrained"
                  else "bo_acquisition")
        reason = (
            f"Criteria not yet met ({progress.improvement_pct:.1f}% improvement, "
            f"{progress.arrays_done} arrays, {progress.locations_visited} locations); "
            f"{progress.budget_remaining} indents remain. Next sites from "
            f"{'the image-analysis ranking' if source == 'image_targets' else 'GP Expected Improvement'}.")
        return OrchestratorDecision(
            "continue", reason,
            proposal_source=source, n_sites=c.arrays_per_round)

    def decide_next_action(
        self,
        state: dict[str, Any],
        progress: ObjectiveProgress,
        surrogate: Any = None,
    ) -> OrchestratorDecision:
        """Deterministic rules first (authoritative); LLM may adjust soft decisions."""
        decision = self._rules_decision(state, progress)
        if decision.action in _HARD_ACTIONS or not self.use_llm:
            return decision

        allowed = ("continue", "refine_hypothesis", "escalate_to_human")
        prompt = (
            "You supervise an autonomous nanoindentation campaign.\n\n"
            f"OBJECTIVE:\n{self.objective.summary_text()}\n\n"
            f"PROGRESS:\n{progress.table().to_string(index=False)}\n\n"
            f"Last measurement critique: {str(state.get('measurement_critique', ''))[:400]}\n\n"
            f"The deterministic supervisor chose: {decision.action} — {decision.reason}\n\n"
            f"If you agree, return that action. You may only choose among {list(allowed)}.\n"
            'Return ONLY JSON: {"action": "...", "reason": "<max 50 words>"}'
        )
        response = self._llm_chat(
            prompt, system="You are the orchestrator of an autonomous materials lab. "
                           "Respond with strict JSON only.")
        if response:
            try:
                from ..registry.tool_catalog import parse_llm_json
                data = parse_llm_json(response)
                action = str(data.get("action", "")).strip()
                comment = str(data.get("reason", "")).strip()[:400]
                if action in allowed:
                    decision.llm_comment = comment
                    if action != decision.action:
                        decision = OrchestratorDecision(
                            action,
                            f"LLM advisor override: {comment} "
                            f"(rules suggested {decision.action}: {decision.reason})",
                            proposal_source=decision.proposal_source,
                            n_sites=decision.n_sites,
                            advisor="llm",
                            llm_comment=comment,
                        )
            except Exception as exc:
                print(f"[orchestrator] decision JSON unusable ({type(exc).__name__}) — keeping rules verdict")
        return decision

    def decision_state_updates(
        self,
        state: dict[str, Any],
        decision: OrchestratorDecision,
        progress: ObjectiveProgress,
    ) -> dict[str, Any]:
        """Full-list state updates (plain-dict state.update() semantics)."""
        record = {
            "round": progress.round_index,
            **decision.to_dict(),
            "improvement_pct": progress.improvement_pct,
            "indents_used": progress.indents_used,
            "max_ei": progress.max_ei,
        }
        return {
            "orchestrator_decisions": list(state.get("orchestrator_decisions") or []) + [record],
            "objective_progress": progress.to_dict(),
            "supervisor_reasoning": decision.reason,
        }
