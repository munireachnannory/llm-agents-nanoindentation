"""Tests for the objective-driven orchestrator (objective, acquisition, critic, decisions)."""

from __future__ import annotations

import numpy as np
import pytest

from nanoindentation_agents.agents.orchestrator_agent import OrchestratorAgent
from nanoindentation_agents.analysis import objective_acquisition as oa
from nanoindentation_agents.critics.measurement_critic import measurement_critic_node
from nanoindentation_agents.objective import (
    ObjectiveConstraints,
    ResearchObjective,
    _objective_from_llm_dict,
    parse_objective_fallback,
)
from nanoindentation_agents.state import create_initial_state


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_array(cx: float, cy: float, label: str, h: float, e: float, n: int = 9) -> dict:
    rows = [
        {
            "X (um)": cx + (i % 3 - 1) * 6.0,
            "Y (um)": cy + (i // 3 - 1) * 6.0,
            "Hardness (GPa)": h + 0.1 * (i % 3 - 1),
            "Modulus (GPa)": e + 1.0 * (i % 3 - 1),
        }
        for i in range(n)
    ]
    return {
        "area_name": label,
        "position": (cx, cy),
        "results": rows,
        "n_indents": n,
        "valid_indent_count": n,
        "hardness_mean": h,
        "hardness_std": 0.2,
        "modulus_mean": e,
        "modulus_std": 2.0,
    }


def objective_hi_hi() -> ResearchObjective:
    return parse_objective_fallback(
        "find regions with higher hardness AND higher elastic modulus on a graded carbon film",
        fov_bounds=(0.0, 100.0, 0.0, 75.0), indent_budget=45, max_rounds=6,
    )


def state_with(history: list[dict], locations: int, objective: ResearchObjective, orch: OrchestratorAgent):
    state = create_initial_state("test")
    state.update(orch.objective_state_updates())
    state["indent_history"] = history
    state["visited_target_keys"] = [f"loc{i}" for i in range(locations)]
    state["decision_history"] = [{}] * 5
    return state


# ---------------------------------------------------------------------------
# objective parsing
# ---------------------------------------------------------------------------


def test_fallback_parse_maximize_both() -> None:
    obj = objective_hi_hi()
    assert {(g.name, g.direction) for g in obj.property_goals} == {
        ("hardness", "maximize"), ("modulus", "maximize")}
    assert "carbon film" in obj.material_system
    assert obj.compiled_by == "fallback"


def test_fallback_parse_minimize_and_range() -> None:
    soft = parse_objective_fallback("locate the softest, most compliant areas of the coating")
    assert {(g.name, g.direction) for g in soft.property_goals} == {
        ("hardness", "minimize"), ("modulus", "minimize")}

    ranged = parse_objective_fallback("map areas where hardness is 8-12 GPa")
    (goal,) = ranged.property_goals
    assert goal.direction == "target_range"
    assert goal.target_range == (8.0, 12.0)


def test_fallback_parse_defaults_when_no_properties() -> None:
    obj = parse_objective_fallback("just explore the sample")
    assert {g.name for g in obj.property_goals} == {"hardness", "modulus"}


def test_objective_dict_round_trip() -> None:
    obj = objective_hi_hi()
    clone = ResearchObjective.from_dict(obj.to_dict())
    assert clone.to_dict() == obj.to_dict()
    assert clone.constraints.fov_bounds == obj.constraints.fov_bounds


def test_llm_dict_validation_clamps_and_drops() -> None:
    fallback = parse_objective_fallback("goal")
    parsed = _objective_from_llm_dict(
        {
            "material_system": "carbon film",
            "property_goals": [
                {"name": "hardness", "direction": "maximize", "weight": 999},
                {"name": "toughness", "direction": "maximize", "weight": 1},   # unknown -> dropped
                {"name": "modulus", "direction": "sideways", "weight": -3},    # coerced
            ],
            "success": {"min_improvement_pct": 5000, "min_confirming_arrays": 0},
        },
        "goal", fallback.constraints,
    )
    assert parsed is not None and parsed.compiled_by == "llm"
    by_name = {g.name: g for g in parsed.property_goals}
    assert set(by_name) == {"hardness", "modulus"}
    assert by_name["hardness"].weight == 10.0
    assert by_name["modulus"].direction == "maximize"
    assert parsed.success.min_improvement_pct == 200.0
    assert parsed.success.min_confirming_arrays == 1

    assert _objective_from_llm_dict({"property_goals": []}, "goal", fallback.constraints) is None


def test_success_criteria_capped_to_budget() -> None:
    # EXPECTATION CORRECTED (GAPS.md G15, 27 July 2026).
    #
    # This test previously derived its constraints from
    # parse_objective_fallback(indent_budget=45) while its comment asserted
    # "3x3 arrays -> at most 5 arrays affordable". Those disagree:
    # parse_objective_fallback leaves ObjectiveConstraints at its 2x2 default,
    # so 45 // 4 = 11 arrays are affordable, min(8, 11) = 8, and the assertion
    # of 5 failed.
    #
    # The production cap in _cap_success_to_budget was verified correct and is
    # unchanged. The bug was in the test: at 2x2 the cap never binds, so the
    # test could not have exercised the behaviour it names even with adjusted
    # numbers. The grid is now stated explicitly so the cap genuinely bites.
    # No reported quantity in the manuscript is affected.
    capped = ObjectiveConstraints(indent_budget=45, grid_nx=3, grid_ny=3)
    assert capped.indents_per_array == 9
    assert capped.indent_budget // capped.indents_per_array == 5

    parsed = _objective_from_llm_dict(
        {"property_goals": [{"name": "hardness", "direction": "maximize", "weight": 1}],
         "success": {"min_confirming_arrays": 8, "min_visited_locations": 10}},
        "goal", capped,
    )
    assert parsed.success.min_confirming_arrays == 5
    assert parsed.success.min_visited_locations == 5

    # Where the budget is ample the cap must NOT bind: 2x2 at 45 indents pays
    # for 11 arrays, so an 8-array requirement passes through untouched.
    roomy = parse_objective_fallback("maximize hardness", indent_budget=45).constraints
    roomy.grid_nx = roomy.grid_ny = 2
    assert roomy.indent_budget // roomy.indents_per_array == 11
    uncapped = _objective_from_llm_dict(
        {"property_goals": [{"name": "hardness", "direction": "maximize", "weight": 1}],
         "success": {"min_confirming_arrays": 8, "min_visited_locations": 10}},
        "goal", roomy,
    )
    assert uncapped.success.min_confirming_arrays == 8
    assert uncapped.success.min_visited_locations == 10

    # Tiny budget: the fallback defaults (2 arrays / 3 locations) must shrink.
    #
    # ALSO CORRECTED (GAPS.md G15). This previously used indent_budget=18,
    # which at the 2x2 default affords 18 // 4 = 4 arrays -- more than the
    # default requirement of 3 locations, so nothing shrank and the assertion
    # was unsatisfiable. It was masked because the earlier assertion in this
    # test failed first. The defaults only bind when the budget affords fewer
    # than 3 arrays, i.e. a budget of at most 11.
    tight = _objective_from_llm_dict(
        {"property_goals": [
            {"name": "hardness", "direction": "maximize", "weight": 1}]},
        "maximize hardness",
        ObjectiveConstraints(indent_budget=8, grid_nx=2, grid_ny=2),
    )
    assert tight.constraints.indent_budget // tight.constraints.indents_per_array == 2
    assert tight.success.min_confirming_arrays <= 2
    assert tight.success.min_visited_locations <= 2


# ---------------------------------------------------------------------------
# acquisition surrogate
# ---------------------------------------------------------------------------


def test_surrogate_gp_mode_and_proposals() -> None:
    if not oa.HAVE_SKLEARN:
        pytest.skip("sklearn not installed")
    obj = objective_hi_hi()
    history = [
        make_array(30, 40, "matrix", 6.4, 166.0),
        make_array(64, 24, "hard", 11.8, 214.0),
        make_array(50, 60, "boundary", 8.2, 178.0),
    ]
    sur = oa.ObjectiveSurrogate(obj).fit(history)
    assert sur.mode == "gaussian-process"
    assert sur.n_points == 27
    assert sur.best_observed()["array"] == "hard"

    proposals = sur.propose_sites(2, round_index=1, stage_position=(50.0, 37.5))
    assert len(proposals) == 2
    spacing = obj.constraints.min_site_spacing_um
    for p in proposals:
        assert p["source"] == "bo_acquisition"
        assert p["acquisition"]["ei"] >= 0.0
        assert 0.0 <= p["x_frac"] <= 1.0 and 0.0 <= p["y_frac"] <= 1.0
        for cx, cy in sur.array_centers:
            assert np.hypot(p["x_um"] - cx, p["y_um"] - cy) >= spacing
    p1, p2 = proposals
    assert np.hypot(p1["x_um"] - p2["x_um"], p1["y_um"] - p2["y_um"]) >= spacing


def test_surrogate_idw_fallback_and_untrained(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = objective_hi_hi()
    monkeypatch.setattr(oa, "HAVE_SKLEARN", False)
    sur = oa.ObjectiveSurrogate(obj).fit([make_array(30, 40, "a", 7.0, 170.0)])
    assert sur.mode == "idw-heuristic"
    assert len(sur.propose_sites(1)) == 1
    maps = sur.posterior_maps(n=20)
    assert maps["ei"].shape == (20, 20)

    empty = oa.ObjectiveSurrogate(obj).fit([])
    assert empty.mode == "untrained"
    assert empty.propose_sites(1) == []
    assert empty.summary()["max_ei"] == 0.0


def test_rank_targets_by_objective_directions() -> None:
    targets = [
        {"label": "matrix_reference", "reason": "dark matrix", "expected_hardness": "lower"},
        {"label": "hard_phase_center", "reason": "bright second-phase grain", "expected_hardness": "higher"},
        {"label": "grain_boundary", "reason": "boundary between grains"},
    ]
    hi = oa.rank_targets_by_objective(targets, objective_hi_hi())
    assert hi[0]["label"] == "hard_phase_center"
    assert hi[0]["priority"] == 1
    assert all("objective_score" in t and "rank_reason" in t for t in hi)

    soft = parse_objective_fallback("locate the softest areas")
    lo = oa.rank_targets_by_objective(targets, soft)
    assert lo[0]["label"] == "matrix_reference"


# ---------------------------------------------------------------------------
# measurement critic: objective criteria vs legacy behavior
# ---------------------------------------------------------------------------


def _critic_state(history: list[dict], criteria: dict | None) -> dict:
    state = create_initial_state("q")
    state["indent_history"] = history
    state["visited_target_keys"] = ["a", "b", "c"]
    state["decision_history"] = [{}] * 5
    if criteria is not None:
        state["objective_criteria"] = criteria
    return state


def test_critic_legacy_path_unchanged_without_criteria() -> None:
    contrasty = [make_array(10, 10, "a", 6.0, 170.0), make_array(50, 50, "b", 12.0, 170.0)]
    out = measurement_critic_node(_critic_state(contrasty, None))
    assert out["measurement_critique_decision"] == "hypothesis_confirmed"

    uniform = [make_array(10, 10, "a", 7.0, 170.0), make_array(50, 50, "b", 7.2, 170.0)]
    out = measurement_critic_node(_critic_state(uniform, None))
    assert out["measurement_critique_decision"] == "hypothesis_invalid"


def test_critic_objective_path() -> None:
    crit = {
        "min_indent_arrays": 2, "min_visited_locations": 3, "min_improvement_pct": 10.0,
        "property_weights": {"hardness": 0.5, "modulus": 0.5},
        "directions": {"hardness": "maximize", "modulus": "maximize"},
        "objective_summary": "maximize hardness and modulus",
    }
    contrasty = [make_array(10, 10, "a", 6.0, 166.0), make_array(50, 50, "b", 12.0, 214.0)]
    out = measurement_critic_node(_critic_state(contrasty, crit))
    assert out["measurement_critique_decision"] == "hypothesis_confirmed"
    assert "[OBJECTIVE CRITERIA]" in out["measurement_critique"]

    flat_few = [make_array(10, 10, "a", 7.0, 170.0), make_array(50, 50, "b", 7.1, 171.0)]
    out = measurement_critic_node(_critic_state(flat_few, crit))
    assert out["measurement_critique_decision"] == "more_data_needed"

    flat_many = [make_array(10 * i, 10 * i, f"a{i}", 7.0 + 0.05 * i, 170.0) for i in range(4)]
    out = measurement_critic_node(_critic_state(flat_many, crit))
    assert out["measurement_critique_decision"] == "hypothesis_invalid"


def test_critic_objective_minimize_direction() -> None:
    crit = {
        "min_indent_arrays": 2, "min_visited_locations": 3, "min_improvement_pct": 10.0,
        "property_weights": {"hardness": 1.0},
        "directions": {"hardness": "minimize"},
    }
    history = [make_array(10, 10, "soft", 6.0, 166.0), make_array(50, 50, "hard", 12.0, 214.0)]
    out = measurement_critic_node(_critic_state(history, crit))
    assert out["measurement_critique_decision"] == "hypothesis_confirmed"
    assert "'soft'" in out["measurement_critique"]


def test_critic_objective_raises_evidence_minimums() -> None:
    crit = {
        "min_indent_arrays": 4, "min_visited_locations": 3, "min_improvement_pct": 10.0,
        "property_weights": {"hardness": 1.0}, "directions": {"hardness": "maximize"},
    }
    history = [make_array(10, 10, "a", 6.0, 166.0), make_array(50, 50, "b", 12.0, 214.0)]
    out = measurement_critic_node(_critic_state(history, crit))
    assert out["measurement_critique_decision"] == "more_data_needed"


# ---------------------------------------------------------------------------
# orchestrator decisions (table-driven rules)
# ---------------------------------------------------------------------------


def test_decision_continue_image_targets_before_data() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    state = state_with([], 0, obj, orch)
    progress = orch.evaluate_progress(state, None)
    decision = orch.decide_next_action(state, progress)
    assert decision.action == "continue"
    assert decision.proposal_source == "image_targets"
    assert decision.advisor == "rules"


def test_decision_switches_to_bo_with_data() -> None:
    if not oa.HAVE_SKLEARN:
        pytest.skip("sklearn not installed")
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    history = [make_array(30, 40, "a", 6.4, 166.0), make_array(64, 24, "b", 7.0, 170.0)]
    state = state_with(history, 2, obj, orch)
    sur = oa.ObjectiveSurrogate(obj).fit(history)
    progress = orch.evaluate_progress(state, sur)
    decision = orch.decide_next_action(state, progress, sur)
    assert decision.action == "continue"
    assert decision.proposal_source == "bo_acquisition"


def test_decision_objective_met() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    history = [
        make_array(30, 40, "matrix", 6.4, 166.0),
        make_array(64, 24, "hard", 11.8, 214.0),
        make_array(50, 60, "boundary", 8.2, 178.0),
    ]
    state = state_with(history, 3, obj, orch)
    sur = oa.ObjectiveSurrogate(obj).fit(history)
    progress = orch.evaluate_progress(state, sur)
    assert progress.all_criteria_met
    assert orch.decide_next_action(state, progress, sur).action == "objective_met"


def test_decision_budget_and_round_exhaustion() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    history = [make_array(10 * i, 10, f"a{i}", 7.0, 170.0) for i in range(5)]  # 45 indents
    state = state_with(history, 2, obj, orch)
    progress = orch.evaluate_progress(state, None)
    assert progress.budget_remaining == 0
    assert orch.decide_next_action(state, progress).action == "budget_exhausted"

    state = state_with(history[:2], 2, obj, orch)
    state["orchestrator_round"] = obj.constraints.max_rounds
    progress = orch.evaluate_progress(state, None)
    assert orch.decide_next_action(state, progress).action == "budget_exhausted"


def test_objective_met_beats_budget_exhaustion() -> None:
    """A met objective is a success even when the remaining budget can't fund another array."""
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    # 5 arrays = full 45-indent budget, with strong contrast -> criteria met AND budget gone
    history = [make_array(15 * i + 10, 12 * i + 8, f"a{i}", 6.0 + 1.6 * i, 165.0 + 12.0 * i)
               for i in range(5)]
    state = state_with(history, 5, obj, orch)
    sur = oa.ObjectiveSurrogate(obj).fit(history)
    progress = orch.evaluate_progress(state, sur)
    assert progress.budget_remaining == 0
    assert progress.all_criteria_met
    assert orch.decide_next_action(state, progress, sur).action == "objective_met"


def test_decision_refine_then_escalate_on_invalid_hypothesis() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    history = [make_array(10, 10, "a", 7.0, 170.0), make_array(50, 50, "b", 7.05, 170.5)]
    state = state_with(history, 2, obj, orch)
    state["measurement_critique_decision"] = "hypothesis_invalid"
    progress = orch.evaluate_progress(state, None)
    decision = orch.decide_next_action(state, progress)
    assert decision.action == "refine_hypothesis"

    state.update(orch.decision_state_updates(state, decision, progress))
    decision2 = orch.decide_next_action(state, progress)
    assert decision2.action == "escalate_to_human"


def test_decision_escalates_on_flat_landscape() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    history = [make_array(10 * i, 10, f"a{i}", 7.0, 170.0) for i in range(3)]
    state = state_with(history, 3, obj, orch)

    class FlatSurrogate:
        def summary(self):
            return {"mode": "gaussian-process", "best": {"scalar_value": 0.5},
                    "baseline_scalar": 0.5, "max_ei": 0.0001, "mean_scalar_sigma": 0.01,
                    "n_points": 27}

    progress = orch.evaluate_progress(state, FlatSurrogate())
    decision = orch.decide_next_action(state, progress)
    assert decision.action == "escalate_to_human"


def test_decision_state_updates_append_full_list() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    state = state_with([], 0, obj, orch)
    progress = orch.evaluate_progress(state, None)
    decision = orch.decide_next_action(state, progress)
    updates = orch.decision_state_updates(state, decision, progress)
    assert len(updates["orchestrator_decisions"]) == 1
    state.update(updates)
    updates2 = orch.decision_state_updates(state, decision, progress)
    assert len(updates2["orchestrator_decisions"]) == 2
    assert updates2["objective_progress"]["indent_budget"] == 45


def test_briefings_cover_all_roles_and_seed_literature() -> None:
    obj = objective_hi_hi()
    orch = OrchestratorAgent(obj, use_llm=False)
    state = create_initial_state("test")
    updates = orch.briefing_state_updates(state)
    briefings = updates["orchestrator_briefings"]
    assert set(briefings) == {"image_analysis", "literature", "hypothesis",
                              "planning", "measurement_strategy", "measurement_critic"}
    assert all(len(text) > 30 for text in briefings.values())
    assert "carbon film" in updates["literature_search_query"]
