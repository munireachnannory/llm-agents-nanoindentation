"""Authority, fallback, validation, and audit tests for the recipe advisor."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nanoindentation_agents.agents.measurement_strategy_agent import (
    GridConstraints,
    MeasurementStrategyAgent,
    StrategyPlanError,
    resolve_grid_plan,
)
from nanoindentation_agents.campaign.journal import CampaignJournal
from nanoindentation_agents.campaign.approval import (
    AUTONOMOUS_ACK,
    AutonomousHardwareApproval,
)
from nanoindentation_agents.campaign.runner import CampaignConfig, CampaignRunner
from nanoindentation_agents.agents.orchestrator_agent import OrchestratorAgent
from nanoindentation_agents.config import load_config
from nanoindentation_agents.method import MethodConstraints
from nanoindentation_agents.objective import ObjectiveConstraints, ResearchObjective
from nanoindentation_agents.state import create_initial_state


class FakeLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def chat(self, **_kwargs) -> str:
        self.calls += 1
        return json.dumps(self.payload)


def _agent(
    payload: dict | None = None,
    *,
    enabled: bool = True,
    objective_grid: tuple[int, int] = (2, 2),
) -> tuple[MeasurementStrategyAgent, object, FakeLLM]:
    cfg = load_config()
    cfg.measurement_strategy.enabled = enabled
    objective = ResearchObjective(
        raw_text="Map particles with as few indents as possible.",
        constraints=ObjectiveConstraints(
            indent_budget=20,
            grid_nx=objective_grid[0],
            grid_ny=objective_grid[1],
            spacing_um=15.0,
        ),
    )
    method = MethodConstraints.from_settings(
        cfg.indent_defaults.method_control,
        cfg.indent_defaults.review_changes,
    )
    grid = GridConstraints.from_settings(
        cfg.measurement_strategy.grid_control,
        runtime_defaults={
            "grid_nx": objective_grid[0],
            "grid_ny": objective_grid[1],
            "spacing_um": objective.constraints.spacing_um,
            "rotation_deg": cfg.indent_defaults.rotation,
        },
    )
    fake = FakeLLM(payload or {})
    agent = MeasurementStrategyAgent(
        objective,
        cfg,
        method,
        grid,
        default_method_type=cfg.indent_defaults.method_type,
        use_llm=True,
        llm=fake,
    )
    return agent, cfg, fake


def _state() -> dict:
    state = create_initial_state("minimum sufficient particle map")
    state["image_analysis"] = {
        "image_quality": {"usable": True, "confidence": 0.9},
        "interpretation": "Several isolated particles in epoxy.",
    }
    return state


def test_disabled_agent_preserves_existing_defaults_without_calling_llm() -> None:
    agent, _cfg, fake = _agent({"action": "skip"}, enabled=False)
    plan = agent.plan({"label": "particle_1"}, _state(), remaining_budget=20)
    assert fake.calls == 0
    assert plan.source == "disabled"
    assert plan.action == "indent"
    assert (plan.grid_plan.grid_nx, plan.grid_plan.grid_ny) == (2, 2)
    assert plan.method_plan.target_depth_nm == 1000.0
    assert not plan.agent_features


def test_agent_controls_only_individually_delegated_features() -> None:
    payload = {
        "action": "indent",
        "method_type": "constant",
        "control_mode": "load_controlled",
        "grid": {"grid_nx": 1, "grid_ny": 1, "spacing_um": 20, "rotation_deg": 45},
        "method": {"target depth": 600, "target load": 500, "strain rate": 0.5},
        "confidence": 0.91,
        "reason": "One central indent is sufficient for this small particle.",
        "feature_reasons": {"grid_nx": "minimum sufficient coverage"},
    }
    agent, cfg, fake = _agent(payload)
    cfg.measurement_strategy.site_action.mode = "agent"
    for key in ("grid_nx", "grid_ny", "spacing_um"):
        agent.grid_constraints.params[key].mode = "agent"
    # Rotation, method type, control mode, and strain rate remain fixed.
    plan = agent.plan(
        {"label": "particle_1", "usable_diameter_um": 25.0},
        _state(),
        remaining_budget=20,
    )
    assert fake.calls == 1
    assert plan.source == "llm"
    assert (plan.grid_plan.grid_nx, plan.grid_plan.grid_ny) == (1, 1)
    assert plan.grid_plan.spacing_um == 20.0
    assert plan.grid_plan.rotation_deg == 0.0
    assert plan.method_type == "eh"
    assert plan.method_plan.control_mode == "depth_controlled"
    assert plan.method_plan.target_depth_nm == 600.0
    assert plan.method_plan.target_load_mN == 500.0
    assert plan.method_plan.strain_rate_per_s == 0.2
    assert plan.feature_decisions()["strain rate"]["adjusted"] is True
    assert set(plan.agent_features) >= {
        "site_action", "grid_nx", "grid_ny", "spacing_um",
        "target depth", "target load",
    }
    assert plan.requires_human_approval


def test_low_confidence_response_falls_back_to_defaults() -> None:
    agent, cfg, _fake = _agent({
        "action": "skip",
        "grid": {"grid_nx": 1, "grid_ny": 1},
        "confidence": 0.2,
        "reason": "uncertain",
    })
    cfg.measurement_strategy.site_action.mode = "agent"
    agent.grid_constraints.params["grid_nx"].mode = "agent"
    agent.grid_constraints.params["grid_ny"].mode = "agent"
    plan = agent.plan({"label": "uncertain"}, _state(), remaining_budget=20)
    assert plan.source == "low_confidence_fallback"
    assert plan.action == "indent"
    assert (plan.grid_plan.grid_nx, plan.grid_plan.grid_ny) == (2, 2)
    assert not plan.agent_features
    action_row = plan.feature_decisions()["site_action"]
    assert action_row["requested_value"] == "skip"
    assert action_row["selected_value"] == "indent"
    assert action_row["adjusted"] is True


def test_over_budget_agent_grid_is_rejected_to_safe_default() -> None:
    agent, _cfg, _fake = _agent({
        "action": "indent",
        "grid": {"grid_nx": 5, "grid_ny": 5},
        "confidence": 0.9,
        "reason": "too many measurements",
    }, objective_grid=(1, 1))
    agent.grid_constraints.params["grid_nx"].mode = "agent"
    agent.grid_constraints.params["grid_ny"].mode = "agent"
    plan = agent.plan({"label": "particle"}, _state(), remaining_budget=4)
    assert plan.source == "validation_fallback"
    assert (plan.grid_plan.grid_nx, plan.grid_plan.grid_ny) == (1, 1)
    assert any("only 4 remain" in warning for warning in plan.warnings)
    assert plan.feature_decisions()["grid_nx"]["requested_value"] == 5


def test_fixed_grid_that_cannot_fit_budget_is_a_hard_error() -> None:
    agent, _cfg, _fake = _agent({}, enabled=False, objective_grid=(2, 2))
    with pytest.raises(StrategyPlanError, match="only 3 remain"):
        agent.plan({"label": "particle"}, _state(), remaining_budget=3)


def test_explicit_particle_footprint_rejects_oversized_agent_grid() -> None:
    agent, _cfg, _fake = _agent({
        "grid": {"grid_nx": 3, "grid_ny": 1, "spacing_um": 15},
        "confidence": 0.9,
        "reason": "sample across particle",
    }, objective_grid=(1, 1))
    for key in ("grid_nx", "grid_ny", "spacing_um"):
        agent.grid_constraints.params[key].mode = "agent"
    plan = agent.plan(
        {"label": "small_particle", "usable_width_um": 20.0, "usable_height_um": 20.0},
        _state(),
        remaining_budget=10,
    )
    assert plan.source == "validation_fallback"
    assert plan.grid_plan.n_indents == 1
    assert any("usable width" in warning for warning in plan.warnings)


def test_agent_selected_depth_spacing_pair_must_clear_plastic_zone_floor() -> None:
    agent, _cfg, _fake = _agent({
        "grid": {"grid_nx": 2, "grid_ny": 2, "spacing_um": 6},
        "method": {"target depth": 600},
        "confidence": 0.9,
        "reason": "dense grid",
    }, objective_grid=(1, 1))
    for key in ("grid_nx", "grid_ny", "spacing_um"):
        agent.grid_constraints.params[key].mode = "agent"
    plan = agent.plan({"label": "particle"}, _state(), remaining_budget=10)
    assert plan.source == "validation_fallback"
    assert plan.grid_plan.n_indents == 1
    assert any("x-depth floor" in warning for warning in plan.warnings)


def test_control_mode_and_rotation_can_be_delegated_explicitly() -> None:
    agent, _cfg, _fake = _agent({
        "control_mode": "load_controlled",
        "grid": {"rotation_deg": 90},
        "confidence": 0.8,
        "reason": "operator enabled both features",
    })
    agent.method_constraints.control_mode_agent_selectable = True
    agent.grid_constraints.params["rotation_deg"].mode = "agent"
    plan = agent.plan({"label": "large_particle"}, _state(), remaining_budget=20)
    assert plan.method_plan.control_mode == "load_controlled"
    assert plan.method_plan.decided_by["control_mode"] == "agent"
    assert plan.grid_plan.rotation_deg == 90.0
    assert plan.grid_plan.decided_by["rotation_deg"] == "agent"


def test_grid_resolver_logs_requested_clamped_and_selected_values() -> None:
    agent, _cfg, _fake = _agent({}, objective_grid=(1, 1))
    agent.grid_constraints.params["spacing_um"].mode = "agent"
    plan = resolve_grid_plan(
        agent.grid_constraints,
        {"spacing_um": 1000},
        remaining_budget=10,
    )
    row = plan.feature_decisions()["spacing_um"]
    assert row["requested_value"] == 1000
    assert row["selected_value"] == 100.0
    assert row["adjusted"] is True


def test_journal_exposes_summary_and_per_feature_audit_tables(tmp_path) -> None:
    agent, cfg, _fake = _agent({
        "action": "indent",
        "grid": {"grid_nx": 1, "grid_ny": 1},
        "confidence": 0.9,
        "reason": "minimum sufficient",
    })
    cfg.measurement_strategy.site_action.mode = "agent"
    agent.grid_constraints.params["grid_nx"].mode = "agent"
    agent.grid_constraints.params["grid_ny"].mode = "agent"
    plan = agent.plan({"label": "particle_1"}, _state(), remaining_budget=20)
    journal = CampaignJournal(tmp_path, "strategy_test")
    journal.append(
        "MEASUREMENT_STRATEGY_DECISION",
        target="particle_1",
        **plan.to_journal(),
    )
    summary = journal.measurement_strategy_history()
    features = journal.measurement_strategy_feature_history()
    assert summary.iloc[0]["grid"] == "1x1"
    assert {"site_action", "grid_nx", "target depth", "execution_tools"} <= set(
        features["feature"])
    assert any(str(name).startswith("review:") for name in features["feature"])
    assert features.loc[features["feature"] == "grid_nx", "decided_by"].iloc[0] == "agent"


def test_campaign_runner_uses_resolved_recipe_and_journals_it(tmp_path) -> None:
    cfg = load_config()
    objective = ResearchObjective(
        raw_text="Map particles with minimum indents.",
        constraints=ObjectiveConstraints(
            indent_budget=20, grid_nx=2, grid_ny=2, spacing_um=15.0),
    )
    state = _state()
    journal = CampaignJournal(tmp_path, "runner_strategy")
    runner = CampaignRunner(
        session=object(),
        motion=object(),
        exporter=object(),
        journal=journal,
        objective=objective,
        orchestrator=OrchestratorAgent(objective, cfg, use_llm=False),
        state=state,
        app_cfg=cfg,
        campaign=CampaignConfig(
            use_simulator=True,
            run_id="runner_strategy",
            use_measurement_strategy_llm=False,
        ),
    )
    plan = runner._plan_site(
        {"label": "particle_1", "x_frac": 0.5, "y_frac": 0.5},
        (100.0, 200.0),
        source="image_targets",
        remaining_budget=20,
    )
    assert plan is not None
    assert (plan["grid_nx"], plan["grid_ny"]) == (2, 2)
    assert plan["setup"]["target depth"] == 1000.0
    assert plan["rotation_deg"] == 0.0
    assert plan["measurement_strategy"].source == "disabled"
    assert journal.events("MEASUREMENT_STRATEGY_DECISION")
    indent_card = plan["cards"][1]
    assert indent_card["arguments"]["strategy_source"] == "disabled"
    assert indent_card["arguments"]["agent_features"] == "none"


class _GateStub:
    def __init__(self) -> None:
        self.decision = None
        self.approved_by = ""
        self.note = ""

    def approve(self, *, approved_by: str, note: str) -> None:
        self.decision = "approve"
        self.approved_by = approved_by
        self.note = note


class _GenericNoninteractivePolicy:
    interactive = False

    def approve(self, gate) -> None:
        gate.approve(approved_by="generic automation", note="generic")


def test_real_hardware_generic_policy_cannot_approve_agent_recipe() -> None:
    runner = object.__new__(CampaignRunner)
    runner.c = SimpleNamespace(use_simulator=False)
    plans = [{"measurement_strategy": SimpleNamespace(
        requires_human_approval=True)}]
    with pytest.raises(PermissionError, match="not authorized"):
        runner._approve_with_policy(
            _GateStub(), plans, _GenericNoninteractivePolicy())


def test_acknowledged_hardware_policy_approves_agent_recipe() -> None:
    runner = object.__new__(CampaignRunner)
    runner.c = SimpleNamespace(use_simulator=False)
    plans = [{"measurement_strategy": SimpleNamespace(
        requires_human_approval=True)}]
    gate = _GateStub()
    policy = AutonomousHardwareApproval("operator", ack=AUTONOMOUS_ACK)
    runner._approve_with_policy(gate, plans, policy)
    assert gate.decision == "approve"
    assert "acknowledged by operator" in gate.approved_by
