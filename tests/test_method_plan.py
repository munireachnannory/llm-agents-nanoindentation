"""Validated method parameters with fixed/default and delegated-agent control.

Every parameter ships mode="fixed" — these tests exist to prove that change is
behavior-identical to today's unguarded `dict(cfg.indent_defaults.setup)`
(the golden tests), and that the new safety chokepoint actually blocks the
two real hazards already found in this codebase: an unsafe approach distance,
and the "Depth to Cite Results" trap that silently reports Hardness=Modulus=0
(see AI_Agents_for_Nanoindentation--main-1's `target depth: 200` against a
fixed `cite: 250`).
"""
import pytest

from nanoindentation_agents.campaign.journal import CampaignJournal
from nanoindentation_agents.config import IndentDefaultSettings, load_config
from nanoindentation_agents.method import (
    CITE_DEPTH_KEY,
    MIN_DEPTH_KEY,
    MethodConstraints,
    MethodPlan,
    MethodPlanError,
    ParameterSpec,
    predict_binding_limit,
    resolve_plan,
    validate_method_plan,
)


def _plain_plan(**overrides) -> MethodPlan:
    """A MethodPlan with today's config defaults, for validator-only tests."""
    base = dict(
        control_mode="depth_controlled", target_depth_nm=1000.0, target_load_mN=1000.0,
        poisson=0.3, metal=0, drift=0, surface_approach_distance_nm=2000.0,
        surface_approach_velocity_nm_s=100.0, strain_rate_per_s=0.2,
        review_changes={MIN_DEPTH_KEY: 30, CITE_DEPTH_KEY: 250},
    )
    base.update(overrides)
    return MethodPlan(**base)


def _constraints(**overrides) -> MethodConstraints:
    defaults = IndentDefaultSettings()
    return MethodConstraints.from_settings(
        overrides.pop("method_control", defaults.method_control),
        overrides.pop("review_changes", defaults.review_changes))


# --------------------------------------------------------------- golden tests

def test_default_resolve_plan_matches_todays_factory_setup():
    """Stage 0 must change no behavior: every param ships mode=fixed, so
    resolve_plan()'s setup dict must equal today's unguarded factory dict."""
    defaults = IndentDefaultSettings()
    constraints = MethodConstraints.from_settings(defaults.method_control, defaults.review_changes)
    plan = resolve_plan(constraints)
    assert plan.to_setup() == {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in defaults.setup.items()}


def test_default_resolve_plan_matches_default_yaml():
    """The method_control block added to config/default.yaml must resolve to
    the same setup dict as the (untouched) `indent_defaults.setup` block."""
    cfg = load_config()
    constraints = MethodConstraints.from_settings(
        cfg.indent_defaults.method_control, cfg.indent_defaults.review_changes)
    plan = resolve_plan(constraints)
    for key, value in cfg.indent_defaults.setup.items():
        assert plan.to_setup()[key] == pytest.approx(float(value)), key


def test_default_cite_depth_matches_todays_fixed_value():
    """target depth 1000nm * cite_depth_frac 0.25 = 250nm — today's fixed
    config value. Confirms the derivation doesn't silently change reporting
    for anyone who hasn't touched target depth yet."""
    defaults = IndentDefaultSettings()
    constraints = MethodConstraints.from_settings(defaults.method_control, defaults.review_changes)
    plan = resolve_plan(constraints)
    assert plan.review_changes[CITE_DEPTH_KEY] == pytest.approx(250.0)


# -------------------------------------------------------------- safety floor

def test_approach_distance_below_floor_raises_not_clamps():
    plan = _plain_plan(surface_approach_distance_nm=20.0)
    with pytest.raises(MethodPlanError, match="safety floor"):
        validate_method_plan(plan, _constraints())


def test_approach_distance_at_floor_is_fine():
    plan = _plain_plan(surface_approach_distance_nm=2000.0)
    validate_method_plan(plan, _constraints())  # must not raise


# ------------------------------------------------------------- range clamps

def test_out_of_range_value_is_clamped_and_warned():
    constraints = _constraints()
    plan = _plain_plan(target_depth_nm=5000.0)  # above configured max=2000
    out = validate_method_plan(plan, constraints)
    assert out.target_depth_nm == 2000.0
    assert any("target depth" in w for w in out.warnings)


# -------------------------------------------------------- cite-depth trap

def test_cite_depth_at_or_above_target_depth_raises():
    """The exact hazard on disk in --main-1: target depth 200nm against a
    fixed cite depth of 250nm silently reports Hardness = Modulus = 0."""
    constraints = _constraints()
    constraints.derive_cite_depth = False   # simulate the un-derived/legacy case
    plan = _plain_plan(target_depth_nm=200.0,
                       review_changes={MIN_DEPTH_KEY: 30, CITE_DEPTH_KEY: 250})
    with pytest.raises(MethodPlanError, match="Hardness = Modulus = 0|Modulus = 0"):
        validate_method_plan(plan, constraints)


def test_derive_cite_depth_recomputes_and_avoids_the_trap():
    constraints = _constraints()
    constraints.derive_cite_depth = True
    constraints.cite_depth_frac = 0.25
    plan = _plain_plan(target_depth_nm=200.0,
                       review_changes={MIN_DEPTH_KEY: 30, CITE_DEPTH_KEY: 250})
    out = validate_method_plan(plan, constraints)
    assert out.review_changes[CITE_DEPTH_KEY] == pytest.approx(50.0)


def test_min_depth_above_cite_depth_raises():
    constraints = _constraints()
    constraints.derive_cite_depth = False
    plan = _plain_plan(target_depth_nm=1000.0,
                       review_changes={MIN_DEPTH_KEY: 300, CITE_DEPTH_KEY: 250})
    with pytest.raises(MethodPlanError, match="never qualify"):
        validate_method_plan(plan, constraints)


# -------------------------------------------------------- binding-limit math

def test_predict_binding_limit_hand_calculation():
    """H=5 GPa, 1000nm depth -> P = 5 * 24.5 * 1.0^2 = 122.5 mN, well under a
    1000 mN ceiling -> depth-bound."""
    limit, predicted = predict_binding_limit(5.0, 1000.0, 1000.0)
    assert limit == "depth"
    assert predicted == pytest.approx(122.5)


def test_predict_binding_limit_load_bound_when_load_ceiling_too_low():
    limit, predicted = predict_binding_limit(5.0, 1000.0, 50.0)
    assert limit == "load"
    assert predicted == 50.0


def test_control_mode_consistency_warns_when_load_would_bind_first():
    constraints = _constraints()
    plan = _plain_plan(target_depth_nm=2000.0, target_load_mN=1000.0)
    out = validate_method_plan(plan, constraints, hardness_estimate_gpa=11.0)
    # H=11 GPa, 2000nm -> P = 11 * 24.5 * 2.0^2 = 1078 mN > 1000 mN ceiling
    assert out.binding_limit == "load"
    assert any("bind before target depth" in w for w in out.warnings)


# ----------------------------------------------------- spacing floor (warn)

def test_spacing_below_floor_warns_not_raises():
    constraints = _constraints()
    plan = _plain_plan(target_depth_nm=1000.0)
    out = validate_method_plan(plan, constraints, current_spacing_um=5.0)
    assert any("spacing" in w for w in out.warnings)


def test_spacing_above_floor_is_silent():
    constraints = _constraints()
    plan = _plain_plan(target_depth_nm=1000.0)
    out = validate_method_plan(plan, constraints, current_spacing_um=100.0)
    assert not any("spacing" in w for w in out.warnings)


# ------------------------------------------------------- agent_allowed lock

def test_agent_allowed_false_with_mode_agent_raises_at_config_load():
    bad = MethodConstraints(params={
        "poisson": ParameterSpec(name="poisson", mode="agent", default=0.3, agent_allowed=False),
    })
    with pytest.raises(MethodPlanError, match="agent_allowed=False"):
        bad.validate_config()


def test_agent_allowed_true_with_mode_agent_is_fine():
    ok = MethodConstraints(params={
        "target depth": ParameterSpec(name="target depth", mode="agent", default=1000,
                                      min=150, max=2000, agent_allowed=True),
    })
    ok.validate_config()  # must not raise


def test_agent_mode_uses_override_when_supplied():
    constraints = _constraints()
    constraints.params["target depth"].mode = "agent"
    plan = resolve_plan(constraints, overrides={"target depth": 600.0})
    assert plan.target_depth_nm == 600.0
    assert plan.decided_by["target depth"] == "agent"


def test_fixed_mode_ignores_override():
    constraints = _constraints()  # "target depth" stays mode="fixed"
    plan = resolve_plan(constraints, overrides={"target depth": 600.0})
    assert plan.target_depth_nm == 1000.0
    assert plan.decided_by["target depth"] == "operator"


# ------------------------------------------------------- card + journal visibility

def test_card_arguments_collapses_uniform_decided_by():
    """When every param was decided the same way (Stage 0: always), the card
    shows one word, not an 8-key dict — keeps the approval card's JSON under
    the display truncation limit and legible."""
    constraints = _constraints()
    plan = resolve_plan(constraints)
    assert plan.to_card_arguments()["decided_by"] == "operator"


def test_card_arguments_json_fits_under_display_truncation():
    """notebook_ui.ApprovalGate._card_html truncates the args JSON blob —
    the method-plan fields must fit, or they render cut off mid-string."""
    import json
    constraints = _constraints()
    plan = resolve_plan(constraints, current_spacing_um=15.0)  # triggers a warning
    args = {"sample_name": "test_01", "grid": "2x2", "spacing_um": 15.0,
            "method_type": "eh", **plan.to_card_arguments()}
    assert len(json.dumps(args, default=str)) < 2000


def test_method_plan_history_reads_back_from_journal(tmp_path):
    """The post-hoc audit view: ARRAY_PLACED events carry the full resolved
    MethodPlan, and method_plan_history() turns them into a readable table."""
    journal = CampaignJournal(tmp_path, "test_run")
    constraints = _constraints()
    plan = resolve_plan(constraints)
    journal.append("ARRAY_PLACED", sample="test_01", label="grain_boundary",
                   grid="2x2", method_plan=plan.to_journal())
    df = journal.method_plan_history()
    assert len(df) == 1
    assert df.iloc[0]["target_depth_nm"] == 1000.0
    assert df.iloc[0]["control_mode"] == "depth_controlled"
