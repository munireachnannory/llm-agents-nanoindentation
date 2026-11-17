"""
method.py
=========
The indentation METHOD PARAMETERS (target load, target depth, approach,
strain rate, poisson/metal/drift, and the post-processing `review_changes`)
as a validated, agent-controllable model.

Historically these values were a static `dict[str, Any]` copied unchanged from
config into every round. This module is now the single chokepoint:
`resolve_plan()` applies each feature's fixed/agent authority contract, and
`validate_method_plan()` checks the resolved value before it can reach the
instrument.

Physical model
--------------
The iMicro is controlled by depth AND load simultaneously — whichever limit
is reached first terminates loading. So `target depth` and `target load` are
never independent: one is the SETPOINT (what you're measuring against) and
the other is a CEILING (set loose enough not to bind). `control_mode` picks
which is which. `predict_binding_limit()` makes the "which one actually
binds" question explicit and testable, using a Berkovich contact-area
estimate (A = 24.5*hc^2, hc in um) — units work out with no conversion
factor: 1 mN/um^2 = 1 GPa, so H[GPa] * A[um^2] = P[mN].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# The exact InView key spelling `apply_method_parameters_ui` expects (all
# lowercase — the locator table for the "eh" method is all-lowercase too, so
# this must stay lowercase or the value silently fails to match and InView
# keeps whatever was already typed).
CANONICAL_SETUP_KEYS = (
    "target depth",
    "target load",
    "surface approach distance",
    "surface approach velocity",
    "strain rate",
    "poisson",
    "metal",
    "drift",
)

# canonical setup key -> MethodPlan attribute name
_KEY_TO_ATTR = {
    "target depth": "target_depth_nm",
    "target load": "target_load_mN",
    "surface approach distance": "surface_approach_distance_nm",
    "surface approach velocity": "surface_approach_velocity_nm_s",
    "strain rate": "strain_rate_per_s",
    "poisson": "poisson",
    "metal": "metal",
    "drift": "drift",
}
_ATTR_TO_KEY = {v: k for k, v in _KEY_TO_ATTR.items()}

# Matches today's `IndentDefaultSettings.setup` factory in config.py — used
# only as a defensive fallback if a MethodConstraints is missing a key.
_FACTORY_DEFAULTS: dict[str, float] = {
    "poisson": 0.3,
    "metal": 0,
    "drift": 0,
    "target load": 1000,
    "target depth": 1000,
    "surface approach distance": 2000,
    "surface approach velocity": 100,
    "strain rate": 0.2,
}

CITE_DEPTH_KEY = "Depth to Cite Results (nm)"
MIN_DEPTH_KEY = "Min Depth for Results"

# Below this the tip crashes the surface at speed. This is an absolute
# physical floor, independent of whatever range a config file declares —
# checked directly on the resolved value, never just clamped into range.
SAFETY_MIN_APPROACH_DISTANCE_NM = 2000.0

# Berkovich projected contact area A = k * hc^2 (hc in um, A in um^2).
BERKOVICH_AREA_CONSTANT = 24.5


class MethodPlanError(ValueError):
    """A method plan violates a hard safety or physical-validity rule."""


@dataclass
class ParameterSpec:
    """One parameter: who decides it, within what range, and what the
    operator knows. `mode="fixed"` means always use `default`; `mode="agent"`
    means an agent may choose within `[min, max]`, seeded by `default` and
    the `candidates` menu, unless `agent_allowed=False` — which makes
    `mode="agent"` a config-time error, not a silently-ignored setting."""

    name: str
    mode: str = "fixed"                        # "fixed" | "agent"
    default: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    candidates: Optional[list[float]] = None
    units: str = ""
    agent_allowed: bool = True
    note: str = ""

    def clamp(self, value: float) -> tuple[float, bool]:
        """Returns (clamped_value, was_clamped)."""
        v = value
        if self.min is not None and v < self.min:
            v = self.min
        if self.max is not None and v > self.max:
            v = self.max
        return v, (v != value)


@dataclass
class MethodConstraints:
    """The operator's contract with the agent — bounds and prior knowledge,
    not values. Built once per campaign from config."""

    control_mode: str = "depth_controlled"       # "depth_controlled" | "load_controlled"
    control_mode_agent_selectable: bool = False
    params: dict[str, ParameterSpec] = field(default_factory=dict)
    review_changes: dict[str, Any] = field(default_factory=dict)   # operator-locked
    derive_cite_depth: bool = True
    cite_depth_frac: float = 0.25
    spacing_depth_multiplier: float = 20.0        # advisory only until Stage 4

    def validate_config(self) -> None:
        """Config-time check: `agent_allowed=False` params must not be set to
        `mode="agent"`. Called automatically by `from_settings`."""
        for name, spec in self.params.items():
            if spec.mode not in ("fixed", "agent"):
                raise MethodPlanError(
                    f"'{name}' has unknown mode {spec.mode!r}; use 'fixed' or 'agent'.")
            if spec.mode == "agent" and not spec.agent_allowed:
                raise MethodPlanError(
                    f"'{name}' is agent_allowed=False (safety/material-constant "
                    f"lock) and cannot be set to mode='agent'.")
        if self.control_mode not in ("depth_controlled", "load_controlled"):
            raise MethodPlanError(f"unknown control_mode {self.control_mode!r}")

    @classmethod
    def from_settings(cls, method_control: Any, review_changes: dict[str, Any]) -> "MethodConstraints":
        """Build from the pydantic `MethodControlSettings` in config.py plus
        the (still operator-locked) `review_changes` dict."""
        params: dict[str, ParameterSpec] = {}
        for key in CANONICAL_SETUP_KEYS:
            p = method_control.params.get(key)
            if p is None:
                params[key] = ParameterSpec(name=key, default=_FACTORY_DEFAULTS[key])
                continue
            params[key] = ParameterSpec(
                name=key, mode=p.mode, default=p.default, min=p.min, max=p.max,
                candidates=list(p.candidates) if p.candidates else None,
                units=p.units, agent_allowed=p.agent_allowed, note=p.note)
        out = cls(
            control_mode=method_control.control_mode,
            control_mode_agent_selectable=method_control.control_mode_agent_selectable,
            params=params, review_changes=dict(review_changes),
            derive_cite_depth=method_control.derive_cite_depth,
            cite_depth_frac=method_control.cite_depth_frac,
            spacing_depth_multiplier=method_control.spacing_depth_multiplier)
        out.validate_config()
        return out


@dataclass
class MethodPlan:
    """Fully-resolved, validated parameters for ONE array. `to_setup()` is
    the ONLY sanctioned way to build the dict that reaches the instrument."""

    control_mode: str
    target_depth_nm: float
    target_load_mN: float
    poisson: float
    metal: int
    drift: int
    surface_approach_distance_nm: float
    surface_approach_velocity_nm_s: float
    strain_rate_per_s: float
    review_changes: dict[str, Any] = field(default_factory=dict)
    binding_limit: str = "unknown"                    # "depth" | "load" | "unknown"
    predicted_peak_load_mN: Optional[float] = None
    decided_by: dict[str, str] = field(default_factory=dict)   # key -> "operator" | "agent"
    decision_modes: dict[str, str] = field(default_factory=dict)
    decision_reasons: dict[str, str] = field(default_factory=dict)
    requested_values: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_setup(self) -> dict[str, Any]:
        return {key: getattr(self, attr) for key, attr in _KEY_TO_ATTR.items()}

    def to_card_arguments(self) -> dict[str, Any]:
        """What the human sees on the approval card before approving."""
        args: dict[str, Any] = {
            "control_mode": self.control_mode,
            "target_depth_nm": self.target_depth_nm,
            "target_load_mN": self.target_load_mN,
            "predicted_binding_limit": self.binding_limit,
        }
        if self.predicted_peak_load_mN is not None:
            args["predicted_peak_load_mN"] = round(self.predicted_peak_load_mN, 1)
        # Collapse to one word when every param was decided the same way (the
        # common case until agent control is opted into per-parameter) — the
        # full per-key breakdown is still available via to_journal().
        distinct = set(self.decided_by.values())
        args["decided_by"] = (next(iter(distinct)) if len(distinct) == 1
                              else self.decided_by)
        if self.rationale:
            args["rationale"] = self.rationale
        if self.warnings:
            args["warnings"] = list(self.warnings)
        return args

    def feature_decisions(self) -> dict[str, dict[str, Any]]:
        """Final, per-feature authority record after validation/clamping."""
        selected = {key: getattr(self, attr) for key, attr in _KEY_TO_ATTR.items()}
        selected["control_mode"] = self.control_mode
        out: dict[str, dict[str, Any]] = {}
        for key, value in selected.items():
            requested = self.requested_values.get(key)
            row = {
                "mode": self.decision_modes.get(key, "fixed"),
                "selected_value": value,
                "decided_by": self.decided_by.get(key, "operator"),
                "reason": self.decision_reasons.get(
                    key, "Configured fixed/default value retained."),
            }
            if key in self.requested_values:
                row["requested_value"] = requested
                row["adjusted"] = requested != value
            out[key] = row
        return out

    def to_journal(self) -> dict[str, Any]:
        return {
            **self.to_card_arguments(),
            "setup": self.to_setup(),
            "review_changes": dict(self.review_changes),
            "feature_decisions": self.feature_decisions(),
        }


def predict_binding_limit(
    hardness_gpa: float, target_depth_nm: float, target_load_mN: float,
) -> tuple[str, float]:
    """Which limit stops loading first, and the predicted peak load, given
    an estimated hardness at the site. Berkovich A = 24.5 * hc^2 (hc in um);
    H[GPa] * A[um^2] = P[mN] with no conversion factor."""
    hc_um = target_depth_nm / 1000.0
    predicted_load_mN = hardness_gpa * BERKOVICH_AREA_CONSTANT * hc_um ** 2
    if predicted_load_mN <= target_load_mN:
        return "depth", predicted_load_mN
    return "load", target_load_mN


def resolve_plan(
    constraints: MethodConstraints,
    overrides: Optional[dict[str, float]] = None,
    *,
    control_mode_override: Optional[str] = None,
    override_reasons: Optional[dict[str, str]] = None,
    hardness_estimate_gpa: Optional[float] = None,
    current_spacing_um: Optional[float] = None,
    rationale: str = "operator defaults (no agent control active)",
) -> MethodPlan:
    """Resolve every canonical setup parameter: `mode="fixed"` -> `default`;
    `mode="agent"` -> `overrides[key]` if supplied, else `default` (no agent
    has decided yet). Then validates. This is the single entry point campaign
    code should call to build a `MethodPlan`."""
    constraints.validate_config()
    overrides = overrides or {}
    override_reasons = override_reasons or {}
    values: dict[str, float] = {}
    decided_by: dict[str, str] = {}
    decision_modes: dict[str, str] = {}
    decision_reasons: dict[str, str] = {}
    requested_values: dict[str, Any] = dict(overrides)
    for key in CANONICAL_SETUP_KEYS:
        spec = constraints.params.get(key) or ParameterSpec(name=key, default=_FACTORY_DEFAULTS[key])
        decision_modes[key] = spec.mode
        if spec.mode == "agent" and key in overrides:
            values[key] = overrides[key]
            decided_by[key] = "agent"
            decision_reasons[key] = override_reasons.get(key) or rationale
        else:
            values[key] = spec.default
            decided_by[key] = "operator"
            if spec.mode == "agent":
                decision_reasons[key] = (
                    "No usable agent value was supplied; configured default retained.")
            else:
                decision_reasons[key] = "Feature is fixed by operator configuration."

    if control_mode_override is not None:
        requested_values["control_mode"] = control_mode_override
    decision_modes["control_mode"] = (
        "agent" if constraints.control_mode_agent_selectable else "fixed")
    if (constraints.control_mode_agent_selectable
            and control_mode_override in ("depth_controlled", "load_controlled")):
        control_mode = str(control_mode_override)
        decided_by["control_mode"] = "agent"
        decision_reasons["control_mode"] = (
            override_reasons.get("control_mode") or rationale)
    else:
        control_mode = constraints.control_mode
        decided_by["control_mode"] = "operator"
        decision_reasons["control_mode"] = (
            "Feature is fixed by operator configuration."
            if not constraints.control_mode_agent_selectable
            else "No usable agent value was supplied; configured default retained.")

    plan = MethodPlan(
        control_mode=control_mode,
        target_depth_nm=values["target depth"],
        target_load_mN=values["target load"],
        poisson=values["poisson"],
        metal=int(values["metal"]),
        drift=int(values["drift"]),
        surface_approach_distance_nm=values["surface approach distance"],
        surface_approach_velocity_nm_s=values["surface approach velocity"],
        strain_rate_per_s=values["strain rate"],
        review_changes=dict(constraints.review_changes),
        decided_by=decided_by,
        decision_modes=decision_modes,
        decision_reasons=decision_reasons,
        requested_values=requested_values,
        rationale=rationale,
    )
    return validate_method_plan(
        plan, constraints,
        hardness_estimate_gpa=hardness_estimate_gpa,
        current_spacing_um=current_spacing_um)


def validate_method_plan(
    plan: MethodPlan,
    constraints: MethodConstraints,
    *,
    hardness_estimate_gpa: Optional[float] = None,
    current_spacing_um: Optional[float] = None,
) -> MethodPlan:
    """The single chokepoint. Raises `MethodPlanError` on safety violations,
    clamps-and-warns on range violations. Mutates and returns `plan`."""
    warnings = list(plan.warnings)

    # 1. approach-distance safety floor — RAISE, never clamp. Checked first,
    #    on the raw resolved value, independent of any configured range.
    if plan.surface_approach_distance_nm < SAFETY_MIN_APPROACH_DISTANCE_NM:
        raise MethodPlanError(
            f"surface approach distance {plan.surface_approach_distance_nm:g} nm is below "
            f"the {SAFETY_MIN_APPROACH_DISTANCE_NM:g} nm safety floor — the tip would "
            "crash the surface at speed.")

    # 2. range clamp every scalar against its declared [min, max]
    for key, attr in _KEY_TO_ATTR.items():
        spec = constraints.params.get(key)
        if spec is None:
            continue
        raw = getattr(plan, attr)
        clamped, was_clamped = spec.clamp(raw)
        if was_clamped:
            warnings.append(
                f"{key} {raw:g} clamped to [{spec.min}, {spec.max}] -> {clamped:g}")
            setattr(plan, attr, clamped)

    # 3. control-mode consistency — only checkable with a hardness estimate
    if hardness_estimate_gpa is not None:
        limit, predicted = predict_binding_limit(
            hardness_estimate_gpa, plan.target_depth_nm, plan.target_load_mN)
        plan.binding_limit = limit
        plan.predicted_peak_load_mN = predicted
        if plan.control_mode == "depth_controlled" and limit == "load":
            warnings.append(
                f"predicted peak load {predicted:.1f} mN would bind before "
                f"target depth {plan.target_depth_nm:g} nm is reached — this round "
                "will effectively run load-controlled despite control_mode="
                "'depth_controlled'.")
        elif plan.control_mode == "load_controlled" and limit == "depth":
            warnings.append(
                f"predicted contact at target load {plan.target_load_mN:g} mN needs only "
                f"~{predicted:.1f} mN — target depth {plan.target_depth_nm:g} nm binds "
                "first despite control_mode='load_controlled'.")

    # 4. cite-depth coupling — closes the H = E = 0 trap
    if constraints.derive_cite_depth:
        cite_depth = round(constraints.cite_depth_frac * plan.target_depth_nm, 1)
        plan.review_changes[CITE_DEPTH_KEY] = cite_depth
    else:
        cite_depth = plan.review_changes.get(CITE_DEPTH_KEY)

    if cite_depth is not None:
        if cite_depth >= plan.target_depth_nm:
            raise MethodPlanError(
                f"{CITE_DEPTH_KEY}={cite_depth:g} is >= target depth "
                f"{plan.target_depth_nm:g} nm — InView will report Hardness = "
                "Modulus = 0 because results are cited at a depth the indent "
                "never reaches. Lower the cite depth or raise the target depth.")
        min_depth = plan.review_changes.get(MIN_DEPTH_KEY)
        if min_depth is not None and min_depth >= cite_depth:
            raise MethodPlanError(
                f"{MIN_DEPTH_KEY}={min_depth:g} is >= {CITE_DEPTH_KEY}={cite_depth:g} "
                "— results would never qualify for citation.")

    # 5. spacing floor — advisory only until Stage 4 gives this module
    #    ownership of grid/spacing
    if current_spacing_um is not None:
        floor_um = constraints.spacing_depth_multiplier * (plan.target_depth_nm / 1000.0)
        if current_spacing_um < floor_um:
            warnings.append(
                f"spacing {current_spacing_um:g} um is below the "
                f"{constraints.spacing_depth_multiplier:g}x-depth floor "
                f"({floor_um:.1f} um) for a {plan.target_depth_nm:g} nm indent — "
                "neighboring indents' plastic zones may overlap.")

    plan.warnings = warnings
    return plan
