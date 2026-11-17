"""Bounded, auditable per-site measurement strategy advisor.

The LLM in this module can only propose values. It receives no instrument
tools, and every proposal is resolved through fixed/agent authority contracts
plus deterministic method, grid, budget, and safety validation before a plan
can reach a human approval gate.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Optional

from ..method import (
    CANONICAL_SETUP_KEYS,
    CITE_DEPTH_KEY,
    MethodConstraints,
    MethodPlan,
    ParameterSpec,
    resolve_plan,
)
from ..registry.tool_catalog import parse_llm_json


GRID_KEYS = ("grid_nx", "grid_ny", "spacing_um", "rotation_deg")
EXECUTION_TOOLS = (
    "indent_move",
    "capture_video_panel_remote",
    "starting_tests",
    "start_test_normal",
    "process_review_file",
)


class StrategyPlanError(ValueError):
    """A proposed measurement recipe cannot be made safe or valid."""


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _short(value: Any, limit: int = 400) -> str:
    return str(value or "").strip()[:limit]


@dataclass
class GridConstraints:
    """Operator authority and numeric bounds for one indentation array."""

    params: dict[str, ParameterSpec] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        runtime_defaults: dict[str, float],
    ) -> "GridConstraints":
        params: dict[str, ParameterSpec] = {}
        for key in GRID_KEYS:
            configured = settings.params.get(key)
            default = float(runtime_defaults[key])
            if configured is None:
                params[key] = ParameterSpec(name=key, default=default)
                continue
            params[key] = ParameterSpec(
                name=key,
                mode=configured.mode,
                default=default,
                min=configured.min,
                max=configured.max,
                candidates=list(configured.candidates) if configured.candidates else None,
                units=configured.units,
                agent_allowed=configured.agent_allowed,
                note=configured.note,
            )
        out = cls(params=params)
        out.validate_config()
        return out

    def validate_config(self) -> None:
        for key, spec in self.params.items():
            if spec.mode not in ("fixed", "agent"):
                raise StrategyPlanError(
                    f"'{key}' has unknown mode {spec.mode!r}; use 'fixed' or 'agent'.")
            if spec.mode == "agent" and not spec.agent_allowed:
                raise StrategyPlanError(
                    f"'{key}' is agent_allowed=False and cannot use mode='agent'.")
            default = _finite_float(spec.default)
            if default is None:
                raise StrategyPlanError(f"'{key}' has a non-numeric default.")
            if key in ("grid_nx", "grid_ny") and default < 1:
                raise StrategyPlanError(f"'{key}' must be at least 1.")
            if key == "spacing_um" and default <= 0:
                raise StrategyPlanError("'spacing_um' must be positive.")


@dataclass
class GridPlan:
    grid_nx: int
    grid_ny: int
    spacing_um: float
    rotation_deg: float
    decided_by: dict[str, str] = field(default_factory=dict)
    decision_modes: dict[str, str] = field(default_factory=dict)
    decision_reasons: dict[str, str] = field(default_factory=dict)
    requested_values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_indents(self) -> int:
        return int(self.grid_nx) * int(self.grid_ny)

    def feature_decisions(self) -> dict[str, dict[str, Any]]:
        selected = {
            "grid_nx": self.grid_nx,
            "grid_ny": self.grid_ny,
            "spacing_um": self.spacing_um,
            "rotation_deg": self.rotation_deg,
        }
        out: dict[str, dict[str, Any]] = {}
        for key, value in selected.items():
            row = {
                "mode": self.decision_modes.get(key, "fixed"),
                "selected_value": value,
                "decided_by": self.decided_by.get(key, "operator"),
                "reason": self.decision_reasons.get(
                    key, "Configured fixed/default value retained."),
            }
            if key in self.requested_values:
                requested = self.requested_values[key]
                row["requested_value"] = requested
                row["adjusted"] = requested != value
            out[key] = row
        return out

    def to_card_arguments(self) -> dict[str, Any]:
        distinct = set(self.decided_by.values())
        return {
            "grid": f"{self.grid_nx}x{self.grid_ny}",
            "spacing_um": self.spacing_um,
            "rotation_deg": self.rotation_deg,
            "grid_decided_by": (
                next(iter(distinct)) if len(distinct) == 1 else dict(self.decided_by)),
        }

    def to_journal(self) -> dict[str, Any]:
        return {
            **self.to_card_arguments(),
            "grid_nx": self.grid_nx,
            "grid_ny": self.grid_ny,
            "n_indents": self.n_indents,
            "warnings": list(self.warnings),
            "feature_decisions": self.feature_decisions(),
        }


def resolve_grid_plan(
    constraints: GridConstraints,
    overrides: Optional[dict[str, Any]] = None,
    *,
    override_reasons: Optional[dict[str, str]] = None,
    rationale: str = "operator defaults (no agent control active)",
    remaining_budget: Optional[int] = None,
    target: Optional[dict[str, Any]] = None,
) -> GridPlan:
    """Resolve fixed/agent geometry and enforce hard array/budget checks."""
    constraints.validate_config()
    overrides = dict(overrides or {})
    override_reasons = dict(override_reasons or {})
    values: dict[str, float] = {}
    decided_by: dict[str, str] = {}
    modes: dict[str, str] = {}
    reasons: dict[str, str] = {}
    requested: dict[str, Any] = {}
    warnings: list[str] = []

    for key in GRID_KEYS:
        spec = constraints.params[key]
        modes[key] = spec.mode
        if key in overrides:
            requested[key] = overrides[key]
        proposed = _finite_float(overrides.get(key))
        if spec.mode == "agent" and proposed is not None:
            value, clamped = spec.clamp(proposed)
            values[key] = value
            decided_by[key] = "agent"
            reasons[key] = override_reasons.get(key) or rationale
            if clamped:
                warnings.append(
                    f"{key} {proposed:g} clamped to [{spec.min}, {spec.max}] -> {value:g}")
        else:
            value = _finite_float(spec.default)
            if value is None:
                raise StrategyPlanError(f"'{key}' default is not numeric.")
            values[key] = value
            decided_by[key] = "operator"
            reasons[key] = (
                "Feature is fixed by operator configuration."
                if spec.mode == "fixed"
                else "No usable agent value was supplied; configured default retained.")

    for key in ("grid_nx", "grid_ny"):
        rounded = int(round(values[key]))
        if not math.isclose(values[key], rounded):
            warnings.append(f"{key} {values[key]:g} rounded to integer {rounded}")
        values[key] = rounded

    plan = GridPlan(
        grid_nx=int(values["grid_nx"]),
        grid_ny=int(values["grid_ny"]),
        spacing_um=float(values["spacing_um"]),
        rotation_deg=float(values["rotation_deg"]),
        decided_by=decided_by,
        decision_modes=modes,
        decision_reasons=reasons,
        requested_values=requested,
        warnings=warnings,
    )
    if plan.grid_nx < 1 or plan.grid_ny < 1:
        raise StrategyPlanError("grid dimensions must both be at least 1.")
    if plan.spacing_um <= 0:
        raise StrategyPlanError("grid spacing must be positive.")
    if remaining_budget is not None and plan.n_indents > int(remaining_budget):
        raise StrategyPlanError(
            f"proposed {plan.grid_nx}x{plan.grid_ny} grid needs {plan.n_indents} "
            f"indents but only {int(remaining_budget)} remain.")

    # Enforce an explicitly supplied usable feature footprint. Missing geometry
    # remains a warning because legacy image targets did not record it.
    target = target or {}
    usable_w = _finite_float(target.get("usable_width_um"))
    usable_h = _finite_float(target.get("usable_height_um"))
    diameter = _finite_float(target.get("usable_diameter_um"))
    if diameter is not None:
        usable_w = usable_w if usable_w is not None else diameter
        usable_h = usable_h if usable_h is not None else diameter
    footprint_w = max(0, plan.grid_nx - 1) * plan.spacing_um
    footprint_h = max(0, plan.grid_ny - 1) * plan.spacing_um
    if usable_w is not None and footprint_w > usable_w:
        raise StrategyPlanError(
            f"grid width {footprint_w:g} um exceeds the target's usable width "
            f"{usable_w:g} um.")
    if usable_h is not None and footprint_h > usable_h:
        raise StrategyPlanError(
            f"grid height {footprint_h:g} um exceeds the target's usable height "
            f"{usable_h:g} um.")
    if plan.n_indents > 1 and usable_w is None and usable_h is None:
        plan.warnings.append(
            "Target has no measured usable footprint; confirm this multi-indent grid "
            "stays inside the particle/region at the human approval gate.")
    return plan


@dataclass
class MeasurementStrategyPlan:
    """Fully resolved recipe for one proposed site."""

    action: str
    method_type: str
    grid_plan: GridPlan
    method_plan: MethodPlan
    source: str
    confidence: Optional[float]
    reason: str
    action_decision: dict[str, Any]
    method_type_decision: dict[str, Any]
    agent_proposal: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    require_human_approval_for_agent_changes: bool = True

    @property
    def agent_features(self) -> list[str]:
        return [
            key for key, row in self.feature_decisions().items()
            if row.get("decided_by") == "agent"
        ]

    @property
    def requires_human_approval(self) -> bool:
        return bool(
            self.action == "indent"
            and self.require_human_approval_for_agent_changes
            and self.agent_features)

    def feature_decisions(self) -> dict[str, dict[str, Any]]:
        out = {
            "site_action": dict(self.action_decision),
            "method_type": dict(self.method_type_decision),
            **self.grid_plan.feature_decisions(),
            **self.method_plan.feature_decisions(),
        }
        out["execution_tools"] = {
            "mode": "fixed_workflow",
            "selected_value": list(EXECUTION_TOOLS),
            "decided_by": "workflow",
            "reason": (
                "CampaignRunner owns the validated move, capture, indent, and export "
                "sequence; the strategy agent has no execution tools."),
        }
        for key, value in self.method_plan.review_changes.items():
            derived = key == CITE_DEPTH_KEY
            out[f"review:{key}"] = {
                "mode": "derived" if derived else "fixed",
                "selected_value": value,
                "decided_by": "validator" if derived else "operator",
                "reason": (
                    "Derived from the validated target depth to keep result citation valid."
                    if derived else "Post-processing setting is operator locked."),
            }
        proposed: dict[str, Any] = {}
        if "action" in self.agent_proposal:
            proposed["site_action"] = self.agent_proposal["action"]
        if "method_type" in self.agent_proposal:
            proposed["method_type"] = self.agent_proposal["method_type"]
        if "control_mode" in self.agent_proposal:
            proposed["control_mode"] = self.agent_proposal["control_mode"]
        if isinstance(self.agent_proposal.get("grid"), dict):
            proposed.update(self.agent_proposal["grid"])
        if isinstance(self.agent_proposal.get("method"), dict):
            proposed.update(self.agent_proposal["method"])
        for key, requested in proposed.items():
            if key in out and "requested_value" not in out[key]:
                out[key]["requested_value"] = requested
                out[key]["adjusted"] = requested != out[key].get("selected_value")
        return out

    def to_card_arguments(self) -> dict[str, Any]:
        return {
            "strategy_source": self.source,
            "strategy_action": self.action,
            "agent_features": self.agent_features or "none",
            "strategy_confidence": self.confidence,
            "strategy_reason": self.reason[:300],
        }

    def to_journal(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "method_type": self.method_type,
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "agent_proposal": dict(self.agent_proposal),
            "proposal_applied": self.source == "llm",
            "agent_features": self.agent_features,
            "requires_human_approval": self.requires_human_approval,
            "warnings": list(dict.fromkeys(
                [*self.warnings, *self.grid_plan.warnings, *self.method_plan.warnings])),
            "grid_plan": self.grid_plan.to_journal(),
            "method_plan": self.method_plan.to_journal(),
            "feature_decisions": self.feature_decisions(),
        }


class MeasurementStrategyAgent:
    """LLM advisor whose output is constrained to one site's recipe."""

    ROLE = "measurement_strategy"

    def __init__(
        self,
        objective: Any,
        cfg: Any,
        method_constraints: MethodConstraints,
        grid_constraints: GridConstraints,
        *,
        default_method_type: str,
        use_llm: bool = True,
        llm: Any = None,
    ):
        self.objective = objective
        self.cfg = cfg
        self.settings = cfg.measurement_strategy
        self.method_constraints = method_constraints
        self.grid_constraints = grid_constraints
        self.default_method_type = default_method_type
        self.use_llm = bool(use_llm and self.settings.use_llm)
        self._llm = llm

    def _get_llm(self) -> Any:
        if self._llm is None:
            from ..registry.models import get_llm_for_role
            self._llm = get_llm_for_role(self.ROLE, self.cfg)
        return self._llm

    def _call_llm(self, prompt: str) -> Optional[str]:
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            future = ex.submit(
                self._get_llm().chat,
                user_message=prompt,
                system_message=(
                    "You are a nanoindentation measurement-strategy advisor. "
                    "Minimize irreversible indents while satisfying the human objective. "
                    "Return strict JSON only. You do not operate the instrument."),
            )
            return future.result(timeout=float(self.settings.llm_timeout_s))
        except (FuturesTimeout, Exception) as exc:  # any LLM failure uses fixed defaults
            print(f"[measurement-strategy] LLM unavailable ({type(exc).__name__}); "
                  "using configured defaults")
            return None
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _spec_dict(spec: ParameterSpec) -> dict[str, Any]:
        return {
            "mode": spec.mode,
            "default": spec.default,
            "min": spec.min,
            "max": spec.max,
            "candidates": spec.candidates or [],
            "units": spec.units,
            "agent_allowed": spec.agent_allowed,
            "note": spec.note,
        }

    def _has_agent_features(self) -> bool:
        if self.settings.site_action.mode == "agent":
            return True
        if self.settings.method_type.mode == "agent":
            return True
        if self.method_constraints.control_mode_agent_selectable:
            return True
        return any(s.mode == "agent" for s in self.grid_constraints.params.values()) or any(
            s.mode == "agent" for s in self.method_constraints.params.values())

    def _prompt(self, target: dict[str, Any], state: dict[str, Any], remaining_budget: int) -> str:
        image_analysis = state.get("image_analysis") or {}
        history = []
        for row in list(state.get("indent_history") or [])[-6:]:
            history.append({
                "region": row.get("area_name"),
                "grid": f"{row.get('grid_nx', '?')}x{row.get('grid_ny', '?')}",
                "hardness_gpa": row.get("hardness_mean"),
                "modulus_gpa": row.get("modulus_mean"),
                "valid_indents": row.get("valid_indent_count"),
            })
        grid_contract = {
            key: self._spec_dict(spec) for key, spec in self.grid_constraints.params.items()
        }
        method_contract = {
            key: self._spec_dict(spec) for key, spec in self.method_constraints.params.items()
        }
        choice_contract = {
            "site_action": self.settings.site_action.model_dump(),
            "method_type": {
                **self.settings.method_type.model_dump(),
                "default": self.default_method_type,
            },
            "control_mode": {
                "mode": ("agent" if self.method_constraints.control_mode_agent_selectable
                         else "fixed"),
                "default": self.method_constraints.control_mode,
                "candidates": ["depth_controlled", "load_controlled"],
            },
        }
        directive = (state.get("orchestrator_briefings") or {}).get(
            "measurement_strategy", "")
        context = {
            "target": target,
            "image_quality": image_analysis.get("image_quality"),
            "image_interpretation": image_analysis.get("interpretation"),
            "image_recommendations": image_analysis.get("recommendations"),
            "prior_measurements": history,
            "remaining_indent_budget": int(remaining_budget),
        }
        return (
            "Choose the minimum sufficient measurement recipe for ONE proposed site. "
            "A fixed feature must retain its default even if you would prefer another value. "
            "Use action=skip for epoxy, brass, boundaries, defects, duplicate/low-confidence "
            "sites, or a site that adds no necessary evidence. Prefer 1x1 for a small or "
            "moderate homogeneous particle; use multiple indents only when size or "
            "heterogeneity makes them necessary. Never exceed the remaining budget.\n\n"
            f"HUMAN OBJECTIVE:\n{self.objective.summary_text()}\n\n"
            f"ORCHESTRATOR DIRECTIVE:\n{directive or '(none)'}\n\n"
            f"CURRENT CONTEXT:\n{json.dumps(context, default=str)[:7000]}\n\n"
            f"CHOICE CONTRACTS:\n{json.dumps(choice_contract, default=str)}\n\n"
            f"GRID CONTRACTS:\n{json.dumps(grid_contract, default=str)}\n\n"
            f"METHOD CONTRACTS:\n{json.dumps(method_contract, default=str)}\n\n"
            "Return ONLY this JSON shape:\n"
            '{"action":"indent|skip","method_type":"eh",'
            '"control_mode":"depth_controlled|load_controlled",'
            '"grid":{"grid_nx":1,"grid_ny":1,"spacing_um":15,"rotation_deg":0},'
            '"method":{"target depth":600,"target load":1000,"strain rate":0.2},'
            '"confidence":0.0,"reason":"short overall reason",'
            '"feature_reasons":{"grid_nx":"why","target depth":"why"}}'
        )

    def _request(
        self,
        target: dict[str, Any],
        state: dict[str, Any],
        remaining_budget: int,
    ) -> tuple[dict[str, Any], dict[str, Any], str, Optional[float], str]:
        if not self.settings.enabled:
            return {}, {}, "disabled", None, "Strategy agent disabled; configured defaults retained."
        if not self._has_agent_features():
            return {}, {}, "fixed_configuration", None, "All recipe features are fixed by the operator."
        if not self.use_llm:
            return {}, {}, "llm_disabled_fallback", None, "LLM unavailable/disabled; configured defaults retained."
        response = self._call_llm(self._prompt(target, state, remaining_budget))
        if not response:
            return {}, {}, "llm_unavailable_fallback", None, "LLM unavailable; configured defaults retained."
        try:
            data = parse_llm_json(response)
        except Exception as exc:
            return {}, {}, "parse_fallback", None, (
                f"Agent response was not valid JSON ({type(exc).__name__}); configured defaults retained.")
        confidence = _finite_float(data.get("confidence"))
        if confidence is None or confidence < float(self.settings.min_confidence):
            return {}, data, "low_confidence_fallback", confidence, (
                f"Agent confidence {confidence!r} is below the required "
                f"{float(self.settings.min_confidence):g}; configured defaults retained.")
        return data, data, "llm", min(max(confidence, 0.0), 1.0), (
            _short(data.get("reason")) or "Agent selected the minimum sufficient recipe.")

    @staticmethod
    def _resolve_choice(
        name: str,
        spec: Any,
        requested: Any,
        reason: str,
    ) -> tuple[str, dict[str, Any], list[str]]:
        if spec.mode not in ("fixed", "agent"):
            raise StrategyPlanError(
                f"'{name}' has unknown mode {spec.mode!r}; use 'fixed' or 'agent'.")
        if spec.mode == "agent" and not spec.agent_allowed:
            raise StrategyPlanError(
                f"'{name}' is agent_allowed=False and cannot use mode='agent'.")
        default = str(spec.default).strip()
        candidates = [str(v).strip() for v in spec.candidates]
        requested_text = str(requested).strip() if requested is not None else ""
        warnings: list[str] = []
        if spec.mode == "agent" and requested_text and (
                not candidates or requested_text in candidates):
            selected, who = requested_text, "agent"
            why = reason
        else:
            selected, who = default, "operator"
            why = ("Feature is fixed by operator configuration." if spec.mode == "fixed"
                   else "No usable agent value was supplied; configured default retained.")
            if spec.mode == "agent" and requested_text and candidates:
                warnings.append(
                    f"{name} {requested_text!r} is not in the allowed choices {candidates}; "
                    f"using {default!r}.")
        row = {
            "mode": spec.mode,
            "selected_value": selected,
            "decided_by": who,
            "reason": why,
        }
        if requested is not None:
            row["requested_value"] = requested
            row["adjusted"] = requested_text != selected
        return selected, row, warnings

    def _resolve(
        self,
        data: dict[str, Any],
        *,
        source: str,
        confidence: Optional[float],
        reason: str,
        agent_proposal: dict[str, Any],
        target: dict[str, Any],
        remaining_budget: int,
    ) -> MeasurementStrategyPlan:
        feature_reasons = data.get("feature_reasons") or {}
        if not isinstance(feature_reasons, dict):
            feature_reasons = {}
        action, action_row, action_warnings = self._resolve_choice(
            "site_action", self.settings.site_action, data.get("action"),
            _short(feature_reasons.get("site_action") or reason))
        method_type_spec = self.settings.method_type.model_copy(
            update={"default": self.default_method_type})
        method_type, method_type_row, method_type_warnings = self._resolve_choice(
            "method_type", method_type_spec, data.get("method_type"),
            _short(feature_reasons.get("method_type") or reason))

        grid_raw = data.get("grid") if isinstance(data.get("grid"), dict) else {}
        method_raw = data.get("method") if isinstance(data.get("method"), dict) else {}
        method_overrides = {
            key: value for key, value in method_raw.items()
            if key in CANONICAL_SETUP_KEYS and _finite_float(value) is not None
        }
        method_overrides = {key: float(value) for key, value in method_overrides.items()}
        grid_plan = resolve_grid_plan(
            self.grid_constraints,
            grid_raw,
            override_reasons={key: _short(feature_reasons.get(key) or reason)
                              for key in GRID_KEYS},
            rationale=reason,
            remaining_budget=remaining_budget,
            target=target,
        )
        method_plan = resolve_plan(
            self.method_constraints,
            overrides=method_overrides,
            control_mode_override=data.get("control_mode"),
            override_reasons={key: _short(feature_reasons.get(key) or reason)
                              for key in (*CANONICAL_SETUP_KEYS, "control_mode")},
            current_spacing_um=grid_plan.spacing_um,
            rationale=reason,
        )
        if isinstance(agent_proposal.get("method"), dict):
            method_plan.requested_values.update({
                key: value for key, value in agent_proposal["method"].items()
                if key in CANONICAL_SETUP_KEYS})
        if grid_plan.n_indents > 1:
            floor_um = (self.method_constraints.spacing_depth_multiplier
                        * method_plan.target_depth_nm / 1000.0)
            coupled_agent_choice = (
                grid_plan.decided_by.get("spacing_um") == "agent"
                or method_plan.decided_by.get("target depth") == "agent")
            if coupled_agent_choice and grid_plan.spacing_um < floor_um:
                raise StrategyPlanError(
                    f"agent-selected spacing {grid_plan.spacing_um:g} um is below the "
                    f"{self.method_constraints.spacing_depth_multiplier:g}x-depth floor "
                    f"({floor_um:g} um) for {method_plan.target_depth_nm:g} nm depth.")
        return MeasurementStrategyPlan(
            action=action,
            method_type=method_type,
            grid_plan=grid_plan,
            method_plan=method_plan,
            source=source,
            confidence=confidence,
            reason=reason,
            action_decision=action_row,
            method_type_decision=method_type_row,
            agent_proposal=dict(agent_proposal),
            warnings=[*action_warnings, *method_type_warnings],
            require_human_approval_for_agent_changes=bool(
                self.settings.require_human_approval_for_agent_changes),
        )

    def plan(
        self,
        target: dict[str, Any],
        state: dict[str, Any],
        *,
        remaining_budget: int,
    ) -> MeasurementStrategyPlan:
        data, agent_proposal, source, confidence, reason = self._request(
            target, state, remaining_budget)
        try:
            return self._resolve(
                data, source=source, confidence=confidence, reason=reason,
                agent_proposal=agent_proposal, target=target,
                remaining_budget=remaining_budget)
        except (StrategyPlanError, TypeError, ValueError) as exc:
            # Agent-generated recipes fail closed to the existing defaults. A
            # bad fixed/default configuration still raises on the second pass.
            fallback_reason = (
                f"Agent proposal rejected by validation ({type(exc).__name__}: {exc}); "
                "configured defaults retained.")
            fallback = self._resolve(
                {}, source="validation_fallback", confidence=confidence,
                reason=fallback_reason, agent_proposal=agent_proposal,
                target=target,
                remaining_budget=remaining_budget)
            fallback.warnings.insert(0, str(exc))
            return fallback

    def control_contract_rows(self) -> list[dict[str, Any]]:
        """Notebook-friendly view of every strategy-controlled feature."""
        rows = [
            {
                "feature": "site_action",
                "mode": self.settings.site_action.mode,
                "default": self.settings.site_action.default,
                "allowed": self.settings.site_action.candidates,
                "agent active": bool(
                    self.settings.enabled and self.settings.site_action.mode == "agent"),
                "safety": "human gate remains before indentation",
            },
            {
                "feature": "method_type",
                "mode": self.settings.method_type.mode,
                "default": self.default_method_type,
                "allowed": self.settings.method_type.candidates,
                "agent active": bool(
                    self.settings.enabled and self.settings.method_type.mode == "agent"),
                "safety": "operator allow-list",
            },
            {
                "feature": "control_mode",
                "mode": ("agent" if self.method_constraints.control_mode_agent_selectable
                         else "fixed"),
                "default": self.method_constraints.control_mode,
                "allowed": ["depth_controlled", "load_controlled"],
                "agent active": bool(
                    self.settings.enabled
                    and self.method_constraints.control_mode_agent_selectable),
                "safety": "validated with target depth/load",
            },
        ]
        for key, spec in self.grid_constraints.params.items():
            rows.append({
                "feature": key,
                "mode": spec.mode,
                "default": spec.default,
                "allowed": spec.candidates or f"{spec.min}..{spec.max}",
                "agent active": bool(self.settings.enabled and spec.mode == "agent"),
                "safety": "bounds + budget + feature-footprint check",
            })
        for key, spec in self.method_constraints.params.items():
            rows.append({
                "feature": key,
                "mode": spec.mode,
                "default": spec.default,
                "allowed": spec.candidates or f"{spec.min}..{spec.max}",
                "agent active": bool(self.settings.enabled and spec.mode == "agent"),
                "safety": ("operator locked" if not spec.agent_allowed
                           else "method bounds + physical validation"),
            })
        for key, value in self.method_constraints.review_changes.items():
            derived = key == CITE_DEPTH_KEY and self.method_constraints.derive_cite_depth
            rows.append({
                "feature": f"review:{key}",
                "mode": "derived" if derived else "fixed",
                "default": value,
                "allowed": "derived from target depth" if derived else "operator value",
                "agent active": False,
                "safety": "post-processing validation",
            })
        rows.append({
            "feature": "execution_tools",
            "mode": "fixed_workflow",
            "default": list(EXECUTION_TOOLS),
            "allowed": list(EXECUTION_TOOLS),
            "agent active": False,
            "safety": "agent receives no instrument tools",
        })
        return rows
