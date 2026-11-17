"""
objective.py
============
Structured research objectives for the objective-driven orchestrator.

A ``ResearchObjective`` is compiled from a free-text user goal — via LLM
when a Hugging Face token is available, via a deterministic keyword
parser otherwise — and travels through agent state as a plain dict so
sub-agents, critics and the acquisition module all align to one target.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


VALID_DIRECTIONS = ("maximize", "minimize", "target_range")
KNOWN_PROPERTIES = ("hardness", "modulus")

_PROPERTY_PATTERNS = {
    "hardness": re.compile(r"\bhard(?:er|est|ness)?\b|\bsoft(?:er|est|ness)?\b|\bductile\b", re.IGNORECASE),
    "modulus": re.compile(
        r"\bmodul(?:us|i)\b|\bstiff(?:er|est|ness)?\b|\belastic\b|\bcomplian(?:t|ce)\b|\bflexib(?:le|ility)\b",
        re.IGNORECASE),
}
# property mentions that themselves imply "lower is better"
_INVERSE_MENTIONS = re.compile(r"^(?:soft|ductile|complian|flexib)", re.IGNORECASE)
_MAXIMIZE_WORDS = re.compile(
    r"\bhigh(?:er|est)?\b|\bharder\b|\bstiffer\b|\bmax(?:imum|imize|imise)?\b|"
    r"\bincrease[sd]?\b|\bgreater\b|\blarger\b", re.IGNORECASE)
_MINIMIZE_WORDS = re.compile(
    r"\blow(?:er|est)?\b|\bsofter\b|\bmin(?:imum|imize|imise)?\b|"
    r"\bdecrease[sd]?\b|\bsmaller\b|\bcompliant\b|\bductile\b", re.IGNORECASE)
_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*GPa", re.IGNORECASE)
_MATERIAL_PATTERN = re.compile(
    r"\b(?:on|of|in|across)\s+(?:a|an|the)\s+([a-zA-Z0-9][^,.;?]{2,70})", re.IGNORECASE)


@dataclass
class PropertyGoal:
    """One optimized property (e.g., maximize hardness)."""
    name: str
    direction: str = "maximize"
    weight: float = 1.0
    target_range: Optional[tuple[float, float]] = None
    units: str = "GPa"

    def describe(self) -> str:
        if self.direction == "target_range" and self.target_range:
            return f"{self.name} in {self.target_range[0]:g}-{self.target_range[1]:g} {self.units}"
        return f"{self.direction} {self.name} (weight {self.weight:g})"


@dataclass
class ObjectiveConstraints:
    """Experiment budget and geometry limits."""
    indent_budget: int = 45
    max_rounds: int = 6
    arrays_per_round: int = 1
    grid_nx: int = 1
    grid_ny: int = 1
    spacing_um: float = 15.0
    min_site_spacing_um: float = 18.0
    fov_bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 75.0)
    edge_margin_frac: float = 0.10

    @property
    def indents_per_array(self) -> int:
        return int(self.grid_nx) * int(self.grid_ny)


@dataclass
class SuccessCriteria:
    """Quantitative conditions under which the objective counts as met."""
    min_improvement_pct: float = 10.0
    min_confirming_arrays: int = 2
    min_visited_locations: int = 3
    ei_plateau: float = 0.01
    description: str = ""


@dataclass
class ResearchObjective:
    """Machine-actionable form of the user's research goal."""
    raw_text: str
    material_system: str = ""
    property_goals: list[PropertyGoal] = field(default_factory=list)
    constraints: ObjectiveConstraints = field(default_factory=ObjectiveConstraints)
    success: SuccessCriteria = field(default_factory=SuccessCriteria)
    stop_conditions: list[str] = field(default_factory=list)
    compiled_by: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuples -> lists for clean JSON round-tripping
        d["constraints"]["fov_bounds"] = list(self.constraints.fov_bounds)
        for g in d["property_goals"]:
            if g.get("target_range"):
                g["target_range"] = list(g["target_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchObjective":
        goals = [
            PropertyGoal(
                name=g.get("name", "hardness"),
                direction=g.get("direction", "maximize"),
                weight=float(g.get("weight", 1.0)),
                target_range=tuple(g["target_range"]) if g.get("target_range") else None,
                units=g.get("units", "GPa"),
            )
            for g in d.get("property_goals", [])
        ]
        cons_d = dict(d.get("constraints") or {})
        if cons_d.get("fov_bounds"):
            cons_d["fov_bounds"] = tuple(cons_d["fov_bounds"])
        constraints = ObjectiveConstraints(**{
            k: v for k, v in cons_d.items()
            if k in ObjectiveConstraints.__dataclass_fields__
        })
        succ_d = dict(d.get("success") or {})
        success = SuccessCriteria(**{
            k: v for k, v in succ_d.items()
            if k in SuccessCriteria.__dataclass_fields__
        })
        return cls(
            raw_text=d.get("raw_text", ""),
            material_system=d.get("material_system", ""),
            property_goals=goals,
            constraints=constraints,
            success=success,
            stop_conditions=list(d.get("stop_conditions") or []),
            compiled_by=d.get("compiled_by", "fallback"),
        )

    def normalized_weights(self) -> dict[str, float]:
        total = sum(g.weight for g in self.property_goals) or 1.0
        return {g.name: g.weight / total for g in self.property_goals}

    def directions(self) -> dict[str, str]:
        return {g.name: g.direction for g in self.property_goals}

    def summary_text(self) -> str:
        """Human-readable digest for the Gate 0 card and prompt injection."""
        c, s = self.constraints, self.success
        lines = [f"Research objective: {self.raw_text}"]
        if self.material_system:
            lines.append(f"Material system: {self.material_system}")
        lines.append("Optimization targets: " + "; ".join(g.describe() for g in self.property_goals))
        lines.append(
            f"Budget: {c.indent_budget} indents, {c.max_rounds} rounds max, "
            f"{c.grid_nx}x{c.grid_ny} arrays at {c.spacing_um:g} um spacing"
        )
        lines.append(
            f"Success: best site >= {s.min_improvement_pct:g}% above survey baseline, "
            f">= {s.min_confirming_arrays} arrays and >= {s.min_visited_locations} locations measured"
        )
        if self.stop_conditions:
            lines.append("Stop conditions: " + "; ".join(self.stop_conditions))
        lines.append(f"Compiled by: {self.compiled_by}")
        return "\n".join(lines)


def _cap_success_to_budget(
    success: SuccessCriteria,
    constraints: ObjectiveConstraints,
) -> SuccessCriteria:
    """
    Success criteria must be reachable within the indent budget — each array
    measures one location, so neither requirement can exceed the number of
    arrays the budget can pay for.
    """
    max_arrays = max(1, constraints.indent_budget // max(1, constraints.indents_per_array))
    success.min_confirming_arrays = min(success.min_confirming_arrays, max_arrays)
    success.min_visited_locations = min(success.min_visited_locations, max_arrays)
    return success


def _default_stop_conditions(constraints: ObjectiveConstraints) -> list[str]:
    return [
        "objective success criteria met",
        f"indent budget exhausted ({constraints.indent_budget} indents)",
        f"round limit reached ({constraints.max_rounds} rounds)",
        "operator rejects a proposal or requests escalation",
    ]


def _direction_near(text: str, span: tuple[int, int], window: int = 60) -> str:
    """Infer optimization direction from the nearest direction word."""
    center = (span[0] + span[1]) / 2.0
    best_direction, best_dist = "maximize", float(window)
    for direction, pattern in (("maximize", _MAXIMIZE_WORDS), ("minimize", _MINIMIZE_WORDS)):
        for m in pattern.finditer(text):
            dist = abs((m.start() + m.end()) / 2.0 - center)
            if dist < best_dist:
                best_direction, best_dist = direction, dist
    return best_direction


def parse_objective_fallback(
    raw_text: str,
    *,
    fov_bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 75.0),
    indent_budget: int = 45,
    max_rounds: int = 6,
) -> ResearchObjective:
    """Deterministic keyword parser — never calls an LLM, never raises."""
    text = (raw_text or "").strip()
    goals: list[PropertyGoal] = []

    ranges = [(m.span(), (float(m.group(1)), float(m.group(2)))) for m in _RANGE_PATTERN.finditer(text)]

    for name, pattern in _PROPERTY_PATTERNS.items():
        m = pattern.search(text)
        if not m:
            continue
        if _INVERSE_MENTIONS.match(m.group(0)):
            direction = "minimize"
        else:
            direction = _direction_near(text, m.span())
        goal = PropertyGoal(name=name, direction=direction)
        # attach a "N-M GPa" range if one appears near this property mention
        for span, (lo, hi) in ranges:
            if abs(span[0] - m.span()[1]) < 80:
                goal.direction = "target_range"
                goal.target_range = (min(lo, hi), max(lo, hi))
                break
        goals.append(goal)

    if not goals:
        goals = [PropertyGoal("hardness"), PropertyGoal("modulus")]

    material = ""
    mat_match = _MATERIAL_PATTERN.search(text)
    if mat_match:
        material = mat_match.group(1).strip()

    constraints = ObjectiveConstraints(indent_budget=indent_budget, max_rounds=max_rounds,
                                       fov_bounds=fov_bounds)
    success = _cap_success_to_budget(SuccessCriteria(
        description="Best measured site beats the survey baseline by the required margin "
                    "with sufficient supporting arrays/locations."
    ), constraints)
    return ResearchObjective(
        raw_text=text,
        material_system=material,
        property_goals=goals,
        constraints=constraints,
        success=success,
        stop_conditions=_default_stop_conditions(constraints),
        compiled_by="fallback",
    )


_LLM_OBJECTIVE_PROMPT = """You translate a materials-science research goal into a JSON experiment objective
for an autonomous nanoindentation instrument.

USER GOAL:
{raw_text}

Return ONLY a JSON object with this exact schema (no prose):
{{
  "material_system": "<short material description or empty string>",
  "property_goals": [
    {{"name": "hardness|modulus", "direction": "maximize|minimize|target_range",
      "weight": <float 0-10>, "target_range": [lo, hi] or null}}
  ],
  "success": {{"min_improvement_pct": <float>, "min_confirming_arrays": <int>,
               "min_visited_locations": <int>}},
  "stop_conditions": ["<condition>", "..."]
}}

Rules: only the properties hardness and modulus are measurable. Include a property only
if the goal mentions or implies it. Weights express relative importance. target_range is
in GPa and only used with direction "target_range". min_improvement_pct is the percent by
which the best site must beat the sample-average baseline (default 10)."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _objective_from_llm_dict(
    data: dict[str, Any],
    raw_text: str,
    constraints: ObjectiveConstraints,
) -> Optional[ResearchObjective]:
    """Validate and clamp an LLM-produced objective dict; None if unusable."""
    goals: list[PropertyGoal] = []
    for g in data.get("property_goals") or []:
        name = str(g.get("name", "")).strip().lower()
        if name not in KNOWN_PROPERTIES:
            continue
        direction = str(g.get("direction", "maximize")).strip().lower()
        if direction not in VALID_DIRECTIONS:
            direction = "maximize"
        try:
            weight = _clamp(float(g.get("weight", 1.0)), 0.05, 10.0)
        except (TypeError, ValueError):
            weight = 1.0
        target_range = None
        tr = g.get("target_range")
        if direction == "target_range":
            try:
                lo, hi = float(tr[0]), float(tr[1])
                target_range = (min(lo, hi), max(lo, hi))
            except (TypeError, ValueError, IndexError):
                direction = "maximize"
        goals.append(PropertyGoal(name=name, direction=direction, weight=weight,
                                  target_range=target_range))
    if not goals:
        return None

    succ = data.get("success") or {}
    success = SuccessCriteria()
    try:
        success.min_improvement_pct = _clamp(float(succ.get("min_improvement_pct", 10.0)), 1.0, 200.0)
        success.min_confirming_arrays = int(_clamp(int(succ.get("min_confirming_arrays", 2)), 1, 10))
        success.min_visited_locations = int(_clamp(int(succ.get("min_visited_locations", 3)), 1, 20))
    except (TypeError, ValueError):
        pass
    success.description = (
        "Best measured site beats the survey baseline by the required margin "
        "with sufficient supporting arrays/locations."
    )
    success = _cap_success_to_budget(success, constraints)

    stop_conditions = [str(s)[:200] for s in (data.get("stop_conditions") or [])][:6]
    return ResearchObjective(
        raw_text=raw_text,
        material_system=str(data.get("material_system", ""))[:120],
        property_goals=goals,
        constraints=constraints,
        success=success,
        stop_conditions=stop_conditions or _default_stop_conditions(constraints),
        compiled_by="llm",
    )


def compile_objective(
    raw_text: str,
    *,
    cfg: Any = None,
    fov_bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 75.0),
    indent_budget: int = 45,
    max_rounds: int = 6,
    use_llm: bool = True,
    timeout_s: float = 90.0,
) -> ResearchObjective:
    """
    Compile a free-text goal into a ResearchObjective.

    Tries the "orchestrator" LLM role first; any failure (no token, timeout,
    bad JSON, empty goals) falls back to the deterministic parser. Never raises.
    """
    fallback = parse_objective_fallback(
        raw_text, fov_bounds=fov_bounds, indent_budget=indent_budget, max_rounds=max_rounds)
    if not use_llm:
        return fallback

    try:
        from concurrent.futures import ThreadPoolExecutor

        from .registry.models import get_llm_for_role
        from .registry.tool_catalog import parse_llm_json

        llm = get_llm_for_role("orchestrator", cfg)
        prompt = _LLM_OBJECTIVE_PROMPT.format(raw_text=raw_text)

        def _call() -> str:
            return llm.chat(
                user_message=prompt,
                system_message="You are the orchestrator of an autonomous nanoindentation "
                               "laboratory. Respond with strict JSON only.",
            )

        with ThreadPoolExecutor(max_workers=1) as ex:
            response = ex.submit(_call).result(timeout=timeout_s)
        parsed = _objective_from_llm_dict(parse_llm_json(response), raw_text, fallback.constraints)
        if parsed is not None:
            return parsed
        print("[objective] LLM returned no usable property goals — using fallback parser")
    except Exception as exc:
        print(f"[objective] LLM compile unavailable ({type(exc).__name__}: {exc}) — using fallback parser")
    return fallback
