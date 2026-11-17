"""
spatial_targeting.py
====================
Build target queues from image-analysis regions, track visits, and manage
indentation coordinates for the nanoindentation workflow.
"""

from __future__ import annotations

import math
from typing import Any, Optional

# Minimum separation between indent targets in micrometers
MIN_INDENT_SPACING_UM = 5.0

# Minimum margin from image edge (fraction)
EDGE_MARGIN_FRAC = 0.10


def target_visit_key(target: dict[str, Any]) -> str:
    """Generate a unique key for a target based on label and position."""
    label = str(target.get("label") or target.get("type") or "pt")
    x = float(target.get("x_um", target.get("x", 0.0)))
    y = float(target.get("y_um", target.get("y", 0.0)))
    return f"{label}:{x:.1f}:{y:.1f}"


def _too_close(a: dict[str, Any], b: dict[str, Any], min_sep_um: float) -> bool:
    """Check if two targets are too close together."""
    dx = float(a.get("x_um", 0)) - float(b.get("x_um", 0))
    dy = float(a.get("y_um", 0)) - float(b.get("y_um", 0))
    return math.hypot(dx, dy) < min_sep_um


def normalize_target(
    region: dict[str, Any],
    *,
    image_width_um: float = 100.0,
    image_height_um: float = 100.0,
    source: str = "image_analysis",
) -> Optional[dict[str, Any]]:
    """
    Normalize a region/point into a standard measurement target dict.
    
    Accepts various coordinate formats:
    - x_um/y_um: direct micrometers
    - x_frac/y_frac: fractional (0-1)
    - x/y: assumed micrometers
    - row/col: pixel coordinates (requires image dimensions)
    """
    if not region:
        return None
    
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    
    # Try different coordinate formats
    if "x_um" in region or "y_um" in region:
        x_um = float(region.get("x_um", region.get("x", 0.0)))
        y_um = float(region.get("y_um", region.get("y", 0.0)))
    elif "x_frac" in region or "y_frac" in region:
        x_frac = float(region.get("x_frac", 0.5))
        y_frac = float(region.get("y_frac", 0.5))
        x_um = x_frac * image_width_um
        y_um = y_frac * image_height_um
    elif "x" in region or "y" in region:
        x_um = float(region.get("x", 0.0))
        y_um = float(region.get("y", 0.0))
    elif "row" in region and "col" in region:
        # Assume row/col are pixel coordinates, convert to fractional then um
        # This is a simplified conversion
        x_um = float(region.get("col", 0))
        y_um = float(region.get("row", 0))
    
    if x_um is None or y_um is None:
        return None
    
    label = str(
        region.get("label")
        or region.get("name")
        or region.get("type")
        or f"{source}_{hash(str(region)) % 1000}"
    )
    
    target: dict[str, Any] = {
        "x_um": x_um,
        "y_um": y_um,
        "x_frac": x_um / image_width_um if image_width_um > 0 else 0.5,
        "y_frac": y_um / image_height_um if image_height_um > 0 else 0.5,
        "label": label,
        "type": str(region.get("type") or source),
        "source": source,
        "reason": region.get("reason") or region.get("description", ""),
        "priority": float(region.get("priority", region.get("saliency", 1.0))),
    }
    
    # Copy additional fields
    for key in ("expected_contrast", "expected_hardness", "hypothesis_link",
                "confidence", "uncertainty", "size_estimate", "usable_width_um",
                "usable_height_um", "usable_diameter_um"):
        if region.get(key):
            target[key] = (region[key] if key.startswith("usable_")
                           else str(region[key])[:400])
    
    return target


def merge_targets(
    *target_lists: list[dict[str, Any]],
    min_spacing_um: float = MIN_INDENT_SPACING_UM,
    image_width_um: float = 100.0,
    image_height_um: float = 100.0,
) -> list[dict[str, Any]]:
    """
    Merge multiple target lists, removing duplicates and too-close targets.
    Earlier lists have priority.
    """
    merged: list[dict[str, Any]] = []
    
    for targets in target_lists:
        for region in targets:
            t = normalize_target(
                region,
                image_width_um=image_width_um,
                image_height_um=image_height_um,
                source=region.get("source", "merged"),
            )
            if t is None:
                continue
            
            # Check if too close to existing targets
            if any(_too_close(t, existing, min_spacing_um) for existing in merged):
                continue
            
            # Check if within image bounds (with margin)
            margin_x = EDGE_MARGIN_FRAC * image_width_um
            margin_y = EDGE_MARGIN_FRAC * image_height_um
            if not (margin_x <= t["x_um"] <= image_width_um - margin_x and
                    margin_y <= t["y_um"] <= image_height_um - margin_y):
                continue
            
            merged.append(t)
    
    # Sort by priority (lower is higher priority)
    merged.sort(key=lambda t: float(t.get("priority", 99)))
    return merged


def pick_next_target(
    plan: list[dict[str, Any]],
    visited_keys: list[str],
    *,
    current_target: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Pick the next unvisited target from the plan.
    """
    visited = set(visited_keys)
    
    # Find the next unvisited target
    for t in plan:
        if target_visit_key(t) not in visited:
            return t
    
    # All visited, return first target (or empty)
    return plan[0] if plan else {}


def mark_target_visited(
    visited_keys: list[str],
    target: dict[str, Any],
) -> list[str]:
    """Add a target to the visited list."""
    key = target_visit_key(target)
    if key in visited_keys:
        return visited_keys
    return visited_keys + [key]


def get_unvisited_count(
    plan: list[dict[str, Any]],
    visited_keys: list[str],
) -> int:
    """Count unvisited targets in the plan."""
    visited = set(visited_keys)
    return sum(1 for t in plan if target_visit_key(t) not in visited)


def sync_targets_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Build measurement_plan and current_target from image analysis.
    """
    ia = state.get("image_analysis") or {}
    target_locations = list(ia.get("target_locations") or [])
    visited = list(state.get("visited_target_keys") or [])
    
    # Get image dimensions from state or use defaults
    image_meta = state.get("current_image") or {}
    image_width_um = float(image_meta.get("width_um", 100.0))
    image_height_um = float(image_meta.get("height_um", 100.0))
    
    if not target_locations:
        return {}
    
    plan = merge_targets(
        target_locations,
        image_width_um=image_width_um,
        image_height_um=image_height_um,
    )
    
    if not plan:
        return {}
    
    target = pick_next_target(
        plan,
        visited,
        current_target=state.get("current_target"),
    )
    unvisited = get_unvisited_count(plan, visited)
    
    return {
        "measurement_plan": plan,
        "current_target": target,
        "scratchpad": [
            f"spatial_targeting: {len(plan)} targets, {unvisited} unvisited, "
            f"next={target.get('label', '?')} @ ({target.get('x_um', 0):.1f}, {target.get('y_um', 0):.1f}) um"
        ],
    }


def targets_queue_text(state: dict[str, Any], max_items: int = 10) -> str:
    """Format target queue as text for display/prompts."""
    plan = state.get("measurement_plan") or []
    visited = set(state.get("visited_target_keys") or [])
    
    if not plan:
        return "  (no targets identified)"
    
    cur = state.get("current_target") or {}
    cur_key = target_visit_key(cur) if cur else ""
    
    lines = []
    for t in plan[:max_items]:
        key = target_visit_key(t)
        if key in visited:
            mark = "DONE"
        elif key == cur_key:
            mark = "NEXT"
        else:
            mark = "PENDING"
        
        reason = (t.get("reason") or t.get("type") or "")[:40]
        lines.append(
            f"  [{mark}] {t.get('label', '?')} "
            f"({t.get('x_um', 0):.1f}, {t.get('y_um', 0):.1f}) um "
            f"| {reason}"
        )
    
    remaining = len(plan) - max_items
    if remaining > 0:
        lines.append(f"  ... and {remaining} more targets")
    
    return "\n".join(lines)


def action_context_for_planner(state: dict[str, Any]) -> str:
    """
    Build context hints for the instrument planner LLM.
    """
    lines: list[str] = []
    
    target = state.get("current_target") or {}
    if target:
        lines.append(
            f"Current target: {target.get('label', '?')} at "
            f"({target.get('x_um', 0):.1f}, {target.get('y_um', 0):.1f}) um"
        )
        reason = target.get("reason") or target.get("expected_contrast")
        if reason:
            lines.append(f"  Reason: {reason[:150]}")
    
    plan = state.get("measurement_plan") or []
    visited = state.get("visited_target_keys") or []
    if plan:
        unvisited = get_unvisited_count(plan, visited)
        lines.append(f"Target queue: {len(plan)} total, {unvisited} remaining")
    
    # Indentation history summary
    indent_history = state.get("indent_history") or []
    if indent_history:
        lines.append(f"Indent arrays completed: {len(indent_history)}")
        total_indents = sum(h.get("valid_indent_count", 0) for h in indent_history)
        lines.append(f"Total valid indents: {total_indents}")
        
        # Recent results
        recent = indent_history[-1] if indent_history else {}
        if recent:
            h_mean = recent.get("hardness_mean")
            e_mean = recent.get("modulus_mean")
            if h_mean is not None:
                lines.append(f"Last array: H={h_mean:.2f} GPa, E={e_mean:.2f} GPa")
    
    # Hypothesis context
    hypothesis = state.get("hypothesis", "")
    if hypothesis:
        lines.append(f"Current hypothesis: {hypothesis[:200]}")
    
    return "\n".join(f"  - {ln}" for ln in lines)
