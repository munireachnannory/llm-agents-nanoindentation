"""
exploration_plan.py
===================
Structured exploration backlog: links hypothesis, image-analysis regions,
and indentation measurements for intelligent experiment planning.
"""

from __future__ import annotations

from typing import Any, Optional

from .spatial_targeting import target_visit_key


_VALID_MEASUREMENT_TYPES = frozenset({
    "indent_array",
    "single_indent",
    "line_scan",
    "area_map",
    "reference",
})


def normalize_ia_region(region: dict[str, Any]) -> dict[str, Any]:
    """Coerce vision-LLM region fields into planner-friendly keys."""
    out = dict(region)
    
    # Normalize measurement type
    mtype = (
        out.get("measurement_type")
        or out.get("suggested_action")
        or out.get("type")
        or ""
    )
    mtype = str(mtype).strip().lower().replace(" ", "_")
    
    if mtype in ("array", "grid", "indent_array"):
        mtype = "indent_array"
    elif mtype in ("single", "point", "single_indent"):
        mtype = "single_indent"
    elif mtype in ("line", "transect", "line_scan"):
        mtype = "line_scan"
    elif mtype in ("map", "area", "area_map"):
        mtype = "area_map"
    elif mtype in ("ref", "baseline", "reference"):
        mtype = "reference"
    
    if mtype and mtype not in _VALID_MEASUREMENT_TYPES:
        mtype = "indent_array"  # Default to array
    
    if mtype:
        out["measurement_type"] = mtype
        out["suggested_action"] = mtype
    
    # Normalize priority
    try:
        out["priority"] = float(out.get("priority", out.get("saliency", 3)))
    except (TypeError, ValueError):
        out["priority"] = 3.0
    
    # Truncate long text fields
    for key in ("uncertainty", "hypothesis_link", "success_signal", 
                "failure_signal", "reason", "expected_contrast"):
        if out.get(key):
            out[key] = str(out[key])[:400]
    
    return out


def build_exploration_plan(
    state: dict[str, Any],
    *,
    max_sites: int = 10,
) -> dict[str, Any]:
    """
    Build an exploration plan from hypothesis and image analysis.
    
    The plan links:
    - Hypothesis predictions to target locations
    - Image features to measurement types
    - Success/failure signals for each site
    """
    hypothesis = state.get("hypothesis", "")
    hypothesis_data = state.get("hypothesis_data", {}) or {}
    ia = state.get("image_analysis") or {}
    
    # Extract hypothesis-driven expectations
    success_criteria = (
        hypothesis_data.get("success_criteria")
        or state.get("_hypothesis_success_criteria", "")
    )
    failure_criteria = (
        hypothesis_data.get("failure_criteria")
        or state.get("_hypothesis_failure_criteria", "")
    )
    proposed_measurements = (
        hypothesis_data.get("measurements")
        or state.get("_hypothesis_measurements", "")
    )
    
    # Get targets from image analysis
    target_locations = list(ia.get("target_locations") or [])
    features = list(ia.get("features_detected") or [])
    phases = list(ia.get("phases_identified") or [])
    
    # Build exploration sites
    sites: list[dict[str, Any]] = []
    
    for i, target in enumerate(target_locations[:max_sites]):
        target = normalize_ia_region(target)
        
        site: dict[str, Any] = {
            "label": target.get("label", f"site_{i+1}"),
            "x_um": target.get("x_um", target.get("x", 0)),
            "y_um": target.get("y_um", target.get("y", 0)),
            "x_frac": target.get("x_frac", 0.5),
            "y_frac": target.get("y_frac", 0.5),
            "reason": target.get("reason", "image_analysis"),
            "measurement_type": target.get("measurement_type", "indent_array"),
            "priority": target.get("priority", i + 1),
            "status": "pending",
        }
        
        # Link to hypothesis
        site["hypothesis_link"] = target.get("hypothesis_link", "")
        if not site["hypothesis_link"] and hypothesis:
            # Try to infer link from reason
            reason = site["reason"].lower()
            if "boundary" in reason:
                site["hypothesis_link"] = "Test hardness variation at boundaries"
            elif "phase" in reason:
                site["hypothesis_link"] = "Compare phase-specific properties"
            elif "matrix" in reason or "reference" in reason:
                site["hypothesis_link"] = "Establish baseline properties"
        
        # Set success/failure signals
        site["success_signal"] = target.get(
            "success_signal",
            f"Measurement shows expected variation per hypothesis"
        )
        site["failure_signal"] = target.get(
            "failure_signal",
            f"No significant property difference detected"
        )
        
        # Expected outcome
        site["expected_contrast"] = target.get("expected_contrast", "")
        
        sites.append(site)
    
    # Extract feature types (handle both string and dict formats)
    feature_types = []
    for f in features[:5]:
        if isinstance(f, dict):
            feature_types.append(f.get("type", "feature"))
        elif isinstance(f, str):
            feature_types.append(f)
        else:
            feature_types.append(str(f))
    
    # Extract phase labels (handle both string and dict formats)
    phase_labels = []
    for i, p in enumerate(phases[:5]):
        if isinstance(p, dict):
            phase_labels.append(p.get("label", f"phase_{i}"))
        elif isinstance(p, str):
            phase_labels.append(p)
        else:
            phase_labels.append(f"phase_{i}")
    
    # Build the plan
    plan: dict[str, Any] = {
        "hypothesis_summary": hypothesis[:500] if hypothesis else "",
        "success_criteria": success_criteria[:400] if success_criteria else "",
        "failure_criteria": failure_criteria[:400] if failure_criteria else "",
        "proposed_measurements": proposed_measurements[:400] if proposed_measurements else "",
        "sites": sites,
        "total_sites": len(sites),
        "phases_to_compare": phase_labels,
        "features_of_interest": feature_types,
    }
    
    return plan


def refresh_exploration_plan(state: dict[str, Any]) -> dict[str, Any]:
    """
    Refresh the exploration plan based on current state.
    Called after hypothesis updates or new image analysis.
    """
    plan = build_exploration_plan(state)
    
    # Mark visited sites
    visited_keys = set(state.get("visited_target_keys") or [])
    for site in plan.get("sites", []):
        key = target_visit_key(site)
        if key in visited_keys:
            site["status"] = "visited"
    
    # Count remaining
    pending = sum(1 for s in plan.get("sites", []) if s.get("status") == "pending")
    
    return {
        "exploration_plan": plan,
        "scratchpad": [
            f"exploration_plan: {plan.get('total_sites', 0)} sites, {pending} pending"
        ],
    }


def get_site_for_target(
    plan: dict[str, Any],
    target: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Find the exploration site corresponding to a target."""
    if not target or not plan:
        return None
    
    target_key = target_visit_key(target)
    for site in plan.get("sites", []):
        if target_visit_key(site) == target_key:
            return site
    
    return None


def format_exploration_plan_text(plan: dict[str, Any]) -> str:
    """Format exploration plan for display/prompts."""
    if not plan:
        return "(No exploration plan)"
    
    lines = []
    
    if plan.get("hypothesis_summary"):
        lines.append(f"Hypothesis: {plan['hypothesis_summary'][:200]}")
    
    if plan.get("success_criteria"):
        lines.append(f"Success if: {plan['success_criteria'][:150]}")
    
    if plan.get("failure_criteria"):
        lines.append(f"Refuted if: {plan['failure_criteria'][:150]}")
    
    sites = plan.get("sites", [])
    if sites:
        lines.append(f"\nExploration sites ({len(sites)}):")
        for site in sites[:8]:
            status = site.get("status", "pending").upper()
            lines.append(
                f"  [{status}] {site.get('label', '?')} "
                f"({site.get('x_um', 0):.1f}, {site.get('y_um', 0):.1f}) um - "
                f"{site.get('reason', '?')}"
            )
        if len(sites) > 8:
            lines.append(f"  ... and {len(sites) - 8} more")
    
    return "\n".join(lines)
