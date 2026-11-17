"""
image_analysis_agent.py
=======================
Image Analysis Agent for microscope image processing.

Analyzes captured microscope images to identify:
- Grain boundaries
- Phase regions
- Target locations for indentation
- Fine correction requirements

Uses vision LLM when available, falls back to template analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import get_config
from ..state import AgentState, ImageAnalysisResult
from ..image_analysis.vision_agent import (
    analyze_image_with_vision,
    convert_vision_to_targets,
    fallback_grid_targets,
)
from ..analysis.spatial_targeting import sync_targets_from_state
from ..analysis.exploration_plan import refresh_exploration_plan


def _template_analysis(image_path: str, state: AgentState) -> ImageAnalysisResult:
    """
    Template-based analysis when vision LLM is unavailable.
    Returns a basic grid of targets.
    """
    # Get image dimensions from state or use defaults
    image_meta = state.get("current_image") or {}
    width_um = float(image_meta.get("width_um", 100.0))
    height_um = float(image_meta.get("height_um", 100.0))
    
    # Generate grid targets
    grid_targets = fallback_grid_targets(
        n_x=3, n_y=3,
        image_width_um=width_um,
        image_height_um=height_um,
    )
    
    return ImageAnalysisResult(
        features_detected=[
            "systematic_grid_coverage",
        ],
        grain_boundaries=[],
        phases=[],
        defects=[],
        target_locations=grid_targets,
        statistical_summary={
            "coverage_type": "grid",
            "grid_size": "3x3",
        },
        recommendations=[
            "Systematic grid coverage for initial characterization",
            "Consider targeted analysis after initial results",
        ],
        interpretation=(
            "Vision analysis unavailable. Using systematic 3x3 grid for initial "
            "characterization across the field of view."
        ),
        crosshair_position=(0.0, 0.0),
        fine_correction_needed=False,
        fine_correction_vector=(0.0, 0.0),
    )


def _vision_analysis(image_path: str, state: AgentState) -> ImageAnalysisResult:
    """
    Analyze image using vision LLM.
    """
    cfg = get_config()
    
    # Get context for vision analysis
    research_question = state.get("research_question", "")
    hypothesis = state.get("hypothesis", "")
    experiment_context = state.get("experiment_context", "")
    
    # Get image dimensions
    image_meta = state.get("current_image") or {}
    width_um = float(image_meta.get("width_um", 100.0))
    height_um = float(image_meta.get("height_um", 100.0))
    
    # Run vision analysis
    result = analyze_image_with_vision(
        image_path,
        research_question=research_question,
        hypothesis=hypothesis,
        experiment_context=experiment_context,
        cfg=cfg,
    )
    
    if not result.get("ok") or not result.get("vision", {}).get("ok"):
        # Vision failed, return template
        print(f"[image_analysis] Vision LLM failed: {result.get('vision', {}).get('error', 'unknown')}")
        return _template_analysis(image_path, state)
    
    # Convert vision results to standard format
    targets = convert_vision_to_targets(
        result,
        image_width_um=width_um,
        image_height_um=height_um,
    )
    
    # Build grain boundaries list
    grain_boundaries = []
    for gb in result.get("grain_boundaries", []):
        grain_boundaries.append({
            "x": gb.get("x_frac", 0.5) * width_um,
            "y": gb.get("y_frac", 0.5) * height_um,
            "type": gb.get("type", "unknown"),
        })
    
    # Build phases list
    phases = []
    for phase in result.get("phases_identified", []):
        phases.append({
            "label": phase.get("label", "unknown"),
            "area_fraction": phase.get("area_fraction", 0.0),
            "center_x": phase.get("center_x_frac", 0.5) * width_um,
            "center_y": phase.get("center_y_frac", 0.5) * height_um,
            "expected_hardness": phase.get("expected_hardness", ""),
        })
    
    return ImageAnalysisResult(
        features_detected=[f.get("type", "feature") for f in result.get("features_detected", [])],
        grain_boundaries=grain_boundaries,
        phases=phases,
        defects=[],
        target_locations=targets,
        statistical_summary={
            "vision_used": True,
            "confidence": result.get("image_quality", {}).get("confidence", 0.0),
            "num_targets": len(targets),
            "num_boundaries": len(grain_boundaries),
            "num_phases": len(phases),
        },
        recommendations=result.get("recommendations", []),
        interpretation=result.get("interpretation", ""),
        crosshair_position=(0.0, 0.0),
        fine_correction_needed=False,
        fine_correction_vector=(0.0, 0.0),
    )


def _analyze_image(image_path: str, state: AgentState) -> ImageAnalysisResult:
    """
    Analyze a microscope image.
    Uses vision LLM if available, otherwise falls back to template.
    """
    cfg = get_config()
    
    # Check if image exists
    if not image_path or not Path(image_path).exists():
        print(f"[image_analysis] No image at {image_path}, using template")
        return _template_analysis(image_path, state)
    
    # Try vision analysis
    try:
        print(f"[image_analysis] Analyzing {image_path} with vision LLM...")
        return _vision_analysis(image_path, state)
    except Exception as exc:
        print(f"[image_analysis] Vision analysis failed ({exc}), using template")
        return _template_analysis(image_path, state)


def image_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    Analyze the current microscope image.
    
    Processes the image to identify target locations and
    microstructural features for indentation planning.
    """
    current_image = state.get("current_image")

    if not current_image:
        print("[image_analysis] No image available")
        return {
            "current_stage": "image_analysis",
            "image_analysis": ImageAnalysisResult(
                interpretation="No image available for analysis",
                features_detected=[],
                target_locations=[],
                recommendations=["Capture microscope image first"],
            ),
        }

    # Prefer local_path (downloaded to Mac) over image_path (Windows remote path)
    image_path = current_image.get("local_path") or current_image.get("image_path", "")
    analysis = _analyze_image(image_path, state)
    
    # Convert to dict for state
    analysis_dict = analysis if isinstance(analysis, dict) else dict(analysis)
    
    # Build initial updates
    updates: dict[str, Any] = {
        "current_stage": "image_analysis",
        "image_analysis": analysis_dict,
    }
    
    # Sync targets from analysis
    merged = {**state, **updates}
    target_updates = sync_targets_from_state(merged)
    updates.update(target_updates)
    
    # Build exploration plan
    merged = {**state, **updates}
    plan_updates = refresh_exploration_plan(merged)
    updates.update(plan_updates)
    
    num_targets = len(updates.get("measurement_plan", []))
    print(f"[image_analysis] Identified {num_targets} target locations")
    
    return updates


def opening_image_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    Perform initial image analysis on the opening observation.
    Called after the observe node to analyze the first captured image.
    """
    result = image_analysis_node(state)
    result["current_stage"] = "opening_image_analysis"
    return result


def hypothesis_image_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    Analyze images in context of current hypothesis.
    Called after hypothesis generation to validate hypothesis against visual evidence.
    """
    # Get base analysis
    result = image_analysis_node(state)
    
    # Add hypothesis-specific interpretation
    hypothesis = state.get("hypothesis", "")
    if hypothesis:
        analysis = result.get("image_analysis", {})
        if isinstance(analysis, dict):
            # Check if targets align with hypothesis
            targets = result.get("measurement_plan", [])
            if targets:
                analysis["hypothesis_alignment"] = (
                    f"Identified {len(targets)} target locations that can test the hypothesis "
                    "about spatial variation in mechanical properties."
                )
            else:
                analysis["hypothesis_alignment"] = (
                    "Limited targets identified. Consider capturing additional images "
                    "or adjusting hypothesis scope."
                )
            result["image_analysis"] = analysis
    
    result["current_stage"] = "hypothesis_image_analysis"
    return result
