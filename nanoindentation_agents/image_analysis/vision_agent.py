"""
Vision-LLM facade for nanoindentation microscope image analysis.

This layer uses a vision model to analyze microscope images and identify:
- Grain boundaries
- Phase regions  
- Surface features
- Target locations for indentation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import AppConfig, get_config
from ..registry.models import get_llm_for_role
from ..registry.tool_catalog import parse_llm_json


VISION_SYSTEM_PROMPT = """
You are an image-analysis agent for nanoindentation microscopy.

You receive a microscope image of a material surface. Your job is to identify 
microstructural features relevant for mechanical testing:

1. Grain boundaries - lines/regions between crystallographic grains
2. Phase regions - distinct material phases with different contrast
3. Surface defects - scratches, pores, inclusions
4. Target locations - optimal positions for indentation arrays

Rules:
- Return JSON only
- Be specific about spatial locations (use x_frac, y_frac from 0-1)
- Rate confidence for each detected feature
- Focus on features that would show mechanical property variations
""".strip()


def _build_vision_prompt(
    research_question: str,
    hypothesis: str,
    experiment_context: str,
) -> str:
    return f"""
## Research Context
Research question: {research_question or "(not provided)"}
Hypothesis: {hypothesis or "(not provided)"}
Experiment context: {experiment_context or "(not provided)"}

## Task
Analyze this microscope image for nanoindentation target selection.
Identify microstructural features and recommend indentation locations.

## Required JSON Output
{{
  "vision_used": true,
  "image_quality": {{
    "usable": true,
    "confidence": 0.0,
    "issues": []
  }},
  "features_detected": [
    {{
      "type": "grain_boundary | phase_region | inclusion | pore | scratch | other",
      "description": "Brief description",
      "x_frac": 0.5,
      "y_frac": 0.5,
      "confidence": 0.8,
      "size_estimate": "small | medium | large"
    }}
  ],
  "phases_identified": [
    {{
      "label": "phase_A",
      "description": "Brighter/darker region characteristics",
      "area_fraction": 0.0,
      "center_x_frac": 0.5,
      "center_y_frac": 0.5,
      "expected_hardness": "higher | lower | similar to matrix"
    }}
  ],
  "grain_boundaries": [
    {{
      "type": "high_angle | low_angle | twin",
      "x_frac": 0.5,
      "y_frac": 0.5,
      "orientation": "horizontal | vertical | diagonal",
      "confidence": 0.8
    }}
  ],
  "target_locations": [
    {{
      "label": "target_1",
      "x_frac": 0.5,
      "y_frac": 0.5,
      "reason": "grain_boundary | phase_center | inclusion | matrix_reference",
      "priority": 1,
      "confidence": 0.8,
      "size_estimate": "small | medium | large",
      "usable_diameter_frac": 0.0,
      "expected_contrast": "Why this location should show property variation"
    }}
  ],
  "interpretation": "Overall interpretation of the microstructure and recommended measurement strategy",
  "recommendations": ["List of specific recommendations for indentation placement"]
}}
""".strip()


def analyze_image_with_vision(
    image_path: str | Path,
    *,
    research_question: str = "",
    hypothesis: str = "",
    experiment_context: str = "",
    cfg: AppConfig | None = None,
    role: str = "image_analysis",
) -> dict[str, Any]:
    """
    Analyze a microscope image using vision LLM.

    Returns a JSON-friendly dictionary with:
    - features_detected
    - phases_identified
    - grain_boundaries
    - target_locations
    - interpretation
    - recommendations
    """
    cfg = cfg or get_config()
    image_path = Path(image_path)
    
    result: dict[str, Any] = {
        "ok": False,
        "image_path": str(image_path),
        "vision": {
            "attempted": True,
            "ok": False,
            "used": False,
            "report": None,
            "raw_response": "",
            "error": "",
        },
        "features_detected": [],
        "phases_identified": [],
        "grain_boundaries": [],
        "target_locations": [],
        "interpretation": "",
        "recommendations": [],
    }
    
    if not image_path.exists():
        result["vision"]["error"] = f"Image file not found: {image_path}"
        return result
    
    prompt = _build_vision_prompt(research_question, hypothesis, experiment_context)
    
    try:
        llm = get_llm_for_role(role, cfg)
        raw = llm.chat_with_images(
            prompt,
            image_paths=[str(image_path)],
            system_message=VISION_SYSTEM_PROMPT,
        )
        
        try:
            parsed = parse_llm_json(raw)
        except Exception as parse_exc:
            parsed = {
                "vision_used": True,
                "image_quality": {"usable": True, "confidence": 0.5, "issues": ["Could not parse JSON"]},
                "features_detected": [],
                "phases_identified": [],
                "grain_boundaries": [],
                "target_locations": [],
                "interpretation": raw,
                "recommendations": [],
            }
        
        result["ok"] = True
        result["vision"].update({
            "ok": True,
            "used": True,
            "report": parsed,
            "raw_response": raw,
            "error": "",
        })
        
        # Extract fields from parsed response
        result["features_detected"] = parsed.get("features_detected", [])
        result["phases_identified"] = parsed.get("phases_identified", [])
        result["grain_boundaries"] = parsed.get("grain_boundaries", [])
        result["target_locations"] = parsed.get("target_locations", [])
        result["interpretation"] = parsed.get("interpretation", "")
        result["recommendations"] = parsed.get("recommendations", [])
        result["image_quality"] = parsed.get("image_quality", {})
        
    except Exception as exc:
        result["vision"].update({
            "ok": False,
            "used": False,
            "report": None,
            "raw_response": "",
            "error": f"{type(exc).__name__}: {exc}",
        })
    
    return result


def convert_vision_to_targets(
    vision_result: dict[str, Any],
    *,
    image_width_um: float = 100.0,
    image_height_um: float = 100.0,
) -> list[dict[str, Any]]:
    """
    Convert vision analysis target_locations to measurement targets with coordinates.
    
    Converts fractional coordinates (0-1) to micrometers.
    """
    targets = []
    
    for i, loc in enumerate(vision_result.get("target_locations", [])):
        x_frac = float(loc.get("x_frac", 0.5))
        y_frac = float(loc.get("y_frac", 0.5))
        
        target = {
            "label": loc.get("label", f"target_{i+1}"),
            "x_um": x_frac * image_width_um,
            "y_um": y_frac * image_height_um,
            "x_frac": x_frac,
            "y_frac": y_frac,
            "reason": loc.get("reason", "vision_analysis"),
            "priority": int(loc.get("priority", i + 1)),
            "confidence": loc.get("confidence"),
            "size_estimate": loc.get("size_estimate"),
            "expected_contrast": loc.get("expected_contrast", ""),
            "source": "vision_llm",
        }
        try:
            usable_frac = float(loc.get("usable_diameter_frac"))
        except (TypeError, ValueError):
            usable_frac = 0.0
        if usable_frac > 0:
            target["usable_diameter_um"] = (
                usable_frac * min(image_width_um, image_height_um))
        targets.append(target)
    
    # Also add grain boundary locations as potential targets
    for i, gb in enumerate(vision_result.get("grain_boundaries", [])):
        x_frac = float(gb.get("x_frac", 0.5))
        y_frac = float(gb.get("y_frac", 0.5))
        
        target = {
            "label": f"gb_{i+1}",
            "x_um": x_frac * image_width_um,
            "y_um": y_frac * image_height_um,
            "x_frac": x_frac,
            "y_frac": y_frac,
            "reason": "grain_boundary",
            "priority": 10 + i,  # Lower priority than explicit targets
            "type": gb.get("type", "boundary"),
            "source": "vision_llm_boundary",
        }
        targets.append(target)
    
    # Sort by priority
    targets.sort(key=lambda t: t.get("priority", 99))
    return targets


def fallback_grid_targets(
    n_x: int = 3,
    n_y: int = 3,
    image_width_um: float = 100.0,
    image_height_um: float = 100.0,
    margin_frac: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Generate a regular grid of targets when vision analysis is unavailable.
    """
    targets = []
    
    x_start = margin_frac * image_width_um
    x_end = (1 - margin_frac) * image_width_um
    y_start = margin_frac * image_height_um
    y_end = (1 - margin_frac) * image_height_um
    
    x_step = (x_end - x_start) / max(n_x - 1, 1)
    y_step = (y_end - y_start) / max(n_y - 1, 1)
    
    priority = 1
    for j in range(n_y):
        for i in range(n_x):
            x_um = x_start + i * x_step
            y_um = y_start + j * y_step
            
            targets.append({
                "label": f"grid_{i}_{j}",
                "x_um": x_um,
                "y_um": y_um,
                "x_frac": x_um / image_width_um,
                "y_frac": y_um / image_height_um,
                "reason": "systematic_grid",
                "priority": priority,
                "source": "fallback_grid",
            })
            priority += 1
    
    return targets
