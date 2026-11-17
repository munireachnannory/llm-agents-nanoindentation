"""
hypothesis_critic.py
====================
Hypothesis Critic for evaluating scientific hypotheses.

Validates hypotheses against:
- Available evidence (images, prior data)
- Scientific rigor
- Testability criteria
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..state import AgentState


def hypothesis_critic_node(state: AgentState) -> dict[str, Any]:
    """
    Evaluate the current hypothesis for scientific validity.
    
    Checks if the hypothesis is:
    - Testable with available instrumentation
    - Consistent with available evidence
    - Specific enough to be validated/refuted
    """
    cfg = get_config()
    hypothesis = state.get("hypothesis", "")
    
    if not hypothesis:
        return {
            "hypothesis_critique": "No hypothesis provided to evaluate.",
            "hypothesis_critique_approved": False,
            "current_stage": "hypothesis_critic",
        }
    
    # Evaluate hypothesis
    # In production, this would use an LLM with vision capabilities
    
    critique_parts = []
    approved = True
    
    # Check specificity
    if len(hypothesis) < 100:
        critique_parts.append("Hypothesis may be too brief. Consider adding more specific predictions.")
        approved = False
    
    # Check for measurable predictions
    measurable_keywords = ["hardness", "modulus", "GPa", "variation", "difference", "higher", "lower"]
    has_measurable = any(kw in hypothesis.lower() for kw in measurable_keywords)
    if not has_measurable:
        critique_parts.append("Consider adding specific measurable predictions (e.g., expected hardness/modulus ranges).")
        approved = approved  # Warning but don't reject
    
    # Check alignment with image analysis
    image_analysis = state.get("image_analysis", {})
    if image_analysis:
        features = image_analysis.get("features_detected", [])
        if features:
            critique_parts.append(f"Image analysis detected: {', '.join(features[:3])}. Hypothesis should address these features.")
    
    # Check for testability
    hypothesis_data = state.get("hypothesis_data", {})
    success_criteria = hypothesis_data.get("success_criteria", "")
    failure_criteria = hypothesis_data.get("failure_criteria", "")
    
    if not success_criteria or not failure_criteria:
        critique_parts.append("Include clear success and failure criteria for proper hypothesis testing.")
    
    # Build critique summary
    if approved:
        critique = (
            "APPROVED: Hypothesis is testable and appropriately scoped for nanoindentation experiments. "
            + " ".join(critique_parts)
        )
    else:
        critique = (
            "REVISION NEEDED: " + " ".join(critique_parts)
        )
    
    # Track critic passes
    critic_passes = state.get("hypothesis_critic_passes", 0) + 1
    min_passes = cfg.experiment.min_hypothesis_critic_passes
    
    # Force another pass if below minimum
    if critic_passes < min_passes and approved:
        critique += f" (Critic pass {critic_passes}/{min_passes})"
    
    return {
        "hypothesis_critique": critique,
        "hypothesis_critique_approved": approved,
        "hypothesis_critic_passes": critic_passes,
        "current_stage": "hypothesis_critic",
    }
