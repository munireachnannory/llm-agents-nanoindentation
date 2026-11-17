"""
hypothesis_agent.py
===================
Hypothesis Agent for the Nanoindentation Multi-Agent System.

Generates and refines materials science hypotheses based on:
- Literature context
- Research question
- Previous indentation results
- Critic feedback
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ..config import get_config
from ..registry.models import get_llm_for_role
from ..state import AgentState


def _build_hypothesis_prompt(state: AgentState) -> str:
    """Build the prompt for hypothesis generation."""
    parts = []
    
    # Research question
    research_question = state.get("research_question", "")
    if research_question:
        parts.append(f"## Research Question\n{research_question}")

    # Orchestrator directive (objective-driven mode)
    briefing = (state.get("orchestrator_briefings") or {}).get("hypothesis", "")
    if briefing:
        parts.append(f"## Orchestrator Directive\n{briefing[:600]}")
    
    # Literature context
    literature_context = state.get("literature_context", "")
    if literature_context:
        parts.append(f"## Literature Context\n{literature_context[:2000]}")
    
    # Previous hypotheses
    hypothesis_history = state.get("hypothesis_history", [])
    if hypothesis_history:
        history_text = "\n".join(
            f"- Hypothesis {i+1}: {h[:200]}..." for i, h in enumerate(hypothesis_history[-3:])
        )
        parts.append(f"## Previous Hypotheses\n{history_text}")
    
    # Critic feedback
    hypothesis_critique = state.get("hypothesis_critique", "")
    if hypothesis_critique:
        parts.append(f"## Critic Feedback\n{hypothesis_critique}")
    
    # Indentation results summary
    indent_history = state.get("indent_history", [])
    if indent_history:
        results_summary = []
        for i, indent in enumerate(indent_history[-5:]):
            summary = (
                f"- Test {i+1} at ({indent.get('position', (0,0))[0]:.1f}, {indent.get('position', (0,0))[1]:.1f}): "
                f"H={indent.get('hardness_mean', 'N/A'):.2f}±{indent.get('hardness_std', 0):.2f} GPa, "
                f"E={indent.get('modulus_mean', 'N/A'):.2f}±{indent.get('modulus_std', 0):.2f} GPa"
            )
            results_summary.append(summary)
        parts.append(f"## Indentation Results\n" + "\n".join(results_summary))
    
    # Image analysis
    image_analysis = state.get("image_analysis", {})
    if image_analysis:
        features = image_analysis.get("features_detected", [])
        if features:
            parts.append(f"## Detected Features\n- " + "\n- ".join(features[:5]))
        interpretation = image_analysis.get("interpretation", "")
        if interpretation:
            parts.append(f"## Image Analysis\n{interpretation[:500]}")
    
    # Iteration info
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    parts.append(f"\n*Iteration {iteration + 1} of {max_iterations}*")
    
    # Task instruction
    if hypothesis_history:
        task = """
## Task
Based on the above context and indentation results, please:
1. Evaluate what we have learned about the material's mechanical properties
2. Refine or generate a new hypothesis that addresses gaps in understanding
3. Specify what additional measurements would test this hypothesis
4. Define clear success/failure criteria based on hardness/modulus values

Format your response as:
**HYPOTHESIS:** [Your testable hypothesis about material properties]
**RATIONALE:** [Why this hypothesis is worth testing]
**MEASUREMENTS:** [Specific indentation locations/parameters to validate/refute]
**SUCCESS_CRITERIA:** [Expected hardness/modulus ranges if hypothesis is true]
**FAILURE_CRITERIA:** [What would refute the hypothesis]
"""
    else:
        task = """
## Task
Based on the research question and any available context, please:
1. Generate an initial hypothesis about the material's mechanical properties
2. Be specific about expected hardness and modulus variations
3. Consider what indentation measurements would be most informative

Focus on testable predictions about:
- Spatial variations in hardness/modulus
- Phase-dependent mechanical properties
- Grain boundary effects
- Compositional gradients

Format your response as:
**HYPOTHESIS:** [Your testable hypothesis]
**RATIONALE:** [Why this hypothesis is worth testing]
**MEASUREMENTS:** [Specific measurements to validate/refute]
**SUCCESS_CRITERIA:** [What would confirm the hypothesis]
**FAILURE_CRITERIA:** [What would refute the hypothesis]
"""
    parts.append(task)
    
    return "\n\n".join(parts)


def _parse_hypothesis_response(response: str) -> dict[str, str]:
    """Parse the structured hypothesis response."""
    parsed = {
        "hypothesis": "",
        "rationale": "",
        "measurements": "",
        "success_criteria": "",
        "failure_criteria": "",
    }
    
    current_field = None
    current_content = []
    
    for line in response.split("\n"):
        line = line.strip()
        
        if line.startswith("**HYPOTHESIS:**"):
            if current_field:
                parsed[current_field] = "\n".join(current_content).strip()
            current_field = "hypothesis"
            current_content = [line.replace("**HYPOTHESIS:**", "").strip()]
        elif line.startswith("**RATIONALE:**"):
            if current_field:
                parsed[current_field] = "\n".join(current_content).strip()
            current_field = "rationale"
            current_content = [line.replace("**RATIONALE:**", "").strip()]
        elif line.startswith("**MEASUREMENTS:**"):
            if current_field:
                parsed[current_field] = "\n".join(current_content).strip()
            current_field = "measurements"
            current_content = [line.replace("**MEASUREMENTS:**", "").strip()]
        elif line.startswith("**SUCCESS_CRITERIA:**"):
            if current_field:
                parsed[current_field] = "\n".join(current_content).strip()
            current_field = "success_criteria"
            current_content = [line.replace("**SUCCESS_CRITERIA:**", "").strip()]
        elif line.startswith("**FAILURE_CRITERIA:**"):
            if current_field:
                parsed[current_field] = "\n".join(current_content).strip()
            current_field = "failure_criteria"
            current_content = [line.replace("**FAILURE_CRITERIA:**", "").strip()]
        elif current_field and line:
            current_content.append(line)
    
    if current_field:
        parsed[current_field] = "\n".join(current_content).strip()
    
    # Fallback if parsing failed
    if not parsed["hypothesis"]:
        parsed["hypothesis"] = response.strip()
    
    return parsed


def hypothesis_node(state: AgentState) -> dict[str, Any]:
    """
    Hypothesis generation/refinement node.
    Uses literature context and experiment input via LLM when available.
    """
    cfg = get_config()
    
    # Log what context we're using from literature
    lit_context = state.get("literature_context", "")
    if lit_context:
        print(f"[hypothesis] Using literature context ({len(lit_context)} chars) from literature agent")
    else:
        print("[hypothesis] WARNING: No literature context available - hypothesis may be less grounded")
    
    prompt = _build_hypothesis_prompt(state)

    parsed: dict[str, str]
    hypothesis = ""

    try:
        llm = get_llm_for_role("hypothesis", cfg)
        response = llm.chat(
            user_message=prompt,
            system_message=(
                "You are a materials scientist designing nanoindentation experiments. "
                "Ground your hypothesis in the provided literature context, image analysis, "
                "and indentation data. Be specific about expected hardness/modulus behavior."
            ),
        )
        parsed = _parse_hypothesis_response(response)
        hypothesis = parsed.get("hypothesis", "").strip() or response.strip()
        print(f"[hypothesis] LLM generated hypothesis grounded in literature + experiment context")
    except Exception as exc:
        print(f"[hypothesis] LLM unavailable ({exc}) — using template fallback")
        research_question = state.get("research_question", "material mechanical properties")
        lit = (state.get("literature_context") or "")[:200]
        hypothesis = (
            f"We hypothesize spatially-varying hardness and modulus correlated with "
            f"microstructure. Literature notes: {lit} Research focus: {research_question}"
        )
        parsed = {
            "hypothesis": hypothesis,
            "rationale": "Template fallback (includes literature context)",
            "measurements": "Indent arrays at image-analysis target locations",
            "success_criteria": ">10% property difference between region types",
            "failure_criteria": "Uniform properties across locations",
        }

    hypothesis_history = list(state.get("hypothesis_history", []))
    if hypothesis and (not hypothesis_history or hypothesis_history[-1] != hypothesis):
        hypothesis_history.append(hypothesis)

    messages = list(state.get("messages", []))
    messages.extend([
        HumanMessage(content=prompt[:2000]),
        AIMessage(content=hypothesis),
    ])

    updates: dict[str, Any] = {
        "hypothesis": hypothesis,
        "hypothesis_history": hypothesis_history,
        "current_stage": "hypothesis_generation",
        "messages": messages,
        "hypothesis_refinement_requested": False,
        "_hypothesis_rationale": parsed.get("rationale", ""),
        "_hypothesis_measurements": parsed.get("measurements", ""),
        "_hypothesis_success_criteria": parsed.get("success_criteria", ""),
        "_hypothesis_failure_criteria": parsed.get("failure_criteria", ""),
        "hypothesis_data": parsed,
    }
    # Only reset user approval when generating a fresh hypothesis after rejection
    if not state.get("user_approved"):
        updates["user_approved"] = False
    return updates


def format_hypothesis_for_display(state: AgentState) -> str:
    """Format the current hypothesis for user display."""
    hypothesis = state.get("hypothesis", "No hypothesis generated yet.")
    rationale = state.get("_hypothesis_rationale", "")
    measurements = state.get("_hypothesis_measurements", "")
    success = state.get("_hypothesis_success_criteria", "")
    failure = state.get("_hypothesis_failure_criteria", "")
    
    parts = [
        "=" * 60,
        "HYPOTHESIS",
        "=" * 60,
        "",
        hypothesis,
        "",
    ]
    
    if rationale:
        parts.extend(["RATIONALE:", rationale, ""])
    if measurements:
        parts.extend(["PROPOSED MEASUREMENTS:", measurements, ""])
    if success:
        parts.extend(["SUCCESS CRITERIA:", success, ""])
    if failure:
        parts.extend(["FAILURE CRITERIA:", failure, ""])
    
    parts.extend([
        "-" * 60,
        f"Iteration: {state.get('iteration', 0) + 1} / {state.get('max_iterations', 10)}",
        "-" * 60,
    ])
    
    return "\n".join(parts)
