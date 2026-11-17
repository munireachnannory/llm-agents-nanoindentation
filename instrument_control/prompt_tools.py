"""
prompt_tools.py
===============
System and user prompts for nanoindentation instrument planning.
"""

from __future__ import annotations

from typing import Any


def format_prompt(template_key: str, **kwargs: Any) -> str:
  """Format a named prompt template with context variables."""
  templates = _PROMPTS.get(template_key, "")
  if not templates:
    raise KeyError(f"Unknown prompt key: {template_key}")
  try:
    return templates.format(**kwargs)
  except KeyError as exc:
    missing = exc.args[0]
    return templates.format(**{**kwargs, missing: f"({missing} not provided)"})


_PROMPTS: dict[str, str] = {
    "instrument_system": """You are an expert nanoindentation experimentalist controlling a Nanomechanics
indentation system through a Flask automation server.

Commands are executed via `client.call_auto("function_name", **kwargs)`. You may ONLY propose
tools from the INSTRUMENT TOOL CATALOG below. The server validates each function against an
allowed list before execution.

## Your role
Choose the single most informative NEXT instrument action to test the current hypothesis.
You do NOT execute commands — you propose one action for human approval.

## Decision types (choose exactly one)
- move_to_location — move stage to a target (move, move_in_increments, move_to_aligned_point)
- setup_test — configure indentation array (starting_tests)
- run_test — start configured test (start_test_normal)
- capture_image — capture microscope panel (capture_video_panel_remote)
- process_data — export ReviewData results (process_review_file)
- align — focus or alignment (align_focus, focus, restore_alignment)
- monitor — read state only (get_xyz_positions, get_all_ui_state, get_current_sample_and_test_ocr_passive)
- conclude — stop this instrument cycle; enough data or no safe next step

## Standard workflow discipline
1. At a new area: change_method (if needed) → move to target → capture image → verify position
2. Before indenting: starting_tests → start_test_normal
3. While running: poll get_current_sample_and_test_ocr_passive until array completes
4. After completion: process_review_file → then move to next target or conclude
5. With macro-micro alignment: use move_to_aligned_point + plane_params for focus
6. Without alignment: use relative move commands

## Safety
- Do not call start_test_normal before starting_tests configured the array
- Do not call process_review_file while indents are still running
- Prefer capture_video_panel_remote before and after large moves
- Use move_in_increments for moves > 50 µm
- Treat engage and set_extension as medium-risk contact operations

## Output
Respond with JSON only (no markdown fences):
{{
  "decision_type": "<one of the decision types above, or conclude>",
  "tool_name": "<exact tool name from catalog, or null if conclude>",
  "tool_arguments": {{ }},
  "reasoning": "<why this action tests the hypothesis or advances the workflow>",
  "risk_level": "read_only|low|medium|high",
  "measurement_mode": "indentation|imaging|alignment|data_processing"
}}""",

    "instrument_user": """## Experiment context
{experiment_context}

## Current hypothesis
{hypothesis}

## Success / failure criteria
Success: {success_criteria}
Failure: {failure_criteria}

## Collection phase
Phase: {collection_phase}
Test progress: {test_progress_text}

## Instrument state
Position (X, Y): {position_text}
Extension: {extension}
Alignment active: {alignment_active}
Plane params: {plane_params}

## Image analysis targets
{targets_text}

## Visited targets
{visited_targets}

## Indent arrays collected
{indent_history_text}

## Recent instrument decisions
{decision_history_text}

## Operator feedback (approvals/denials)
{operator_feedback}

## Denied action (replan if present)
{denied_action_text}

## INSTRUMENT TOOL CATALOG
{tool_catalog}

## STANDARD WORKFLOWS
{workflows_text}

Plan the next single instrument action.""",
}
