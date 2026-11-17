"""
state.py
========
Defines the AgentState TypedDict for the LangGraph Nanoindentation Multi-Agent System.
"""

from __future__ import annotations

import operator
from typing import Any, Optional, TypedDict
from typing_extensions import Annotated

try:
    from langchain_core.messages import BaseMessage
except ModuleNotFoundError:
    BaseMessage = Any


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    return left + right


def append_list(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class IndentData(TypedDict, total=False):
    """Data from a single indentation test/array."""
    test_name: str
    sample_name: str
    n_indents: int
    grid_nx: int
    grid_ny: int
    spacing_um: float
    rotation: float
    method_type: str
    setup: dict[str, Any]
    results: list[dict[str, Any]]
    hardness_mean: float
    hardness_std: float
    modulus_mean: float
    modulus_std: float
    valid_indent_count: int
    data_path: str
    export_path: str
    timestamp: str
    position: tuple[float, float]
    area_name: str
    cluster: str


class MicroscopeImage(TypedDict, total=False):
    """Captured microscope/video panel image."""
    image_path: str
    local_path: str
    timestamp: str
    position: tuple[float, float]
    features_detected: list[str]
    target_regions: list[dict[str, Any]]
    crosshair_offset: tuple[float, float]


class AlignmentData(TypedDict, total=False):
    """Macro-micro alignment parameters."""
    area_name: str
    new_origin_macro: tuple[float, float]
    new_origin_micro: tuple[float, float]
    macro_scale_x: float
    macro_scale_y: float
    rotation_angle: float
    macro_image_path: str
    plane_params: list[float]


class MeasurementPlan(TypedDict, total=False):
    """Structure for orchestrator measurement plans."""
    tool_name: str
    tool_arguments: dict[str, Any]
    action_type: str
    parameters: dict[str, Any]
    justification: str
    expected_outcome: str
    risk_level: str
    decision_type: str
    measurement_mode: str


class ImageAnalysisResult(TypedDict, total=False):
    """Results from microscope image analysis."""
    features_detected: list[str]
    grain_boundaries: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    defects: list[dict[str, Any]]
    target_locations: list[dict[str, Any]]
    statistical_summary: dict[str, float]
    recommendations: list[str]
    interpretation: str
    crosshair_position: tuple[float, float]
    fine_correction_needed: bool
    fine_correction_vector: tuple[float, float]


class AgentState(TypedDict, total=False):
    # Hypothesis and literature
    hypothesis: str
    hypothesis_history: Annotated[list[str], append_list]
    literature_context: str
    literature_sources: Annotated[list[str], append_list]
    literature_raw_context: str
    literature_search_query: str
    literature_search_keywords: list[str]
    literature_refresh_requested: bool
    literature_search_meta: dict[str, Any]
    hypothesis_refinement_requested: bool
    hypothesis_refinement_reason: str
    last_hypothesis_refinement_iteration: int
    research_question: str

    # Indentation data
    current_indent: Optional[IndentData]
    indent_history: Annotated[list[IndentData], append_list]
    current_position: tuple[float, float]
    current_extension: float
    plane_params: list[float]
    alignment_data: AlignmentData
    instrument_status: dict[str, Any]

    # Microscope imaging
    current_image: Optional[MicroscopeImage]
    image_history: Annotated[list[MicroscopeImage], append_list]
    image_analysis: ImageAnalysisResult
    image_analysis_artifacts: dict[str, Any]

    # Test progress tracking
    test_progress: dict[str, Any]
    current_test_number: int
    expected_indent_count: int

    # Planning
    orchestrator_plan: MeasurementPlan
    measurement_plan: list[dict[str, Any]]
    current_target: dict[str, Any]
    visited_target_keys: Annotated[list[str], append_list]
    exploration_plan: dict[str, Any]
    site_evidence: dict[str, Any]

    # Critics
    hypothesis_critique: str
    hypothesis_critique_approved: bool
    hypothesis_critic_vision: dict[str, Any]
    hypothesis_critic_passes: int
    measurement_critique: str
    measurement_critique_decision: str
    measurement_critique_recommended_worker: str
    measurement_critique_priority_sites: list[str]

    # Workflow control
    iteration: int
    max_iterations: int
    ic_cycle: int
    ic_max_cycles: int
    should_continue: bool
    user_approved: bool
    current_stage: str
    error_message: Optional[str]

    # Messages
    messages: Annotated[list[BaseMessage], add_messages]

    # Experiment metadata
    experiment_name: str
    run_id: str

    # Decision tracking
    spatial_options: list[str]
    decision_history: Annotated[list[dict[str, Any]], append_list]
    policy: dict[str, Any]
    operator_feedback: Annotated[list[str], append_list]
    stall_count: int
    last_decision_type: str

    # Instrument control
    instrument_tool_catalog: list[dict[str, Any]]
    pending_proposed_action: dict[str, Any]
    last_tool_result: dict[str, Any]
    last_instrument_interpretation: dict[str, Any]
    instrument_conclude: bool
    instrument_approved: bool
    last_denied_action: dict[str, Any]

    # Supervisor routing
    supervisor_next: str
    supervisor_reasoning: str
    scratchpad: Annotated[list[str], append_list]

    # Phase tracking
    experiment_phase: str
    phase_targets_remaining: int
    collection_phase: str
    data_collection_progress: dict[str, Any]
    pending_action_sequence: list[dict[str, Any]]
    current_sequence_step: int

    # Hypothesis data
    hypothesis_data: dict[str, Any]
    experiment_context: str

    # Objective-driven orchestration (notebook 5)
    research_objective: dict[str, Any]
    orchestrator_briefings: dict[str, str]
    objective_criteria: dict[str, Any]
    orchestrator_round: int
    orchestrator_decisions: Annotated[list[dict[str, Any]], append_list]
    measurement_strategy_decisions: Annotated[list[dict[str, Any]], append_list]
    objective_progress: dict[str, Any]
    acquisition_meta: dict[str, Any]


def create_initial_state(
    research_question: str = "",
    experiment_name: str = "nanoindent_experiment",
    max_iterations: int = 10,
    ic_max_cycles: int = 12,
) -> AgentState:
    """Create the initial agent state for a new experiment."""
    return AgentState(
        hypothesis="",
        hypothesis_history=[],
        literature_context="",
        literature_sources=[],
        literature_raw_context="",
        literature_search_query="",
        literature_search_keywords=[],
        literature_refresh_requested=False,
        literature_search_meta={},
        hypothesis_refinement_requested=False,
        hypothesis_refinement_reason="",
        last_hypothesis_refinement_iteration=0,
        research_question=research_question,
        current_indent=None,
        indent_history=[],
        current_position=(0.0, 0.0),
        current_extension=0.0,
        plane_params=[],
        alignment_data={},
        instrument_status={},
        current_image=None,
        image_history=[],
        image_analysis={},
        image_analysis_artifacts={},
        test_progress={},
        current_test_number=0,
        expected_indent_count=0,
        orchestrator_plan={},
        measurement_plan=[],
        current_target={},
        visited_target_keys=[],
        exploration_plan={},
        site_evidence={},
        hypothesis_critique="",
        hypothesis_critique_approved=False,
        hypothesis_critic_vision={},
        hypothesis_critic_passes=0,
        measurement_critique="",
        measurement_critique_decision="",
        measurement_critique_recommended_worker="",
        measurement_critique_priority_sites=[],
        iteration=0,
        max_iterations=max_iterations,
        ic_cycle=0,
        ic_max_cycles=ic_max_cycles,
        should_continue=True,
        user_approved=False,
        current_stage="initialize",
        error_message=None,
        messages=[],
        experiment_name=experiment_name,
        run_id="",
        spatial_options=[],
        decision_history=[],
        policy={},
        operator_feedback=[],
        stall_count=0,
        last_decision_type="",
        instrument_tool_catalog=[],
        pending_proposed_action={},
        last_tool_result={},
        last_instrument_interpretation={},
        instrument_conclude=False,
        instrument_approved=False,
        last_denied_action={},
        supervisor_next="",
        supervisor_reasoning="",
        scratchpad=[],
        experiment_phase="survey",
        phase_targets_remaining=0,
        collection_phase="exploring",
        data_collection_progress={},
        pending_action_sequence=[],
        current_sequence_step=0,
        hypothesis_data={},
        experiment_context="",
        research_objective={},
        orchestrator_briefings={},
        objective_criteria={},
        orchestrator_round=0,
        orchestrator_decisions=[],
        measurement_strategy_decisions=[],
        objective_progress={},
        acquisition_meta={},
    )
