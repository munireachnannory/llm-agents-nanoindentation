"""
constants.py
============
Constants for the nanoindentation multi-agent system.
"""
import os

# Decision types for instrument planning
DECISION_TYPES = (
    "move_to_location",
    "setup_test",
    "run_test",
    "capture_image",
    "process_data",
    "align",
    "conclude",
)

# Escalation mapping when stall detected
DECISION_ESCALATION = {
    "move_to_location": "setup_test",
    "setup_test": "run_test",
    "run_test": "process_data",
    "capture_image": "move_to_location",
    "process_data": "conclude",
    "align": "move_to_location",
}

# Measurement modes
MEASUREMENT_MODES = (
    "indentation",
    "imaging",
    "alignment",
    "data_processing",
)

# Default export directory (InView shared profile on the instrument PC).
# Override with the NANOINDENT_EXPORT_DIR environment variable.
DEFAULT_EXPORT_DIR = os.environ.get(
    "NANOINDENT_EXPORT_DIR",
    r"C:\Users\Public\Documents\Nanomechanics\Profiles\Shared AENI",
)

# Method types
METHOD_TYPES = ("eh", "constant")

# Valid directions for movement
MOVEMENT_DIRECTIONS = ("right", "left", "up", "down")
