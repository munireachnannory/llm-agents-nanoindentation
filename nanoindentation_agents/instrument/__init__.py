"""
Instrument layer — everything that talks to (or interprets data from) the iMicro.

    session.py    InstrumentSession: strict/soft tool-call policy, stage readback,
                  frame capture, and placing one approved indent array.
    screenread.py ScreenReadConfig + LiveScreenRecorder: the live H/E safety net.
    motion.py     MotionController: closed-loop jogs, backlash policy, persisted
                  per-session calibration (sign + gain + reversal cost).
    export.py     ExportPipeline: the InView ReviewData/export contract.
    results.py    Parsing, validation ladder, and per-indent reduction.

The campaign layer (nanoindentation_agents.campaign) composes these; notebooks
should not re-implement anything found here.
"""
from .screenread import ScreenReadConfig, LiveScreenRecorder
from .session import InstrumentSession
from .motion import MotionConfig, MotionController, StageSafetyError
from .export import ExportConfig, ExportPipeline
from . import results

__all__ = [
    "ScreenReadConfig", "LiveScreenRecorder", "InstrumentSession",
    "MotionConfig", "MotionController", "StageSafetyError",
    "ExportConfig", "ExportPipeline", "results",
]
