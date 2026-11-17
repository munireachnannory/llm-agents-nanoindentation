"""
indent_tools.py
===============
Tool wrapper for nanoindentation InstrumentClient.

Provides a unified interface for the multi-agent system to call
instrument commands via the existing Flask server automation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from instrument_control.schemas import ToolOutcome


# Allowed tool names that can be called by the agent
AGENT_SAFE_TOOL_NAMES: tuple[str, ...] = (
    # Motion commands
    "move",
    "move_in_increments",
    "set_extension",
    "engage",
    # Test setup
    "starting_tests",
    "start_test_normal",
    "change_method",
    # Monitoring
    "get_current_sample_and_test_ocr_passive",
    "get_xyz_positions",
    "get_all_ui_state",
    # Data processing
    "process_review_file",
    "capture_video_panel_remote",
    # Alignment
    "align_focus",
    "focus",
    "restore_alignment",
    "move_to_aligned_point",
    # Diagnostics (read-only)
    "report_window_rects",
)


# Tool name mapping for normalized names
TOOL_NAME_MAP: dict[str, str] = {
    "indent_move": "move",
    "indent_move_increments": "move_in_increments",
    "indent_set_extension": "set_extension",
    "indent_engage": "engage",
    "indent_start_tests": "starting_tests",
    "indent_run_test": "start_test_normal",
    "indent_change_method": "change_method",
    "indent_get_test_progress": "get_current_sample_and_test_ocr_passive",
    "indent_get_position": "get_xyz_positions",
    "indent_get_ui_state": "get_all_ui_state",
    "indent_process_review": "process_review_file",
    "indent_capture_image": "capture_video_panel_remote",
    "indent_align_focus": "align_focus",
    "indent_focus": "focus",
    "indent_restore_alignment": "restore_alignment",
    "indent_move_to_aligned": "move_to_aligned_point",
}


class IndentToolkit:
    """
    Toolkit wrapper for nanoindentation InstrumentClient.
    
    Provides a unified interface for the multi-agent system to call
    instrument automation commands.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        token: str = "",
        timeout: int = 300,
        require_approval: bool = True,
    ):
        self.host = host
        self.port = port
        self.token = token
        self.timeout = timeout
        self.require_approval = require_approval
        self._client = None

    @property
    def client(self):
        """Lazy initialization of InstrumentClient."""
        if self._client is None:
            try:
                from instrument_control.instrument_client import InstrumentClient
            except ImportError:
                try:
                    from instrument_client import InstrumentClient  # type: ignore[no-redef]
                except ImportError as e:
                    raise RuntimeError(
                        "InstrumentClient not found. Install instrument_client or start the "
                        "built-in simulator: python -m instrument_control.sim_server"
                    ) from e
            self._client = InstrumentClient(
                host=self.host,
                port=self.port,
                token=self.token,
                timeout=self.timeout,
            )
        return self._client

    def ping(self) -> bool:
        """Check connectivity to the instrument server."""
        try:
            result = self.client.ping()
            return result is not None
        except Exception:
            return False

    def list_available_tools(self) -> list[dict[str, Any]]:
        """Return list of available tool descriptors."""
        tools = []
        for name in AGENT_SAFE_TOOL_NAMES:
            tools.append({
                "name": f"indent_{name}" if not name.startswith("indent_") else name,
                "description": _get_tool_description(name),
                "parameters": _get_tool_parameters(name),
            })
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """
        Execute an instrument tool command.
        
        Parameters
        ----------
        tool_name : str
            Name of the tool to call (with or without 'indent_' prefix)
        arguments : dict
            Arguments to pass to the tool
            
        Returns
        -------
        ToolOutcome
            Result of the tool execution
        """
        # Normalize tool name
        actual_name = TOOL_NAME_MAP.get(tool_name, tool_name)
        if actual_name.startswith("indent_"):
            actual_name = actual_name[7:]  # Remove prefix

        if actual_name not in AGENT_SAFE_TOOL_NAMES:
            return ToolOutcome(
                ok=False,
                error=f"Tool '{tool_name}' is not in the allowed tool list",
                tool_name=tool_name,
                arguments=arguments,
            )

        start_time = time.time()
        try:
            result = self.client.call_auto(actual_name, **arguments)
            elapsed = time.time() - start_time

            # The real iMicro server returns get_xyz_positions as a positional
            # list [x, y, extension]; the simulator returns a dict. Normalize to
            # the dict shape every agent-side consumer expects.
            if actual_name == "get_xyz_positions":
                result = _normalize_xyz_envelope(result)

            # Handle different result formats
            if isinstance(result, dict):
                if "result" in result:
                    return ToolOutcome(
                        ok=result.get("ok", True),
                        result=result.get("result"),
                        error=result.get("error"),
                        elapsed_s=elapsed,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                return ToolOutcome(
                    ok=True,
                    result=result,
                    elapsed_s=elapsed,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            return ToolOutcome(
                ok=True,
                result=result,
                elapsed_s=elapsed,
                tool_name=tool_name,
                arguments=arguments,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            # The Flask server returns HTTP 400 with {"message", "traceback"} for
            # ANY exception raised inside the automation method. requests' own
            # HTTPError string is just "400 Client Error: BAD REQUEST for url:
            # ...", which throws that away and makes a method failure on the
            # instrument PC look like a network problem. Pull the server's own
            # words back out of the response body.
            status_code = server_message = server_tb = None
            resp = getattr(e, "response", None)
            if resp is not None:
                status_code = getattr(resp, "status_code", None)
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if isinstance(body, dict):
                    server_message = body.get("message") or body.get("error")
                    server_tb = body.get("traceback")
            detail = ""
            if server_message:
                detail = f" — server said: {server_message}"
                if server_tb:
                    last = str(server_tb).strip().splitlines()
                    if last:
                        detail += f" [{last[-1].strip()}]"
            return ToolOutcome(
                ok=False,
                error=f"{type(e).__name__}: {e}{detail}",
                elapsed_s=elapsed,
                tool_name=tool_name,
                arguments=arguments,
                status_code=status_code,
                server_message=server_message,
                server_traceback=server_tb,
            )

    def download_file(
        self,
        directory: str,
        filename: str,
        local_dir: str = "downloads",
    ) -> Optional[str]:
        """Download a file from the instrument server."""
        try:
            return self.client.download_file(
                directory=directory,
                filename=filename,
                local_dir=local_dir,
            )
        except Exception as e:
            print(f"Download failed: {e}")
            return None


def _normalize_xyz_envelope(result: Any) -> Any:
    """
    Coerce a ``get_xyz_positions`` response to the ``{"x","y","extension"}`` dict
    shape. The real iMicro server returns a positional ``[x, y, extension]`` list;
    the simulator already returns the dict. Handles both the ``call_auto``
    envelope (``{"status","method","result": ...}``) and a bare payload.
    """
    def _coerce(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            keys = ("x", "y", "extension")
            return {k: (value[i] if i < len(value) else 0.0)
                    for i, k in enumerate(keys)}
        return value

    if isinstance(result, dict) and "result" in result:
        normalized = dict(result)
        normalized["result"] = _coerce(result["result"])
        return normalized
    return _coerce(result)


def _get_tool_description(name: str) -> str:
    """Get description for a tool."""
    descriptions = {
        "move": "Perform relative XY stage movement",
        "move_in_increments": "Split large movement into smaller increments",
        "set_extension": "Control Z extension",
        "engage": "Click engage button in Z control",
        "starting_tests": "Create and configure a new indentation test",
        "start_test_normal": "Start the currently configured indentation test",
        "change_method": "Change the currently active indentation method",
        "get_current_sample_and_test_ocr_passive": "Read current test number without changing UI",
        "get_xyz_positions": "Read current stage position and extension",
        "get_all_ui_state": "Read all available numeric UI state values",
        "process_review_file": "Process completed indentation data in ReviewData",
        "capture_video_panel_remote": "Capture microscope/video panel image",
        "align_focus": "Create a focus plane using three manually selected points",
        "focus": "Use saved plane parameters to calculate proper extension",
        "restore_alignment": "Restore alignment object on server",
        "move_to_aligned_point": "Move to macro-selected point using alignment",
    }
    return descriptions.get(name, f"Execute {name} command")


def _get_tool_parameters(name: str) -> dict[str, Any]:
    """Get parameter schema for a tool."""
    schemas = {
        "move": {
            "amount": {"type": "number", "description": "Movement amount in microns"},
            "direction": {"type": "string", "enum": ["right", "left", "up", "down"]},
            "t": {"type": "number", "description": "Wait time", "default": 2},
        },
        "move_in_increments": {
            "total_amount": {"type": "number", "description": "Total movement amount"},
            "direction": {"type": "string", "enum": ["right", "left", "up", "down"]},
            "increment": {"type": "number", "description": "Increment size", "default": 25},
        },
        "set_extension": {
            "number": {"type": "number", "description": "Extension value"},
            "t": {"type": "number", "description": "Wait time", "default": 2},
        },
        "starting_tests": {
            "sample_name": {"type": "string", "description": "Name of new test/sample"},
            "setup": {"type": "object", "description": "Method parameters"},
            "indents": {"type": "object", "description": "Indent array configuration"},
            "remove_n": {"type": "integer", "description": "Remove existing queued tests", "default": 1},
            "method_type": {"type": "string", "description": "Method type", "default": "eh"},
            "use_import": {"type": "integer", "description": "0=manual entry, 1=import", "default": 0},
        },
        "start_test_normal": {
            "t": {"type": "number", "description": "Wait time", "default": 2},
        },
        "process_review_file": {
            "sample_names": {"type": "array", "description": "List of sample names"},
            "open_dir": {"type": "string", "description": "Directory to open"},
            "current_sample_file_name": {"type": "string"},
            "export_dir": {"type": "string"},
            "export_file_name": {"type": "string"},
            "review_changes": {"type": "object"},
            "review_type": {"type": "string", "default": "eh"},
            "export_format": {"type": "string", "default": "excel"},
        },
        "move_to_aligned_point": {
            "area_name": {"type": "string", "description": "Alignment area name"},
            "x": {"type": "number", "description": "X coordinate in macro pixels"},
            "y": {"type": "number", "description": "Y coordinate in macro pixels"},
            "plane_params": {"type": "array", "description": "Focus plane parameters [a, b, c]"},
        },
    }
    return schemas.get(name, {})


# Global toolkit instance management
_toolkit_instance: Optional[IndentToolkit] = None


def get_toolkit() -> IndentToolkit:
    """Get the global toolkit instance."""
    if _toolkit_instance is None:
        raise RuntimeError(
            "IndentToolkit not initialized. Call set_toolkit() first."
        )
    return _toolkit_instance


def set_toolkit(toolkit: IndentToolkit) -> None:
    """Set the global toolkit instance."""
    global _toolkit_instance
    _toolkit_instance = toolkit
