"""
observe_agent.py
================
Observe Agent for initial data acquisition.

Captures the initial microscope image and instrument state.
"""

from __future__ import annotations

import time
from pathlib import Path, PureWindowsPath
from typing import Any

import requests as _requests

from ..config import get_config
from ..state import AgentState
from ..tools.indent_tools import get_toolkit


def _download_image(toolkit, remote_path: str) -> str:
    """
    Download a captured image from the iMicro PC to the local Mac.
    Returns the local file path, or "" if all strategies fail.
    """
    if not remote_path:
        return ""

    try:
        win_path = PureWindowsPath(remote_path)
        filename = win_path.name
        directory = str(win_path.parent)
    except Exception:
        filename = Path(remote_path).name
        directory = str(Path(remote_path).parent)

    local_dir = Path("./downloaded_images")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    client = toolkit.client
    base_url = client.base_url
    # Use the session from InstrumentClient so X-API-Token is included
    session = getattr(client, "session", None)

    def _get(url, **kwargs):
        if session:
            return session.get(url, **kwargs)
        return _requests.get(url, **kwargs)

    def _post(url, **kwargs):
        if session:
            return session.post(url, **kwargs)
        return _requests.post(url, **kwargs)

    # Strategy 1: POST /download_file with directory + filename
    try:
        r = _post(f"{base_url}/download_file",
                  json={"directory": directory, "filename": filename},
                  timeout=30)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            print(f"[observe] Image downloaded to: {local_path.resolve()}")
            return str(local_path.resolve())
    except Exception:
        pass

    # Strategy 2: POST /download_file with filename only
    try:
        r = _post(f"{base_url}/download_file",
                  json={"filename": filename},
                  timeout=30)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            print(f"[observe] Image downloaded to: {local_path.resolve()}")
            return str(local_path.resolve())
    except Exception:
        pass

    # Strategy 3: GET /download_file with path param
    try:
        r = _get(f"{base_url}/download_file",
                 params={"path": remote_path, "filename": filename},
                 timeout=30)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            print(f"[observe] Image downloaded to: {local_path.resolve()}")
            return str(local_path.resolve())
    except Exception:
        pass

    # Strategy 4: GET /video_panel (some servers expose this directly)
    try:
        r = _get(f"{base_url}/video_panel", timeout=30)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            print(f"[observe] Image downloaded to: {local_path.resolve()}")
            return str(local_path.resolve())
    except Exception:
        pass

    print(f"[observe] Could not download image. File is on iMicro PC at: {remote_path}")
    return ""


def observe_node(state: AgentState) -> dict[str, Any]:
    """
    Initial observation node.

    Captures microscope image and reads instrument state to
    establish baseline for the experiment.
    """
    cfg = get_config()
    updates: dict[str, Any] = {
        "current_stage": "observe",
    }

    try:
        toolkit = get_toolkit()

        # Get current position
        pos_result = toolkit.call_tool("get_xyz_positions", {})
        if pos_result.ok:
            result = pos_result.result or {}
            updates["current_position"] = (
                result.get("x", 0.0),
                result.get("y", 0.0),
            )
            updates["current_extension"] = result.get("extension", 0.0)

        # Get UI state
        ui_result = toolkit.call_tool("get_all_ui_state", {})
        if ui_result.ok:
            updates["instrument_status"] = ui_result.result or {}

        # Capture initial image
        img_result = toolkit.call_tool("capture_video_panel_remote", {})
        if img_result.ok:
            result_data = img_result.result if isinstance(img_result.result, dict) else {}
            # Real server returns "save_path"; simulator returns "path"
            remote_path = result_data.get("path") or result_data.get("save_path") or ""

            local_path = _download_image(toolkit, remote_path)

            img_data = {
                "image_path": remote_path,
                "local_path": local_path,
                "timestamp": str(time.time()),
                "position": updates.get("current_position", (0.0, 0.0)),
            }
            updates["current_image"] = img_data
            updates["image_history"] = [img_data]

        updates["scratchpad"] = ["observe: captured initial state and image"]

    except RuntimeError:
        # Toolkit not initialized - running in test mode
        updates["instrument_status"] = {"mode": "test"}
        updates["current_position"] = (0.0, 0.0)
        updates["scratchpad"] = ["observe: toolkit not available - test mode"]

    return updates
