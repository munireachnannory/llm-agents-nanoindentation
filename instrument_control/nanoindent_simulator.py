"""
nanoindent_simulator.py
=======================
In-memory simulator for nanoindenter automation commands.

Mirrors the command surface used by ``IndentToolkit`` / ``call_auto`` so agent
workflows can run without the real Flask instrument server or InView UI.
"""

from __future__ import annotations

import csv
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .synthetic_microstructure import (
    FOV_HEIGHT_UM,
    FOV_WIDTH_UM,
    properties_at,
    render_fov,
)

# Commands exposed to the multi-agent planner (keep in sync with indent_tools.py).
AGENT_SAFE_TOOL_NAMES: tuple[str, ...] = (
    "move",
    "move_in_increments",
    "set_extension",
    "engage",
    "starting_tests",
    "start_test_normal",
    "change_method",
    "get_current_sample_and_test_ocr_passive",
    "get_xyz_positions",
    "get_all_ui_state",
    "process_review_file",
    "capture_video_panel_remote",
    "align_focus",
    "focus",
    "restore_alignment",
    "move_to_aligned_point",
)

# Extra server commands referenced in the automation reference notebook.
EXTRA_COMMANDS: tuple[str, ...] = (
    "apply_review_parameters_ui",
    "click_review_button",
    "open_sample_from_directory",
    "save_review_file_as",
    "get_current_sample_and_test_ocr",
    "capture_video_panel",
    "align_focus_remote",
)

ALLOWED_COMMANDS: frozenset[str] = frozenset(AGENT_SAFE_TOOL_NAMES) | frozenset(EXTRA_COMMANDS)

_DIRECTION_DELTA: dict[str, tuple[float, float]] = {
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
}


@dataclass
class AlignmentState:
    area_name: str
    new_origin_macro: tuple[float, float]
    new_origin_micro: tuple[float, float]
    macro_scale_x: float
    macro_scale_y: float
    macro_image_path: str = ""
    rotation_deg: float = 0.0


@dataclass
class TestState:
    sample_name: str = ""
    method_type: str = "eh"
    setup: dict[str, Any] = field(default_factory=dict)
    indents: dict[str, Any] = field(default_factory=dict)
    status: str = "idle"  # idle | configured | running | complete
    expected_indents: int = 0
    test_number: int = 0
    queued_tests_removed: int = 0


@dataclass
class NanoindentSimulatorState:
    x_um: float = 1000.0
    y_um: float = 1000.0
    extension_um: float = 12.0
    engaged: bool = False
    method_type: str = "eh"
    plane_params: tuple[float, float, float] = (0.0, 0.0, 12.0)
    alignments: dict[str, AlignmentState] = field(default_factory=dict)
    test: TestState = field(default_factory=TestState)
    ui_state: dict[str, float] = field(default_factory=dict)
    last_export: dict[str, str] = field(default_factory=dict)
    command_log: list[dict[str, Any]] = field(default_factory=list)


class NanoindentSimulator:
    """Stateful simulator for nanoindenter ``call_auto`` commands."""

    def __init__(
        self,
        data_dir: str | Path = "./sim_instrument_data",
        run_seed: int | None = None,
    ):
        """
        Parameters
        ----------
        run_seed
            Run-level seed for *measurement noise only*. The ground-truth
            microstructure is generated from a separate fixed global seed in
            ``synthetic_microstructure.py`` and is deliberately NOT affected by
            this value, so every run measures the same virtual specimen.

            Before GAPS.md G9a was closed, noise was seeded from stage position
            alone, so revisiting a site returned a byte-identical value and
            repeated campaigns were not independent replicates -- no standard
            error across simulated runs was meaningful, and any paired
            comparison between strategies was confounded. Passing a distinct
            ``run_seed`` per run makes repeated campaigns genuine replicates.

            ``None`` reproduces the legacy position-only behaviour, retained so
            that campaigns logged before this fix remain reproducible.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "captures").mkdir(exist_ok=True)
        (self.data_dir / "exports").mkdir(exist_ok=True)
        (self.data_dir / "review").mkdir(exist_ok=True)
        self.run_seed = run_seed
        self.state = NanoindentSimulatorState()
        self._refresh_ui_state()

    def reset(self, run_seed: int | None = None) -> None:
        """Reset instrument state. Optionally re-seed the noise generator."""
        if run_seed is not None:
            self.run_seed = run_seed
        self.state = NanoindentSimulatorState()
        self._refresh_ui_state()

    def _noise_rng(self, x_um: float, y_um: float) -> "np.random.Generator":
        """
        Measurement-noise generator for an array centred at (x_um, y_um).

        Position still contributes, so that within a single run the spatial
        noise field is coherent; the run seed is mixed in so that two runs of
        the same strategy over the same specimen are independent (GAPS.md G9a).
        """
        pos = (int(x_um * 11.0) ^ int(y_um * 17.0)) & 0xFFFFFFFF
        if self.run_seed is None:
            return np.random.default_rng(pos)
        # SeedSequence mixes the two entropy sources properly; XOR would
        # correlate nearby positions across runs.
        return np.random.default_rng(
            np.random.SeedSequence([int(self.run_seed) & 0xFFFFFFFF, pos])
        )

    def dispatch(self, command: str, **kwargs: Any) -> Any:
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"Command '{command}' is not allowed on the simulator")

        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            return self._cmd_stub(command, **kwargs)

        t_wait = float(kwargs.pop("t", 0) or 0)
        if t_wait > 0:
            # Real server sleeps; simulator uses a short cap so tests stay fast.
            time.sleep(min(t_wait, 0.05))

        result = handler(**kwargs)
        self._log_command(command, kwargs, result)
        self._refresh_ui_state()
        return result

    # ------------------------------------------------------------------
    # Motion / Z
    # ------------------------------------------------------------------

    def _cmd_move(self, amount: float, direction: str, **_: Any) -> dict[str, Any]:
        direction = str(direction).lower()
        if direction not in _DIRECTION_DELTA:
            raise ValueError(f"Invalid direction: {direction}")
        dx, dy = _DIRECTION_DELTA[direction]
        self.state.x_um += float(amount) * dx
        self.state.y_um += float(amount) * dy
        return {
            "ok": True,
            "message": f"Moved {amount} µm {direction}",
            "x": self.state.x_um,
            "y": self.state.y_um,
        }

    def _cmd_move_in_increments(
        self,
        total_amount: float,
        direction: str,
        increment: float = 25.0,
        **_: Any,
    ) -> dict[str, Any]:
        remaining = float(total_amount)
        steps = 0
        while remaining > 0:
            step = min(float(increment), remaining)
            self._cmd_move(amount=step, direction=direction)
            remaining -= step
            steps += 1
        return {
            "ok": True,
            "message": f"Completed {steps} incremental moves {direction}",
            "x": self.state.x_um,
            "y": self.state.y_um,
        }

    def _cmd_set_extension(self, number: float, **_: Any) -> dict[str, Any]:
        self.state.extension_um = float(number)
        return {
            "ok": True,
            "message": f"Extension set to {self.state.extension_um:.3f} µm",
            "extension": self.state.extension_um,
        }

    def _cmd_engage(self, **_: Any) -> dict[str, Any]:
        self.state.engaged = True
        return {"ok": True, "message": "Engage clicked (simulated)"}

    # ------------------------------------------------------------------
    # Method / tests
    # ------------------------------------------------------------------

    def _cmd_change_method(
        self,
        add: int = 1,
        method_dir: Optional[str] = None,
        method_file_name: Optional[str] = None,
        method_type: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        if method_type:
            self.state.method_type = str(method_type)
        return {
            "ok": True,
            "message": "Method loaded (simulated)",
            "method_type": self.state.method_type,
            "add": int(add),
            "method_dir": method_dir,
            "method_file_name": method_file_name,
        }

    def _cmd_starting_tests(
        self,
        sample_name: str,
        setup: dict[str, Any],
        indents: dict[str, Any],
        remove_n: int = 1,
        method_type: str = "eh",
        use_import: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        nx = int(indents.get("X Indents", 5))
        ny = int(indents.get("Y Indents", 5))
        self.state.test = TestState(
            sample_name=str(sample_name),
            method_type=str(method_type),
            setup=dict(setup or {}),
            indents=dict(indents or {}),
            status="configured",
            expected_indents=nx * ny,
            test_number=0,
            queued_tests_removed=int(remove_n),
        )
        self.state.method_type = str(method_type)
        return {
            "ok": True,
            "message": f"Configured test '{sample_name}' ({nx}x{ny} indents)",
            "sample_name": sample_name,
            "expected_indents": nx * ny,
            "use_import": int(use_import),
        }

    def _cmd_start_test_normal(self, **_: Any) -> dict[str, Any]:
        test = self.state.test
        if test.status not in ("configured", "running", "complete"):
            raise RuntimeError("No configured test — call starting_tests first")
        test.status = "running"
        if test.test_number == 0:
            test.test_number = 0
        return {
            "ok": True,
            "message": f"Started test '{test.sample_name}' (simulated)",
            "sample_name": test.sample_name,
        }

    def _cmd_get_current_sample_and_test_ocr_passive(
        self,
        debug: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        test = self.state.test
        if test.status == "running" and test.test_number < test.expected_indents:
            test.test_number += 1
            if test.test_number >= test.expected_indents:
                test.status = "complete"
        payload = {
            "test_number": test.test_number,
            "sample_name": test.sample_name,
            "expected_indents": test.expected_indents,
            "status": test.status,
            "debug": bool(debug),
        }
        return {"ok": True, "result": payload, **payload}

    def _cmd_get_current_sample_and_test_ocr(self, **kwargs: Any) -> dict[str, Any]:
        return self._cmd_get_current_sample_and_test_ocr_passive(**kwargs)

    # ------------------------------------------------------------------
    # State readback
    # ------------------------------------------------------------------

    def _cmd_get_xyz_positions(self, **_: Any) -> dict[str, float]:
        return {
            "x": self.state.x_um,
            "y": self.state.y_um,
            "extension": self.state.extension_um,
        }

    def _cmd_get_all_ui_state(self, **_: Any) -> dict[str, float]:
        return dict(self.state.ui_state)

    # ------------------------------------------------------------------
    # Imaging
    # ------------------------------------------------------------------

    def _cmd_capture_video_panel_remote(self, **_: Any) -> dict[str, Any]:
        return self._capture_panel(prefix="video_panel")

    def _cmd_capture_video_panel(self, **_: Any) -> dict[str, Any]:
        return self._capture_panel(prefix="video_panel")

    def _capture_panel(self, prefix: str) -> dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}.png"
        path = self.data_dir / "captures" / filename
        self._write_synthetic_microscope_image(path)
        rel_dir = "captures"
        return {
            "ok": True,
            "message": "Captured video panel (simulated)",
            "path": str(path.resolve()),
            "directory": rel_dir,
            "filename": filename,
            "remote_dir": rel_dir,
            "remote_name": filename,
            "x": self.state.x_um,
            "y": self.state.y_um,
            "fov_width_um": FOV_WIDTH_UM,
            "fov_height_um": FOV_HEIGHT_UM,
            "simulated": True,
        }

    # ------------------------------------------------------------------
    # Review / export
    # ------------------------------------------------------------------

    def _cmd_process_review_file(
        self,
        sample_names: list[str],
        open_dir: str,
        current_sample_file_name: str,
        export_dir: str,
        export_file_name: str,
        review_changes: dict[str, Any],
        review_type: str = "eh",
        export_format: str = "excel",
        load_wait: int = 30,
        recalc_wait: int = 30,
        save_wait: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        test = self.state.test
        if test.status not in ("running", "complete") and not sample_names:
            raise RuntimeError("No completed test to process")

        export_fmt = str(export_format).lower()
        ext = ".csv" if export_fmt == "csv" else ".xlsx"
        if export_file_name.endswith((".csv", ".xlsx", ".xls")):
            export_name = export_file_name
        else:
            export_name = f"{export_file_name}{ext}"

        export_path = self.data_dir / "exports" / export_name
        rows = self._synthetic_indent_rows(
            sample_name=current_sample_file_name or (sample_names[0] if sample_names else "sim_sample"),
            n=max(test.expected_indents, 1),
        )
        if export_name.endswith(".csv"):
            self._write_csv(export_path, rows)
        else:
            # Lightweight stand-in: write CSV with xlsx extension for agent path tests.
            self._write_csv(export_path.with_suffix(".csv"), rows)
            export_path = export_path.with_suffix(".csv")
            export_name = export_path.name

        self.state.last_export = {
            "directory": "exports",
            "filename": export_name,
            "path": str(export_path.resolve()),
            "sample_name": current_sample_file_name,
        }
        return {
            "ok": True,
            "message": "ReviewData processed and exported (simulated)",
            "export_path": str(export_path.resolve()),
            "export_dir": str((self.data_dir / "exports").resolve()),
            "export_file_name": export_name,
            "review_type": review_type,
            "review_changes": review_changes,
            "load_wait": load_wait,
            "recalc_wait": recalc_wait,
            "save_wait": save_wait,
            "simulated": True,
        }

    # ------------------------------------------------------------------
    # Alignment / focus
    # ------------------------------------------------------------------

    def _cmd_align_focus(self, **_: Any) -> list[float]:
        # Mild plane tilt based on current position.
        a, b, c = 0.001, -0.0008, self.state.extension_um
        self.state.plane_params = (a, b, c)
        return [a, b, c]

    def _cmd_align_focus_remote(self, **_: Any) -> list[float]:
        return self._cmd_align_focus()

    def _cmd_focus(self, plane_params: list[float], **_: Any) -> dict[str, Any]:
        if len(plane_params) != 3:
            raise ValueError("plane_params must be [a, b, c]")
        a, b, c = (float(plane_params[0]), float(plane_params[1]), float(plane_params[2]))
        self.state.plane_params = (a, b, c)
        z = a * self.state.x_um + b * self.state.y_um + c
        self.state.extension_um = z
        return {
            "ok": True,
            "message": "Focus applied from plane (simulated)",
            "extension": z,
            "plane_params": [a, b, c],
        }

    def restore_alignment(
        self,
        area_name: str,
        new_origin_macro: list[float],
        new_origin_micro: list[float],
        macro_scale_x: float,
        macro_scale_y: float,
        macro_image_path: str = "",
    ) -> dict[str, Any]:
        if len(new_origin_macro) < 2 or len(new_origin_micro) < 2:
            raise ValueError("origin lists must have at least two values")
        alignment = AlignmentState(
            area_name=str(area_name),
            new_origin_macro=(float(new_origin_macro[0]), float(new_origin_macro[1])),
            new_origin_micro=(float(new_origin_micro[0]), float(new_origin_micro[1])),
            macro_scale_x=float(macro_scale_x),
            macro_scale_y=float(macro_scale_y),
            macro_image_path=str(macro_image_path or ""),
        )
        self.state.alignments[alignment.area_name] = alignment
        return {
            "ok": True,
            "message": f"Alignment restored for '{area_name}' (simulated)",
            "area_name": area_name,
        }

    def move_to_aligned_point(
        self,
        area_name: str,
        x: float,
        y: float,
        plane_params: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        alignment = self.state.alignments.get(area_name)
        if alignment is None:
            raise RuntimeError(f"Alignment '{area_name}' not found — call restore_alignment first")

        ox_m, oy_m = alignment.new_origin_macro
        ox_s, oy_s = alignment.new_origin_micro
        dx = (float(x) - ox_m) * alignment.macro_scale_x
        dy = (float(y) - oy_m) * alignment.macro_scale_y
        self.state.x_um = ox_s + dx
        self.state.y_um = oy_s + dy

        focus_result = None
        if plane_params:
            focus_result = self._cmd_focus(plane_params=plane_params)

        return {
            "ok": True,
            "message": f"Moved to aligned point ({x}, {y}) in '{area_name}' (simulated)",
            "x": self.state.x_um,
            "y": self.state.y_um,
            "macro_x": float(x),
            "macro_y": float(y),
            "focus": focus_result,
        }

    def _cmd_move_to_aligned_point(
        self,
        area_name: str,
        x: float,
        y: float,
        plane_params: Optional[list[float]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self.move_to_aligned_point(
            area_name=area_name,
            x=x,
            y=y,
            plane_params=plane_params,
        )

    def _cmd_restore_alignment(
        self,
        area_name: str,
        new_origin_macro: list[float],
        new_origin_micro: list[float],
        macro_scale_x: float,
        macro_scale_y: float,
        macro_image_path: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        return self.restore_alignment(
            area_name=area_name,
            new_origin_macro=new_origin_macro,
            new_origin_micro=new_origin_micro,
            macro_scale_x=macro_scale_x,
            macro_scale_y=macro_scale_y,
            macro_image_path=macro_image_path,
        )

    # ------------------------------------------------------------------
    # Stubs for ancillary review UI commands
    # ------------------------------------------------------------------

    def _cmd_stub(self, command: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "message": f"{command} acknowledged (simulated stub)",
            "kwargs": kwargs,
            "simulated": True,
        }

    def _cmd_apply_review_parameters_ui(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "message": "Review parameters applied (simulated)", "kwargs": kwargs}

    def _cmd_click_review_button(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "message": "Review button clicked (simulated)", "kwargs": kwargs}

    def _cmd_open_sample_from_directory(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "message": "Sample opened (simulated)", "kwargs": kwargs}

    def _cmd_save_review_file_as(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "message": "Review file saved (simulated)", "kwargs": kwargs}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_download_path(self, directory: str, filename: str) -> Path:
        base = self.data_dir / directory
        path = (base / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Simulated file not found: {directory}/{filename}")
        if self.data_dir.resolve() not in path.parents and path != self.data_dir.resolve():
            raise ValueError("Invalid download path")
        return path

    def _refresh_ui_state(self) -> None:
        test = self.state.test
        self.state.ui_state = {
            "Extension": self.state.extension_um,
            "X Position": self.state.x_um,
            "Y Position": self.state.y_um,
            "Force": 0.45 if self.state.engaged else 0.0,
            "Displacement": 120.0 if test.status == "running" else 0.0,
            "Test Number": float(test.test_number),
            "Expected Indents": float(test.expected_indents),
        }

    def _log_command(self, command: str, kwargs: dict[str, Any], result: Any) -> None:
        self.state.command_log.append(
            {
                "ts": time.time(),
                "command": command,
                "kwargs": kwargs,
                "result": result,
            }
        )

    def _write_synthetic_microscope_image(self, path: Path) -> None:
        render_fov(
            x_center_um=self.state.x_um,
            y_center_um=self.state.y_um,
            extension_um=self.state.extension_um,
            path=path,
        )

    def _synthetic_indent_rows(self, sample_name: str, n: int) -> list[dict[str, Any]]:
        """
        Sample the ground-truth microstructure at each indent site so results
        depend on where the agents chose to indent (phase, boundaries, pores).
        """
        indents = self.state.test.indents or {}
        nx = int(indents.get("X Indents", max(int(math.sqrt(n)), 1)))
        ny = int(indents.get("Y Indents", max(int(math.ceil(n / max(nx, 1))), 1)))
        sx = float(indents.get("X Spacing", indents.get("Spacing", 5.0)) or 5.0)
        sy = float(indents.get("Y Spacing", indents.get("Spacing", 5.0)) or 5.0)

        rng = self._noise_rng(self.state.x_um, self.state.y_um)
        rows: list[dict[str, Any]] = []
        for i in range(n):
            gi, gj = i % max(nx, 1), i // max(nx, 1)
            x = self.state.x_um + (gi - (nx - 1) / 2.0) * sx
            y = self.state.y_um + (gj - (ny - 1) / 2.0) * sy

            truth = properties_at(x, y)
            hardness = truth["hardness_gpa"] * float(rng.normal(1.0, 0.03))
            modulus = truth["modulus_gpa"] * float(rng.normal(1.0, 0.02))
            # Deeper contact at a fixed target load in softer material.
            depth_nm = 250.0 * math.sqrt(8.0 / max(hardness, 0.2))

            rows.append(
                {
                    "Sample": sample_name,
                    "Indent": i + 1,
                    "X (um)": round(x, 3),
                    "Y (um)": round(y, 3),
                    "Hardness (GPa)": round(hardness, 3),
                    "Modulus (GPa)": round(modulus, 3),
                    "Contact Depth (nm)": round(depth_nm, 1),
                    "Region (sim truth)": truth["phase"]
                    + ("+boundary" if truth["near_boundary"] else ""),
                }
            )
        return rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            rows = [{"Sample": "empty", "Indent": 0}]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
