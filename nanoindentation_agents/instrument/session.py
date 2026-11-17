"""
InstrumentSession — one object that owns the connection contract.

Two call tiers (the load-bearing safety idea from notebook 5, kept verbatim):

* ``call()``  — STRICT, raises. Used only while aborting is still safe
  (before ``start_test_normal`` fires).
* ``call_soft()`` — TOLERANT, returns (ok, result), never raises. Used inside
  the monitor loop after indents have started: a dropped VPN packet must not
  abandon data for indents that cannot be re-run.

``place_array()`` places ONE approved array and screen-reads while it runs.
It NEVER moves the stage — positioning belongs to MotionController, which
removed the double-move / wrong-direction bug class from notebook 5 cell D.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import notebook_ui as ui
from .screenread import (LiveScreenRecorder, ScreenReadConfig, TestSequenceTracker,
                         parse_current_test_index, screen_metric)


class InstrumentCommError(RuntimeError):
    """Raised when the instrument cannot be reached at a point where it is
    still safe to stop (nothing irreversible has happened yet)."""


class InstrumentMethodError(InstrumentCommError):
    """The server ANSWERED and the automation method raised on the instrument PC.

    Distinct from a transport failure on purpose: a 400 from /call_auto means
    the iMicro tried and threw (a window it could not click, a dialog in the
    way, a bad argument) — the network is fine, and telling the operator to
    check the VPN sends them to the wrong machine. Subclasses
    InstrumentCommError so existing handlers keep working.
    """

    def __init__(self, message: str, *, tool: str = "", status_code=None,
                 server_message: str = "", server_traceback: str = ""):
        super().__init__(message)
        self.tool = tool
        self.status_code = status_code
        self.server_message = server_message
        self.server_traceback = server_traceback


def _raise_tool_failure(tool: str, out) -> None:
    """Raise the RIGHT exception for a failed ToolOutcome."""
    if getattr(out, "server_side_failure", False):
        msg = (f"{tool} failed ON THE INSTRUMENT PC "
               f"(HTTP {out.status_code}, the network is fine): "
               f"{out.server_message or out.error}")
        tb = str(getattr(out, "server_traceback", "") or "").strip().splitlines()
        if tb:
            msg += f"\n    instrument traceback (last line): {tb[-1].strip()}"
        msg += ("\n    This is almost always the InView window: a leftover modal "
                "dialog, ReviewData in front of RunTest, or the window moved. "
                "Check the instrument SCREEN, clear it, then re-run — no indents "
                "were placed by the failed call.")
        raise InstrumentMethodError(msg, tool=tool, status_code=out.status_code,
                                    server_message=out.server_message or "",
                                    server_traceback=str(
                                        getattr(out, "server_traceback", "") or ""))
    raise InstrumentCommError(f"{tool} failed: {out.error}")


class InstrumentSession:
    """Wraps an IndentToolkit with the session-level semantics the campaign
    needs: strict/soft calls, stage readback, capture, and array placement."""

    def __init__(self, toolkit, *, download_dir, run_id: str,
                 screen_config: Optional[ScreenReadConfig] = None,
                 journal=None):
        self.toolkit = toolkit
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.screen_config = screen_config or ScreenReadConfig(csv_dir=str(self.download_dir))
        if not self.screen_config.csv_dir or self.screen_config.csv_dir == ".":
            self.screen_config.csv_dir = str(self.download_dir)
        self.journal = journal
        self.run_records: dict[str, dict] = {}   # sample -> frozen run_record

    # ------------------------------------------------------------ basics
    def ping(self) -> bool:
        return self.toolkit.ping()

    def call(self, tool: str, args: dict, tool_log: Optional[list] = None):
        """STRICT: raises. Use only where aborting is safe."""
        out = self.toolkit.call_tool(tool, args)
        if tool_log is not None:
            tool_log.append({"tool": tool,
                             "arguments": json.dumps(args, default=str)[:220],
                             "ok": out.ok,
                             "elapsed_s": round(getattr(out, "elapsed_s", 0.0) or 0.0, 3),
                             "summary": str(out.result if out.ok else out.error)[:180]})
        if not out.ok:
            _raise_tool_failure(tool, out)
        return out.result

    def call_soft(self, tool: str, args: dict, tool_log: Optional[list] = None,
                  suppress_log: bool = False):
        """TOLERANT: returns (ok, result), never raises. Only AFTER indents start."""
        try:
            out = self.toolkit.call_tool(tool, args)
            ok = bool(out.ok)
            payload = out.result if ok else {}
            note = str(out.result if ok else out.error)[:180]
            secs = round(getattr(out, "elapsed_s", 0.0) or 0.0, 3)
        except Exception as exc:
            ok, payload, note, secs = False, {}, f"{type(exc).__name__}: {exc}"[:180], 0.0
        if tool_log is not None and not suppress_log:
            tool_log.append({"tool": tool,
                             "arguments": json.dumps(args, default=str)[:220],
                             "ok": ok, "elapsed_s": secs, "summary": note})
        return ok, payload

    # ------------------------------------------------------------ stage / frames
    def read_stage_xy(self, *, what: str = "stage position", attempts: int = 4,
                      pause_s: float = 1.5) -> tuple[float, float]:
        """Live (x, y) in um, or raise after `attempts` unreadable replies.

        A single NaN reply is transient and must not kill a round; a persistent
        one is fatal because an indent placed without a readable position could
        never be located afterwards.
        """
        last = None
        for i in range(attempts):
            out = self.toolkit.call_tool("get_xyz_positions", {})
            if out.ok:
                r = out.result if isinstance(out.result, dict) else {}
                x = float(r.get("x", float("nan")))
                y = float(r.get("y", float("nan")))
                if math.isfinite(x) and math.isfinite(y):
                    return x, y
                last = f"NaN reply {r}"
            else:
                last = f"tool error {out.error}"
            if i + 1 < attempts:
                print(f"    position read {i + 1}/{attempts} failed ({last}) — retrying")
                time.sleep(pause_s)
        raise InstrumentCommError(
            f"{what} unreadable after {attempts} tries ({last}) — an indent placed "
            "now could not be located afterwards")

    def capture_frame(self, *, fov_w_um: float, fov_h_um: float, show: bool = True,
                      title: str = "Live frame", note: str = "",
                      um_per_px: Optional[float] = None):
        """Grab the video panel at the CURRENT position.
        Returns (frame_pos, image_info | None)."""
        frame_pos = self.read_stage_xy(what="stage position at capture")
        cap = self.toolkit.call_tool("capture_video_panel_remote", {})
        if not cap.ok:
            print(f"  [warn] capture failed ({cap.error}) — keeping the previous frame.")
            return frame_pos, None
        img = ui.download_capture(self.toolkit,
                                  cap.result if isinstance(cap.result, dict) else {},
                                  self.download_dir)
        if not img.get("local_path"):
            print("  [warn] frame downloaded empty — keeping the previous frame.")
            return frame_pos, None
        img["width_um"], img["height_um"] = fov_w_um, fov_h_um
        if um_per_px is not None:
            img["um_per_px"] = um_per_px
        if show:
            ui.show_capture(img, title=title, note=note,
                            fov_width_um=fov_w_um, fov_height_um=fov_h_um)
        return frame_pos, img

    def read_screen_metrics(self):
        """(depth, load, hardness, modulus, latency_s) from get_all_ui_state."""
        try:
            st = self.toolkit.call_tool("get_all_ui_state", {})
            ui_state = st.result if st.ok else {}
            lat = round(getattr(st, "elapsed_s", 0.0) or 0.0, 3)
        except Exception:
            ui_state, lat = {}, float("nan")
        mk = self.screen_config.metric_keys
        return (screen_metric(ui_state, mk["depth"]),
                screen_metric(ui_state, mk["load"]),
                screen_metric(ui_state, mk["hardness"]),
                screen_metric(ui_state, mk["modulus"]), lat)

    def screen_read_diagnostic(self) -> None:
        """Print the raw get_all_ui_state keys vs what the recorder will look for."""
        st = self.toolkit.call_tool("get_all_ui_state", {})
        u = st.result if st.ok else {}
        print(f"get_all_ui_state ok={st.ok}  keys={len(u)}  "
              f"latency={getattr(st, 'elapsed_s', 0):.2f}s")
        if not st.ok:
            print(f"  ERROR: {st.error}")
        for k, v in sorted(u.items() if isinstance(u, dict) else []):
            print(f"    {k!r:<42} = {v!r}")
        print("\nWhat the recorder will look for:")
        for metric, keys in self.screen_config.metric_keys.items():
            hit = next((k for k in keys if isinstance(u, dict) and k in u), None)
            print(f"  {metric:<9} {keys} -> "
                  f"{'MATCH ' + repr(hit) if hit else 'NO MATCH — live read will be nan'}")

    # ------------------------------------------------------------ the array
    def place_array(self, plan: dict, *, fov_w_um: float = 100.0, fov_h_um: float = 75.0,
                    test_timeout_s: Optional[float] = None, show_frames: bool = True,
                    done_probe: Optional[Any] = None, skip_screening: bool = False,
                    indent_duration_s: float = 150.0) -> dict:
        """Place ONE approved array at the CURRENT stage position and wait
        for it to finish. Irreversible; never moves the stage.

        The caller (CampaignRunner) is responsible for positioning first and
        arming ``plan['destination']`` at the live readback. If the plan and
        the live position disagree by more than 1 um this refuses to start —
        that mismatch means the positioning step was skipped.

        `done_probe` is an optional zero-argument callable answering "has InView
        written this test's result file yet?". Its EXISTENCE alone is not
        sufficient completion evidence on a multi-indent array — see
        `_monitor_screen_read` / `LiveScreenRecorder.grid_complete` —
        except when `skip_screening=True`, where the operator has deliberately
        chosen to trust it outright (see CampaignConfig.skip_screening).

        `skip_screening=True` bypasses the live screen-read monitor entirely:
        no OCR/depth polling, no per-indent H/E capture. Process Review is
        then the SOLE data source for this array — `run_record["recorder"]`
        comes back None, which `finalize_indent_data` already treats as an
        empty screen-read table, not an error.
        """
        tool_log: list[dict] = []

        # ---- 0. verify the plan is armed at the live position (strict zone) ----
        pos = self.read_stage_xy(what=f"stage position before '{plan['sample_name']}'")
        dest = plan.get("destination") or pos
        drift = math.hypot(pos[0] - float(dest[0]), pos[1] - float(dest[1]))
        if drift > 1.0:
            raise InstrumentCommError(
                f"plan '{plan['sample_name']}' is armed at {tuple(dest)} but the stage "
                f"reads ({pos[0]:.2f}, {pos[1]:.2f}) — {drift:.1f} um apart. Position "
                "with MotionController.jog_to() and arm_plan_at() first; place_array() "
                "never moves the stage.")
        plan["destination"] = (round(pos[0], 2), round(pos[1], 2))

        # ---- 1. confirmation capture at the target (strict zone) ----
        cap = self.call("capture_video_panel_remote", {}, tool_log)
        image_info = ui.download_capture(self.toolkit,
                                         cap if isinstance(cap, dict) else {},
                                         self.download_dir)
        if show_frames and image_info.get("local_path"):
            ui.show_capture(image_info, title=f"Confirmation frame — {plan['label']}",
                            note="Captured at the armed position, before indenting.",
                            fov_width_um=fov_w_um, fov_height_um=fov_h_um)

        # ---- 2. configure + START (the point of no return) ----
        nx, ny, sp = plan["grid_nx"], plan["grid_ny"], plan["spacing_um"]
        rotation = float(plan.get("rotation_deg", 0.0))
        self.call("starting_tests",
                  {"sample_name": plan["sample_name"], "setup": plan.get("setup") or {},
                   "indents": {"X Indents": nx, "Y Indents": ny,
                               "X Spacing": sp, "Y Spacing": sp,
                               "Rotation": rotation},
                   "method_type": plan.get("method_type", "eh"),
                   "remove_n": int(plan.get("remove_n", 0))}, tool_log)
        self.call("start_test_normal", {"t": 0}, tool_log)

        # ---- 3. wait for completion: screen-read monitor, or a file-only wait ----
        expected = nx * ny
        timeout_s = float(os.environ.get("NB5_TEST_TIMEOUT_S",
                                         test_timeout_s or 90.0 * expected + 120.0))
        # InView writes the result file after the FIRST indent of the array, so
        # its appearance leaves (expected - 1) indents still to run. Hold for
        # that long before Process Review may touch the data.
        post_file_wait_s = max(0.0, (expected - 1) * float(indent_duration_s))
        if post_file_wait_s:
            needed = post_file_wait_s + 600.0
            if timeout_s < needed:
                # Otherwise a big grid would hit the monitor deadline before its
                # own estimated wait finished (a 5x5 needs 60 min; the default
                # timeout is 30) and export mid-array anyway.
                print(f"  [note] monitor timeout raised {timeout_s:.0f}s -> {needed:.0f}s "
                      f"to cover the {post_file_wait_s / 60:.1f} min estimated wait "
                      f"for {expected} indents.")
                timeout_s = needed
        if skip_screening:
            recorder = None
            mon = self._wait_for_result_file(plan["sample_name"], expected=expected,
                                             timeout_s=timeout_s, done_probe=done_probe,
                                             post_file_wait_s=post_file_wait_s)
        else:
            recorder = LiveScreenRecorder(plan, self.screen_config, self.run_id)
            mon = self._monitor_screen_read(recorder, expected=expected, timeout_s=timeout_s,
                                            done_probe=done_probe, tool_log=tool_log,
                                            post_file_wait_s=post_file_wait_s)
        result_dir = mon["result_dir"]
        done_reason = mon["done_reason"]
        n_done = mon["n_done"]
        status = mon["status"]
        monitor_lost_connection = mon["monitor_lost_connection"]

        # ---- 4. freeze everything the export/finalize steps will need ----
        # export_name is FROZEN here, never rebuilt later: InView derives its
        # export file names from what was typed into Save As.
        run_record = {
            "sample_name": plan["sample_name"],
            "export_name": f"{plan['sample_name']}_{self.run_id}_export",
            # The folder the completion probe found the result in, when it was
            # the probe that ended the wait — the export step then skips its own
            # search entirely.
            "sample_dir": result_dir,
            "plan": plan, "label": plan["label"],
            "method_type": plan.get("method_type", "eh"),
            "review_changes": dict(
                getattr(plan.get("method_plan"), "review_changes", {})
                or {}),
            "grid_nx": nx, "grid_ny": ny, "spacing_um": sp,
            "rotation_deg": rotation, "expected": expected,
            "position": (round(pos[0], 2), round(pos[1], 2)),
            "destination": plan["destination"],
            "recorder": recorder, "image": image_info, "tool_log": tool_log,
            "monitor_status": status, "n_done": n_done, "done_reason": done_reason,
            "monitor_lost_connection": monitor_lost_connection,
            "run_id": self.run_id, "download_dir": str(self.download_dir),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.run_records[plan["sample_name"]] = run_record
        screen_csv = str(recorder.csv_path) if recorder is not None else None
        screen_records = len(recorder.records) if recorder is not None else 0
        if self.journal is not None:
            method_plan = plan.get("method_plan")
            grid_plan = plan.get("grid_plan")
            strategy_plan = plan.get("measurement_strategy")
            self.journal.append("ARRAY_PLACED",
                                sample=plan["sample_name"],
                                export_name=run_record["export_name"],
                                position=run_record["position"],
                                grid=f"{nx}x{ny}", spacing_um=sp,
                                rotation_deg=rotation,
                                expected=expected, n_done=n_done,
                                done_reason=done_reason,
                                monitor_lost_connection=monitor_lost_connection,
                                skip_screening=skip_screening,
                                screen_csv=screen_csv,
                                screen_records=screen_records,
                                method_type=run_record["method_type"],
                                label=plan["label"],
                                method_plan=(method_plan.to_journal()
                                            if method_plan is not None
                                            else {"setup": plan.get("setup") or {}}),
                                grid_plan=(grid_plan.to_journal()
                                           if grid_plan is not None else {}),
                                measurement_strategy=(strategy_plan.to_journal()
                                                      if strategy_plan is not None else {}))
        source_note = ("skip_screening — Process Review is the sole source"
                       if recorder is None else f"{screen_records}/{expected} indents seen on screen")
        print(f"PLACED: '{plan['sample_name']}' — {source_note}, export name frozen as "
              f"'{run_record['export_name']}'.")
        return run_record

    @staticmethod
    def _announce_post_file_wait(expected: int, post_file_wait_s: float) -> None:
        """The one place the estimated-wait banner is worded."""
        print("  Process Review file detected after first indentation.")
        print(f"  Total indentations requested: {expected}")
        print(f"  Remaining indentations: {max(0, expected - 1)}")
        print(f"  Estimated waiting time: {post_file_wait_s:.0f} seconds "
              f"({post_file_wait_s / 60:.1f} minutes)")
        print("  Waiting for indentation sequence to finish...")

    def _wait_for_result_file(self, sample_name: str, *, expected: int, timeout_s: float,
                              done_probe: Optional[Any],
                              post_file_wait_s: float = 0.0) -> dict:
        """skip_screening's completion wait: poll ONLY the result-file probe —
        no OCR, no depth, no per-indent tracking.

        The file appears after the FIRST indent, so finding it is the START of
        the wait, not the end: the remaining (expected - 1) indents are given
        `post_file_wait_s` to run before Process Review is allowed."""
        print(f"[skip_screening] '{sample_name}' ({expected} indent(s) planned) — "
              "waiting for the result file, then the estimated indent time.")
        deadline = time.monotonic() + timeout_s
        poll_s = max(1.0, float(self.screen_config.ocr_interval_s))
        result_dir = None
        while done_probe is not None and result_dir is None and time.monotonic() < deadline:
            try:
                result_dir = done_probe()
            except Exception as exc:
                print(f"  [warn] completion probe failed ({exc}) — retrying.")
            if result_dir is None:
                time.sleep(poll_s)
        if result_dir:
            self._announce_post_file_wait(expected, post_file_wait_s)
            waited = self._hold_for_remaining_indents(post_file_wait_s, deadline)
            done_reason = (f"InView wrote the result file to {result_dir}, then waited "
                           f"{waited:.0f}s for the remaining {max(0, expected - 1)} "
                           "indent(s) (skip_screening)")
        elif done_probe is None:
            done_reason = ("skip_screening: no result-file probe available "
                           "(e.g. simulator) — proceeding immediately")
        else:
            done_reason = f"skip_screening: result file did not appear within {timeout_s:.0f}s"
        print(f"  {done_reason}")
        return {"result_dir": result_dir, "done_reason": done_reason, "n_done": expected,
                "status": {}, "monitor_lost_connection": False}

    @staticmethod
    def _hold_for_remaining_indents(post_file_wait_s: float, deadline: float) -> float:
        """Sleep out the estimated wait, reporting progress; returns seconds held."""
        started = time.monotonic()
        next_note = started + 60.0
        while True:
            waited = time.monotonic() - started
            if waited >= post_file_wait_s:
                break
            if time.monotonic() >= deadline:
                print(f"  [warn] monitor deadline reached {waited:.0f}s into the "
                      f"{post_file_wait_s:.0f}s estimated wait — going on to the export.")
                break
            if time.monotonic() >= next_note:
                next_note += 60.0
                left = post_file_wait_s - waited
                print(f"  ...waiting for the indentation sequence: {waited / 60:.1f} of "
                      f"{post_file_wait_s / 60:.1f} min elapsed ({left / 60:.1f} min left)")
            time.sleep(min(5.0, max(0.1, post_file_wait_s - waited)))
        if post_file_wait_s:
            print("  Waiting period complete — starting Process Review.")
        return time.monotonic() - started

    def _monitor_screen_read(self, recorder: LiveScreenRecorder, *, expected: int,
                             timeout_s: float, done_probe: Optional[Any],
                             tool_log: list, post_file_wait_s: float = 0.0) -> dict:
        """The screen-read completion monitor: OCR polling, live depth/H/E
        capture, and the depth-driven multi-signal completion decision.
        Returns the pieces `place_array` folds into `run_record`."""
        scfg = self.screen_config
        sample_dt = float(scfg.sample_interval_s)
        ocr_dt = float(scfg.ocr_interval_s)
        phrases = [str(p).lower() for p in (scfg.complete_phrases or []) if p]
        deadline = time.monotonic() + timeout_s

        status: dict = {}
        n_done, last_ocr, stable_complete = 0, 0.0, 0
        lat_samples: list[float] = []
        done_reason = None
        ocr_miss, ocr_ever_parsed = 0, False
        # WHICH test the instrument is on — the primary completion criterion.
        tests = TestSequenceTracker(expected)
        announced_complete = False
        last_progress = time.monotonic()
        monitor_lost_connection = False
        outage = {"since": None, "logged": False}
        # ACTIVITY, not just the counter: on a 1x1 array the counter never moves
        # after the first tick, so a stall rule watching only the counter is a
        # plain egg-timer. A depth readout that is still changing is positive
        # evidence the tool is working; one that has gone flat is evidence it
        # has stopped. `result_dir` is the definitive answer when available.
        last_activity = time.monotonic()
        last_depth = float("nan")
        peak_depth = 0.0
        last_heartbeat = time.monotonic()
        result_dir = None
        probed_result_dir = None
        file_seen_t = None            # when the result file first appeared
        wait_done_announced = False

        print(f"Screen-read safety net active for '{recorder.sample}' "
              f"({recorder.nx}x{recorder.ny} = {expected} indents; accept depth >= "
              f"{recorder.accept_depth_nm:.0f} nm) -> {recorder.csv_path.name}")
        while True:
            now = time.monotonic()
            fresh_ocr = (not status) or (now - last_ocr >= ocr_dt)
            if fresh_ocr:
                ok, payload = self.call_soft("get_current_sample_and_test_ocr_passive", {},
                                             tool_log, suppress_log=outage["logged"])
                last_ocr = time.monotonic()
                if ok:
                    if outage["since"] is not None:
                        print(f"  reconnected after {now - outage['since']:.0f}s offline — "
                              f"resuming monitoring at indent {recorder.current_index}"
                              f"/{expected} (depth-tracked).")
                        outage.update(since=None, logged=False)
                    status = payload if isinstance(payload, dict) else {}
                    # Re-parse the full-screen OCR text ourselves: the server's
                    # own regex can pick up the Sample Definition Panel's
                    # "# TESTS" TOTAL instead of the header's current index.
                    parsed = parse_current_test_index(status.get("raw_text"))
                    if parsed is None:
                        parsed = status.get("test_number")
                    if parsed is None:
                        # No "Tests: N" in this frame. NOT zero progress.
                        ocr_miss += 1
                        if ocr_miss == 3 and not tests.ever_parsed:
                            print("  NOTE: the OCR cannot find 'Tests: N' on the InView "
                                  "screen — bring the RunTest panel to the front on the "
                                  "instrument. Completion still works from the H/E ramps "
                                  "alone; only the test counter below is affected.")
                    else:
                        ocr_miss = 0
                    was_on = tests.current
                    tests.observe(parsed, physical_started=recorder.indent_in_progress)
                    ocr_ever_parsed = tests.ever_parsed
                    n_done = tests.current or 0
                    if tests.current is not None and tests.current != was_on:
                        last_progress = last_activity = time.monotonic()
                        print(f"  Current indentation: {tests.current}/{expected} — "
                              f"waiting for remaining tests"
                              if tests.current < expected else
                              f"  Current indentation: {tests.current}/{expected} — "
                              "final test running")
                else:
                    # Ride the outage out: aborting here discards results for
                    # indents already placed.
                    if outage["since"] is None:
                        outage["since"] = now
                    offline_s = now - outage["since"]
                    if not outage["logged"]:
                        print(f"  WARNING: lost contact with the instrument at indent "
                              f"{recorder.current_index}/{expected} — retrying for up to "
                              f"{scfg.net_grace_s:.0f}s. The array keeps running on the "
                              "tool; check the VPN / Wi-Fi link now.")
                        outage["logged"] = True
                    if offline_s >= scfg.net_grace_s or time.monotonic() >= deadline:
                        print(f"  stopped waiting after {offline_s:.0f}s offline — going "
                              "on to the export step so the run can still be recovered.")
                        monitor_lost_connection, done_reason = True, "connection lost"
                        break
                    time.sleep(scfg.net_retry_s)
                    continue

            depth, load, hardness, modulus, lat = self.read_screen_metrics()
            if math.isfinite(lat):
                lat_samples.append(lat)
            recorder.observe(n_done, depth, load, hardness, modulus)
            if math.isfinite(depth):
                if (not math.isfinite(last_depth)
                        or abs(depth - last_depth) >= scfg.activity_depth_nm):
                    last_activity = time.monotonic()
                last_depth = depth
                peak_depth = max(peak_depth, depth)

            if fresh_ocr:
                raw = str(status.get("raw_text") or "").lower()
                phrase_hit = next((p for p in phrases if p in raw), None)
                quiet_s = time.monotonic() - last_activity
                if done_probe is not None and probed_result_dir is None:
                    try:
                        probed_result_dir = done_probe()
                    except Exception as exc:
                        print(f"  [warn] completion probe failed ({exc}) — falling "
                              "back to the screen heuristics.")
                        done_probe = None
                    if probed_result_dir and file_seen_t is None:
                        file_seen_t = time.monotonic()
                        self._announce_post_file_wait(expected, post_file_wait_s)
                # THE ESTIMATED-TIME FLOOR. The file lands after indent #1, so
                # nothing below may declare the array finished until the other
                # (expected - 1) indents have had their time. This is the
                # backstop for every completion signal, not a signal itself.
                holding = False
                if file_seen_t is not None:
                    waited = time.monotonic() - file_seen_t
                    if waited < post_file_wait_s:
                        holding = True
                    elif not wait_done_announced:
                        wait_done_announced = True
                        print("  Waiting period complete — starting Process Review.")
                # Never cut off an indent that is visibly still loading, even if
                # the estimate says the array should be over by now.
                if holding or recorder.indent_in_progress:
                    if time.monotonic() - last_heartbeat >= 30.0:
                        last_heartbeat = time.monotonic()
                        left = max(0.0, post_file_wait_s - (time.monotonic() - file_seen_t)) \
                            if file_seen_t is not None else 0.0
                        print(f"  ...waiting for the indentation sequence: current "
                              f"indentation {tests.current or '?'}/{expected}, "
                              f"{recorder.completed_indents} finished, "
                              f"{left / 60:.1f} min of estimated wait left")
                    if time.monotonic() >= deadline:
                        print(f"WARNING: '{recorder.sample}' hit the {timeout_s:.0f}s "
                              "deadline during the estimated wait; using what was captured.")
                        done_reason = "deadline"
                        result_dir = probed_result_dir
                        break
                    if sample_dt > 0:
                        time.sleep(sample_dt)
                    continue
                # THE RESULT FILE IS NEVER PROOF THE SEQUENCE FINISHED. On the
                # 2026-08-18 runs it existed seconds into indent #1 of 4 and
                # was still being updated while the tool ran — using it alone
                # launched Process Review during test 2. It is corroboration
                # only; the INSTRUMENT'S OWN TEST STATE decides.
                #
                # Two independent progress signals, both grid-size-agnostic:
                #   A. the header test counter  -> tests.reached_end
                #   B. each indent's H/E ramp   -> recorder.grid_complete
                # Whichever signals are actually AVAILABLE must all agree the
                # sequence is over; a signal that never appeared cannot veto,
                # and a signal that is merely lagging holds the round back —
                # waiting is the safe direction, and the stall/deadline paths
                # below are the backstop.
                # The counter says WHICH test is running, never that it
                # FINISHED — on a 1x1 array it reads 1 from the moment the
                # test is queued, so "reached the end" alone would fire before
                # the tip ever touched down. Both questions must be answered:
                #   counter_ok  — the instrument is on the LAST test
                #   finished_ok — that last test has actually completed
                counter_ok = (tests.reached_end if tests.armed else True)
                if recorder.saw_any_indent:
                    finished_ok = recorder.grid_complete
                elif probed_result_dir is not None:
                    # H/E unreadable: the written result file plus a screen
                    # that has gone quiet is the only evidence left.
                    finished_ok = quiet_s >= scfg.settle_quiet_s
                else:
                    finished_ok = False     # no evidence at all — keep waiting
                sequence_done = counter_ok and finished_ok
                if sequence_done and not announced_complete:
                    announced_complete = True
                    print(f"  Indentation sequence complete: {expected}/{expected} "
                          "— Process Review can now begin")
                # SINGLE INDENT: nothing remains after it, so the estimated wait
                # is zero by construction and the file is the cleanest hand-off.
                if expected == 1 and sequence_done and probed_result_dir:
                    result_dir = probed_result_dir
                    done_reason = (f"the single indent finished and InView wrote "
                                   f"the result file to {result_dir}")
                    print(f"  array complete ({done_reason}).")
                    break
                # THE ESTIMATED WAIT HAS ELAPSED (and no ramp is running, or we
                # would have held above). Every remaining indent has been given
                # its time, so hand off — this is the path that carries the
                # round when the screen signals never confirm.
                if file_seen_t is not None:
                    result_dir = probed_result_dir
                    held = time.monotonic() - file_seen_t
                    per = post_file_wait_s / max(1, expected - 1)
                    # Say which one actually carried the round. "Confirmed by
                    # the instrument" is only true when the counter or the H/E
                    # ramps really saw the whole grid; a blind screen means the
                    # ESTIMATE is all that stands behind this result.
                    state_confirmed = ((tests.armed and tests.reached_end)
                                       or recorder.grid_complete)
                    done_reason = (
                        f"indentation sequence complete: {expected}/{expected} confirmed "
                        f"by the instrument's test state, after the {held / 60:.1f} min "
                        f"estimated wait; result file at {result_dir}" if state_confirmed
                        else
                        f"waited {held / 60:.1f} min after the result file appeared for "
                        f"{max(0, expected - 1)} remaining indent(s) at {per:.0f}s each "
                        f"(screen signals unconfirmed — the ESTIMATE is the only "
                        f"evidence); result file at {result_dir}")
                    print(f"  array complete ({done_reason}).")
                    break
                grid_confirmed = bool(sequence_done)
                # 3. Nothing has moved for a long time — a fallback for when the
                #    depth readout itself cannot be trusted (occluded screen,
                #    wrong panel focused). Only trustworthy once a finite depth
                #    has actually been read: if the screen reader is blind,
                #    "nothing changed" says nothing about the tool.
                stalled = math.isfinite(last_depth) and quiet_s >= scfg.stall_s
                is_complete = grid_confirmed or (phrase_hit is not None) or stalled
                stable_complete = stable_complete + 1 if is_complete else 0
                if stable_complete >= 2:
                    result_dir = probed_result_dir   # may be None; export will search
                    file_note = f", result file at {result_dir}" if result_dir else ""
                    done_reason = (f"screen shows {phrase_hit!r}" if phrase_hit
                                   else f"indentation sequence complete: {expected}/"
                                        f"{expected} confirmed by the instrument's test "
                                        f"state{file_note}"
                                   if grid_confirmed
                                   else f"nothing changed on screen for {quiet_s:.0f}s "
                                        f"(on test {tests.current or '?'}/{expected}, "
                                        f"{recorder.completed_indents} indent(s) finished)"
                                        " — TIMED OUT, not a completion signal")
                    print(f"  array complete ({done_reason}).")
                    break
                if time.monotonic() - last_heartbeat >= 30.0:
                    last_heartbeat = time.monotonic()
                    if probed_result_dir:
                        waiting = (f"result file found at {probed_result_dir}, "
                                   "waiting for the remaining indents to finish")
                    elif done_probe is not None:
                        waiting = "result file not written yet"
                    else:
                        waiting = "no result-file probe wired up"
                    print(f"  ...still running: current indentation "
                          f"{tests.current or '?'}/{expected}, "
                          f"{recorder.completed_indents} finished, "
                          f"{len(recorder.records)} recorded, last screen change "
                          f"{quiet_s:.0f}s ago, {waiting} "
                          f"({deadline - time.monotonic():.0f}s before timeout)")

            if time.monotonic() >= deadline:
                print(f"WARNING: '{recorder.sample}' did not report complete within "
                      f"{timeout_s:.0f}s ({recorder.completed_indents}/{expected} "
                      "indents completed); using whatever was captured.")
                done_reason = "deadline"
                break
            if sample_dt > 0:
                time.sleep(sample_dt)

        if lat_samples:
            print(f"Screen-read latency: ~{np.median(lat_samples) * 1000:.0f} ms/read "
                  f"({recorder.n_samples} reads, {len(recorder.samples)} accepted).")

        return {"result_dir": result_dir, "done_reason": done_reason, "n_done": n_done,
                "status": status, "monitor_lost_connection": monitor_lost_connection}
