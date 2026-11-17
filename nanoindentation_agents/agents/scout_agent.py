"""
scout_agent.py
==============
Autonomous **Scout Agent** for initial field-of-view selection.

Optional front-end to the Observe stage: instead of the operator manually framing a
field, the Scout drives the microscope stage (XY only — it never engages/indents),
captures and analyses each field with the classical-CV + vision-LLM stack, scores it
against the research objective, and lets a vision LLM plan the next move (direction +
how far) within a bounded move budget and safety envelope. It returns the best field
and parks the stage there for the downstream capture-position lock.

Design notes
------------
* Movement is CLOSED-LOOP but SINGLE-SHOT (one relative move per axis, no retry): the
  real iMicro stage direction mapping is suspected inverted and a retry loop diverged in
  the past, so a badly-landed field is SKIPPED, never re-driven.
* Field suitability uses ABSOLUTE, cross-field-comparable property norms (not the
  per-image normalisation in ``rank_targets_by_objective``, which makes every field's best
  target ≈ 1.0) so different fields get genuinely different scores.
* The true field of view is calibrated from the burned-in 50 µm red scale bar BEFORE any
  move, so step sizes and the safety envelope are in real micrometres, not the nominal FOV.

Typical use (notebook)::

    from nanoindentation_agents.agents.scout_agent import ScoutAgent, ScoutConfig
    scout = ScoutAgent(objective, ScoutConfig(fov_w_um=FOV_W, fov_h_um=FOV_H, ...),
                       cfg=cfg, toolkit=toolkit, download_dir=DOWNLOAD_DIR,
                       briefings=briefings, msg_log=msg_log)
    result = scout.run(state)          # moves stage, shows each field, updates state
    if result.winner is not None:
        FOV_W, FOV_H = result.fov_w_um, result.fov_h_um
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..config import get_config
from ..tools.indent_tools import get_toolkit
from .. import notebook_ui as ui
from ..analysis.spatial_targeting import sync_targets_from_state
from ..analysis.exploration_plan import refresh_exploration_plan
from ..analysis.objective_acquisition import (
    rank_targets_by_objective,
    scalarize_properties,
    _estimate_target_properties,
)
from ..registry.models import get_llm_for_role
from ..registry.tool_catalog import parse_llm_json


# Absolute, cross-field-comparable property norms (vmin, range) spanning the phase
# estimates in objective_acquisition._PHASE_ESTIMATES. Fixed norms (rather than per-image
# min/max) are what let the Scout tell fields apart.
_SCOUT_NORMS = {"hardness": (1.0, 11.0), "modulus": (30.0, 185.0)}


@dataclass
class ScoutConfig:
    """Operator-tunable search parameters (all also settable via NB5_* env in the notebook)."""
    max_locations: int = 5            # hard cap on the number of stage moves / fields visited
    overlap_frac: float = 0.9         # one field-width step = FOV * this
    max_jump_fields: float = 3.0      # most field-widths the agent may jump in ONE move
    satisfy_score: float = 1.01       # early-stop threshold on suitability [0..1]; >1 disables it
    move_tol_um: float = 1.0          # landing tolerance for closed-loop moves
    settle_s: int = 2                 # settle time between moves (use 0 in the simulator)
    radius_abs_um: Optional[float] = None   # hard cap on wander distance; else derived from FOV+budget
    fov_w_um: float = 100.0           # nominal FOV; overwritten by scale-bar calibration on real hw
    fov_h_um: float = 75.0
    use_simulator: bool = False
    use_vision_llm: bool = True


@dataclass
class ScoutResult:
    """Outcome of a search. ``winner`` is None when no valid field was found."""
    winner: Optional[dict]
    candidates: list
    fov_w_um: float
    fov_h_um: float


class ScoutAgent:
    """Autonomous, LLM-guided field-of-view scout. See module docstring."""

    def __init__(
        self,
        objective,
        scout_cfg: ScoutConfig,
        *,
        cfg=None,
        toolkit=None,
        download_dir: Any = ".",
        briefings: Optional[dict] = None,
        msg_log=None,
        verbose: bool = True,
    ):
        self.objective = objective
        self.sc = scout_cfg
        self.cfg = cfg or get_config()
        self.toolkit = toolkit or get_toolkit()
        self.download_dir = download_dir
        self.briefings = briefings or {}
        self.msg_log = msg_log
        self.verbose = verbose
        self.fov_w = scout_cfg.fov_w_um
        self.fov_h = scout_cfg.fov_h_um
        self._llm = None

    # ------------------------------------------------------------------ utilities
    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _call(self, tool: str, args: dict):
        """Single instrument tool call that raises on failure."""
        out = self.toolkit.call_tool(tool, args)
        if not out.ok:
            raise RuntimeError(f"{tool} failed: {out.error}")
        return out.result

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm_for_role("image_analysis", self.cfg)
        return self._llm

    # ------------------------------------------------------------------ movement
    def _goto(self, dest_x, dest_y, label: str = ""):
        """SINGLE-SHOT XY move (one relative move per axis, NO retry). Prints the exact input
        of every move command and reports the landing error so the caller can SKIP a badly
        landed field rather than re-driving the stage. XY only -- never engages the indenter."""
        tol, settle = self.sc.move_tol_um, self.sc.settle_s
        tag = f" {label}" if label else ""
        p = self._call("get_xyz_positions", {})
        cx, cy = float(p["x"]), float(p["y"])
        dx, dy = dest_x - cx, dest_y - cy
        if abs(dx) < tol and abs(dy) < tol:
            self._log(f"   [move{tag}] already within {tol} um of target "
                      f"({dest_x:.1f}, {dest_y:.1f}) -- no move needed.")
            return cx, cy, 0.0
        self._log(f"   [move{tag}] at ({cx:.1f}, {cy:.1f}) -> target ({dest_x:.1f}, {dest_y:.1f})  "
                  f"| residual dx={dx:+.2f} dy={dy:+.2f} um")
        if abs(dx) >= 0.05:
            args = {"amount": round(abs(dx), 2), "direction": "right" if dx > 0 else "left", "t": settle}
            self._log(f"        move input -> {args}")
            self._call("move", args)
        if abs(dy) >= 0.05:
            args = {"amount": round(abs(dy), 2), "direction": "up" if dy > 0 else "down", "t": settle}
            self._log(f"        move input -> {args}")
            self._call("move", args)
        p = self._call("get_xyz_positions", {})
        cx, cy = float(p["x"]), float(p["y"])
        err = ((cx - dest_x) ** 2 + (cy - dest_y) ** 2) ** 0.5
        self._log(f"   [move{tag}] landed at ({cx:.1f}, {cy:.1f})  | error {err:.2f} um")
        return cx, cy, err

    # ------------------------------------------------------------------ scoring
    def _objective_score(self, ranked) -> float:
        """ABSOLUTE objective fit [0..1] of a field: score the best-ranked target's estimated
        H/E against FIXED reference norms, so scores are comparable across fields."""
        if not ranked:
            return 0.0
        h, e, _note = _estimate_target_properties(ranked[0])
        props = {"hardness": np.array([h], dtype=float), "modulus": np.array([e], dtype=float)}
        return float(scalarize_properties(self.objective, props, norms=_SCOUT_NORMS)[0])

    def _suitability(self, best_obj_score, ranked, cal_ok):
        """Pre-indent suitability proxy in [0,1]: objective fit + grid room - calibration penalty."""
        n_needed = max(1, self.objective.constraints.grid_nx * self.objective.constraints.grid_ny)
        room = min(len(ranked), n_needed) / n_needed
        cal_pen = 0.0 if cal_ok else 0.25
        score = max(0.0, min(1.0, 0.75 * float(best_obj_score) + 0.25 * room - cal_pen))
        return round(score, 3), {
            "best_objective_score": round(float(best_obj_score), 3),
            "grid_room": round(room, 3),
            "calibrated": bool(cal_ok),
        }

    def _capture_and_score(self, tag: str, state: dict):
        """Capture the current field, run CV + Vision-LLM + objective ranking, return a scored candidate."""
        cap = self._call("capture_video_panel_remote", {})
        img = ui.download_capture(self.toolkit, cap if isinstance(cap, dict) else {}, self.download_dir)
        if not img.get("local_path"):
            return None
        img["width_um"], img["height_um"] = self.fov_w, self.fov_h
        img_path = ui.resolve_image_path(img)

        cal_ok = True
        if not self.sc.use_simulator:
            try:
                umpp, _c = ui.calibrate_um_per_px(img_path)   # raises if the scale bar can't be read
                img["um_per_px"] = umpp
            except Exception as exc:
                cal_ok = False
                self._log(f"   [scout:{tag}] scale bar unreadable ({type(exc).__name__}) -- field penalized.")

        cv_result = ui.classical_cv_analysis(img_path, self.fov_w, self.fov_h)
        vision_result = None
        if self.sc.use_vision_llm:
            from ..image_analysis.vision_agent import analyze_image_with_vision
            directive = self.briefings.get("image_analysis", "")
            vision_result = ui.run_with_timeout(
                lambda: analyze_image_with_vision(
                    str(img_path),
                    research_question=state.get("research_question", ""),
                    experiment_context=state.get("experiment_context", "")
                        + "\n\n[ORCHESTRATOR DIRECTIVE]\n" + directive
                        + "\n\n[SCOUT TASK] Judge how suitable THIS field of view is for indentation "
                          "toward the objective.",
                    cfg=self.cfg),
                seconds=120, label=f"Vision LLM (scout {tag})")
        analysis, _prov = ui.combine_image_analysis(cv_result, vision_result)

        # Score against the objective using a throwaway copy of state (does not mutate the live run).
        scratch = dict(state)
        scratch["current_image"] = img
        scratch["image_analysis"] = analysis
        scratch.update(sync_targets_from_state(scratch))
        scratch.update(refresh_exploration_plan(scratch))
        ranked = rank_targets_by_objective(scratch.get("measurement_plan", []), self.objective, analysis)
        best = self._objective_score(ranked)
        score, breakdown = self._suitability(best, ranked, cal_ok)
        summary = (analysis.get("interpretation", "") or "")[:120]
        return {"tag": tag, "image": img, "analysis": analysis, "ranked": ranked,
                "score": score, "breakdown": breakdown, "cal_ok": cal_ok, "summary": summary}

    # ------------------------------------------------------------------ LLM decision
    def _decide_next_move(self, cand, visited, remaining, disp_fields, env_fields):
        """The vision LLM looks at the CURRENT field and PLANS its next stage move: a DIRECTION and
        HOW FAR (1..max_jump_fields field-widths) -- so it can chain its remaining moves to reach a
        farther region -- or STOP. Returns {action, direction, distance_fields, reason, assessment,
        source}. Deterministic survey-pattern fallback when the LLM is off/unavailable/unparseable."""
        fallback = {"action": "move", "direction": None, "distance_fields": 1.0, "source": "survey-pattern",
                    "assessment": cand.get("summary", "") or "(no vision summary)",
                    "reason": "LLM disabled -- following the fixed survey pattern."}
        if not self.sc.use_vision_llm:
            return fallback
        img_path = ui.resolve_image_path(cand["image"])
        goals = "; ".join(g.describe() for g in self.objective.property_goals)
        hist = ", ".join(f"{c['tag']}(fit {c['breakdown']['best_objective_score']:.2f})" for c in visited) or "none yet"
        prompt = (
            "You are the SCOUT AGENT steering a nanoindenter's microscope stage to find the BEST field of "
            "view to indent for this research objective:\n"
            f"  {self.objective.summary_text()[:280]}\n"
            f"  Property goals: {goals}\n\n"
            "MOVEMENT: each move you jump the stage in ONE direction (up/down/left/right) by a distance you "
            f"choose, from 1 to {self.sc.max_jump_fields:g} field-widths. A larger jump covers more ground but "
            "skips what lies between; a 1-field jump inspects the immediate neighbour. PLAN across your remaining "
            "moves -- it is perfectly fine to spend two or more moves travelling the same direction to reach a "
            "promising far region instead of stopping short.\n"
            f"BUDGET: {remaining} move(s) left. You are currently ~{disp_fields:.1f} field(s) from the start and "
            f"must stay within ~{env_fields:.1f} field(s) of it (a safety envelope).\n"
            f"Fields visited so far (higher fit = better for the objective): {hist}.\n\n"
            "Study THIS image, then decide. Reply with ONLY a JSON object, no prose:\n"
            '{"assessment": "<phases/features you see and how well this field fits the objective>", '
            '"action": "move" or "stop", '
            '"direction": "up" | "down" | "left" | "right" (empty string if action is stop), '
            '"distance_fields": <number of field-widths to jump this move, e.g. 1, 1.5, 2>, '
            '"reason": "<why this direction AND distance, what you expect to find there, or why you stop here>"}'
        )
        try:
            raw = ui.run_with_timeout(
                lambda: self._get_llm().chat_with_images(
                    prompt, image_paths=[str(img_path)],
                    system_message="You are an expert microscopist guiding indentation site selection. "
                                   "Plan efficiently within the move budget; output only the JSON object."),
                seconds=120, label="Scout move decision (LLM)")
            d = parse_llm_json(raw)
            action = str(d.get("action", "move")).lower().strip()
            direction = str(d.get("direction", "")).lower().strip()
            if action not in ("move", "stop"):
                action = "move"
            if action == "move" and direction not in ("up", "down", "left", "right"):
                direction = None
            try:
                dist = float(d.get("distance_fields", 1.0))
            except (TypeError, ValueError):
                dist = 1.0
            dist = max(0.5, min(float(self.sc.max_jump_fields), dist))
            return {"action": action, "direction": direction or None, "distance_fields": dist,
                    "reason": str(d.get("reason", "")).strip() or "(no reason given)",
                    "assessment": str(d.get("assessment", "")).strip() or (cand.get("summary", "") or ""),
                    "source": "llm"}
        except Exception as exc:
            fb = dict(fallback)
            fb["reason"] = f"LLM decision unavailable ({type(exc).__name__}) -- using survey pattern."
            return fb

    # ------------------------------------------------------------------ main loop
    def run(self, state: dict) -> ScoutResult:
        """Explore up to ``max_locations`` fields, pick the best, park the stage there, and update
        ``state['current_image']`` / ``['current_position']`` / ``['image_analysis']``."""
        ui.agent_banner("Scout Agent",
                        f"autonomously exploring up to {self.sc.max_locations} fields to pick the best "
                        f"indentation site")
        p0 = self._call("get_xyz_positions", {})
        start_xy = (float(p0["x"]), float(p0["y"]))

        # --- Calibrate the TRUE field of view from the 50 um red scale bar BEFORE any move ---
        if not self.sc.use_simulator:
            home_img = ui.resolve_image_path(state.get("current_image") or {})
            try:
                umpp0, cal0 = ui.calibrate_um_per_px(home_img) if home_img else (None, None)
                if cal0:
                    self.fov_w, self.fov_h = cal0["fov_w_um"], cal0["fov_h_um"]
                    self._log(f"[scout] scale-bar calibration: bar {cal0['bar_len_px']}px = {cal0['bar_um']} um "
                              f"({cal0['bar_um_source']}) -> {umpp0:.4f} um/px | TRUE FOV "
                              f"{self.fov_w:.1f} x {self.fov_h:.1f} um. Scout steps use this.")
            except Exception as exc:
                self._log(f"[scout] WARNING: scale bar unreadable on the home frame ({type(exc).__name__}: {exc}) "
                          f"-- stepping with nominal FOV {self.fov_w:.1f} x {self.fov_h:.1f} um.")

        step_x, step_y = self.fov_w * self.sc.overlap_frac, self.fov_h * self.sc.overlap_frac
        field_um = max(step_x, step_y)
        radius = float(self.sc.radius_abs_um) if self.sc.radius_abs_um \
            else round((self.sc.max_locations + 0.5) * field_um, 1)
        DIR = {"right": (step_x, 0.0), "left": (-step_x, 0.0), "up": (0.0, step_y), "down": (0.0, -step_y)}
        survey = ["right", "up", "left", "down", "right"]
        self._log(f"[scout] start ({start_xy[0]:.1f}, {start_xy[1]:.1f}) um | one field ~= "
                  f"({step_x:.1f}, {step_y:.1f}) um | envelope radius {radius:.0f} um")

        candidates, cur_xy, winner = [], start_xy, None
        try:
            for step_i in range(self.sc.max_locations + 1):        # field 0 (home) + up to max_locations moves
                tag = "home" if step_i == 0 else f"loc{step_i}"

                c = self._capture_and_score(tag, state)
                if not c:
                    self._log(f"[scout] {tag}: capture failed -- ending exploration.")
                    break
                c["pos"] = cur_xy
                candidates.append(c)
                self._log(f"\n================  Scout field '{tag}'  @ ({cur_xy[0]:.1f}, {cur_xy[1]:.1f}) um  "
                          f"|  suitability {c['score']:.3f}  ================")
                ui.show_capture(
                    c["image"],
                    title=f"Scout field '{tag}' @ ({cur_xy[0]:.1f}, {cur_xy[1]:.1f}) um -- suitability {c['score']:.3f}",
                    note=f"objective fit {c['breakdown']['best_objective_score']:.3f}, "
                         f"grid room {c['breakdown']['grid_room']:.2f}, calibrated {c['breakdown']['calibrated']}",
                    fov_width_um=self.fov_w, fov_height_um=self.fov_h)

                remaining = self.sc.max_locations - step_i
                disp_um = ((cur_xy[0] - start_xy[0]) ** 2 + (cur_xy[1] - start_xy[1]) ** 2) ** 0.5
                decision = self._decide_next_move(c, candidates, remaining,
                                                  disp_um / field_um, radius / field_um)
                self._log(f"[Scout Agent thinks] ({decision['source']}) {decision['assessment']}")

                if c["score"] >= self.sc.satisfy_score:
                    self._log(f"[Scout Agent] DECISION: STOP -- suitability {c['score']:.3f} >= "
                              f"{self.sc.satisfy_score} already satisfies the objective.")
                    break
                if remaining <= 0:
                    self._log(f"[Scout Agent] move budget spent ({self.sc.max_locations} moves) -- ending exploration.")
                    break
                if decision["action"] == "stop":
                    self._log(f"[Scout Agent] DECISION: STOP here.  Why: {decision['reason']}")
                    break

                direction = decision["direction"] or survey[min(step_i, len(survey) - 1)]
                dist = decision["distance_fields"]
                ox, oy = DIR[direction]
                tx, ty = cur_xy[0] + ox * dist, cur_xy[1] + oy * dist
                # Smart envelope handling: CLAMP the jump to the safety boundary instead of giving up.
                d = ((tx - start_xy[0]) ** 2 + (ty - start_xy[1]) ** 2) ** 0.5
                clamped = ""
                if d > radius and d > 0:
                    s = radius / d
                    tx = start_xy[0] + (tx - start_xy[0]) * s
                    ty = start_xy[1] + (ty - start_xy[1]) * s
                    clamped = f"  (clamped to the {radius:.0f} um envelope)"

                self._log(f"[Scout Agent] DECISION: move {direction.upper()} {dist:.1f} field(s) -> "
                          f"(~{tx:.1f}, {ty:.1f}) um{clamped}.  Why: {decision['reason']}")
                lx, ly, err = self._goto(tx, ty, label=direction)
                if err > max(3 * self.sc.move_tol_um, 2.0):
                    self._log(f"[scout] landed {err:.1f} um off the intended target (stage direction may be "
                              f"inverted) -- scoring wherever we actually landed and continuing.")
                cur_xy = (lx, ly)

            winner = max(candidates, key=lambda c: c["score"]) if candidates else None

            if candidates:
                import pandas as pd
                from IPython.display import display
                self._log(f"\n[scout] visited {len(candidates)} field(s); comparison:")
                display(pd.DataFrame([{
                    "field": c["tag"], "x (um)": round(c["pos"][0], 1), "y (um)": round(c["pos"][1], 1),
                    "suitability": c["score"], "obj. fit": c["breakdown"]["best_objective_score"],
                    "grid room": c["breakdown"]["grid_room"], "calib": c["breakdown"]["calibrated"],
                    "winner": "*" if winner is c else "",
                    "vision summary": c["summary"],
                } for c in candidates]))

            if winner is None:
                self._log("[scout] no valid field found -- returning to start; use the manual frame.")
                self._goto(*start_xy, label="return")
            else:
                wx, wy = winner["pos"]
                self._log(f"\n[scout] --> returning to winning field '{winner['tag']}' at ({wx:.1f}, {wy:.1f}) um")
                self._goto(wx, wy, label="winner")
                cap = self._call("capture_video_panel_remote", {})
                wimg = ui.download_capture(self.toolkit, cap if isinstance(cap, dict) else {}, self.download_dir)
                if wimg.get("local_path"):
                    wimg["width_um"], wimg["height_um"] = self.fov_w, self.fov_h
                    if winner["image"].get("um_per_px") is not None:
                        wimg["um_per_px"] = winner["image"]["um_per_px"]
                else:
                    wimg = winner["image"]
                state["current_image"] = wimg
                state["current_position"] = (wx, wy)
                state["image_analysis"] = winner["analysis"]
                self._log(f"[scout] SELECTED field '{winner['tag']}' at ({wx:.1f}, {wy:.1f}) um "
                          f"-- suitability {winner['score']:.3f}. Stage parked here for the capture_pos lock.")
                if self.msg_log is not None:
                    self.msg_log.log(
                        "Scout Agent", "Orchestrator Agent",
                        f"Autonomous LLM-guided search visited {len(candidates)} field(s); selected "
                        f"'{winner['tag']}' at ({wx:.1f}, {wy:.1f}) um (suitability {winner['score']:.3f}).",
                        detail=winner["summary"])
        except Exception as exc:
            self._log(f"[scout] ABORTED ({type(exc).__name__}: {exc}) -- returning to start, falling back to manual.")
            try:
                self._goto(*start_xy, label="abort-return")
            except Exception:
                pass
            winner = max(candidates, key=lambda c: c["score"]) if candidates else None

        return ScoutResult(winner=winner, candidates=candidates, fov_w_um=self.fov_w, fov_h_um=self.fov_h)


def scout_search(state: dict, objective, scout_cfg: ScoutConfig, **kwargs) -> ScoutResult:
    """Convenience wrapper: build a :class:`ScoutAgent` and run one search. See ``ScoutAgent``."""
    return ScoutAgent(objective, scout_cfg, **kwargs).run(state)
