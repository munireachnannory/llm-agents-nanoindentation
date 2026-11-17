"""
CampaignRunner — ONE round implementation for the whole campaign.

Round 1, operator-gated adaptive rounds, and the autonomous loop all execute
the same two calls:

    proposal = runner.propose_round()          # decide -> frame -> analyse ->
                                               # propose sites -> open the gate
    runner.execute_round(proposal, ...)        # position -> indent -> export ->
                                               # validate -> record -> critique

Notebook 5 implemented this three times (Section 12, cells A–D, Section 14)
and nearly every bug of 7/29–8/03 was drift between the copies. Here the
autonomous loop is literally::

    while True:
        p = runner.propose_round()
        if p.terminal or not p.plans: break
        policy.approve(p.gate)
        runner.execute_round(p, policy=policy)

Every transition is appended to the CampaignJournal, so a kernel restart
loses nothing that is needed to recover data already on the instrument.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import notebook_ui as ui
from ..analysis.objective_acquisition import ObjectiveSurrogate, rank_targets_by_objective
from ..analysis.spatial_targeting import sync_targets_from_state, target_visit_key
from ..analysis.exploration_plan import refresh_exploration_plan
from ..critics.measurement_critic import measurement_critic_node
from ..instrument import results as R
from ..instrument.motion import StageSafetyError
from ..instrument.session import InstrumentCommError, InstrumentMethodError
from ..method import MethodConstraints
from ..agents.measurement_strategy_agent import (
    GridConstraints,
    MeasurementStrategyAgent,
)
from .approval import apply_decision


@dataclass
class CampaignConfig:
    """Operator-facing campaign knobs (the section-1 switches, typed)."""
    operator: str = "operator"
    use_simulator: bool = True
    run_id: str = ""
    sample_prefix: str = "test"       # zero-padded names: test_01, test_02, ...
                                      # (the ReviewData window is picked by
                                      # SUBSTRING match — unpadded collides at 10)
    fov_w_um: float = 100.0
    fov_h_um: float = 75.0
    um_per_px: Optional[float] = None
    # IMAGE -> STAGE conversion: stage_delta = site_gain * image_offset_from_centre,
    # with image x RIGHT-positive and image y DOWN-positive. The sign and the
    # scale both live in this one number, so it is directly measurable.
    #
    # SIGN is measured and cross-checked three ways (2026-08-04):
    #   1. probe moves registered against captured frames (sign proved by
    #      imagecal.self_test against a known shift, not by reasoning);
    #   2. the operator watching the stage — "when it should move left it
    #      moves right" — which is what caught the error;
    #   3. NB5 `v2-backup-2` Cell C1/2, which picks the arrow straight from the
    #      image offset (target left -> click "left") and agrees.
    # All three say NEGATIVE: a target LEFT of centre needs "left", and one
    # BELOW centre needs "down". The old hard-coded (+1.0, -1.0) had X
    # INVERTED, which drove targets out of the field entirely.
    #
    # MAGNITUDE is deliberately 1.0, NOT the 1.03/1.38 the probe suggested.
    # Re-registering the Y probe across four crops gave 166/242/247/280 px for
    # the same move, with neighbouring peaks scoring within 2% of the best --
    # that measurement is not determined, so a 1.38 "correction" would be
    # inventing precision. 1.0 is what NB5 used, it matches the frame geometry,
    # and it is the honest default until a sharper measurement exists.
    # Re-measure per objective/zoom with runner.calibrate_image_to_stage().
    site_gain_x: float = -1.0
    site_gain_y: float = -1.0
    # AIM AT THE INDENTER AXIS, NOT THE MIDDLE OF THE PICTURE. InView burns a red
    # crosshair into the video panel at the indenter axis, and on this instrument
    # (2026-08-05) it sits 2.9 um ABOVE the geometric centre of the captured
    # frame — a systematic bias on every site, identical between frames. Located
    # per frame by imagecal.find_crosshair(); falls back to the centre.
    aim_at_crosshair: bool = True
    # ---- PRE-INDENT VISUAL VERIFICATION ----------------------------------
    # jog_to() proves the stage reached the requested STAGE COORDINATE; it
    # cannot prove that coordinate was the right one. Any error in the
    # image->stage map (a wrong gain, a wrong SIGN) produces a jog that
    # converges to 0.07 um and still lands ~2x the offset away from the feature
    # the agent chose. This check closes that loop in IMAGE space: register the
    # confirmation frame against the frame the target was picked from, and
    # confirm the feature is now under the crosshair before anything indents.
    #   "enforce" — correct, then refuse to indent if still out of tolerance
    #   "warn"    — measure and report, indent anyway
    #   "off"     — skip (always skipped on the simulator)
    # A check that CANNOT RUN (registration unavailable, textureless field)
    # warns and proceeds in every mode — absence of evidence is not evidence of
    # a fault, and a guard that cannot be satisfied is a guard that gets
    # switched off.
    verify_arrival: str = "enforce"
    arrival_tol_um: float = 8.0        # registration is +-1 px (~0.2 um) but the
                                       # gains are only known to ~5%
    max_visual_corrections: int = 1
    show_frames: bool = True
    capture_after_indent: bool = True
    recenter_each_round: bool = False  # long jogs risk losing this stage
    site_keepout_um: Optional[float] = None    # None -> objective min_site_spacing
    move_envelope_um: Optional[float] = None   # None -> one FOV diagonal
    test_timeout_s: float = 1800.0
    # HOW LONG ONE INDENT TAKES on this instrument, in seconds. Measured at
    # ~2 min 30 s per indent (2026-08-18). InView writes the result file after
    # the FIRST indent of an array, so once that file appears there are still
    # (nx*ny - 1) indents to go: the monitor holds for
    # (nx*ny - 1) * indent_duration_s before Process Review may run.
    # Deliberately a tunable estimate, not a measurement — raise it if a round
    # ever exports mid-array, lower it to stop wasting instrument time.
    indent_duration_s: float = 150.0
    # SKIP_SCREENING: bypass the live screen-read monitor (OCR/depth polling,
    # per-indent H/E capture) entirely and trust the instrument's result file
    # the moment it appears, with Process Review as the sole data source —
    # no screen-read fallback table. Operator-accepted tradeoff: this file-
    # exists signal has, on real hardware, fired before a multi-indent grid
    # actually finished (see InstrumentSession.place_array's screen-read
    # path); with screening off there is nothing left to catch that.
    # Applies uniformly to manual rounds and the Section 8 autonomous loop —
    # both call CampaignRunner._execute_plan(), the one place this is read.
    skip_screening: bool = False
    use_vision_llm: bool = False
    use_measurement_strategy_llm: bool = False
    require_motion_calibration: bool = True
    calibration_max_age_h: float = 12.0
    annotated_dir: Optional[str] = None


@dataclass
class RoundProposal:
    round_no: int
    decision: Any                       # OrchestratorDecision
    progress: Any                       # ObjectiveProgress
    surrogate: Any                      # ObjectiveSurrogate
    frame_pos: tuple
    proposals: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    gate: Any = None

    @property
    def terminal(self) -> bool:
        return self.decision.action != "continue"


class CampaignRunner:
    """Composes session + motion + export + journal + orchestrator into
    gated, journaled, recoverable rounds."""

    def __init__(self, *, session, motion, exporter, journal, objective,
                 orchestrator, state, app_cfg, campaign: CampaignConfig,
                 msg_log=None):
        self.session = session
        self.motion = motion
        self.exporter = exporter
        self.journal = journal
        self.objective = objective
        self.orch = orchestrator
        self.state = state
        self.app_cfg = app_cfg
        self.c = campaign
        self.msg_log = msg_log
        # Bounds, defaults, and authority modes for method-parameter control
        # (target depth/load, approach, strain rate, ...). Each parameter keeps
        # its configured fixed value unless explicitly delegated to the agent.
        self.method_constraints = MethodConstraints.from_settings(
            app_cfg.indent_defaults.method_control, app_cfg.indent_defaults.review_changes)
        self.grid_constraints = GridConstraints.from_settings(
            app_cfg.measurement_strategy.grid_control,
            runtime_defaults={
                "grid_nx": objective.constraints.grid_nx,
                "grid_ny": objective.constraints.grid_ny,
                "spacing_um": objective.constraints.spacing_um,
                "rotation_deg": app_cfg.indent_defaults.rotation,
            },
        )
        self.measurement_strategy = MeasurementStrategyAgent(
            objective,
            app_cfg,
            self.method_constraints,
            self.grid_constraints,
            default_method_type=app_cfg.indent_defaults.method_type,
            use_llm=campaign.use_measurement_strategy_llm,
        )
        self.executions: list[dict] = []
        self.capture_pos: Optional[tuple] = None
        self.crosshair_um: Optional[tuple] = None   # indenter axis in image um
        self._round_frame: Optional[tuple] = None   # (snapshot path, frame_pos)
        self._preflighted = False
        self._sample_seq = journal.next_sample_index(campaign.sample_prefix) - 1
        if self._sample_seq:
            print(f"[journal] resuming sample numbering at "
                  f"{campaign.sample_prefix}_{self._sample_seq + 1:02d} "
                  "(earlier names found in this run's journal)")

    # ------------------------------------------------------------ small utils
    def _log(self, sender: str, recipient: str, subject: str, detail: str = ""):
        if self.msg_log is not None:
            self.msg_log.log(sender, recipient, subject, detail=detail)

    def next_sample_name(self) -> str:
        self._sample_seq += 1
        return f"{self.c.sample_prefix}_{self._sample_seq:02d}"

    @property
    def _keepout_um(self) -> float:
        if self.c.site_keepout_um is not None:
            return float(self.c.site_keepout_um)
        return float(self.objective.constraints.min_site_spacing_um)

    @property
    def _envelope_um(self) -> float:
        if self.c.move_envelope_um is not None:
            return float(self.c.move_envelope_um)
        return math.hypot(self.c.fov_w_um, self.c.fov_h_um)

    # ------------------------------------------------------------- setup
    def lock_search_area(self) -> tuple:
        """Fix the campaign's coordinate frame from the CURRENT frame.

        Real hardware: calibrate um/px from the burned-in scale bar — never
        move an indenter on an unknown scale (raises if the bar is unreadable).
        Simulator: keep the nominal FOV (its synthetic frame has no bar).
        """
        img = self.state.get("current_image") or {}
        if not self.c.use_simulator:
            cal_img = ui.resolve_image_path(img)
            assert cal_img, "No captured frame to calibrate against — run Observe first."
            um_per_px, cal = ui.calibrate_um_per_px(cal_img)   # raises if unreadable
            self.c.fov_w_um, self.c.fov_h_um = cal["fov_w_um"], cal["fov_h_um"]
            self.c.um_per_px = um_per_px
            print(f"Scale calibration: bar {cal['bar_len_px']}px = {cal['bar_um']} um "
                  f"-> {um_per_px:.4f} um/px | FOV {self.c.fov_w_um:.1f} x "
                  f"{self.c.fov_h_um:.1f} um")
        else:
            print(f"Simulator — nominal FOV {self.c.fov_w_um:.1f} x "
                  f"{self.c.fov_h_um:.1f} um (no scale bar).")
        if img:
            img["width_um"], img["height_um"] = self.c.fov_w_um, self.c.fov_h_um
            if self.c.um_per_px is not None:
                img["um_per_px"] = self.c.um_per_px
            self._refresh_crosshair(img)

        pos = tuple(self.state.get("current_position", (0.0, 0.0)))
        self.capture_pos = pos
        self.objective.constraints.fov_bounds = (
            pos[0] - self.c.fov_w_um / 2, pos[0] + self.c.fov_w_um / 2,
            pos[1] - self.c.fov_h_um / 2, pos[1] + self.c.fov_h_um / 2)
        self.state["research_objective"] = self.objective.to_dict()
        b = self.objective.constraints.fov_bounds
        self.journal.append("SEARCH_AREA", capture_pos=pos, fov_bounds=b,
                            fov_w_um=self.c.fov_w_um, fov_h_um=self.c.fov_h_um,
                            um_per_px=self.c.um_per_px)
        print(f"BO search bounds (stage um): x {b[0]:.0f}..{b[1]:.0f}, "
              f"y {b[2]:.0f}..{b[3]:.0f}")
        return pos

    def preflight(self) -> None:
        """Everything that must be true BEFORE the first indent. Raises on
        failure — a campaign never starts on hope."""
        assert self.session.ping(), "Instrument server unreachable."
        assert self.capture_pos is not None, \
            "Search area not locked — run lock_search_area() after Observe."
        if not self.c.use_simulator:
            confirmed = self.exporter.verify_session_dir()
            if confirmed is None and self.exporter.open_dir is None:
                raise RuntimeError(
                    "The InView sitting folder could not be confirmed OR derived — "
                    "fix ExportConfig.test_number before indenting.")
            if self.c.require_motion_calibration:
                self.motion.require_calibration(self.c.calibration_max_age_h)
        self.journal.append(
            "PREFLIGHT",
            simulator=self.c.use_simulator,
            session_dir=self.exporter.open_dir,
            motion=self.motion.summary(),
            site_gain_x=self.c.site_gain_x, site_gain_y=self.c.site_gain_y,
            objective=self.objective.summary_text()[:400],
            indent_budget=self.objective.constraints.indent_budget,
            max_rounds=self.objective.constraints.max_rounds,
            # the gate every round's arrival was judged against — a reported
            # error means nothing later without the tolerance it was measured to
            verify_arrival=self.c.verify_arrival,
            arrival_tol_um=self.c.arrival_tol_um)
        self._preflighted = True
        print("PREFLIGHT OK —", self.motion.summary())

    # ------------------------------------------------------- frame + analysis
    def analyze_current_frame(self, round_no: int = 0, *, tag: str = "initial",
                              refresh_plan: bool = False) -> list[dict]:
        """The image workflow on state['current_image']: CV (+ optional vision
        LLM) -> combine -> targets -> objective ranking -> annotated PNG."""
        img = self.state.get("current_image") or {}
        img_path = ui.resolve_image_path(img)
        if not img_path:
            print(f"  [round {round_no}] no frame to analyze — keeping previous plan.")
            return list(self.state.get("measurement_plan") or [])

        cv_result = ui.classical_cv_analysis(img_path, self.c.fov_w_um, self.c.fov_h_um)
        vision_result = None
        if self.c.use_vision_llm:
            try:
                from ..image_analysis.vision_agent import analyze_image_with_vision
                directive = (self.state.get("orchestrator_briefings") or {}).get(
                    "image_analysis", "")
                vision_result = ui.run_with_timeout(
                    lambda: analyze_image_with_vision(
                        str(img_path),
                        research_question=self.state["research_question"],
                        experiment_context=self.state["experiment_context"]
                        + "\n\n[ORCHESTRATOR DIRECTIVE]\n" + directive,
                        cfg=self.app_cfg),
                    seconds=120, label=f"Vision LLM (round {round_no})")
            except Exception as e:
                print(f"  [round {round_no}] vision LLM skipped: {e}")

        analysis, provenance = ui.combine_image_analysis(cv_result, vision_result)
        self.state["image_analysis"] = analysis
        self.state.update(sync_targets_from_state(self.state))
        if refresh_plan:
            self.state.update(refresh_exploration_plan(self.state))
        ranked = rank_targets_by_objective(self.state["measurement_plan"],
                                           self.objective, analysis)
        self.state["measurement_plan"] = ranked
        if ranked and self.c.annotated_dir:
            png = Path(self.c.annotated_dir) / f"nb7_round{round_no:02d}_targets.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            ui.annotate_targets(img, analysis, ranked[:4],
                                f"Round {round_no} — objective-ranked targets ({tag})",
                                fov_width_um=self.c.fov_w_um,
                                fov_height_um=self.c.fov_h_um, save_path=png)
        self._log("Image Analysis Agent", "Orchestrator Agent",
                  f"Round {round_no} ({tag}): {len(ranked)} targets ranked "
                  f"[{provenance}]", detail=str(analysis.get("interpretation", ""))[:250])
        return ranked

    def _fresh_frame(self, round_no: int) -> tuple:
        """Capture where the stage is (real hw) and re-analyse; the simulator
        keeps its fixed frame. Returns frame_pos."""
        if self.c.use_simulator:
            return tuple(self.capture_pos)
        if self.c.recenter_each_round:
            _, err, _, _ = self.motion.jog_to(self.capture_pos, label="FOV reference")
            if err > self.motion.cfg.abort_um:
                raise StageSafetyError(
                    f"Cannot return to the FOV reference ({err:.0f} um off) — "
                    "the campaign's coordinate frame is no longer valid.")
        frame_pos, img = self.session.capture_frame(
            fov_w_um=self.c.fov_w_um, fov_h_um=self.c.fov_h_um,
            show=self.c.show_frames, um_per_px=self.c.um_per_px,
            title=f"Round {round_no} — live frame before choosing a site",
            note="Captured BEFORE this round's move; the analysis drives it.")
        if img:
            self.state["current_image"] = img
            self._refresh_crosshair(img)
            # KEEP A COPY. Every capture downloads to the same filename, so the
            # confirmation frame taken minutes later overwrites the very frame
            # the targets were chosen from — and arrival verification has
            # nothing left to register against.
            self._round_frame = (self._snapshot(img, f"round{round_no:02d}_pre"),
                                 tuple(frame_pos))
            self.analyze_current_frame(round_no, tag="pre-proposal")
        return frame_pos

    def _snapshot(self, img_or_path, name: str) -> Optional[Path]:
        """Copy a downloaded frame somewhere the next capture will not clobber."""
        import shutil
        src = ui.resolve_image_path(img_or_path) if isinstance(img_or_path, dict) \
            else img_or_path
        if not src:
            return None
        try:
            outdir = Path(self.session.download_dir) / "arrival"
            outdir.mkdir(parents=True, exist_ok=True)
            # Time-stamped: re-proposing a round reuses the round number, and
            # on 2026-08-05 the retry's frame overwrote the one belonging to the
            # attempt that had just failed — deleting the evidence needed to
            # explain the failure.
            stamp = datetime.now().strftime("%H%M%S")
            dst = outdir / f"{self.c.run_id or 'run'}_{name}_{stamp}.png"
            shutil.copy(src, dst)
            return dst
        except Exception as exc:
            print(f"  [warn] could not snapshot the frame ({exc}) — arrival "
                  "verification will have nothing to compare against.")
            return None

    # ------------------------------------------------------------ site maths
    def image_reference_um(self) -> tuple:
        """Where the indenter points IN THE FRAME (image um): the located
        crosshair, else the geometric centre. Every image<->stage conversion in
        the campaign — aiming, the planning overlay, arrival verification — is
        measured from this one point, so they cannot disagree."""
        if self.crosshair_um is not None and self.c.aim_at_crosshair:
            return self.crosshair_um
        return (self.c.fov_w_um / 2.0, self.c.fov_h_um / 2.0)

    def _refresh_crosshair(self, img: Optional[dict]) -> None:
        """Locate InView's burned-in red indenter marker in the current frame."""
        if self.c.use_simulator or not self.c.aim_at_crosshair or not img:
            return
        path = ui.resolve_image_path(img)
        if not path:
            return
        from PIL import Image
        from ..instrument import imagecal as ic
        try:
            arr = np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            return
        px = ic.find_crosshair(arr)
        umpp = self.c.um_per_px or (self.c.fov_w_um / arr.shape[1])
        if px is None:
            if self.crosshair_um is None:
                print("  [warn] indenter crosshair not found in the frame — aiming "
                      "at the geometric centre of the capture.")
            return
        was = self.crosshair_um
        self.crosshair_um = (round(px[0] * umpp, 2), round(px[1] * umpp, 2))
        if was is None or math.hypot(was[0] - self.crosshair_um[0],
                                     was[1] - self.crosshair_um[1]) > 1.0:
            print(f"  indenter crosshair at image ({self.crosshair_um[0]:.1f}, "
                  f"{self.crosshair_um[1]:.1f}) um — "
                  f"({self.crosshair_um[0] - self.c.fov_w_um / 2:+.1f}, "
                  f"{self.crosshair_um[1] - self.c.fov_h_um / 2:+.1f}) um from the "
                  "frame centre; sites are aimed at THIS point.")

    def _site_destination(self, target: dict, frame_pos, *, source: str):
        """Absolute stage destination in the RIGHT frame for its source:
        image targets carry FRAME-LOCAL fractions; BO proposals are already
        absolute (their x_frac round-trip gets clipped, so absolute wins).

        The image->stage conversion uses the MEASURED site gains rather than
        the assumed "image right = +x / image up = +y" convention. See
        CampaignConfig.site_gain_x/_y — the y half of that assumption was
        backwards on this instrument and drove every target the wrong way.
        """
        from ..instrument.imagecal import stage_delta_from_image
        if source == "bo_acquisition" and target.get("x_um") is not None:
            return (round(float(target["x_um"]), 2), round(float(target["y_um"]), 2))
        xf, yf = float(target.get("x_frac", 0.5)), float(target.get("y_frac", 0.5))
        ref_x, ref_y = self.image_reference_um()
        dx_img = xf * self.c.fov_w_um - ref_x     # image RIGHT positive
        dy_img = yf * self.c.fov_h_um - ref_y     # image DOWN  positive
        dsx, dsy = stage_delta_from_image(dx_img, dy_img,
                                          self.c.site_gain_x, self.c.site_gain_y)
        return (round(frame_pos[0] + dsx, 2), round(frame_pos[1] + dsy, 2))

    # ------------------------------------------------- image->stage calibration
    def calibrate_image_to_stage(self, probe_um: float = 60.0,
                                 *, apply: bool = True) -> dict:
        """MEASURE site_gain_x / site_gain_y. XY only, NO indents, ~1 minute.

        Captures a frame, drives a known move on each axis, re-captures, and
        registers the frames to see which way (and how far) the scene actually
        moved. Runs its own controls first and REFUSES to report a gain from a
        field it cannot register, rather than returning a confident wrong sign.
        """
        import shutil
        from PIL import Image
        from ..instrument import imagecal as ic

        outdir = self.session.download_dir / "imagecal"
        outdir.mkdir(parents=True, exist_ok=True)

        def grab(tag):
            cap = self.session.toolkit.call_tool("capture_video_panel_remote", {})
            info = ui.download_capture(self.session.toolkit,
                                       cap.result if cap.ok else {},
                                       self.session.download_dir)
            if not info.get("local_path"):
                raise RuntimeError("capture failed during image-to-stage calibration")
            dst = outdir / f"{tag}.png"
            shutil.copy(info["local_path"], dst)
            return ic.crop_video_area(
                np.asarray(Image.open(dst).convert("L"), dtype=float))

        home = self.session.read_stage_xy(what="image-cal home")
        A = grab("A_home")
        ok, why = ic.self_test(A)
        print(f"  registration controls: {why}")
        if not ok:
            raise RuntimeError(
                "Cannot calibrate image->stage on this field: " + why +
                "\n  Move to a field with more microstructural contrast and retry, "
                "or set CampaignConfig.site_gain_x/_y from a known-good measurement.")

        umpp = self.c.um_per_px or (self.c.fov_w_um / A.shape[1])
        out = {"probe_um": probe_um, "um_per_px_assumed": umpp}
        for axis, dest in (("x", (home[0] + probe_um, home[1])),
                           ("y", (home[0], home[1] + probe_um))):
            self.motion.jog_to(dest, label=f"image-cal +{axis}")
            moved = self.session.read_stage_xy(what=f"image-cal +{axis}")
            actual = (moved[0] - home[0]) if axis == "x" else (moved[1] - home[1])
            B = grab(f"B_plus_{axis}")
            dy, dx, score = ic.register(A, B)
            shift_px = dx if axis == "x" else dy
            gain, true_umpp = ic.gain_from_probe(actual, shift_px, umpp)
            out[axis] = {"stage_um": round(actual, 2), "scene_shift_px": shift_px,
                         "scene_shift_um": round(shift_px * umpp, 2),
                         "site_gain": round(gain, 3),
                         "true_um_per_px": round(true_umpp, 4),
                         "cross_axis_px": (dy if axis == "x" else dx),
                         "score": round(score, 3)}
            print(f"  {axis}: stage {actual:+.1f} um -> scene {shift_px:+d} px "
                  f"({shift_px * umpp:+.1f} um)  =>  site_gain_{axis} = {gain:+.3f}  "
                  f"(true {true_umpp:.4f} um/px, score {score:.2f})")
            self.motion.jog_to(home, label="image-cal home")

        if apply:
            self.c.site_gain_x = out["x"]["site_gain"]
            self.c.site_gain_y = out["y"]["site_gain"]
            print(f"\nAPPLIED: site_gain_x={self.c.site_gain_x:+.3f}, "
                  f"site_gain_y={self.c.site_gain_y:+.3f}")
        self.journal.append("IMAGE_STAGE_CAL", **out)
        return out

    def _clamp_to_search_area(self, dest, *, margin_um: float = 10.0):
        b = getattr(self.objective.constraints, "fov_bounds", None)
        if not b:
            return tuple(dest), False
        x = min(max(dest[0], b[0] - margin_um), b[1] + margin_um)
        y = min(max(dest[1], b[2] - margin_um), b[3] + margin_um)
        return ((round(x, 2), round(y, 2)),
                abs(x - dest[0]) > 0.01 or abs(y - dest[1]) > 0.01)

    def _far_enough(self, dest) -> bool:
        """Keep new sites clear of what was already measured — distance in
        STAGE space, because frame-local keys change every round."""
        for d in self.state.get("indent_history") or []:
            p = d.get("position")
            if p and math.hypot(dest[0] - p[0], dest[1] - p[1]) < self._keepout_um:
                return False
        return True

    def _plan_site(
        self,
        target: dict,
        frame_pos,
        *,
        source: str,
        remaining_budget: int,
    ) -> Optional[dict]:
        """Build one action plan pointed at the true destination, with honest
        approval cards (the gate shows the site that will be driven to)."""
        strategy_plan = self.measurement_strategy.plan(
            target, self.state, remaining_budget=remaining_budget)
        strategy_record = {
            "target": target.get("label", "target"),
            "target_source": source,
            **strategy_plan.to_journal(),
        }
        self.state.setdefault("measurement_strategy_decisions", []).append(strategy_record)
        self.journal.append("MEASUREMENT_STRATEGY_DECISION", **strategy_record)
        agent_features = ", ".join(strategy_plan.agent_features) or "none"
        print(
            f"    strategy '{target.get('label', '?')}': {strategy_plan.action.upper()} | "
            f"{strategy_plan.grid_plan.grid_nx}x{strategy_plan.grid_plan.grid_ny} @ "
            f"{strategy_plan.grid_plan.spacing_um:g} um | source={strategy_plan.source} | "
            f"agent features={agent_features}\n"
            f"      {strategy_plan.reason}")
        self._log(
            "Measurement Strategy Agent", "Orchestrator Agent",
            f"{target.get('label', 'target')}: {strategy_plan.action}, "
            f"{strategy_plan.grid_plan.grid_nx}x{strategy_plan.grid_plan.grid_ny}, "
            f"agent features: {agent_features}",
            detail=strategy_plan.reason,
        )
        if strategy_plan.action == "skip":
            return None

        grid_plan = strategy_plan.grid_plan
        method_plan = strategy_plan.method_plan
        plan = ui.plan_target_actions(
            target, frame_pos, sample_name=self.next_sample_name(),
            grid_nx=grid_plan.grid_nx, grid_ny=grid_plan.grid_ny,
            spacing_um=grid_plan.spacing_um, rotation_deg=grid_plan.rotation_deg,
            setup=method_plan.to_setup(),
            method_type=strategy_plan.method_type,
            fov_width_um=self.c.fov_w_um, fov_height_um=self.c.fov_h_um,
            expected_outcome=target.get("reason", target.get("rank_reason",
                                                             "objective-driven site")),
            site_gain_x=self.c.site_gain_x, site_gain_y=self.c.site_gain_y,
            method_plan_args=method_plan.to_card_arguments(),
            strategy_plan_args=strategy_plan.to_card_arguments())
        plan["method_plan"] = method_plan
        plan["grid_plan"] = grid_plan
        plan["measurement_strategy"] = strategy_plan
        dest, clamped = self._clamp_to_search_area(
            self._site_destination(target, frame_pos, source=source))
        if clamped:
            print(f"    '{plan['label']}' clamped back into the search area -> {dest}")
        plan["destination"] = dest
        # The frame this site was chosen FROM — what arrival verification
        # registers the confirmation frame against.
        if self._round_frame:
            plan["frame_path"] = str(self._round_frame[0] or "")
            plan["frame_pos"] = self._round_frame[1]
        for card in plan["cards"]:
            card["target"] = f"{plan['label']} @ stage ({dest[0]:.1f}, {dest[1]:.1f}) um"
        plan["cards"][0]["arguments"] = {
            "destination_um": list(dest),
            "how": "closed-loop MotionController.jog_to() from the live position"}
        plan["moves"] = []
        return plan

    # ------------------------------------------------- arrival verification
    def _register_against(self, pre_path, post_path) -> dict:
        """Measure how far the SCENE moved between two frames, in image um.

        Returns ``{"status": "ok"|"unknown", ...}``. Every way of NOT knowing —
        a missing file, mismatched panel sizes, a field too flat to register,
        a weak correlation peak — is "unknown", never a number.
        """
        from PIL import Image
        from ..instrument import imagecal as ic
        try:
            pre = np.asarray(Image.open(pre_path).convert("L"), dtype=float)
            post = np.asarray(Image.open(post_path).convert("L"), dtype=float)
        except Exception as exc:
            return {"status": "unknown", "why": f"could not open a frame: {exc}"}
        if pre.shape != post.shape:
            return {"status": "unknown",
                    "why": f"panel size changed between frames "
                           f"({pre.shape} vs {post.shape}) — the InView window "
                           "was resized or moved"}
        a, b = ic.crop_video_area(pre), ic.crop_video_area(post)
        ok, why = ic.self_test(a)
        if not ok:
            return {"status": "unknown", "why": f"registration controls failed: {why}"}
        dy, dx, score = ic.register(a, b)
        # `score` is a true correlation coefficient now; every good registration
        # measured on this instrument came in above 0.9.
        if score < 0.45:
            return {"status": "unknown",
                    "why": f"correlation peak too weak (r={score:.2f}) — this "
                           "field has too little texture to register"}
        umpp = self.c.um_per_px or (self.c.fov_w_um / a.shape[1])
        return {"status": "ok", "shift_um": (dx * umpp, dy * umpp),
                "shift_px": (dx, dy), "score": round(score, 3)}

    # How far a measured scene shift may sit from the commanded one before the
    # measurement is disbelieved. Good registrations on this instrument land
    # within ~1-6 um of the command; 15 um is comfortably outside that and well
    # inside the tens-of-um errors that matter.
    _ARRIVAL_SLACK_UM = 15.0

    def _trust_registration(self, leg_stage, leg_scene) -> tuple:
        """Does a measured scene shift AGREE with the move that produced it?

        Returns ``("ok" | "flipped" | "untrusted", detail)``.

        THE CHECK NEEDS A CHECK. On 2026-08-05 a registration failure reported
        that the scene had not moved in x at all while the stage moved 41.7 um;
        the arrival logic believed it, "corrected" a stage that was already
        0.2 um from its target, and drove it 49 um away before refusing to
        indent. A measurement is only usable if it is consistent with the
        commanded move, and the two ways of being inconsistent mean OPPOSITE
        things:

        * right magnitude, WRONG SIGN -> the image->stage map is genuinely
          backwards. Credible, dangerous, and exactly what this check exists to
          catch: escalate.
        * magnitude nowhere near the command (the scene "did not move", or moved
          several times too far) -> the REGISTRATION failed, not the stage. Say
          so and change nothing. Correcting on it would be acting on noise.

        EVERY AXIS IS JUDGED, however small the move. An earlier version skipped
        axes commanded less than 10 um on the grounds that the expectation was
        within registration noise — which left the SECOND check of a correction
        cycle (a few um on each axis, by construction) completely unguarded. On
        2026-08-05 a 5.8 um correction registered as a 96 um scene shift, sailed
        through unjudged, and stopped the campaign 113 um "off target" when it
        was on it. A small commanded move simply gets an ABSOLUTE tolerance
        instead of a proportional one.
        """
        gains = {"x": self.c.site_gain_x, "y": self.c.site_gain_y}
        detail = []
        verdict = "ok"
        for i, ax in enumerate(("x", "y")):
            if abs(gains[ax]) < 1e-9:
                continue
            expected = -leg_stage[i] / gains[ax]      # scene um this move implies
            err = leg_scene[i] - expected
            # Slack: whichever is larger of the registration's own error floor
            # and half the commanded distance (gains are known to ~10-20%).
            slack = max(self._ARRIVAL_SLACK_UM, 0.5 * abs(expected))
            detail.append(f"{ax}: scene {leg_scene[i]:+.1f} um vs {expected:+.1f} "
                          f"expected (off by {err:+.1f})")
            if abs(err) <= slack:
                continue
            # Wrong by a lot. A near-perfect MIRROR of the expected move is the
            # signature of a backwards map; anything else is a failed read.
            mirrored = (expected != 0 and leg_scene[i] * expected < 0
                        and 0.5 <= abs(leg_scene[i] / expected) <= 2.0)
            if mirrored and verdict != "untrusted":
                verdict = "flipped"
            elif not mirrored:
                verdict = "untrusted"
        return verdict, "; ".join(detail)

    def _verify_arrival(self, plan: dict, achieved, round_no: int) -> tuple:
        """IS THE CHOSEN FEATURE ACTUALLY UNDER THE INDENTER? Returns the
        (possibly corrected) achieved position and a report dict.

        jog_to() answers a different question — "did the stage reach the
        coordinate I asked for" — and answers it very well (0.07 um on
        2026-08-05). It is blind by construction to the coordinate being wrong,
        because it verifies the stage against the same map that computed the
        destination. A sign error in image->stage therefore converges perfectly
        onto a site ~2x the offset away from the feature, and reports success.

        This measures the thing that actually matters, independently of the
        stage readback: register the live frame against the frame the target
        was chosen from, and see where the target has ENDED UP relative to the
        indenter crosshair. Out of tolerance -> correct and re-measure; still
        out -> refuse to indent (mode "enforce").
        """
        mode = (self.c.verify_arrival or "off").lower()
        if mode == "off" or self.c.use_simulator:
            return achieved, {"status": "skipped"}
        pre_path, frame_pos = plan.get("frame_path"), plan.get("frame_pos")
        if not pre_path or not Path(pre_path).exists() or not frame_pos:
            print("  [warn] arrival NOT verified — no snapshot of the frame this "
                  "site was chosen from. Indenting on the stage readback alone.")
            return achieved, {"status": "unknown", "why": "no reference frame"}

        from ..instrument.imagecal import image_delta_from_stage
        gx, gy = self.c.site_gain_x, self.c.site_gain_y
        requested = tuple(plan.get("requested_destination") or plan["destination"])
        ref = self.image_reference_um()
        # Where the target sat in the reference frame (image um from its corner).
        dxi, dyi = image_delta_from_stage(requested[0] - frame_pos[0],
                                          requested[1] - frame_pos[1], gx, gy)
        target_pre = (ref[0] + dxi, ref[1] + dyi)

        report: dict = {"status": "unknown", "target_image_um": target_pre}
        # Chain the registrations: each leg is measured against the PREVIOUS
        # frame, not the original. Legs are small, and a small shift registers
        # far more reliably than a large one (a 100 um move leaves barely a
        # third of the frame in common with where it started).
        ref_path, ref_stage, cum = pre_path, tuple(frame_pos), [0.0, 0.0]
        for attempt in range(int(self.c.max_visual_corrections) + 1):
            post = self._snapshot(self._capture_now(),
                                  f"{plan['sample_name']}_arrive{attempt}")
            if post is None:
                print("  [warn] arrival NOT verified — the confirmation capture "
                      "failed. Indenting on the stage readback alone.")
                report["why"] = "capture failed"
                break
            m = self._register_against(ref_path, post)
            if m["status"] != "ok":
                print(f"  [warn] arrival NOT verified — {m['why']}. Indenting on "
                      "the stage readback alone.")
                report.update(why=m["why"])
                break

            leg_stage = (achieved[0] - ref_stage[0], achieved[1] - ref_stage[1])
            leg_scene = m["shift_um"]
            verdict, detail = self._trust_registration(leg_stage, leg_scene)
            if verdict == "untrusted":
                # NOT a stage fault and NOT a reason to stop: the picture, not
                # the instrument, is what failed to be read.
                print(f"  [warn] arrival NOT verified — the registration disagrees "
                      f"with the move that was just made ({detail}), so it is "
                      "measuring something other than the stage. Changing NOTHING "
                      "and indenting on the stage readback, which landed "
                      f"{math.hypot(*[achieved[i] - requested[i] for i in (0, 1)]):.2f}"
                      " um from the requested site.")
                report.update(why=f"registration inconsistent with the move ({detail})",
                              scene_shift_um=(round(leg_scene[0], 1),
                                              round(leg_scene[1], 1)),
                              score=m["score"], frame=str(post), attempt=attempt)
                break

            cum[0] += leg_scene[0]; cum[1] += leg_scene[1]
            ref_path, ref_stage = post, tuple(achieved)
            live = {ax: round(-leg_stage[i] / leg_scene[i], 2)
                    for i, ax in enumerate(("x", "y"))
                    if abs(leg_stage[i]) >= 10.0 and abs(leg_scene[i]) >= 1.0}
            now = (target_pre[0] + cum[0], target_pre[1] + cum[1])
            ref_now = self._crosshair_from(post) or ref
            res = (now[0] - ref_now[0], now[1] - ref_now[1])
            dist = math.hypot(*res)
            report.update(status="ok", scene_shift_um=(round(cum[0], 1), round(cum[1], 1)),
                          residual_um=(round(res[0], 1), round(res[1], 1)),
                          error_um=round(dist, 1), score=m["score"],
                          live_gain=live, frame=str(post), attempt=attempt)
            print(f"    VISUAL ARRIVAL: scene moved ({cum[0]:+.1f}, {cum[1]:+.1f}) um "
                  f"(score {m['score']:.2f}); the chosen feature is "
                  f"{dist:.1f} um from the indenter axis "
                  f"[live gain {', '.join(f'{k}={v:+.2f}' for k, v in live.items()) or 'n/a'}]")

            # A map whose SIGN is backwards is not a tolerance problem — every
            # site this session is aimed at its mirror image. Correcting the
            # position would hide it. `_trust_registration` only reaches this
            # verdict when the MAGNITUDE matched, so it is the map, not noise.
            if verdict == "flipped":
                msg = (f"image->stage sign is WRONG: {detail}, against a configured "
                       f"({gx:+.2f}, {gy:+.2f}). The scene moved the right distance "
                       "the WRONG WAY, so every site is being aimed at its mirror "
                       "image. NOT indenting — fix with "
                       "runner.calibrate_image_to_stage().")
                self.journal.append("ARRIVAL_FAILED", round=round_no,
                                    sample=plan["sample_name"], reason="sign", **report)
                if mode == "enforce":
                    raise StageSafetyError(msg)
                print(f"  [warn] {msg}")
                break

            if dist <= self.c.arrival_tol_um:
                break
            if attempt >= int(self.c.max_visual_corrections):
                self.journal.append("ARRIVAL_FAILED", round=round_no,
                                    sample=plan["sample_name"],
                                    reason="out_of_tolerance", **report)
                msg = (f"'{plan['label']}' is {dist:.1f} um from the indenter axis "
                       f"after {attempt + 1} visual check(s) — tolerance "
                       f"{self.c.arrival_tol_um:g} um. The stage reached its "
                       "requested coordinate, so this is the image->stage map, not "
                       "the stage. NOT indenting.")
                if mode == "enforce":
                    raise StageSafetyError(msg)
                print(f"  [warn] {msg} (verify_arrival='warn' — indenting anyway)")
                break

            # Correct: the feature is `res` away in the image, so move the stage
            # by gain * res — the same forward map, no new convention.
            from ..instrument.imagecal import stage_delta_from_image
            cdx, cdy = stage_delta_from_image(res[0], res[1], gx, gy)
            print(f"    correcting by ({cdx:+.1f}, {cdy:+.1f}) um and re-checking.")
            achieved, _, _, cdiag = self.motion.jog_to(
                (achieved[0] + cdx, achieved[1] + cdy),
                label=f"{plan['label']} visual correction")
            if cdiag["inverted_axes"]:
                raise StageSafetyError(
                    f"axis {'/'.join(cdiag['inverted_axes'])} moved OPPOSITE to a "
                    "clean command during visual correction — NOT indenting.")
            report["corrected_by_um"] = (round(cdx, 1), round(cdy, 1))

        # Name the event for what actually happened — a journal that logs
        # "VERIFIED" for a check that could not run is worse than no journal.
        ok = (report.get("status") == "ok"
              and report.get("error_um", float("inf")) <= self.c.arrival_tol_um)
        self.journal.append("ARRIVAL_VERIFIED" if ok else "ARRIVAL_UNVERIFIED",
                            round=round_no, sample=plan["sample_name"], **report)
        return achieved, report

    def _capture_now(self) -> Optional[str]:
        """Grab the video panel right now; return the local path (or None)."""
        try:
            cap = self.session.toolkit.call_tool("capture_video_panel_remote", {})
            info = ui.download_capture(self.session.toolkit,
                                       cap.result if cap.ok else {},
                                       self.session.download_dir)
            return info.get("local_path") or None
        except Exception as exc:
            print(f"  [warn] capture failed during arrival check: {exc}")
            return None

    def _crosshair_from(self, path) -> Optional[tuple]:
        """The indenter crosshair in THIS frame, in image um."""
        if not self.c.aim_at_crosshair:
            return None
        from PIL import Image
        from ..instrument import imagecal as ic
        try:
            arr = np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            return None
        px = ic.find_crosshair(arr)
        if px is None:
            return None
        umpp = self.c.um_per_px or (self.c.fov_w_um / arr.shape[1])
        return (px[0] * umpp, px[1] * umpp)

    @staticmethod
    def _arm_plan_at(plan: dict, achieved, requested) -> dict:
        """Point the plan at the position the stage ACTUALLY reached, so
        place_array() sees zero drift; keep the requested site for provenance."""
        plan["requested_destination"] = (round(float(requested[0]), 2),
                                         round(float(requested[1]), 2))
        plan["destination"] = (round(float(achieved[0]), 2),
                               round(float(achieved[1]), 2))
        plan["position_error_um"] = round(
            math.hypot(achieved[0] - requested[0], achieved[1] - requested[1]), 2)
        return plan

    # --------------------------------------------------------------- propose
    def propose_round(self, *, show: Optional[bool] = None) -> RoundProposal:
        """(A) orchestrator decides; (frame) fresh capture + analysis;
        (B) sites proposed + gate opened. Idempotent: re-running replaces an
        undecided gate with the same id and never touches the instrument
        beyond a capture."""
        show = self.c.show_frames if show is None else show
        surrogate = ObjectiveSurrogate(self.objective).fit(
            self.state["indent_history"])
        progress = self.orch.evaluate_progress(self.state, surrogate)
        decision = self.orch.decide_next_action(self.state, progress, surrogate)
        round_no = progress.round_index + 1

        print(f"\n{'=' * 76}\nORCHESTRATOR (round {round_no}): "
              f"{decision.action.upper()}   [advisor: {decision.advisor}]\n"
              f"{decision.reason}\n{'=' * 76}")
        self._log("Orchestrator Agent", "Human Operator",
                  f"Round {round_no}: {decision.action} — {decision.reason[:120]}",
                  detail=decision.reason)
        self.journal.append("ROUND_DECISION", round=round_no,
                            action=decision.action, reason=decision.reason[:400],
                            proposal_source=decision.proposal_source,
                            advisor=decision.advisor,
                            improvement_pct=progress.improvement_pct,
                            indents_used=progress.indents_used,
                            # the acquisition value the decision was made on —
                            # without it the journal cannot reproduce the BO
                            # convergence plot after the session is gone
                            max_ei=progress.max_ei)

        if decision.action != "continue":
            decs = self.state.get("orchestrator_decisions") or []
            if not (decs and decs[-1].get("round") == progress.round_index
                    and decs[-1].get("action") == decision.action):
                self.state.update(self.orch.decision_state_updates(
                    self.state, decision, progress))
            return RoundProposal(round_no, decision, progress, surrogate,
                                 tuple(self.capture_pos or (0.0, 0.0)))

        frame_pos = self._fresh_frame(round_no)
        ranked = list(self.state.get("measurement_plan") or [])

        want = int(decision.n_sites)
        if decision.proposal_source == "bo_acquisition":
            proposals = surrogate.propose_sites(
                want + 3, round_index=round_no, stage_position=self.capture_pos,
                fov_width_um=self.c.fov_w_um, fov_height_um=self.c.fov_h_um)
            if show and proposals:
                self._plot_acquisition(surrogate, proposals, round_no)
        else:
            visited = self.state.get("visited_target_keys") or []
            proposals = [dict(t) for t in ranked
                         if target_visit_key(t) not in visited][:want + 3]

        plans = []
        for p in proposals:
            if len(plans) >= want:
                break
            remaining_for_site = progress.budget_remaining - sum(
                pl["grid_nx"] * pl["grid_ny"] for pl in plans)
            if remaining_for_site <= 0:
                break
            dest = self._clamp_to_search_area(
                self._site_destination(p, frame_pos,
                                       source=decision.proposal_source))[0]
            if not self._far_enough(dest):
                print(f"    skipping '{p.get('label', '?')}' at {dest} — within "
                      f"{self._keepout_um:g} um of a measured site.")
                continue
            site_plan = self._plan_site(
                p, frame_pos, source=decision.proposal_source,
                remaining_budget=remaining_for_site)
            if site_plan is not None:
                plans.append(site_plan)

        if plans:
            from IPython.display import display
            display(pd.DataFrame([{
                "site": pl["label"], "sample": pl["sample_name"],
                "stage x (um)": pl["destination"][0],
                "stage y (um)": pl["destination"][1],
                "strategy": pl["measurement_strategy"].source,
                "grid": f"{pl['grid_nx']}x{pl['grid_ny']}",
                "depth (nm)": pl["method_plan"].target_depth_nm,
                "load (mN)": pl["method_plan"].target_load_mN,
                "agent-decided": ", ".join(
                    pl["measurement_strategy"].agent_features) or "none",
                "why": str(pl["target"].get("reason")
                           or pl["target"].get("rank_reason", ""))[:90],
            } for pl in plans]))
            if show and self.state.get("current_image"):
                ui.plot_planned_grids(
                    self.state["current_image"], plans, frame_pos,
                    title=f"Round {round_no} — sites from {decision.proposal_source}",
                    fov_width_um=self.c.fov_w_um, fov_height_um=self.c.fov_h_um,
                    site_gain_x=self.c.site_gain_x, site_gain_y=self.c.site_gain_y,
                    crosshair_um=self.image_reference_um())
        else:
            print("    no proposable sites left this round.")

        cost = sum(pl["grid_nx"] * pl["grid_ny"] for pl in plans)
        gate = None
        if plans:
            gid = f"gate-round-{round_no}"
            ui.ApprovalGate.registry[:] = [
                g for g in ui.ApprovalGate.registry
                if not (g.gate_id == gid and g.decision is None)]
            gate = ui.ApprovalGate(
                gid, f"Round {round_no} — {decision.proposal_source} site(s)",
                description=(f"Orchestrator: {decision.reason[:220]}\n"
                             f"Approving executes {cost} indents at "
                             f"{len(plans)} location(s).\n"
                             "The cards show every agent-selected recipe feature; "
                             "the validated values on those cards are what will execute."),
                action_cards=[card for pl in plans for card in pl["cards"]])
            self.journal.append("ROUND_PROPOSED", round=round_no,
                                source=decision.proposal_source,
                                n_plans=len(plans), cost_indents=cost,
                                sites=[{"sample": pl["sample_name"],
                                        "label": pl["label"],
                                        "dest": pl["destination"],
                                        "method_plan": pl["method_plan"].to_journal(),
                                        "grid_plan": pl["grid_plan"].to_journal(),
                                        "measurement_strategy": pl[
                                            "measurement_strategy"].to_journal()}
                                       for pl in plans])
            self.state["acquisition_meta"] = {**surrogate.summary(),
                                              "round": round_no,
                                              "proposal_source": decision.proposal_source,
                                              "proposals": proposals}
        return RoundProposal(round_no, decision, progress, surrogate,
                             frame_pos, proposals, plans, gate)

    def _plot_acquisition(self, surrogate, proposals, round_no) -> None:
        import matplotlib.pyplot as plt
        maps = surrogate.posterior_maps(n=60)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for ax, key, title, cmap in [
                (axes[0], "scalar_mean", "objective posterior mean", "viridis"),
                (axes[1], "scalar_sigma", "posterior uncertainty (sigma)", "magma"),
                (axes[2], "ei", "Expected Improvement", "cividis")]:
            im = ax.imshow(maps[key], origin="lower", extent=maps["extent"],
                           aspect="equal", cmap=cmap)
            ax.scatter(surrogate.X[:, 0], surrogate.X[:, 1], s=14, c="white",
                       edgecolor="k", linewidths=0.4)
            for p in proposals:
                ax.scatter([p["x_um"]], [p["y_um"]], marker="*", s=280, c="red",
                           edgecolor="k", linewidths=0.6)
            ax.set_title(title)
            ax.set_xlabel("stage x (um)"); ax.set_ylabel("stage y (um)")
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(f"Round {round_no} — GP posterior + acquisition "
                     f"({surrogate.mode})", y=1.03)
        plt.tight_layout(); plt.show()

    # --------------------------------------------------------------- execute
    def _approve_with_policy(self, gate, plans: list[dict], policy) -> None:
        """Apply a noninteractive policy without bypassing recipe authority.

        On real hardware, a policy must explicitly declare that it covers
        agent-selected recipes. ``AutonomousHardwareApproval`` earns that
        capability only after the operator supplies the acknowledgement phrase.
        """
        agent_changes = any(
            pl["measurement_strategy"].requires_human_approval for pl in plans)
        if (not self.c.use_simulator and agent_changes
                and not getattr(policy, "allows_agent_recipe_changes", False)):
            raise PermissionError(
                "This round contains Measurement Strategy Agent changes, but the "
                "current approval policy is not authorized for agent-selected "
                "recipes on real hardware. Review and approve this round manually, "
                "or arm Section 8 with AutonomousHardwareApproval and the explicit "
                "AUTONOMOUS_ACK acknowledgement.")
        policy.approve(gate)

    def execute_round(self, proposal: RoundProposal, *, decision: Optional[str] = None,
                      policy=None) -> list[dict]:
        """Position -> indent -> export -> validate -> record, for every
        approved plan; then the measurement critic and the round advance.
        Refuses to double-execute an already-executed round."""
        if proposal.terminal:
            print(f"Nothing to execute — the orchestrator decided "
                  f"'{proposal.decision.action}'.")
            return []
        if not proposal.plans:
            print("Nothing to execute — no plans were proposed.")
            return []
        if not self._preflighted:
            self.preflight()

        gate = proposal.gate
        if policy is not None:
            self._approve_with_policy(gate, proposal.plans, policy)
        elif decision is not None and gate.decision is None:
            apply_decision(gate, decision, self.c.operator)
        if gate.decision is not None:
            self.journal.append("GATE", gate_id=gate.gate_id,
                                approved=gate.is_approved,
                                by=getattr(gate, "approved_by", ""),
                                note=getattr(gate, "note", ""))
        if gate.decision is None:
            return []          # PENDING — apply_decision already said how to proceed
        if not gate.is_approved:
            print(f"Round {proposal.round_no} REJECTED — nothing will execute. "
                  "Re-run propose_round() for a different proposal.")
            return []

        if any(dd.get("approved_via") == gate.gate_id
               for dd in self.state["decision_history"]):
            print(f"Round {proposal.round_no} was already executed — "
                  "call propose_round() for the next round.")
            return []

        round_no = proposal.round_no
        in_list = int(self.state.get("imicro_queue_count",
                                     len(self.state["indent_history"])))
        results, executed = [], 0
        for plan in proposal.plans:
            plan["remove_n"] = in_list if executed == 0 else 0
            res = self._execute_plan(plan, round_no, gate)
            results.append(res)
            executed += 1

        if executed:
            self.state["imicro_queue_count"] = executed
            from IPython.display import display
            display(ui.indent_results_table(self.state["indent_history"]))

            self.state.update(measurement_critic_node(self.state))
            ui.critic_panel(
                f"Measurement Critic — after round {round_no}",
                self.state["measurement_critique"],
                approved=self.state["measurement_critique_decision"]
                == "hypothesis_confirmed")
            self._log("Measurement Critic", "Orchestrator Agent",
                      f"Decision: {self.state['measurement_critique_decision']} "
                      "(objective criteria)",
                      detail=str(self.state["measurement_critique"])[:250])

            self.state.update(self.orch.decision_state_updates(
                self.state, proposal.decision, proposal.progress))
            self.state["orchestrator_round"] = round_no
            self.journal.append("ROUND_DONE", round=round_no, executed=executed,
                                critic=self.state.get("measurement_critique_decision"))
            print(f"\nRound {round_no} complete — call propose_round() for "
                  f"round {round_no + 1}.")
        return results

    def _execute_plan(self, plan: dict, round_no: int, gate) -> dict:
        """One site end to end. Raises StageSafetyError rather than indent
        blind; export failures degrade to screen data + a journaled recovery."""
        requested = tuple(plan["destination"])
        print("\n" + "=" * 78)
        print(f"PLAN '{plan['sample_name']}' — {plan['label']}  ->  stage "
              f"({requested[0]:.1f}, {requested[1]:.1f}) um")
        print("=" * 78)

        # ---- MOVE (the only move in the round) ----
        achieved, err, on_target, mdiag = self.motion.jog_to(
            requested, label=plan["label"])
        self.journal.append("POSITIONED", round=round_no,
                            sample=plan["sample_name"], requested=requested,
                            achieved=achieved, error_um=round(err, 2),
                            jogs=len(mdiag["attempts"]),
                            inverted_axes=mdiag["inverted_axes"])
        if mdiag["inverted_axes"]:
            raise StageSafetyError(
                f"axis {'/'.join(mdiag['inverted_axes'])} moved OPPOSITE to a clean "
                "command — NOT indenting. Run motion.calibrate(), then retry.")
        from_centre = math.hypot(achieved[0] - self.capture_pos[0],
                                 achieved[1] - self.capture_pos[1])
        if from_centre > self._envelope_um:
            raise StageSafetyError(
                f"stage is {from_centre:.0f} um from the campaign centre (envelope "
                f"{self._envelope_um:.0f} um) — outside the area being mapped. NOT "
                "indenting. Recover with motion.jog_to(runner.capture_pos).")
        if err > self.motion.cfg.accept_um:
            print(f"    landed {err:.1f} um from the requested site (backlash/gain). "
                  "Indenting HERE — the recorded position is the live readback, so "
                  "BO gets a TRUE measurement, just not at the exact site it picked.")
        elif not on_target:
            print(f"    {err:.1f} um from the requested site — indenting HERE and "
                  "recording THIS position.")

        # ---- VERIFY IN IMAGE SPACE (the stage cannot check its own map) ----
        achieved, arrival = self._verify_arrival(plan, achieved, round_no)
        plan["arrival"] = arrival
        self._arm_plan_at(plan, achieved, requested)

        # ---- INDENT (irreversible; place_array never moves) ----
        # Hand the monitor a DEFINITIVE completion signal. Without it a 1x1
        # array can only be timed out: the OCR counter is the current test
        # index, so it reads 1 from the moment the test is queued.
        probe = (None if self.c.use_simulator else
                 (lambda: self.exporter.result_ready(plan["sample_name"])))
        run_record = self.session.place_array(
            plan, fov_w_um=self.c.fov_w_um, fov_h_um=self.c.fov_h_um,
            test_timeout_s=self.c.test_timeout_s, show_frames=self.c.show_frames,
            done_probe=probe, skip_screening=self.c.skip_screening,
            indent_duration_s=self.c.indent_duration_s)

        # ---- EXPORT (recoverable; failure degrades, never aborts) ----
        csv_result = None
        if self.c.use_simulator:
            try:
                csv_result = self.exporter.fetch_result_csv(run_record)
            except Exception as exc:
                print(f"    export failed ({type(exc).__name__}: {exc}) — "
                      "screen data stands for this round.")
        else:
            sdir = self.exporter.wait_for_nmd(
                plan["sample_name"],
                preferred_dirs=(run_record.get("sample_dir"),))
            if sdir:
                run_record["sample_dir"] = sdir
                try:
                    csv_result = self.exporter.fetch_result_csv(
                        run_record,
                        review_changes=dict(
                            plan["method_plan"].review_changes),
                        open_dir=sdir)
                except Exception as exc:
                    print(f"    export failed ({type(exc).__name__}: {exc}) — the "
                          "indents ARE on the instrument; recover_pending() later, "
                          "no re-indenting.")
                    self.journal.append("EXPORT_FAILED", round=round_no,
                                        sample=plan["sample_name"], sample_dir=sdir,
                                        kind=type(exc).__name__, reason=str(exc)[:400])
            else:
                print("    screen data stands for this round — recover_pending() "
                      "once InView has written the file.")
                # A missing EXPORTED record used to be the ONLY trace of this, so
                # the journal could say a round fell back but never why. Say why.
                self.journal.append(
                    "EXPORT_MISSING", round=round_no, sample=plan["sample_name"],
                    waited_s=self.exporter.cfg.nmd_wait_s,
                    folders_probed=len(self.exporter.candidate_session_dirs()),
                    last_confirmed_sitting=self.exporter.sitting_dir,
                    reason=f"{plan['sample_name']}{self.exporter.cfg.sample_ext} never "
                           "appeared in any candidate sitting folder")
        if csv_result:
            self.journal.append("EXPORTED", sample=plan["sample_name"],
                                sample_dir=csv_result.get("dir"),
                                via=csv_result.get("via"),
                                files=sorted(csv_result.get("local_paths", {})))

        # ---- VALIDATE + RECORD ----
        res = R.finalize_indent_data(run_record, csv_result,
                                     screen_config=self.session.screen_config,
                                     plot_curves=self.c.show_frames)
        self.record_execution(res, plan, gate)

        if self.c.capture_after_indent and not self.c.use_simulator:
            _, aimg = self.session.capture_frame(
                fov_w_um=self.c.fov_w_um, fov_h_um=self.c.fov_h_um,
                show=self.c.show_frames, um_per_px=self.c.um_per_px,
                title=f"Round {round_no} — after '{plan['sample_name']}'",
                note="Post-indent frame at the measured site.")
            if aimg:
                self.state["image_history"].append(aimg)

        d = res["indent_data"]
        vis = ("" if arrival.get("status") == "skipped" else
               f", {arrival['error_um']:.1f} um from the chosen feature (verified "
               "in the image)" if arrival.get("error_um") is not None else
               ", arrival NOT verified in the image")
        print(f"    {d['area_name']}: H={d['hardness_mean']:.2f} GPa, "
              f"E={d['modulus_mean']:.1f} GPa  [source: {d.get('data_source')}]  "
              f"at ({d['position'][0]:.1f}, {d['position'][1]:.1f}) um, "
              f"{plan.get('position_error_um', 0.0):.1f} um from the requested "
              f"stage coordinate{vis}")
        return res

    # ------------------------------------------------------------- recording
    def record_execution(self, res: dict, plan: dict, gate) -> dict:
        """Append one executed array to state + journal + message log."""
        self.executions.append(res)
        d = res["indent_data"]
        src = d.get("data_source", "result_file_download")
        self.state["indent_history"].append(d)
        if res.get("image"):
            self.state["image_history"].append(res["image"])
        self.state["visited_target_keys"].append(target_visit_key(plan["target"]))
        self.state["current_position"] = d["position"]
        self.state["decision_history"].append({
            "cycle": len(self.state["decision_history"]) + 1,
            "decision_type": "indent_array",
            "tool_name": "starting_tests/start_test_normal",
            "approved_via": gate.gate_id,
            "arguments": {"sample": d["test_name"],
                          "grid": f"{d['grid_nx']}x{d['grid_ny']}",
                          "agent_features": plan[
                              "measurement_strategy"].agent_features,
                          "data_source": src},
            "result_summary": f"H={d['hardness_mean']:.2f}±{d['hardness_std']:.2f} "
                              f"GPa, E={d['modulus_mean']:.1f} GPa "
                              f"({d['valid_indent_count']} valid) [source: {src}]",
        })
        self.journal.append("RESULT_RECORDED", sample=d["sample_name"],
                            round=self.state.get("orchestrator_round", 0) + 1,
                            gate_id=gate.gate_id, indent_data=d)
        self._log("Instrument Executor", "Measurement Critic",
                  f"{d['area_name']}: H = {d['hardness_mean']:.2f} ± "
                  f"{d['hardness_std']:.2f} GPa, E = {d['modulus_mean']:.1f} GPa "
                  f"({d['valid_indent_count']}/{d['n_indents']} valid) [source: {src}]",
                  detail=f"data: {d['data_path']}")
        return d

    def replace_execution(self, res: dict) -> bool:
        """Slot a RECOVERED result in place of the screen-read entry for the
        same sample — never append a duplicate the GP would double-count."""
        d = res["indent_data"]
        name = d["sample_name"]
        hist = self.state["indent_history"]
        idx = next((i for i, e in enumerate(hist)
                    if e.get("sample_name") == name), None)
        if idx is None:
            print(f"  {name}: no earlier entry to replace — nothing changed.")
            return False
        old_src = hist[idx].get("data_source")
        hist[idx] = d
        ex_idx = next((i for i, e in enumerate(self.executions)
                       if e["indent_data"].get("sample_name") == name), None)
        if ex_idx is not None:
            self.executions[ex_idx] = res
        self.state["decision_history"].append({
            "cycle": len(self.state["decision_history"]) + 1,
            "decision_type": "data_source_upgrade",
            "tool_name": "process_review_file/download_file",
            "approved_via": "recovery",
            "arguments": {"sample": name, "from": old_src, "to": d["data_source"]},
            "result_summary": f"H={d['hardness_mean']:.2f} GPa "
                              f"[{old_src} -> {d['data_source']}]",
        })
        self.journal.append("RESULT_UPGRADED", sample=name,
                            from_source=old_src, to_source=d["data_source"],
                            indent_data=d)
        print(f"  {name}: state updated  {old_src} -> {d['data_source']}")
        return True

    # ------------------------------------------------------------ autonomous
    def run(self, policy, *, show: Optional[bool] = None) -> str:
        """The hands-free loop — the SAME propose/execute path as manual mode.
        `policy` must be a non-interactive approval policy (AutoApproval for
        the simulator, AutonomousHardwareApproval for real hardware)."""
        assert getattr(policy, "interactive", True) is False, \
            "runner.run() needs a non-interactive approval policy."
        if not self._preflighted:
            self.preflight()
        max_rounds = int(self.objective.constraints.max_rounds)
        stop_reason = "no rounds ran"
        round_log: list[dict] = []

        for _pass in range(max_rounds + 2):          # hard safety cap
            proposal = self.propose_round(show=show)
            if proposal.terminal:
                stop_reason = proposal.decision.action
                break
            if not proposal.plans:
                stop_reason = "no clear sites left"
                break
            try:
                results = self.execute_round(proposal, policy=policy)
            except PermissionError as exc:
                print(f"    HUMAN APPROVAL REQUIRED: {exc}")
                stop_reason = "human approval required for agent-selected recipe"
                self.journal.append("ROUND_ABORTED", round=proposal.round_no,
                                    kind="approval_required", reason=str(exc)[:600])
                break
            except StageSafetyError as exc:
                print(f"    STAGE SAFETY STOP: {exc}")
                stop_reason = "stage safety stop"
                self.journal.append("ROUND_ABORTED", round=proposal.round_no,
                                    reason=str(exc)[:300])
                break
            except InstrumentMethodError as exc:
                # The server answered; the automation method threw. Say so —
                # calling this a comms failure sends the operator to the network
                # when the problem is on the instrument's screen.
                print(f"    INSTRUMENT REJECTED A COMMAND: {exc}")
                stop_reason = f"instrument rejected {exc.tool or 'a command'}"
                self.journal.append("ROUND_ABORTED", round=proposal.round_no,
                                    kind="server_method_error", tool=exc.tool,
                                    status_code=exc.status_code,
                                    server_message=(exc.server_message or "")[:400],
                                    reason=str(exc)[:600])
                break
            except InstrumentCommError as exc:
                print(f"    INSTRUMENT COMMUNICATION STOP: {exc}")
                stop_reason = "instrument communication failed"
                self.journal.append("ROUND_ABORTED", round=proposal.round_no,
                                    kind="transport", reason=str(exc)[:300])
                break
            for res in results:
                d = res["indent_data"]
                pl = res["run_record"]["plan"]
                round_log.append({
                    "round": proposal.round_no, "sample": d["sample_name"],
                    "source": proposal.decision.proposal_source,
                    "miss (um)": pl.get("position_error_um"),
                    "H (GPa)": round(d["hardness_mean"], 2),
                    "E (GPa)": round(d["modulus_mean"], 1),
                    "data source": d.get("data_source")})
        else:
            stop_reason = "safety cap on passes"

        self.journal.append("CAMPAIGN_STOP", reason=stop_reason,
                            rounds=self.state.get("orchestrator_round", 0),
                            arrays=len(self.state["indent_history"]))
        print(f"\n{'=' * 78}\nCampaign finished: {stop_reason} after "
              f"{self.state.get('orchestrator_round', 0)} round(s), "
              f"{len(self.state['indent_history'])} array(s) total.")
        if round_log:
            from IPython.display import display
            display(pd.DataFrame(round_log))
        pending = [d["sample_name"] for d in self.state["indent_history"]
                   if d.get("needs_recovery")]
        if pending:
            print(f"Rounds still on screen data: {pending} — run "
                  "runner.recover_pending() (no new indents).")
        print("=" * 78)
        return stop_reason

    # -------------------------------------------------------------- recovery
    def recover_pending(self, *, wait_s: float = 60.0, extra_days: tuple = ()) -> dict:
        """Re-export every round that fell back to screen data. Places NO
        indents and moves nothing; works even after a kernel restart, using
        run records rebuilt from the journal."""
        pending = [d for d in self.state["indent_history"]
                   if d.get("needs_recovery")
                   or d.get("data_source") == "screen_read_fallback"]
        print(f"RECOVERY — {len(pending)} of {len(self.state['indent_history'])} "
              "round(s) still on screen data")
        lite = self.journal.run_records_lite()
        fixed, failed = [], []
        for d0 in pending:
            name = d0["sample_name"]
            print("\n" + "-" * 74 + f"\nRECOVER '{name}'")
            rec = self.session.run_records.get(name) or lite.get(name)
            if not rec:
                print(f"  no run record in memory OR journal for '{name}' — use "
                      "exporter.recover(name, sample_dir=...) manually.")
                failed.append(name)
                continue
            sdir = self.exporter.wait_for_nmd(
                name, preferred_dirs=(rec.get("sample_dir"),),
                wait_s=wait_s, extra_days=extra_days)
            if not sdir and not self.c.use_simulator:
                failed.append(name)
                continue
            if sdir:
                rec["sample_dir"] = sdir
            try:
                csv = self.exporter.fetch_result_csv(
                    rec,
                    review_changes=dict(
                        rec.get("review_changes")
                        or self.app_cfg.indent_defaults.review_changes),
                    open_dir=rec.get("sample_dir"))
            except Exception as exc:
                print(f"  export failed ({type(exc).__name__}: {exc}) — "
                      "screen data stands.")
                failed.append(name)
                continue
            res = R.finalize_indent_data(rec, csv,
                                         screen_config=self.session.screen_config,
                                         plot_curves=self.c.show_frames)
            if res["indent_data"]["data_source"] == "screen_read_fallback":
                print("  still screen data — the export did not validate.")
                failed.append(name)
            else:
                self.replace_execution(res)
                fixed.append(name)
        print("\n" + "=" * 74)
        print(f"RECOVERY done — upgraded {len(fixed)}: {fixed or '(none)'}")
        if failed:
            print(f"           still on screen data: {failed}")
        return {"fixed": fixed, "failed": failed}

    # ---------------------------------------------------------------- resume
    def resume_from_journal(self) -> int:
        """After a kernel restart: rebuild indent history (and lightweight
        executions) from the journal so the GP, budget, and recovery all pick
        up where the crashed session stopped."""
        records = self.journal.recorded_results()
        n = 0
        for d in records:
            if any(e.get("sample_name") == d.get("sample_name")
                   for e in self.state["indent_history"]):
                continue
            self.state["indent_history"].append(d)
            rows = pd.DataFrame(d.get("results") or [])
            self.executions.append({"indent_data": d, "rows": rows,
                                    "image": None, "tool_log": [],
                                    "monitor_status": {}, "recorder": None,
                                    "csv_result": None,
                                    "run_record": {"position": d.get("position")}})
            pos = d.get("position")
            if pos:
                self.state["current_position"] = tuple(pos)
            n += 1
        rounds = max((e.get("round") or 0)
                     for e in self.journal.events("ROUND_DONE")) \
            if self.journal.events("ROUND_DONE") else 0
        if rounds:
            self.state["orchestrator_round"] = int(rounds)
        self.state["imicro_queue_count"] = 0
        print(f"Resumed {n} array(s) from {self.journal.path.name}; "
              f"orchestrator_round={self.state.get('orchestrator_round', 0)}. "
              "NOTE: visited image-target keys are not rebuilt — the stage-space "
              "keepout still prevents re-measuring the same sites.")
        return n

    # ---------------------------------------------------------------- output
    def results_rows(self) -> pd.DataFrame:
        return R.combined_rows(self.executions)
