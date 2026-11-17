"""
MotionController — closed-loop stage jogs with PERSISTED calibration that the
campaign can enforce, plus a window-position guard.

THE MEASUREMENTS HAVE BEEN CONTRADICTORY, and that contradiction is itself
the important fact here, not any one number:

* An ISOLATED bench test (2026-08-03, nothing else running): dir_x=-1
  ("right" decreases X), dir_y=+1, gain EXACTLY 1.00 on both axes, zero
  cross-axis coupling. Repeatable across three separate probes.
* `calibrate()` run FROM INSIDE a live campaign session (2026-08-03/04,
  after Observe + context agents had already run): dir_x=+1 (the OPPOSITE
  sign), gain 1.14x/4.0-5.2x, and 85-169 um of CROSS-AXIS coupling — asking
  for "right" moved Y by 85-169 um. Reversing cost 100-160 um.

Cross-axis coupling that large on a single-axis command is not a stage
property changing with position; it is the signature of clicks landing on
the WRONG on-screen control — i.e. the InView RunTest window has drifted off
the screen coordinates the click tables assume. `window_check()` below tests
exactly that, using the server's `report_window_rects` diagnostic, and
`calibrate()` now refuses to run until it confirms the window is where the
click coordinates expect it.

Because a bad calibration is a "confidently wrong" failure mode (jog_to()
learns gains from whatever calibrate() gave it and cannot tell those gains
are lying), `pin_calibration()` lets the operator ADOPT numbers from a
trusted external measurement (e.g. the isolated bench test) instead of
re-running the possibly-contaminated in-session probe. Pinning skips the
PROBE, never the runtime safety net: jog_to() still detects an axis that
responds opposite to what was pinned and refuses to indent on it.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class StageSafetyError(RuntimeError):
    """The stage cannot be trusted to indent (inverted axis, lost position,
    uncalibrated on real hardware)."""


@dataclass
class MotionConfig:
    # DEFAULTS FROM THE ISOLATED BENCH TEST (8/03, stage driven with nothing else running):
    #   3 reads, no motion  -> identical to 0.01 um: the readback is stable, not noisy
    #   capture             -> (+0.00, +0.00): capturing does not move the stage
    #   "right" 20.0        -> (-20.00, +0.00)   twice in a row, exactly
    #   "right" 13.0        -> (-13.00, +0.00)
    #   "down"  5.1         -> (+0.00, -5.14)
    # So: X is INVERTED, Y is not, and the stage moves EXACTLY what it is asked (gain 1.00,
    # residual ~0.04 um). No over-travel, no excursion.
    #
    # Earlier defaults here (gain 2.4/2.8, approach 0.6, best_effort, no_reversal) were
    # inferred from CAMPAIGN logs, where large unexplained jumps appeared between jogs.
    # The bench test isolates the move and shows the move itself is clean, so those numbers
    # were wrong and are corrected below. Whatever produced the jumps happens around the
    # jogs, not in them — see the note in `calibrate()`. The safety rails (abort_um, the
    # divergence bail, refusing to indent when lost) are KEPT so a recurrence is caught.
    dir_x: int = -1          # measured: clicking "right" DECREASES stage X
    dir_y: int = +1          # measured: clicking "down" DECREASES stage Y
    gain_seed: dict = field(default_factory=lambda: {"x": 1.0, "y": 1.0})   # measured 1.00
    tol_um: float = 1.0      # achievable: one jog lands within ~0.05 um
    accept_um: float = 5.0   # indent anyway inside this; the readback is what gets recorded
    abort_um: float = 120.0  # beyond this the stage is lost -> stop, do NOT indent
    mode: str = "converge"        # "converge" | "best_effort"
    max_attempts: int = 3
    settle_t: int = 2        # `t` for the move tool. NEVER 0 — t=0 types into a
                             # dialog box that may not have focus yet.
    settle_s: float = 2.5    # WAIT after a move before reading the position back.
                             # `t` only paces the clicks INSIDE the Relative Move
                             # dialog; the call returns before the stage has
                             # arrived. THIS IS WHY THE TWO PROTOCOLS DISAGREED:
                             # the bench test that measured cleanly (gain 1.00,
                             # zero cross-talk, X sign -1) slept 3 s before every
                             # readback; calibrate() slept 0 and measured the X
                             # sign BACKWARDS three times running. A readback taken
                             # mid-flight is indistinguishable, in the numbers
                             # alone, from a genuinely inverted or badly-geared
                             # axis. Never read a position without settling first.
    min_step_um: float = 0.05
    max_jog_um: float = 150.0     # cap on ONE command; hitting it means the stage
                                  # is outside the search area — reported, not retried
    gain_limits: tuple = (0.25, 4.0)
    approach: float = 1.0    # the stage tracks 1:1, so ask for the whole residual
    no_reversal: bool = False     # NOT verified either way: the bench test never drove an
                                  # axis back the way it came. Blocking reversals would
                                  # make most sites unreachable, so they are allowed —
                                  # but a reversing jog is still excluded from gain
                                  # learning and from the axis-inversion verdict.

    @classmethod
    def for_simulator(cls) -> "MotionConfig":
        """The simulator's stage is ideal: converge normally."""
        return cls(dir_x=+1, dir_y=+1, gain_seed={"x": 1.0, "y": 1.0},
                   tol_um=0.5, accept_um=2.0, abort_um=500.0, mode="converge",
                   max_attempts=4, settle_t=0, settle_s=0.0, approach=1.0,
                   no_reversal=False)

    def __post_init__(self):
        if self.mode == "best_effort":
            self.max_attempts = min(self.max_attempts, 1) or 1


class MotionController:
    """Owns jog semantics + calibration state. All stage motion goes through
    here; nothing else in the codebase is allowed to call the move tool."""

    def __init__(self, session, config: Optional[MotionConfig] = None, *,
                 calibration_path: Optional[str | Path] = None,
                 use_simulator: bool = False):
        self.session = session
        self.use_simulator = use_simulator
        self.cfg = config or (MotionConfig.for_simulator() if use_simulator
                              else MotionConfig())
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.calibration: Optional[dict] = None
        # Which way each axis was driven LAST — across moves, not just within
        # one: backlash is a property of the stage.
        self._last_jog_dir = {"x": 0, "y": 0}
        if self.calibration_path and self.calibration_path.exists():
            try:
                self.calibration = json.loads(self.calibration_path.read_text())
            except Exception:
                self.calibration = None

    # ---------------------------------------------------------- calibration
    @property
    def calibration_age_h(self) -> float:
        if not self.calibration:
            return float("inf")
        try:
            ts = datetime.fromisoformat(self.calibration["ts"])
        except Exception:
            return float("inf")
        return (datetime.now() - ts).total_seconds() / 3600.0

    def window_check(self, *, required: bool = True) -> dict:
        """READ-ONLY: is the InView RunTest window where the click-coordinate
        tables assume it is?

        Every jog is a blind click at a fixed screen offset — there is no
        visual feedback loop. If the window has moved, resized, lost focus, or
        sits behind something else, clicks land on the WRONG control and the
        instrument does something unrelated to what was asked. That is
        indistinguishable, from the position readback alone, from a genuinely
        inverted or badly-geared axis — which is exactly the ambiguity behind
        the 8/03-8/04 measurements (see the module docstring). This is the one
        check that can actually tell the two apart, so `calibrate()` runs it
        before spending any probe jogs.

        ONLY A POSITIVE FAULT HALTS. Anything that leaves the answer UNKNOWN —
        the endpoint not deployed, a server error, different arguments, an
        unrecognised reply shape — returns ``ok: None`` and never raises. An
        earlier version tried to classify the failure by string-matching the
        error text; that misjudged this server's actual message and blocked
        calibration outright. Absence of evidence is not evidence of danger,
        and a guard that cannot be satisfied is a guard that gets switched off.
        Only a window we can SEE, and can see is misplaced or hidden, raises.
        """
        def _unknown(reason: str, extra: Any = None) -> dict:
            print(f"  [warn] window position UNKNOWN — {reason}")
            print("         Window drift cannot be ruled out, so this calibration is "
                  "tagged window_verified=false and rests on its own consistency "
                  "checks. Deploying report_window_rects on the iMicro server is "
                  "what would make this check real.")
            return {"ok": None, "available": False, "detail": extra}

        try:
            out = self.session.toolkit.call_tool("report_window_rects",
                                                 {"contains": "InView"})
        except Exception as exc:
            return _unknown(f"the call raised {type(exc).__name__}: {exc}"[:200])
        if not out.ok:
            return _unknown(f"report_window_rects did not answer: "
                            f"{str(out.error or '')[:200]}")

        data = out.result if isinstance(out.result, dict) else {}
        wins = data.get("windows")
        if not isinstance(wins, list):
            # Reply shape we do not recognise: that is UNKNOWN, not "no window".
            return _unknown(f"unexpected reply shape: {str(out.result)[:200]}")

        run_test = next((w for w in wins
                         if isinstance(w, dict)
                         and "runtest" in str(w.get("title", "")).lower()), None)
        if run_test is None:
            # Definitive: the tool listed the windows and RunTest is not among
            # them. Clicks WILL land somewhere else.
            msg = ("no 'InView RunTest' window found on screen "
                   f"(saw: {[str(w.get('title', '?'))[:40] for w in wins][:6]})")
            if required:
                raise StageSafetyError(msg + " — is InView open and not minimized?")
            print(f"  [warn] {msg}")
            return {"ok": False, "available": True, "windows": wins}

        vis = run_test.get("visible")
        on_cal = run_test.get("on_calibrated_monitor")
        if vis is None and on_cal is None:
            return _unknown("the reply does not report visibility or calibrated "
                            f"position for {str(run_test.get('title'))[:60]!r}",
                            run_test)
        ok = (bool(on_cal) if on_cal is not None else True) and \
             (bool(vis) if vis is not None else True)
        print(f"  window check: {str(run_test.get('title'))!r} "
              f"rect={run_test.get('rect')} visible={vis} "
              f"on_calibrated_monitor={on_cal}")
        if not ok and required:
            raise StageSafetyError(
                "InView RunTest is NOT at its calibrated screen position "
                f"(rect {run_test.get('rect')}, expected {data.get('calibrated_rect')}) "
                "or is hidden/not visible. Every click-based jog will be aimed at "
                "the wrong place until this is fixed on the instrument PC — bring "
                "the window to the front / restore its position, then retry.")
        return {"ok": ok, "available": True, "window": run_test,
                "expected": data.get("calibrated_rect")}

    _HOWTO = ("  Either:\n"
              "    motion.calibrate()          # 6 small jogs, NO indents, ~2 min\n"
              "    motion.pin_calibration(dir_x, dir_y, gain_x, gain_y, source=...)\n"
              "                                # adopt a trusted external measurement")

    def require_calibration(self, max_age_h: float = 12.0) -> None:
        """Raise unless a calibration that is both FRESH and DEFENSIBLE exists.

        Freshness is not correctness: a calibration measured an hour ago
        through clicks that were landing on the wrong control is more
        dangerous than no calibration at all, because jog_to() drives on it
        with confidence. So a record flagged with cross-talk is refused on its
        merits, not on the clock. The simulator's ideal stage is exempt.
        """
        if self.use_simulator:
            return
        age = self.calibration_age_h
        if age > max_age_h:
            raise StageSafetyError(
                "No fresh stage calibration "
                f"({'none on file' if not self.calibration else f'{age:.1f} h old'}, "
                f"max {max_age_h:g} h).\n" + self._HOWTO)
        cal = self.calibration or {}
        if cal.get("cross_talk_suspected"):
            raise StageSafetyError(
                "The stage calibration on file is REJECTED on its merits, not its "
                f"age ({age:.1f} h old): it recorded "
                f"{abs(float((cal.get('axes', {}).get('x', {}) or {}).get('other_axis_um', 0))):.0f}"
                " um of cross-axis motion from a single-axis command, which is the "
                "signature of clicks landing on the wrong control rather than a real "
                "stage property. Driving on it would aim every jog wrongly while "
                "reporting success.\n" + self._HOWTO)
        if cal.get("problems"):
            raise StageSafetyError(
                "The stage calibration on file recorded unresolved problems:\n    - "
                + "\n    - ".join(str(p) for p in cal["problems"]) + "\n" + self._HOWTO)

    def _move_and_measure(self, amount: float, direction: str, *, what: str) -> tuple:
        """One move -> SETTLE -> read. Returns the (dx, dy) actually observed.

        The single place a command is turned into a measured displacement.
        Settling is not optional: the move tool's `t` paces the clicks inside
        InView's Relative Move dialog and returns BEFORE the stage has
        arrived, so a readback taken immediately afterwards can catch the
        stage mid-flight — which is indistinguishable, in the numbers alone,
        from a wrong axis map. See MotionConfig.settle_s.
        """
        x0, y0 = self.session.read_stage_xy(what=f"{what} before")
        r = self.session.toolkit.call_tool(
            "move", {"amount": round(abs(float(amount)), 2), "direction": direction,
                     "t": max(self.cfg.settle_t, 1)})
        if not r.ok:
            raise StageSafetyError(f"move {amount} {direction} failed: {r.error}")
        if self.cfg.settle_s > 0:
            time.sleep(self.cfg.settle_s)
        x1, y1 = self.session.read_stage_xy(what=f"{what} after")
        return (x1 - x0, y1 - y0)

    def calibrate(self, probe_um: float = 20.0, apply: bool = True, *,
                  skip_window_check: bool = False,
                  max_cross_talk_um: Optional[float] = None,
                  strict: bool = True) -> dict:
        """Measure, per axis: which arrow moves which way, HOW FAR per um asked,
        and what a direction REVERSAL costs. Six small jogs, NO indents.

        Every probe move is measured INDIVIDUALLY (move -> settle -> read), so
        a single bad move is visible instead of being averaged into a two-move
        total where the axis responsible cannot be identified. This also makes
        the procedure identical to the isolated bench protocol that produced a
        clean, repeatable axis map.

        REFUSES TO SAVE A RESULT IT CANNOT DEFEND (strict=True). Earlier
        versions fitted numbers to whatever they observed and wrote them to
        disk with a confident fresh timestamp; jog_to() then drove on those
        numbers believing them, which is how a backwards X sign reached the
        stage three times running. Now an implausible probe raises instead:
        wrong is worse than missing, because missing is obvious.

        Checks the InView window position first when available (see
        `window_check`); if that endpoint is not deployed the probe still runs
        and the result records ``window_verified: false``.
        """
        wc: dict = {"ok": None, "available": False}
        if not skip_window_check and not self.use_simulator:
            wc = self.window_check(required=True)
        window_verified = bool(wc.get("ok"))

        tol = (float(max_cross_talk_um) if max_cross_talk_um is not None
               else max(15.0, 0.2 * probe_um))
        out: dict[str, Any] = {}
        problems: list[str] = []

        for axis, word, back in (("x", "right", "left"), ("y", "up", "down")):
            steps, net_x, net_y = [], 0.0, 0.0
            for k in (1, 2):          # two jogs the SAME way, each measured alone
                dx, dy = self._move_and_measure(probe_um, word,
                                                what=f"{axis} probe {k}")
                net_x += dx; net_y += dy
                on = dx if axis == "x" else dy
                cross = dy if axis == "x" else dx
                steps.append({"cmd_um": probe_um, "on_axis_um": round(on, 2),
                              "cross_axis_um": round(cross, 2),
                              "gain": round(abs(on) / probe_um, 3),
                              "sign": 1 if on > 0 else -1})
                print(f"  {axis} step {k}: '{word}' {probe_um:g} um -> {axis} "
                      f"{on:+.2f} um, cross {cross:+.2f} um "
                      f"(gain {steps[-1]['gain']:.2f}x)")
                if abs(cross) > tol:
                    problems.append(
                        f"{axis} step {k}: {abs(cross):.1f} um of CROSS-AXIS motion "
                        f"from a single-axis command (tolerance {tol:.0f} um)")

            # reversal, measured the same way
            rdx, rdy = self._move_and_measure(2 * probe_um, back,
                                              what=f"{axis} reversal")
            net_x += rdx; net_y += rdy
            back_err = math.hypot(net_x, net_y)

            g1, g2 = steps[0]["gain"], steps[1]["gain"]
            signs_agree = steps[0]["sign"] == steps[1]["sign"]
            gain_spread = abs(g1 - g2) / max(g1, g2, 1e-6)
            moved_total = sum(s["on_axis_um"] for s in steps)
            cross_total = sum(s["cross_axis_um"] for s in steps)

            out[axis] = {
                "probe_um": probe_um, "arrow": word, "steps": steps,
                "moved_um": round(moved_total, 2),
                "other_axis_um": round(cross_total, 2),
                "sign": steps[0]["sign"],
                "gain": round((g1 + g2) / 2.0, 3),
                "gain_spread_frac": round(gain_spread, 3),
                "signs_agree": signs_agree,
                "reversal_error_um": round(back_err, 2),
                "cross_talk_suspected": abs(cross_total) > tol,
            }
            print(f"  {axis}: sign {out[axis]['sign']:+d}, gain "
                  f"{out[axis]['gain']:.2f}x (spread {gain_spread * 100:.0f}%), "
                  f"reversal net error {back_err:.1f} um")

            if not signs_agree:
                problems.append(
                    f"{axis}: the two identical probe jogs moved in OPPOSITE "
                    "directions — the stage is not tracking commands")
            if gain_spread > 0.5:
                problems.append(
                    f"{axis}: the two identical probe jogs disagree on gain by "
                    f"{gain_spread * 100:.0f}% ({g1:.2f}x vs {g2:.2f}x)")
            if abs(moved_total) < 0.5:
                problems.append(f"{axis}: barely moved ({moved_total:+.2f} um) — is "
                                "the InView RunTest window frontmost?")

        # ---- refuse to publish a measurement we cannot defend ----------------
        if problems:
            detail = "\n    - ".join(problems)
            msg = ("Calibration REJECTED — the probe did not behave like a stage:\n"
                   f"    - {detail}\n"
                   + ("  The window check could not run (report_window_rects is not "
                      "deployed), so click drift cannot be ruled out.\n"
                      if not wc.get("available") else
                      "  The InView window checked out, so this is not window drift.\n")
                   + "  NOTHING was saved; the previous calibration is untouched.\n"
                     "  Options: raise MotionConfig.settle_s (a readback taken before "
                     "the stage settles produces exactly this), re-run with the "
                     "InView window frontmost, or adopt a trusted external "
                     "measurement with pin_calibration().")
            if strict:
                raise StageSafetyError(msg)
            print(msg)

        if apply:
            self.cfg.dir_x, self.cfg.dir_y = out["x"]["sign"], out["y"]["sign"]
            lo, hi = self.cfg.gain_limits
            self.cfg.gain_seed = {"x": max(lo, min(hi, out["x"]["gain"])),
                                  "y": max(lo, min(hi, out["y"]["gain"]))}
            print(f"\nAPPLIED: dir_x={self.cfg.dir_x:+d}, dir_y={self.cfg.dir_y:+d}, "
                  f"gain_seed={self.cfg.gain_seed}")

        self.calibration = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "dir_x": self.cfg.dir_x, "dir_y": self.cfg.dir_y,
            "gain_seed": dict(self.cfg.gain_seed),
            "source": "calibrate() probe",
            "window_verified": window_verified,
            "settle_s": self.cfg.settle_s,
            "cross_talk_suspected": any(out[a]["cross_talk_suspected"]
                                        for a in ("x", "y")),
            "problems": problems, "axes": out,
        }
        if self.calibration_path:
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            self.calibration_path.write_text(json.dumps(self.calibration, indent=2))
            print(f"calibration saved -> {self.calibration_path}")
        if not window_verified:
            print("NOTE: window_verified=false — report_window_rects was unavailable, "
                  "so this calibration is trusted on its own consistency alone.")
        worst = max(out["x"]["reversal_error_um"], out["y"]["reversal_error_um"])
        print(f"Worst reversal net error {worst:.1f} um "
              f"(no_reversal={self.cfg.no_reversal}); the RECORDED position next to "
              "every indent is always the live readback.")
        return out

    def apply_saved_calibration(self) -> bool:
        """Load dir/gain from the persisted calibration into the live config."""
        if not self.calibration:
            return False
        self.cfg.dir_x = int(self.calibration.get("dir_x", self.cfg.dir_x))
        self.cfg.dir_y = int(self.calibration.get("dir_y", self.cfg.dir_y))
        gs = self.calibration.get("gain_seed")
        if isinstance(gs, dict):
            self.cfg.gain_seed = {"x": float(gs["x"]), "y": float(gs["y"])}
        return True

    def pin_calibration(self, dir_x: int, dir_y: int, gain_x: float, gain_y: float, *,
                        source: str = "manual", note: str = "") -> dict:
        """Adopt TRUSTED numbers directly, WITHOUT running the probe jogs.

        Use this when you already have a clean external measurement (e.g. an
        isolated bench test with nothing else running) and don't want
        `calibrate()`'s in-session probe overwriting it — that probe has, on
        this instrument, twice measured something very different from the
        isolated test, most likely because of window drift (see the module
        docstring and `window_check()`).

        Satisfies `require_calibration()` immediately (writes a fresh
        timestamp). Does NOT touch jog_to()'s runtime safety net: if the
        pinned numbers are wrong, the first real jog still detects an axis
        responding opposite to what was pinned and refuses to indent on it
        (`StageSafetyError`) — pinning is a shortcut past the probe, not past
        the check that catches a wrong pin.
        """
        lo, hi = self.cfg.gain_limits
        self.cfg.dir_x, self.cfg.dir_y = int(dir_x), int(dir_y)
        self.cfg.gain_seed = {"x": max(lo, min(hi, float(gain_x))),
                              "y": max(lo, min(hi, float(gain_y)))}
        self.calibration = {"ts": datetime.now().isoformat(timespec="seconds"),
                            "dir_x": self.cfg.dir_x, "dir_y": self.cfg.dir_y,
                            "gain_seed": dict(self.cfg.gain_seed),
                            "source": source, "note": note}
        if self.calibration_path:
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            self.calibration_path.write_text(json.dumps(self.calibration, indent=2))
            print(f"calibration PINNED (not measured) -> {self.calibration_path}")
        print(f"PINNED: {self.summary()}")
        print("This bypasses the probe jogs, NOT the runtime safety net — jog_to() "
              "still refuses to indent if a move responds opposite to this pin.")
        return self.calibration

    # ---------------------------------------------------------------- jogs
    def _dir_word(self, axis: str, signed_amount: float) -> str:
        """Arrow that moves `axis` in the requested STAGE direction."""
        if axis == "x":
            pos, neg = ("right", "left") if self.cfg.dir_x > 0 else ("left", "right")
        else:
            pos, neg = ("up", "down") if self.cfg.dir_y > 0 else ("down", "up")
        return pos if signed_amount > 0 else neg

    def _send_jog(self, axis: str, signed_amount: float) -> bool:
        """One relative jog. Returns True if it REVERSED the axis vs the stage's
        last direction. Raises on a failed tool call (nothing indented yet)."""
        amount = round(abs(float(signed_amount)), 2)
        if amount < self.cfg.min_step_um:
            return False
        want = (signed_amount > 0) - (signed_amount < 0)
        reversed_now = bool(self._last_jog_dir[axis]) and want != self._last_jog_dir[axis]
        out = self.session.toolkit.call_tool(
            "move", {"amount": amount, "direction": self._dir_word(axis, signed_amount),
                     "t": max(self.cfg.settle_t, 0)})
        if not out.ok:
            raise StageSafetyError(
                f"move {amount} um ({self._dir_word(axis, signed_amount)}) failed: "
                f"{out.error}")
        self._last_jog_dir[axis] = want
        return reversed_now

    def jog_to(self, dest, *, label: str = "", tol_um: Optional[float] = None,
               max_attempts: Optional[int] = None, verbose: bool = True):
        """Closed-loop relative jog to an ABSOLUTE stage destination.

        Returns ``(achieved_xy, error_um, on_target, diag)``. Raises only when
        the stage position cannot be read or a move call fails. A stage that
        refuses to arrive is REPORTED — the caller decides whether that is
        good enough to indent.
        """
        c = self.cfg
        tol_um = c.tol_um if tol_um is None else float(tol_um)
        max_attempts = c.max_attempts if max_attempts is None else int(max_attempts)
        dest_x, dest_y = float(dest[0]), float(dest[1])

        x, y = self.session.read_stage_xy(
            what=f"stage position before moving to {label or 'site'}")
        err = math.hypot(dest_x - x, dest_y - y)
        gain = dict(c.gain_seed)
        diag = {"start": (x, y), "dest": (dest_x, dest_y), "attempts": [],
                "bailed": None, "inverted_axes": []}
        lo, hi = c.gain_limits
        sent_dir = {"x": 0, "y": 0}   # which way each axis has gone THIS move

        for attempt in range(1, max_attempts + 1):
            if err <= tol_um:
                break
            resid_x, resid_y = dest_x - x, dest_y - y
            cmd_x = c.approach * resid_x / gain["x"]
            cmd_y = c.approach * resid_y / gain["y"]
            capped = max(abs(cmd_x), abs(cmd_y)) > c.max_jog_um
            cmd_x = max(-c.max_jog_um, min(c.max_jog_um, cmd_x))
            cmd_y = max(-c.max_jog_um, min(c.max_jog_um, cmd_y))
            if capped and verbose:
                print(f"    residual ({resid_x:+.0f}, {resid_y:+.0f}) um exceeds the "
                      f"{c.max_jog_um:g} um cap — the stage is outside the search "
                      "area, not just off the site.")

            # NO REVERSALS: an axis that has moved one way this move stays put.
            if c.no_reversal:
                for axis, cmd in (("x", cmd_x), ("y", cmd_y)):
                    want = (cmd > 0) - (cmd < 0)
                    if sent_dir[axis] and want and want != sent_dir[axis]:
                        if verbose:
                            print(f"    {axis}: closing the last "
                                  f"{abs(cmd * gain[axis]):.1f} um would need a "
                                  "reversal — not worth ~80 um of backlash. "
                                  "Leaving this axis where it is.")
                        if axis == "x":
                            cmd_x = 0.0
                        else:
                            cmd_y = 0.0
                if abs(cmd_x) < c.min_step_um and abs(cmd_y) < c.min_step_um:
                    diag["bailed"] = "would need a reversal"
                    break

            rev = {"x": self._send_jog("x", cmd_x), "y": self._send_jog("y", cmd_y)}
            # SETTLE before reading back — the move call returns before the stage
            # has arrived (see MotionConfig.settle_s). Without this the readback
            # can catch the stage mid-flight, and the learned gain / inversion
            # verdict below would be computed from a position that is not final.
            if c.settle_s > 0 and (abs(cmd_x) >= c.min_step_um
                                   or abs(cmd_y) >= c.min_step_um):
                time.sleep(c.settle_s)
            for axis, cmd in (("x", cmd_x), ("y", cmd_y)):
                if abs(cmd) >= c.min_step_um:
                    sent_dir[axis] = (cmd > 0) - (cmd < 0)
            if any(rev.values()) and verbose:
                print(f"    (axis {'/'.join(a for a in ('x', 'y') if rev[a])} changed "
                      "direction since the last jog — expect a backlash excursion; "
                      "this jog is not used to learn the gain)")

            x1, y1 = self.session.read_stage_xy(
                what=f"stage position after moving to {label or 'site'}")
            moved_x, moved_y = x1 - x, y1 - y
            for axis, cmd, moved in (("x", cmd_x, moved_x), ("y", cmd_y, moved_y)):
                # A jog that reversed the axis has backlash mixed in — its
                # "gain" is meaningless (8/03: asked -4.9, moved +79.6).
                # Learn only from clean, same-direction jogs.
                if not rev[axis] and abs(cmd) >= 1.0 and abs(moved) >= 0.2:
                    g = moved / cmd                 # SIGNED: catches inversion
                    gain[axis] = math.copysign(min(hi, max(lo, abs(g))), g)
            x, y = x1, y1
            prev_err, err = err, math.hypot(dest_x - x, dest_y - y)
            diag["attempts"].append({
                "attempt": attempt, "cmd_x": round(cmd_x, 2),
                "moved_x": round(moved_x, 2), "cmd_y": round(cmd_y, 2),
                "moved_y": round(moved_y, 2), "err_before": round(prev_err, 2),
                "err_after": round(err, 2), "gain_x": round(gain["x"], 2),
                "gain_y": round(gain["y"], 2)})
            if verbose:
                print(f"    jog {attempt}/{max_attempts} {label}: asked "
                      f"({cmd_x:+.1f}, {cmd_y:+.1f}) um, moved "
                      f"({moved_x:+.1f}, {moved_y:+.1f}) um -> ({x:.1f}, {y:.1f}) um "
                      f"| {prev_err:.2f} -> {err:.2f} um from target")

            # THE INVERSION VERDICT ONLY EVER COMES FROM A CLEAN JOG — a
            # reversal is backlash, not an inversion, and reading it as one
            # is what stopped the 8/03 run.
            clean = [a for a, cmd in (("x", cmd_x), ("y", cmd_y))
                     if not rev[a] and abs(cmd) >= 1.0 and gain[a] < 0]
            if clean:
                diag["inverted_axes"] = sorted(set(diag["inverted_axes"]) | set(clean))
                if verbose:
                    print(f"    axis {'/'.join(clean)} responded OPPOSITE to a clean, "
                          "same-direction command — the direction map is wrong. Run "
                          "motion.calibrate() and retry.")

            if err > prev_err + 1.0 and attempt < max_attempts:
                # (+1 um so readback noise on a converged move is not divergence)
                diag["bailed"] = "error grew"
                if verbose:
                    print(f"    STOPPING this move: the error GREW "
                          f"({prev_err:.1f} -> {err:.1f} um).")
                break

        diag.update(achieved=(x, y), error_um=err, on_target=err <= tol_um,
                    gain=dict(gain))
        if verbose and not diag["on_target"] and diag["bailed"] is None:
            print(f"    {label or 'site'}: {err:.2f} um away after "
                  f"{len(diag['attempts'])} jog(s).")
        return (x, y), err, diag["on_target"], diag

    def summary(self) -> str:
        c = self.cfg
        cal = ("simulator (ideal stage)" if self.use_simulator else
               f"calibrated {self.calibration_age_h:.1f} h ago"
               if self.calibration else "NOT CALIBRATED")
        return (f"dir_x={c.dir_x:+d} (right {'increases' if c.dir_x > 0 else 'DECREASES'} x), "
                f"dir_y={c.dir_y:+d}, gain {c.gain_seed}, mode={c.mode}, "
                f"accept<={c.accept_um:g} um, abort>{c.abort_um:g} um | {cal}")
