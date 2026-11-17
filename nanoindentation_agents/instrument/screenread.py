"""
Live screen reading — the safety net that captures H/E while an array runs.

Lifted from notebook 5 cell 12.1 (v2, 2026-08). The physics constraints it
encodes were all measured on the real instrument:

* On Advanced Dynamic E and H the InView panel shows dynamic H/E only during
  the loading RAMP and NaN at peak depth, so the per-indent estimate is the
  MEAN of all accepted ramp samples — the settled values exist only in the
  export file, which is why this is a fallback, never the primary source.
* The completion OCR reply is ONLY {"test_number", "raw_text"}; there is no
  status field. The counter is the CURRENT test index, not a completed count.
* A frame Tesseract cannot parse must not read as "back to zero".
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ScreenReadConfig:
    """Tunables for the live screen reader. Defaults are the values proven on
    the real iMicro across the 7/29–8/03 sessions."""

    # Which server UI-state keys hold each live metric (first match wins).
    metric_keys: dict = field(default_factory=lambda: {
        "depth":    ["DEPTH", "Depth", "Displacement"],   # nm
        "load":     ["LOAD", "Load", "Force"],            # mN
        "hardness": ["HARDNESS", "Hardness"],             # GPa
        "modulus":  ["MODULUS", "Modulus"],               # GPa
    })
    # Timing — the real ceiling is the get_all_ui_state round-trip latency.
    sample_interval_s: float = 0.0
    ocr_interval_s: float = 5.0
    print_interval_s: float = 5.0
    # Network resilience DURING the array: once start_test_normal has fired the
    # indents cannot be un-run, so a dropped VPN must not abort monitoring.
    net_grace_s: float = 180.0
    net_retry_s: float = 3.0
    # Completion detection. THE COUNTER CANNOT DETECT COMPLETION ON A 1x1 ARRAY:
    # "Tests: N" is the CURRENT test index, so it reads 1 as soon as the test is
    # queued and never changes again. The real signal is the result file (see
    # ExportPipeline.result_ready, wired in as place_array's `done_probe`);
    # everything below is the fallback for when that probe is unavailable, and
    # it now watches the DEPTH READOUT rather than the counter.
    stall_s: float = 300.0          # depth flat this long -> assume finished
    settle_quiet_s: float = 15.0    # counter at the end AND depth flat this long
    activity_depth_nm: float = 10.0  # depth change that counts as "still working"
    # How long H/E must stay INVALID before the indent whose ramp was being
    # watched is declared finished. Debounces the edge detector, nothing more:
    # InView blanks H/E to NaN at peak depth *within* an indent, so a burst
    # must not be closed on the first invalid sample. The gap BETWEEN indents
    # (retract + move + surface-find) is tens of seconds, so this is generous.
    indent_gap_s: float = 15.0
    # A new indent's loading ramp RESTARTS low. Depth climbs monotonically
    # through one indent's H/E ramp (149 -> 503 -> 967 nm on the real tool),
    # so a valid-H/E sample this much SHALLOWER than the previous one is the
    # next indent beginning — a timing-free transition marker.
    ramp_restart_drop_nm: float = 50.0
    complete_phrases: list = field(default_factory=list)
    # Two-stage detection + acceptance.
    accept_depth_frac: float = 0.10
    depth_reset_drop_nm: float = 400.0
    hardness_valid_range: tuple = (0.3, 100.0)
    # Where the continuously-flushed screen CSV lands.
    csv_dir: str = "."
    # Fallback target depth when the plan's setup does not carry one (nm).
    default_target_depth_nm: float = 1000.0


_TESTS_RE = re.compile(r"(?:^|[^#\w])Tests\s*[:;]\s*(\d+)", re.IGNORECASE | re.MULTILINE)


def parse_current_test_index(raw_text: Any) -> Optional[int]:
    """Pull the CURRENT test index out of a full-screen OCR dump.

    The InView header reads ``Sample: test_01 / Tests: 2`` — the index of the
    test running right now. The Sample Definition Panel separately shows a
    ``# TESTS`` column holding the array TOTAL (4). Both contain the word
    "TESTS", so the match is anchored on the colon the header uses and on NOT
    being preceded by '#', which is what distinguishes the two on screen.
    """
    if not isinstance(raw_text, str):
        return None
    hits = _TESTS_RE.findall(raw_text)
    if not hits:
        return None
    try:
        return int(hits[0])
    except (TypeError, ValueError):
        return None


class TestSequenceTracker:
    """Which test of the array is the instrument on RIGHT NOW?

    Reads the header counter, which does progress 1 -> 2 -> 3 -> 4 (confirmed
    by the 2026-08-18 screenshot: "Tests: 2" while the panel total read 4).
    Two traps make a naive read useless, and both burned this project:

    * STALE. When a new array is queued the header still shows the PREVIOUS
      array's final index, so an unguarded read says "4/4" before test 1 has
      even started. Worse, the old code latched that with ``if parsed >
      n_done``, so it could never come back down — every log therefore showed
      4/4 from the first sample. This tracker stays UNARMED until it has
      evidence this array really started: a reading of 1, a DECREASE from the
      stale value, or a physically observed indent.
    * NOISE. Single-frame OCR misreads. A value must repeat before it is
      trusted, and once armed the index may only step forward by one, unless
      a larger jump holds steady long enough to be real (slow polling can
      genuinely miss an index).
    """

    __test__ = False        # not a pytest class, despite the leading "Test"

    def __init__(self, expected: int, *, confirm_reads: int = 2,
                 jump_confirm_reads: int = 4):
        self.expected = max(1, int(expected))
        self.confirm_reads = confirm_reads
        self.jump_confirm_reads = jump_confirm_reads
        self.current: Optional[int] = None
        self.armed = False
        self.ever_parsed = False
        self._first_seen: Optional[int] = None
        self._cand: Optional[int] = None
        self._hits = 0

    @property
    def reached_end(self) -> bool:
        """The instrument is on the LAST test of the array (not yet done)."""
        return self.armed and self.current is not None and self.current >= self.expected

    def observe(self, parsed: Optional[int], *, physical_started: bool = False) -> None:
        if parsed is not None:
            self.ever_parsed = True
            v = int(parsed)
            if v == self._cand:
                self._hits += 1
            else:
                self._cand, self._hits = v, 1
            if self._hits >= self.confirm_reads:
                self._fold(v)
        # A ramp is physically running, so whatever the counter shows now
        # belongs to THIS array — but trust the physical evidence for the
        # index, not the possibly-stale number.
        if physical_started and not self.armed:
            self.armed = True
            self.current = 1

    def _fold(self, v: int) -> None:
        if self._first_seen is None:
            self._first_seen = v
        if not self.armed:
            if v <= 1 or v < self._first_seen:
                self.armed, self.current = True, max(1, min(v, self.expected))
            return
        if self.current is None:
            self.current = min(v, self.expected)
        elif v == self.current + 1:
            self.current = min(v, self.expected)          # normal advance
        elif v > self.current + 1 and self._hits >= self.jump_confirm_reads:
            self.current = min(v, self.expected)          # polling missed one


def screen_metric(ui_state: Any, aliases: list) -> float:
    """First finite numeric value among alias keys, else NaN."""
    if not isinstance(ui_state, dict):
        return float("nan")
    for k in aliases:
        if k in ui_state:
            try:
                return float(ui_state[k])
            except (TypeError, ValueError):
                continue
    return float("nan")


class LiveScreenRecorder:
    """Aggregate live DEPTH/HARDNESS/MODULUS samples into one AVERAGED record
    per indent, flushing a CSV continuously so a crash loses nothing."""

    _CSV_COLS = ["sample", "test_number", "grid_row", "grid_col", "grid_label",
                 "X (um)", "Y (um)", "Depth (nm)", "Load (mN)", "Hardness (GPa)",
                 "Modulus (GPa)", "n_samples", "timestamp", "validation_status",
                 "data_source"]

    def __init__(self, plan: dict, config: ScreenReadConfig, run_id: str):
        self.cfg = config
        self.sample = plan["sample_name"]
        self.nx, self.ny = int(plan["grid_nx"]), int(plan["grid_ny"])
        self.spacing = float(plan["spacing_um"])
        self.dest_x, self.dest_y = plan["destination"]
        self.n_indents = self.nx * self.ny
        setup = plan.get("setup") or {}
        self.target_depth = float(setup.get("target depth", config.default_target_depth_nm))
        self.accept_depth_nm = config.accept_depth_frac * self.target_depth
        self.h_lo, self.h_hi = config.hardness_valid_range
        self.records: dict[int, dict] = {}
        self.samples: list[dict] = []
        self._by_idx: dict[int, list] = {}
        self.started: dict[int, bool] = {}
        self.max_depth_seen: dict[int, float] = {}
        self.n_samples = 0
        # ---- per-indent completion tracking (see _fold_indent_progress) ----
        self._completed = 0            # indents whose H/E burst has CLOSED
        self._in_burst = False         # currently inside an indent's H/E ramp
        self._burst_peak = 0.0         # deepest finite depth seen in this burst
        self._last_valid_depth = None  # depth of the previous valid-H/E sample
        self._last_indent_t = 0.0      # monotonic time of the last valid H/E sample
        self.clock = time.monotonic    # injectable so tests can drive the debounce
        self._last_print = 0.0
        self.csv_path = Path(config.csv_dir) / f"{self.sample}_screenread_{run_id}.csv"

    # ------------------------------------------------------------- geometry
    def grid_position(self, idx: int) -> tuple[int, int, float, float]:
        """1-based indent number -> (row, col, x_um, y_um). Row-major raster."""
        i0 = idx - 1
        row, col = i0 // self.nx, i0 % self.nx
        gx = self.dest_x + (col - (self.nx - 1) / 2.0) * self.spacing
        gy = self.dest_y + (row - (self.ny - 1) / 2.0) * self.spacing
        return row, col, round(gx, 2), round(gy, 2)

    # ------------------------------------------------------------ acceptance
    @staticmethod
    def classify(depth: float, hardness: float, modulus: float) -> str:
        h_ok = math.isfinite(hardness) and hardness > 0
        e_ok = math.isfinite(modulus) and modulus > 0
        return "indentation" if (h_ok and e_ok) else "surface_finding"

    def _acceptable(self, idx, depth, hardness, modulus, stage):
        if not self.started.get(idx, False):
            return False, "indentation stage not started"
        if stage != "indentation":
            return False, "surface-finding (H/E NaN)"
        if not (math.isfinite(hardness) and hardness > 0):
            return False, "H not finite/positive"
        if not (math.isfinite(modulus) and modulus > 0):
            return False, "E not finite/positive"
        if not (math.isfinite(depth) and depth >= self.accept_depth_nm):
            return False, f"depth < {self.accept_depth_nm:.0f} nm"
        return True, "ok"

    @property
    def current_index(self) -> int:
        """1-based indent the recorder is currently attributing samples to."""
        return min(self._completed + 1, self.n_indents)

    @property
    def completed_indents(self) -> int:
        """How many indents have DEMONSTRABLY finished (closed H/E bursts)."""
        return self._completed

    @property
    def indent_in_progress(self) -> bool:
        """An indent's loading ramp is being watched right now."""
        return self._in_burst

    @property
    def saw_any_indent(self) -> bool:
        """Has a real (deep enough) indent ramp EVER been observed?

        Distinguishes "the H/E readout is blind" from "no indent yet", so the
        caller can tell an unavailable signal from a pending one.
        """
        return self._completed > 0 or self._burst_peak >= self.accept_depth_nm

    @property
    def grid_complete(self) -> bool:
        """True once every indent in the requested grid has finished.

        Scales to any nx*ny with no hard-coded count and no timer.
        """
        return self._completed >= self.n_indents

    def _fold_indent_progress(self, depth: float, stage: str) -> None:
        """Count indents from the ONE signal that marks them individually:
        each indent's ramp of VALID dynamic H/E.

        Why not the alternatives, all of which were tried on real hardware:

        * The OCR "Tests: N" counter reports the array's TOTAL planned count
          from the moment the array is queued, on grids of any size — it never
          progresses 1/2/3/4 (2026-08-18, two runs).
        * A large DEPTH DROP is not one-per-indent: InView runs surface-find
          approach ramps inside every indent, so depth falls by >400 nm
          several times per indent (345->1568->97.8, then 2206->149.2 in the
          same test). Counting those over-counted the grid and tripped
          "complete" during indent 2 of 4 — the bug this replaces.

        On Advanced Dynamic E and H the panel shows H/E ONLY during an
        indent's loading ramp, blanking to NaN at peak depth, and shows
        H=E=0.00 during surface-finding — so one indent is exactly one
        contiguous burst of `stage == "indentation"`. A burst closes when the
        tip has demonstrably left that indent: either depth returns below the
        acceptance threshold (retracted) or H/E stay invalid for
        `indent_gap_s` (debounce for the NaN-at-peak blank, which is brief,
        versus the tens of seconds between indents).

        Only bursts that actually reached the acceptance depth count, so an
        H/E flicker at 5 nm is not mistaken for an indent.
        """
        now = self.clock()
        if stage == "indentation":
            # The next indent's ramp restarting is itself the boundary: within
            # one indent depth climbs, so a valid sample this much shallower
            # than the last one closes the previous indent. Needs no timer,
            # and survives the screen reader missing the retraction entirely.
            if (self._in_burst and self._last_valid_depth is not None
                    and math.isfinite(depth)
                    and depth < self._last_valid_depth - self.cfg.ramp_restart_drop_nm):
                self._close_burst()
            if not self._in_burst:
                self._in_burst, self._burst_peak = True, 0.0
            self._last_indent_t = now
            if math.isfinite(depth):
                self._burst_peak = max(self._burst_peak, depth)
                self._last_valid_depth = depth
            return
        if not self._in_burst:
            return
        # No ramp is running. The tip pulling back out, or H/E staying invalid
        # long enough to rule out the brief NaN-at-peak blank, ends the indent
        # — this is what closes the LAST indent of the grid.
        retracted = math.isfinite(depth) and depth < self.accept_depth_nm
        if retracted or (now - self._last_indent_t) >= self.cfg.indent_gap_s:
            self._close_burst()

    def _close_burst(self) -> None:
        """End the current indent, counting it only if it was a real one."""
        self._in_burst = False
        self._last_valid_depth = None
        if self._burst_peak >= self.accept_depth_nm:
            self._completed += 1
        self._burst_peak = 0.0

    def observe(self, test_number, depth, load, hardness, modulus) -> str:
        """Fold one live screen sample in; return a short status string.

        `test_number` is the raw OCR reading — kept for the caller's
        completion logic, but NOT used to identify which indent this sample
        belongs to (see `_advance_indent_if_needed`)."""
        self.n_samples += 1
        stage = self.classify(depth, hardness, modulus)
        self._fold_indent_progress(depth, stage)
        idx = self.current_index

        prev_max = self.max_depth_seen.get(idx, 0.0)
        if math.isfinite(depth):
            if prev_max - depth >= self.cfg.depth_reset_drop_nm:
                self.max_depth_seen[idx] = depth
            else:
                self.max_depth_seen[idx] = max(prev_max, depth)
        if stage == "indentation":
            self.started[idx] = True

        accepted, reason = self._acceptable(idx, depth, hardness, modulus, stage)
        status = "accepted" if accepted else f"rejected ({reason})"
        if accepted:
            self.samples.append({
                "test_number": idx, "Depth (nm)": round(depth, 2),
                "Load (mN)": round(load, 4), "Hardness (GPa)": round(hardness, 4),
                "Modulus (GPa)": round(modulus, 3),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            self._by_idx.setdefault(idx, []).append((depth, load, hardness, modulus))
            self._recompute(idx)
            self._flush_csv()
        self._maybe_print(idx, depth, hardness, modulus, stage, status)
        return status

    def _recompute(self, idx: int) -> None:
        """Rebuild indent idx's record as the MEAN of its accepted samples."""
        pts = self._by_idx[idx]
        depths = [p[0] for p in pts]; loads = [p[1] for p in pts]
        hs = [p[2] for p in pts]; es = [p[3] for p in pts]
        row, col, gx, gy = self.grid_position(idx)
        self.records[idx] = {
            "sample": self.sample, "test_number": idx,
            "grid_row": row, "grid_col": col, "grid_label": f"r{row + 1}c{col + 1}",
            "X (um)": gx, "Y (um)": gy,
            "Depth (nm)": round(max(depths), 2),
            "Load (mN)": round(max(loads), 4),
            "Hardness (GPa)": round(float(np.mean(hs)), 4),
            "Modulus (GPa)": round(float(np.mean(es)), 3),
            "n_samples": len(pts),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "validation_status": "valid", "data_source": "screen_read_fallback",
        }

    def _maybe_print(self, idx, depth, hardness, modulus, stage, status) -> None:
        now = time.monotonic()
        if now - self._last_print >= self.cfg.print_interval_s:
            self._last_print = now
            n_acc = sum(len(v) for v in self._by_idx.values())
            print(f"  [screen] indent {idx:>2}/{self.n_indents} | "
                  f"D={depth:8.1f} nm  H={hardness:6.2f} GPa  E={modulus:8.1f} GPa | "
                  f"{stage:<15s} | {status:<28s} | samples: {n_acc} over "
                  f"{len(self.records)} indent(s)")

    def _flush_csv(self) -> None:
        rows = [self.records[k] for k in sorted(self.records)]
        pd.DataFrame(rows, columns=self._CSV_COLS).to_csv(self.csv_path, index=False)

    def to_result_rows(self) -> pd.DataFrame:
        """Repackage AVERAGED per-indent records into the canonical column shape."""
        cols = ["Sample", "Indent", "X (um)", "Y (um)", "Load (mN)", "Hardness (GPa)",
                "Modulus (GPa)", "Contact Depth (nm)", "Region (sim truth)"]
        out = []
        for k in sorted(self.records):
            r = self.records[k]
            out.append({
                "Sample": self.sample, "Indent": r["test_number"],
                "X (um)": r["X (um)"], "Y (um)": r["Y (um)"],
                "Load (mN)": r.get("Load (mN)"),
                "Hardness (GPa)": r["Hardness (GPa)"],
                "Modulus (GPa)": r["Modulus (GPa)"],
                "Contact Depth (nm)": r["Depth (nm)"],
                "Region (sim truth)": "screen_read",
            })
        return pd.DataFrame(out, columns=cols)
