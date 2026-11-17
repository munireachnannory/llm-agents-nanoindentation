"""
CampaignJournal — an append-only JSONL event log, one file per run.

This is the durable spine of the campaign. Every state transition writes one
line: gates, positioning, array placement, exports, recorded results, stops.
It replaces the kernel-bound RUN_RECORDS / executions globals of notebook 5:

* a kernel restart no longer orphans physical indents — the export names,
  sitting folders, and screen-CSV paths needed for recovery are on disk;
* "what happened and why" is greppable and diffable after the fact;
* the run report is a rendering of the journal, not a separate artifact.

Events are plain dicts; anything non-JSON (numpy scalars, Paths, DataFrames)
is coerced on the way in. NaN is written as null so every line stays valid.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


def _jsonable(obj: Any) -> Any:
    """Coerce to something json.dumps handles, NaN -> None."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):                 # numpy scalar
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    return str(obj)


class CampaignJournal:
    """Append-only JSONL journal for one campaign run."""

    def __init__(self, journal_dir: str | Path, run_id: str):
        self.dir = Path(journal_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = self.dir / f"campaign_{run_id}.jsonl"
        self._seq = sum(1 for _ in self._iter_lines()) if self.path.exists() else 0

    # ------------------------------------------------------------- writing
    def append(self, event: str, **payload) -> dict:
        rec = {"seq": self._seq, "ts": datetime.now().isoformat(timespec="seconds"),
               "run_id": self.run_id, "event": str(event), **_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, allow_nan=False, default=str) + "\n")
        self._seq += 1
        return rec

    # ------------------------------------------------------------- reading
    def _iter_lines(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue          # a torn line from a crash mid-write

    def events(self, event: Optional[str] = None) -> list[dict]:
        return [e for e in self._iter_lines()
                if event is None or e.get("event") == event]

    # ----------------------------------------------------------- semantics
    def used_sample_names(self) -> list[str]:
        return [e["sample"] for e in self.events("ARRAY_PLACED") if e.get("sample")]

    def next_sample_index(self, prefix: str) -> int:
        """1 + the highest N seen in '<prefix>_NN' across THIS journal — so a
        restarted kernel cannot mint a colliding sample name."""
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        best = 0
        for name in self.used_sample_names():
            m = pat.match(name)
            if m:
                best = max(best, int(m.group(1)))
        return best + 1

    def run_records_lite(self) -> dict[str, dict]:
        """sample -> minimal run_record rebuilt from ARRAY_PLACED events.
        Enough for ExportPipeline.recover() after a kernel restart (the live
        recorder object is gone; the screen CSV path is kept for reference)."""
        out: dict[str, dict] = {}
        for e in self.events("ARRAY_PLACED"):
            grid = str(e.get("grid") or "1x1")
            method_plan = e.get("method_plan") or {}
            try:
                nx, ny = (int(v) for v in grid.split("x"))
            except Exception:
                nx = ny = 1
            out[e["sample"]] = {
                "sample_name": e["sample"], "export_name": e.get("export_name"),
                "sample_dir": e.get("sample_dir"), "label": e.get("label", e["sample"]),
                "method_type": e.get("method_type", "eh"),
                "review_changes": method_plan.get("review_changes") or {},
                "grid_nx": nx, "grid_ny": ny,
                "spacing_um": float(e.get("spacing_um") or 0.0),
                "rotation_deg": float(e.get("rotation_deg") or 0.0),
                "expected": int(e.get("expected") or nx * ny),
                "position": tuple(e.get("position") or (0.0, 0.0)),
                "destination": tuple(e.get("position") or (0.0, 0.0)),
                "recorder": None, "image": None, "tool_log": [],
                "monitor_status": {}, "n_done": e.get("n_done"),
                "done_reason": e.get("done_reason"),
                "monitor_lost_connection": bool(e.get("monitor_lost_connection")),
                "run_id": e.get("run_id", self.run_id),
                "screen_csv": e.get("screen_csv"),
                "timestamp": e.get("ts"),
                "measurement_strategy": e.get("measurement_strategy") or {},
            }
        # later EXPORTED events know the sitting folder
        for e in self.events("EXPORTED"):
            rec = out.get(e.get("sample"))
            if rec and e.get("sample_dir"):
                rec["sample_dir"] = e["sample_dir"]
        return out

    def recorded_results(self) -> list[dict]:
        """The latest indent_data per sample, upgrades applied, in placement
        order — i.e. state['indent_history'] rebuilt from disk."""
        order: list[str] = []
        latest: dict[str, dict] = {}
        for e in self._iter_lines():
            if e.get("event") in ("RESULT_RECORDED", "RESULT_UPGRADED"):
                d = e.get("indent_data")
                if not isinstance(d, dict):
                    continue
                name = d.get("sample_name")
                if name not in latest:
                    order.append(name)
                latest[name] = d
        return [latest[n] for n in order]

    def pending_recovery_samples(self) -> list[str]:
        return [d["sample_name"] for d in self.recorded_results()
                if d.get("needs_recovery")
                or d.get("data_source") == "screen_read_fallback"]

    def method_plan_history(self):
        """One row per PLACED array: which method parameters (control mode,
        target depth/load, predicted binding limit, who decided each one, and
        why) actually reached the instrument. `journal.summary()` only counts
        events by type and `ApprovalGate.audit_table()` only shows gate
        approve/reject bookkeeping — neither surfaces parameter content, even
        though every ARRAY_PLACED event has carried the full resolved
        MethodPlan since Stage 0. This is the missing post-hoc view; the
        live per-round view is the approval gate card in Section 7."""
        import pandas as pd
        rows = []
        for e in self.events("ARRAY_PLACED"):
            mp = e.get("method_plan") or {}
            rows.append({
                "sample": e.get("sample"), "label": e.get("label"), "grid": e.get("grid"),
                "control_mode": mp.get("control_mode"),
                "target_depth_nm": mp.get("target_depth_nm"),
                "target_load_mN": mp.get("target_load_mN"),
                "predicted_binding_limit": mp.get("predicted_binding_limit"),
                "decided_by": mp.get("decided_by"),
                "rationale": mp.get("rationale"),
                "warnings": "; ".join(mp.get("warnings") or []),
                "ts": e.get("ts"),
            })
        return pd.DataFrame(rows)

    def measurement_strategy_history(self):
        """One concise row per proposed target, including skipped targets."""
        import pandas as pd
        rows = []
        for e in self.events("MEASUREMENT_STRATEGY_DECISION"):
            gp = e.get("grid_plan") or {}
            mp = e.get("method_plan") or {}
            rows.append({
                "target": e.get("target"),
                "action": e.get("action"),
                "source": e.get("source"),
                "confidence": e.get("confidence"),
                "grid": gp.get("grid"),
                "spacing_um": gp.get("spacing_um"),
                "rotation_deg": gp.get("rotation_deg"),
                "method_type": e.get("method_type"),
                "control_mode": mp.get("control_mode"),
                "target_depth_nm": mp.get("target_depth_nm"),
                "target_load_mN": mp.get("target_load_mN"),
                "agent_features": ", ".join(e.get("agent_features") or []),
                "human_approval_required": e.get("requires_human_approval"),
                "reason": e.get("reason"),
                "warnings": "; ".join(e.get("warnings") or []),
                "ts": e.get("ts"),
            })
        return pd.DataFrame(rows)

    def measurement_strategy_feature_history(self):
        """One row per feature decision: requested value, selected value, and why."""
        import pandas as pd
        rows = []
        for e in self.events("MEASUREMENT_STRATEGY_DECISION"):
            for feature, decision in (e.get("feature_decisions") or {}).items():
                if not isinstance(decision, dict):
                    continue
                rows.append({
                    "target": e.get("target"),
                    "feature": feature,
                    "mode": decision.get("mode"),
                    "requested_value": decision.get("requested_value"),
                    "selected_value": decision.get("selected_value"),
                    "decided_by": decision.get("decided_by"),
                    "adjusted": decision.get("adjusted", False),
                    "reason": decision.get("reason"),
                    "strategy_source": e.get("source"),
                    "ts": e.get("ts"),
                })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        ev = list(self._iter_lines())
        kinds: dict[str, int] = {}
        for e in ev:
            kinds[e.get("event", "?")] = kinds.get(e.get("event", "?"), 0) + 1
        stops = [e for e in ev if e.get("event") == "CAMPAIGN_STOP"]
        return {"path": str(self.path), "events": len(ev), "by_event": kinds,
                "samples": self.used_sample_names(),
                "pending_recovery": self.pending_recovery_samples(),
                "stop_reason": stops[-1].get("reason") if stops else None}

    # -------------------------------------------------------------- listing
    @classmethod
    def list_runs(cls, journal_dir: str | Path) -> list[dict]:
        d = Path(journal_dir)
        out = []
        for p in sorted(d.glob("campaign_*.jsonl")):
            run_id = p.stem.replace("campaign_", "", 1)
            out.append({"run_id": run_id, "path": str(p),
                        "size_kb": round(p.stat().st_size / 1024, 1),
                        "modified": time.strftime(
                            "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
        return out
