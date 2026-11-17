"""
ExportPipeline — the InView ReviewData/export contract, in one place.

Facts this module encodes (established on the real instrument, 7/29–8/03):

1. In CSV mode InView writes a FAMILY of files, never a bare one:
   <NAME>_Results.CSV, _SampleSummary.CSV, _Test<N>.CSV, _SampleInformation.CSV.
   In EXCEL mode it writes ONE WORKBOOK, <NAME>.xlsx, whose sheets carry the
   same tables (see results.read_inview_workbook). The two formats therefore
   differ in SHAPE, not just extension: in Excel mode there is no _Results file
   to probe for, and probing for one finds nothing — indistinguishable from an
   export that never ran. Probe the bare name FIRST when cfg.is_excel.
2. InView chooses the FOLDER, we only choose the NAME — `export_dir` reaches
   the Save As dialog only; the Export clicks land files beside the open sample.
3. /last_export stays stale — it is a diagnostic, never a gate.
4. The Open dialog is driven blind: a missing file leaves a modal box the
   automation cannot dismiss, so the .NMD is verified over HTTP FIRST.
5. /find_files may not be deployed; existence is then probed per-file with a
   streamed /download_file that is closed before any body transfers.
6. Right after an array finishes, InView has usually NOT written the .NMD yet
   — every export must wait for it (this is what silently broke notebook 5's
   adaptive rounds while round 1 "worked").
7. InView opens a NEW SITTING FOLDER FOR EACH TEST, so within one campaign the
   trailing number tracks the ROUND, not the session: round 1 lands in
   <profile>_<date>_1, round 5 in ..._5. Anything that probes a FIXED window of
   sitting numbers goes blind the moment a campaign outlives the window — which
   is exactly how test_05 of runs 0805_170826 and 0806_131828 fell back to
   screen data while rounds 1-4 exported cleanly (max_sitting was 4). Follow the
   last CONFIRMED sitting upward instead of trusting a constant.

Everything here is re-runnable: no motors, no indents.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import results as R


def _shape(df) -> str:
    """'9 rows x 12 cols', for a table that may be None or empty."""
    if df is None or getattr(df, "empty", True):
        return "0 rows"
    return f"{len(df)} rows x {len(df.columns)} cols"


@dataclass
class ExportConfig:
    """Where InView keeps sessions and how the export is driven."""
    shared_root: str = field(default_factory=lambda: os.environ.get(
        "NANOINDENT_EXPORT_DIR",
        r"C:\Users\Public\Documents\Nanomechanics\Profiles\Shared AENI",
    ))
    profile: str = "Practice"
    session_date: str = ""            # YYYYMMDD; "" -> today at construction
    test_number: int = 1              # the ONE knob: trailing sitting number
    sample_ext: str = ".NMD"
    # "csv" clicks File > Export > CSV in ReviewData; ANYTHING else ("excel",
    # "xlsx") clicks File > Export > Excel — the server normalizes with the same
    # rule (instrument_server.normalize_export_format), so the two ends cannot
    # disagree. Excel is the slower click on the instrument (the automation
    # waits 30 s after it versus 5 s for CSV) and needs openpyxl locally.
    export_format: str = "csv"
    # "_Results" is the per-indent H/E table (what the BO loop needs);
    # "_SampleSummary" is per-DEPTH-BIN and only a last resort (results.py
    # rejects it by shape). "_Test<N>" are the raw curves.
    want_suffixes: list = field(default_factory=lambda: ["_Results", "_SampleSummary"])
    curve_suffix: str = "_Test"
    call_timeout_s: float = 900.0     # process_review_file drives a GUI
    load_wait: int = 30
    recalc_wait: int = 30
    save_wait: int = 5
    # .NMD waiting (fact 6)
    nmd_wait_s: float = 180.0
    nmd_poll_s: float = 15.0
    # Blind fallback window for <profile>_<date>_1.._N (padded + not). This is
    # only the floor: candidate_session_dirs() extends it past the last sitting
    # actually confirmed, so a campaign longer than the window still resolves.
    # It was 4, which is why round 5 was the first round that could not export.
    max_sitting: int = 12
    sitting_lookahead: int = 4        # sittings to probe PAST the confirmed one
    use_simulator: bool = False
    sim_export_dir: str = "exports"

    def __post_init__(self):
        if not self.session_date:
            self.session_date = time.strftime("%Y%m%d")

    @property
    def is_excel(self) -> bool:
        return str(self.export_format).strip().lower() != "csv"

    @property
    def export_exts(self) -> list:
        """Extensions the export may land under, most likely first.

        A LIST, not a string: which Excel extension InView writes is its own
        choice (.xlsx on a modern InView, .xls on an older one), and probing the
        wrong one looks exactly like "the export never happened". Case does not
        need covering — the instrument is Windows and both /find_files and the
        download probe match case-insensitively.
        """
        return [".CSV"] if not self.is_excel else [".xlsx", ".xls"]

    def session_dir(self, test_number: Optional[int] = None, *, date: Optional[str] = None,
                    profile: Optional[str] = None, pad: bool = False) -> str:
        """Build the InView session folder. Only the trailing number changes.
        pad=False -> ..._1 (what InView actually writes); pad=True probed too."""
        n = self.test_number if test_number is None else int(test_number)
        d = date or self.session_date
        p = profile or self.profile
        tail = f"{n:02d}" if pad else f"{n}"
        return rf"{self.shared_root}\Samples\{p}\{p}_{d}_{tail}"


class ExportPipeline:
    """Drives ReviewData, locates the export family, downloads and parses it."""

    def __init__(self, session, config: ExportConfig, *, download_dir):
        self.session = session          # InstrumentSession
        self.cfg = config
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.open_dir: Optional[str] = None if config.use_simulator \
            else config.session_dir()
        self._sitting_cache: Optional[str] = None   # folder that worked last
        self._find_files_ok: Optional[bool] = None  # None = not tried yet

    # ------------------------------------------------------------ raw HTTP
    @property
    def _client(self):
        return self.session.toolkit.client

    def remote_exists(self, directory: str, filename: str, timeout: int = 120):
        """Does <directory>\\<filename> exist on the instrument PC?
        -> (bool, status_code). Streamed probe; nothing transfers (fact 5)."""
        c = self._client
        r = c.session.post(f"{c.base_url}/download_file",
                           json={"directory": directory, "filename": filename},
                           timeout=timeout, stream=True)
        try:
            return r.status_code == 200, r.status_code
        finally:
            r.close()

    def find_files(self, pattern: str, directory: Optional[str] = None, limit: int = 50,
                   timeout: int = 180):
        """POST /find_files -> newest-first list, or None when not deployed."""
        c = self._client
        try:
            r = c.session.post(f"{c.base_url}/find_files",
                               json={"directory": directory or self.cfg.shared_root,
                                     "pattern": pattern, "limit": limit},
                               timeout=timeout)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        try:
            body = r.json()
        except ValueError:
            return None
        return body.get("files") if isinstance(body, dict) else None

    def _find_files_status(self, directory: str, pattern: str = "*"):
        """('ok', files) | ('missing', None) | ('unavailable', None).
        Distinguishes "folder not there" from "endpoint never deployed"."""
        c = self._client
        try:
            r = c.session.post(f"{c.base_url}/find_files",
                               json={"directory": directory, "pattern": pattern,
                                     "limit": 50},
                               timeout=120)
        except Exception:
            return "unavailable", None
        if r.status_code == 200:
            try:
                return "ok", (r.json() or {}).get("files") or []
            except ValueError:
                return "unavailable", None
        if r.status_code in (403, 404):
            try:
                msg = str((r.json() or {}).get("message", "")).lower()
            except ValueError:
                msg = ""
            return ("missing", None) if ("not a directory" in msg or "outside" in msg
                                         or r.status_code == 404) else ("unavailable", None)
        return "unavailable", None

    def _call_auto_raw(self, method: str, timeout_s: Optional[float] = None, **kwargs):
        """POST /call_auto and surface the SERVER's own message + traceback."""
        c = self._client
        r = c.session.post(f"{c.base_url}/call_auto",
                           json={"method": method, "args": [], "kwargs": kwargs},
                           timeout=timeout_s or c.timeout)
        try:
            body = r.json()
        except ValueError:
            body = {}
        bad = r.status_code != 200 or (isinstance(body, dict)
                                       and (body.get("status") == "error"
                                            or body.get("ok") is False))
        if bad:
            msg = (body.get("message") or body.get("error") or r.text[:300]) \
                if isinstance(body, dict) else r.text[:300]
            tb = str(body.get("traceback") or "").strip().splitlines() \
                if isinstance(body, dict) else []
            detail = f"\n      server: {tb[-1].strip()}" if tb else ""
            raise RuntimeError(f"{method} -> HTTP {r.status_code}: {msg}{detail}")
        return body.get("result", body) if isinstance(body, dict) else body

    def last_export_diagnostic(self):
        """GET /last_export as a DIAGNOSTIC ONLY. Never gate on it (fact 3)."""
        try:
            c = self._client
            r = c.session.get(f"{c.base_url}/last_export", timeout=30)
            return r.json() if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as exc:
            return f"unavailable ({exc})"

    def _download_named(self, remote_dir: str, filename: str) -> Path:
        out = self._client.download_file(remote_dir, filename,
                                         local_dir=str(self.download_dir))
        return Path(out["local_path"] if isinstance(out, dict) else out)

    # ------------------------------------------------------- session folder
    @staticmethod
    def _sitting_index(directory: Optional[str]) -> Optional[int]:
        """Trailing sitting number of a session folder ( ..._4 -> 4, ..._04 -> 4 ).

        Bounded to 3 digits on purpose: a folder that ends in the DATE
        (<profile>_20260806) must not read as sitting 20,260,806.
        """
        if not directory:
            return None
        m = re.search(r"_(\d{1,3})$", str(directory).rstrip("\\/ "))
        return int(m.group(1)) if m else None

    @property
    def sitting_dir(self) -> Optional[str]:
        """The sitting folder that last answered a probe (diagnostics/journal)."""
        return self._sitting_cache

    def candidate_session_dirs(self, extra_days: tuple = ()) -> list[str]:
        """Every plausible sitting folder, MOST LIKELY FIRST.

        Ordering is load-bearing, not cosmetic: `remote_exists` stops at the
        first hit and this list is walked once per poll inside wait_for_nmd, so
        the common case must resolve in a couple of probes.

        Follows the last confirmed sitting upward (fact 7) before falling back to
        the blind 1..max_sitting window, and probes unpadded names ahead of
        padded ones because unpadded is what InView actually writes.
        """
        seen = (self._sitting_index(self._sitting_cache)
                or self._sitting_index(self.open_dir) or 0)
        dirs = [self._sitting_cache, self.open_dir, self.cfg.session_dir()]
        # Where the test that has just finished most likely lives: InView opened
        # a new sitting for it, one or more past the one we last confirmed.
        dirs += [self.cfg.session_dir(k)
                 for k in range(seen + 1, seen + 1 + max(0, self.cfg.sitting_lookahead))]
        hi = min(99, max(self.cfg.max_sitting, seen + self.cfg.sitting_lookahead))
        for pad in (False, True):
            for day in (self.cfg.session_date, *[d for d in extra_days if d]):
                dirs += [self.cfg.session_dir(k, date=day, pad=pad)
                         for k in range(1, hi + 1)]
        return [d for d in dict.fromkeys(dirs) if d]

    def discover_sample_dir(self, sample: str) -> Optional[str]:
        """ASK the instrument where <sample>.NMD is, instead of guessing names.

        One recursive /find_files call scoped to the profile folder. Returns None
        when the endpoint is not deployed (fact 5 — it is not, on the iMicro PC
        as of 2026-08-06, which is why every EXPORTED record reads `via: probe`),
        and then remembers that so a poll loop does not pay for it every pass.
        """
        if self.cfg.use_simulator or self._find_files_ok is False:
            return None
        nmd = f"{sample}{self.cfg.sample_ext}"
        hits = self.find_files(nmd, directory=rf"{self.cfg.shared_root}\Samples"
                                              rf"\{self.cfg.profile}",
                               limit=5, timeout=45)
        if hits is None:
            self._find_files_ok = False
            return None
        self._find_files_ok = True
        for h in hits:                                   # newest first
            if str(h.get("name", "")).lower() == nmd.lower() and h.get("directory"):
                self._sitting_cache = h["directory"]
                return h["directory"]
        return None

    def verify_session_dir(self, test_number: Optional[int] = None, *,
                           search_neighbours: bool = True) -> Optional[str]:
        """Confirm the sitting folder really exists on the instrument BEFORE any
        indenting; rebind open_dir so every later step follows. Returns the
        confirmed folder, or None (with a fix-it listing)."""
        if self.cfg.use_simulator:
            print("Simulator mode — no InView session folder to verify.")
            return None
        n = self.cfg.test_number if test_number is None else int(test_number)
        for pad in (False, True):
            cand = self.cfg.session_dir(n, pad=pad)
            state_, files = self._find_files_status(cand, f"*{self.cfg.sample_ext}")
            if state_ == "ok":
                print(f"session folder CONFIRMED (test {n}): {cand}")
                print(f"  contains {len(files)} {self.cfg.sample_ext} file(s)"
                      + (f" — newest: {files[0]['name']}" if files
                         else " (empty — a fresh sitting)"))
                self.open_dir = self._sitting_cache = cand
                return cand
            if state_ == "unavailable":
                print("/find_files is not deployed on the iMicro PC, so the folder "
                      "cannot be verified up front. Using the derived path and relying "
                      f"on the per-sample preflight:\n    {cand}")
                self.open_dir = cand
                return cand
        print(f">> No session folder for TEST_NUMBER = {n} on {self.cfg.session_date}:")
        print(f"     {self.cfg.session_dir(n)}")
        if search_neighbours:
            found = []
            for k in range(1, 13):
                for pad in (False, True):
                    cand = self.cfg.session_dir(k, pad=pad)
                    if self._find_files_status(cand, f"*{self.cfg.sample_ext}")[0] == "ok":
                        found.append((k, cand))
                        break
            if found:
                print("   These sittings DO exist — set test_number to one of:")
                for k, cand in found:
                    print(f"     test_number = {k}   ->  {cand}")
            else:
                print(f"   No sitting found for {self.cfg.session_date} at all. If "
                      "resuming an earlier day, set ExportConfig.session_date.")
        return None

    def result_ready(self, sample: str, *, preferred_dirs: tuple = ()) -> Optional[str]:
        """HAS INVIEW FINISHED THIS TEST? -> its folder, or None. One cheap probe.

        The only DEFINITIVE completion signal this instrument offers. Everything
        else the monitor can see is ambiguous: the OCR "Tests: N" counter is the
        CURRENT test index, so on a 1x1 array it reads 1 from the moment the test
        is queued and never changes again — it carries no completion information
        at all. Before 2026-08-05 every 1x1 round therefore ended by TIMING OUT
        on the 300 s stall rule (5m40s-6m50s per indent, every run, whatever the
        indent actually took) while reporting "counter reached 1/1".

        Deliberately probes only a HANDFUL of folders — the sitting that worked
        last, the configured open folder, and the one sitting past each (fact 7:
        the test now finishing was given a NEW folder, so last round's sitting is
        the wrong place to look, which is why every round after the first fell
        back to the screen heuristics). This runs every few seconds inside the
        monitor loop, so it must not walk the full candidate list; `wait_for_nmd`
        still does the exhaustive search afterwards.
        """
        if self.cfg.use_simulator:
            return None
        nmd = f"{sample}{self.cfg.sample_ext}"
        nxt = [self.cfg.session_dir(i + 1) for i in
               (self._sitting_index(self._sitting_cache),
                self._sitting_index(self.open_dir)) if i is not None]
        for d in dict.fromkeys([x for x in (*preferred_dirs, self._sitting_cache,
                                            self.open_dir, *nxt) if x]):
            try:
                ok, _code = self.remote_exists(d, nmd, timeout=20)
            except Exception:
                continue          # a probe failure is "don't know", not "not done"
            if ok:
                self._sitting_cache = d
                return d
        return None

    def wait_for_nmd(self, sample: str, *, preferred_dirs: tuple = (),
                     wait_s: Optional[float] = None,
                     extra_days: tuple = ()) -> Optional[str]:
        """Block until InView has written <sample>.NMD; return its folder or None.

        Asks the instrument outright when /find_files is deployed; otherwise
        probes candidate sittings, nearest-first, re-deriving the list each pass
        so a sitting confirmed mid-wait pulls the lookahead along with it
        (facts 6 and 7).
        """
        if self.cfg.use_simulator:
            return None
        nmd = f"{sample}{self.cfg.sample_ext}"
        wait_s = self.cfg.nmd_wait_s if wait_s is None else float(wait_s)
        deadline = time.time() + max(0.0, wait_s)
        dirs: list[str] = []
        while True:
            found = self.discover_sample_dir(sample)
            if found:
                print(f"    {nmd} found in {found}  (/find_files)")
                return found
            dirs = [d for d in dict.fromkeys(
                [d for d in preferred_dirs if d]
                + self.candidate_session_dirs(extra_days)) if d]
            for d in dirs:
                try:
                    ok, _code = self.remote_exists(d, nmd, timeout=30)
                except Exception as exc:
                    print(f"    probe failed on {d} ({type(exc).__name__}) — skipping")
                    continue
                if ok:
                    print(f"    {nmd} found in {d}")
                    self._sitting_cache = d
                    return d
            if time.time() >= deadline:
                print(f"    {nmd} never appeared in {len(dirs)} folder(s) "
                      f"after {wait_s:.0f}s. Highest sitting probed: "
                      f"{self._sitting_index(dirs[-1]) if dirs else '?'}; last "
                      f"confirmed sitting: {self._sitting_cache or '(none)'}.")
                return None
            print(f"    waiting for InView to write {nmd} "
                  f"({deadline - time.time():.0f}s left)")
            time.sleep(self.cfg.nmd_poll_s)

    # ------------------------------------------------------------ the export
    def fetch_result_csv(self, run_record: dict, *, review_changes: Optional[dict] = None,
                         open_dir: Optional[str] = None,
                         want_suffixes: Optional[list] = None,
                         fetch_curves: bool = True) -> Optional[dict]:
        """Post-process in ReviewData and bring the export family back.

        Places NO indents and moves nothing; safe to re-run after fixing a
        path, a wedged dialog, or the network. Returns None (never raises up)
        when nothing came back, so the caller falls back to screen data.
        """
        cfg = self.cfg
        sample = run_record["sample_name"]
        export_name = run_record["export_name"]      # FROZEN at placement
        sample_dir = open_dir or run_record.get("sample_dir") or self.open_dir
        exts = cfg.export_exts
        suffixes = list(want_suffixes or cfg.want_suffixes)
        eff_open_dir = None if cfg.use_simulator else sample_dir
        export_dir = (cfg.sim_export_dir if cfg.use_simulator
                      else (sample_dir or cfg.shared_root))

        print(f"EXPORT for '{sample}'  (export name '{export_name}')")

        # ---- preflight: does the .NMD really exist? (fact 4) ----
        if eff_open_dir:
            nmd = f"{sample}{cfg.sample_ext}"
            ok, code = self.remote_exists(eff_open_dir, nmd)
            if ok:
                print(f"  preflight   : OK  {nmd} is in {eff_open_dir}")
            elif code == 403:
                print(f"  preflight   : CANNOT VERIFY (403 — outside the shared root): "
                      f"{eff_open_dir}")
            else:
                print(f"  preflight   : FAILED — '{nmd}' is not in {eff_open_dir} "
                      f"(HTTP {code}). NOTHING was sent; no dialog is open on the "
                      "instrument. Fix the sitting folder and re-run.")
                return None

        # ---- drive ReviewData ----
        try:
            t0 = time.time()
            pr = self._call_auto_raw(
                "process_review_file", timeout_s=cfg.call_timeout_s,
                # [] = take the single visible ReviewData window; a name here is
                # substring-matched against window titles AND doubles as a file
                # name, so short names like "test_01" can drive the wrong window.
                sample_names=[],
                open_dir=eff_open_dir,
                current_sample_file_name=(None if eff_open_dir is None
                                          else f"{sample}{cfg.sample_ext}"),
                export_dir=export_dir,
                export_file_name=export_name,
                review_changes=review_changes or {},
                review_type=run_record.get("method_type", "eh"),
                export_format=cfg.export_format.lower(),
                load_wait=cfg.load_wait, recalc_wait=cfg.recalc_wait,
                save_wait=cfg.save_wait,
            )
            print(f"  review      : OK in {time.time() - t0:.0f}s  {pr}")
            run_record.setdefault("tool_log", []).append(
                {"tool": "process_review_file", "ok": True,
                 "arguments": f"sample={sample}, export_name={export_name}",
                 "elapsed_s": round(time.time() - t0, 3), "summary": str(pr)[:180]})
        except Exception as exc:
            print(f"  review      : FAILED — {exc}")
            print("                Check the INSTRUMENT SCREEN for a leftover modal "
                  "dialog and clear it (OK, then Cancel) before re-running.")
            return None

        # ---- locate the family: /find_files if deployed, else probe folders ----
        found: dict[str, tuple] = {}
        hits = None
        for ext in exts:
            hits_ext = self.find_files(f"{export_name}*{ext}")
            if hits_ext is None:            # endpoint not deployed — stop asking
                break
            hits = (hits or []) + hits_ext
        if hits is not None:
            print(f"  locate      : /find_files -> {len(hits)} match(es)")
            for h in hits:
                stem = Path(h["name"]).stem
                if stem.lower().startswith(export_name.lower()):
                    found[stem[len(export_name):]] = (h["directory"], h["name"])
        else:
            probe = list(dict.fromkeys(
                d for d in ([eff_open_dir, export_dir, cfg.shared_root]
                            + self.candidate_session_dirs()) if d))
            print("  locate      : /find_files not deployed — probing folders")
            # "" first in Excel mode: that export is a single workbook named
            # exactly <export_name>, so the family suffixes never match it.
            wanted = ([""] if cfg.is_excel else []) + suffixes + \
                [f"{cfg.curve_suffix}{i}"
                 for i in range(1, int(run_record.get("expected", 1)) + 1)]
            for d in probe:
                for suf in wanted:
                    if suf in found:
                        continue
                    for ext in exts:
                        fn = f"{export_name}{suf}{ext}"
                        try:
                            ok, _ = self.remote_exists(d, fn)
                        except Exception:
                            continue
                        if ok:
                            found[suf] = (d, fn)
                            break
                if found:
                    break

        if not found:
            print(f"  locate      : NOTHING named '{export_name}*{'/'.join(exts)}' "
                  "was found.")
            print(f"                /last_export says: {self.last_export_diagnostic()} "
                  "(expected stale — never gated on)")
            return None
        print(f"  found       : {sorted(found)}")

        # ---- download the per-indent summary (preferred) ----
        # The Excel export is ONE WORKBOOK named exactly <export_name>: its
        # "Results" sheet is the _Results table and its "Test <n>" sheets are
        # the _Test<n> curves, so a single download carries everything the CSV
        # family needs several files for.
        local_paths, summary_rows, summary_name, curves = {}, None, None, {}
        for suf in ([s for s in suffixes if s in found] or
                    ([""] if "" in found else [])):
            d, fn = found[suf]
            p = self._download_named(d, fn)
            local_paths[suf or "_workbook"] = str(p)
            summary_name = fn
            if p.suffix.lower() in R.EXCEL_EXTS:
                summary_rows, sheets = R.read_inview_workbook(p)
                summary_rows = R.normalize_result_columns(summary_rows)
                if fetch_curves:
                    curves = {i: R.normalize_result_columns(t)
                              for i, t in sheets.items()}
                print(f"  downloaded  : {fn} ({p.stat().st_size:,} bytes) -> "
                      f"Results {_shape(summary_rows)} + {len(sheets)} curve sheet(s)")
            else:
                summary_rows = R.normalize_result_columns(R.read_inview_table(p))
                print(f"  downloaded  : {fn} ({p.stat().st_size:,} bytes) -> "
                      f"{_shape(summary_rows)}")
            break

        # ---- download the per-indent curves for plots + fallback H/E ----
        # Skipped when a workbook already supplied them as sheets.
        import re as _re
        if fetch_curves and not curves:
            for suf, (d, fn) in sorted(found.items()):
                m = _re.fullmatch(rf"{_re.escape(cfg.curve_suffix)}(\d+)", suf)
                if not m:
                    continue
                p = self._download_named(d, fn)
                local_paths[suf] = str(p)
                curves[int(m.group(1))] = R.normalize_result_columns(R.read_inview_table(p))
        if curves:
            print(f"  curves      : {len(curves)} indent curve(s)")

        if summary_rows is not None and getattr(summary_rows, "empty", False):
            summary_rows = None            # an empty table is not a result
        if summary_rows is None and not curves:
            print("  >> the export ran but neither a summary nor a curve file came back.")
            return None

        return {"summary_rows": summary_rows, "summary_name": summary_name,
                "curves": curves, "dir": next(iter(found.values()))[0],
                "local_paths": local_paths, "export_name": export_name,
                "sample_name": sample,
                "via": "/find_files" if hits is not None else "probe",
                "process_result": pr}

    # ------------------------------------------------------------- recovery
    def recover(self, sample_name: str, *, run_record: Optional[dict] = None,
                sample_dir: Optional[str] = None, export_name: Optional[str] = None,
                review_changes: Optional[dict] = None, expected: int = 25,
                run_id: Optional[str] = None) -> Optional[dict]:
        """Re-export + download a sample that already ran. Places NO new indents.
        Works from a live run_record, a journal-rebuilt one, or from scratch
        (a sample placed by an earlier session)."""
        rec = dict(run_record or {})
        if not rec:
            rec = {"sample_name": sample_name, "expected": expected,
                   "method_type": "eh"}
        if export_name:
            rec["export_name"] = export_name
        rec.setdefault("export_name",
                       f"{sample_name}_{run_id or self.session.run_id}_export")
        if sample_dir:
            rec["sample_dir"] = sample_dir
        rec.setdefault("sample_dir", self.open_dir)
        rec.setdefault("expected", expected)
        return self.fetch_result_csv(rec, review_changes=review_changes,
                                     open_dir=rec.get("sample_dir"))
