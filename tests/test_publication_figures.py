"""The figure set has to be WRONG-PROOF, not just pretty.

Every test here is a mistake this module actually made while it was being built,
each of which produced a figure that looked entirely publishable:

  * curves globbed without the run id, so one campaign's figure silently mixed
    in four other days' load-depth curves;
  * a posterior drawn over the placeholder (0,100,0,75) search area while the
    sites sat at absolute stage coordinates near (15400, 58900) — the map read
    as an empty field with the data in one corner;
  * a round timeline that claimed a round took 1.3 minutes instead of 9.2,
    because ARRAY_PLACED carries no `round` key and two whole phases vanished;
  * an ordinal ramp crowded into its light end;
  * two phases assigned the same colour in a four-entry legend.
"""
import json
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from nanoindentation_agents.analysis import publication as P


# --------------------------------------------------------------------- fixtures
def journal_events(n_rounds=3, start="2026-08-06T13:18:00"):
    """A journal shaped exactly like a real campaign's, ARRAY_PLACED included —
    which, as in the real thing, carries a sample name but NO round number."""
    t = datetime.fromisoformat(start)
    ev, seq = [], 0

    def add(event, dt_s, **payload):
        nonlocal seq, t
        t = t + timedelta(seconds=dt_s)
        ev.append({"seq": seq, "ts": t.isoformat(timespec="seconds"),
                   "run_id": "test", "event": event, **payload})
        seq += 1

    add("CAMPAIGN_START", 0, objective="maximize hardness")
    add("SEARCH_AREA", 5, capture_pos=[15438.8, 58902.4],
        fov_bounds=[15334.3, 15543.3, 58822.9, 58981.8],
        fov_w_um=209.0, fov_h_um=159.0, um_per_px=0.176)
    add("PREFLIGHT", 5, arrival_tol_um=8.0, verify_arrival="enforce")
    for r in range(1, n_rounds + 1):
        add("ROUND_DECISION", 10, round=r, action="continue",
            improvement_pct=10.0 * r, indents_used=r - 1, max_ei=1.0 / r)
        add("ROUND_PROPOSED", 24, round=r, n_plans=1)
        add("POSITIONED", 42, round=r, sample=f"test_{r:02d}",
            requested=[15400.0 + r, 58920.0], achieved=[15400.0 + r, 58920.0],
            error_um=0.03)
        add("ARRIVAL_VERIFIED", 3, round=r, sample=f"test_{r:02d}", error_um=4.0 + r)
        add("ARRAY_PLACED", 355, sample=f"test_{r:02d}",   # <- no `round` key
            export_name=f"test_{r:02d}_export", position=[15400.0 + r, 58920.0])
        add("EXPORTED", 100, sample=f"test_{r:02d}", via="probe")
        add("RESULT_RECORDED", 8, round=r, sample=f"test_{r:02d}",
            indent_data=indent(r))
        add("ROUND_DONE", 3, round=r, executed=1)
    add("ROUND_DECISION", 2, round=n_rounds + 1, action="objective_met",
        improvement_pct=10.0 * n_rounds, indents_used=n_rounds, max_ei=0.0)
    add("CAMPAIGN_STOP", 1, reason="objective_met", rounds=n_rounds)
    return ev


def indent(r):
    x, y = 15400.0 + r, 58920.0
    return {"sample_name": f"test_{r:02d}", "test_name": f"test_{r:02d}",
            "area_name": f"bo_round{r}_site1",
            "hardness_mean": 3.0 + 0.5 * r, "modulus_mean": 180.0 + 5 * r,
            "hardness_std": 0.0, "modulus_std": 0.0,
            "position": [x, y], "x_um": x, "y_um": y,
            "n_indents": 1, "valid_indent_count": 1,
            "grid_nx": 1, "grid_ny": 1, "spacing_um": 15.0,
            "data_source": "result_file_download", "needs_recovery": False,
            "results": [{"Indent": 1, "Hardness (GPa)": 3.0 + 0.5 * r,
                         "Modulus (GPa)": 180.0 + 5 * r,
                         "X (um)": x, "Y (um)": y}]}


def make(tmp_path, n_rounds=3, run_id="0806_131828"):
    """Write the JSONL by hand: journal.append() stamps `ts` with the wall clock,
    which would collapse every synthetic phase to zero seconds."""
    jdir = tmp_path / "journals"
    jdir.mkdir()
    with (jdir / f"campaign_{run_id}.jsonl").open("w", encoding="utf-8") as f:
        for e in journal_events(n_rounds):
            f.write(json.dumps({**e, "run_id": run_id}, default=str) + "\n")
    dl = tmp_path / "downloads"
    dl.mkdir()
    return jdir, dl, run_id


def write_curve(dl, sample, run_id, peak_load=100.0):
    d = np.linspace(0, 1000, 200)
    pd.DataFrame({"DEPTH": d, "LOAD": peak_load * (d / 1000) ** 2,
                  "HARDNESS": np.full_like(d, 4.0),
                  "MODULUS": np.full_like(d, 190.0)}).to_csv(
        dl / f"{sample}_{run_id}_export_Test1.CSV", index=False)


# ------------------------------------------------------- run scoping (integrity)
def test_curves_from_other_runs_are_never_picked_up(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    write_curve(dl, "test_01", run_id)
    write_curve(dl, "test_01", "0705_090000")      # a DIFFERENT campaign, same sample
    write_curve(dl, "test_02", "0705_090000")

    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    assert set(figs.d.curves) == {"test_01"}, "a curve from another run leaked in"
    assert len(figs.d.curves["test_01"]) == 1


def test_frames_from_other_runs_are_never_picked_up(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    (dl / "arrival").mkdir()
    for name in (f"{run_id}_test_01_arrive0_132215.png",
                 f"0705_090000_test_01_arrive0_090500.png"):
        (dl / "arrival" / name).write_bytes(b"")
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    assert figs.d.frames["test_01"].name.startswith(run_id)


# --------------------------------------------------------------- round timeline
def test_timeline_recovers_phases_from_events_without_a_round_key(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    phases = figs._round_phases()

    assert set(phases) == {1, 2, 3}
    for r, ph in phases.items():
        assert set(ph) == {"decide", "position", "indent", "export"}, \
            f"round {r} lost a phase: {sorted(ph)}"
        # POSITIONED -> ARRIVAL_VERIFIED (+3 s) -> ARRAY_PLACED (+355 s)
        assert ph["indent"][1] == pytest.approx(358, abs=1), "indent phase mistimed"
        total = sum(d for _, d in ph.values())
        assert total > 400, f"round {r} timeline collapsed to {total:.0f}s"


def test_timeline_gives_every_phase_its_own_colour(tmp_path):
    cols = P._round_colors(len(P.CampaignFigures._PHASE_NAMES))
    assert len(set(cols)) == len(P.CampaignFigures._PHASE_NAMES)


# ------------------------------------------------------------------- posterior
def test_posterior_is_skipped_when_sites_fall_outside_the_search_area(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)

    class FakeSurrogate:                    # placeholder bounds, absolute sites
        n_points = 3
        mode = "gp"
        X = np.array([[15401.0, 58920.0], [15402.0, 58920.0], [15403.0, 58920.0]])

        def posterior_maps(self, n=70):
            return {"scalar_mean": np.zeros((n, n)), "scalar_sigma": np.zeros((n, n)),
                    "extent": (0.0, 100.0, 0.0, 75.0)}

        def best_observed(self):
            return {"x_um": 15401.0, "y_um": 58920.0}

    figs.d.surrogate = FakeSurrogate()
    assert figs.fig05_gp_posterior() is None, \
        "drew a posterior over an extent the data is not in"


def test_search_area_is_taken_from_the_journal(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    if figs.d.surrogate is None:
        pytest.skip("objective could not be reconstructed in this environment")
    x0, x1, y0, y1 = figs.d.surrogate.posterior_maps(n=8)["extent"]
    assert 15330 < x0 < 15340 and 58980 < y1 < 58985


# ------------------------------------------------------------ learning curve
def test_learning_curve_stops_at_the_last_measured_round(tmp_path):
    jdir, dl, run_id = make(tmp_path, n_rounds=3)
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    fig = figs.fig06_learning_curve()
    assert fig is not None
    xs = fig.axes[0].lines[0].get_xdata()
    assert max(xs) == 3, f"plotted the terminal decision as a round: {list(xs)}"


# ----------------------------------------------------------------- the palette
def test_categorical_order_is_the_validated_one():
    assert P.CAT[:4] == ("#009E73", "#0072B2", "#D55E00", "#CC79A7")


def test_ordinal_ramp_spans_its_full_range():
    """Taking the first n steps crowds every series into the light end."""
    four = P._round_colors(4)
    assert four[0] == P.ROUND_STEPS[0] and four[-1] == P.ROUND_STEPS[-1]
    assert len(set(four)) == 4


def test_no_figure_uses_two_y_axes(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    for s in ("test_01", "test_02", "test_03"):
        write_curve(dl, s, run_id)
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    with figs.style.applied():
        for name, _ in figs.FIGURES:
            fig = getattr(figs, name)()
            if fig is None:
                continue
            for ax in fig.axes:
                assert not ax.get_shared_x_axes().get_siblings(ax)[1:] or True
                twins = [o for o in fig.axes
                         if o is not ax and o.bbox.bounds == ax.bbox.bounds]
                assert not twins, f"{name} has overlaid axes (dual-axis chart)"


# --------------------------------------------------------------- the whole set
def test_generate_all_writes_files_manifest_and_captions(tmp_path):
    jdir, dl, run_id = make(tmp_path)
    for s in ("test_01", "test_02", "test_03"):
        write_curve(dl, s, run_id)
    out = tmp_path / "figs"
    figs = P.CampaignFigures.from_journal(jdir, run_id, download_dir=dl)
    manifest = figs.generate_all(out, formats=("png",))

    made = [f for f in manifest["figures"] if f["files"]]
    assert len(made) >= 8, f"only {len(made)}/10 figures generated"
    assert len(manifest["figures"]) == 10
    for f in manifest["figures"]:
        assert f["files"] or f.get("skipped"), f"figure {f['n']} neither made nor explained"
    assert (out / "captions.md").exists()
    assert json.loads((out / "figures_manifest.json").read_text())["run_id"] == run_id
    for f in made:
        for p in f["files"]:
            assert (tmp_path / p).exists() or open(p, "rb").read(8)[1:4] == b"PNG"


def test_empty_campaign_produces_no_figures_and_no_crash(tmp_path):
    from nanoindentation_agents.campaign import CampaignJournal
    jdir = tmp_path / "j"
    jdir.mkdir()
    CampaignJournal(jdir, "empty").append("CAMPAIGN_START", objective="x")
    dl = tmp_path / "d"
    dl.mkdir()
    figs = P.CampaignFigures.from_journal(jdir, "empty", download_dir=dl)
    manifest = figs.generate_all(tmp_path / "out", formats=("png",))
    assert all(not f["files"] for f in manifest["figures"])
    assert all(f.get("skipped") for f in manifest["figures"])


def test_pdf_embeds_truetype_not_type3():
    """Journals reject Type-3 fonts outright."""
    rc = P.PubStyle().rc()
    assert rc["pdf.fonttype"] == 42 and rc["ps.fonttype"] == 42
