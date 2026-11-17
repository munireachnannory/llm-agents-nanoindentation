"""Tests for the nanoindent instrument simulator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from instrument_control.nanoindent_simulator import NanoindentSimulator
from instrument_control.sim_server import create_app


@pytest.fixture
def sim() -> NanoindentSimulator:
    with tempfile.TemporaryDirectory() as tmp:
        yield NanoindentSimulator(data_dir=tmp)


def test_move_and_position(sim: NanoindentSimulator) -> None:
    sim.dispatch("move", amount=10, direction="right")
    pos = sim.dispatch("get_xyz_positions")
    assert pos["x"] == pytest.approx(1010.0)
    assert pos["y"] == pytest.approx(1000.0)


def test_indent_workflow(sim: NanoindentSimulator) -> None:
    sim.dispatch(
        "starting_tests",
        sample_name="test_area",
        setup={"poisson": 0.3, "target load": 500},
        indents={"X Indents": 2, "Y Indents": 2},
    )
    sim.dispatch("start_test_normal")
    for expected in (1, 2, 3, 4):
        r = sim.dispatch("get_current_sample_and_test_ocr_passive")
        assert r["test_number"] == expected
    assert sim.state.test.status == "complete"


def test_capture_and_export(sim: NanoindentSimulator) -> None:
    cap = sim.dispatch("capture_video_panel_remote")
    assert Path(cap["path"]).is_file()


def test_capture_at_negative_coordinates(sim: NanoindentSimulator) -> None:
    sim.state.x_um = -920.0
    sim.state.y_um = 1000.0
    cap = sim.dispatch("capture_video_panel_remote")
    assert Path(cap["path"]).is_file()

    sim.dispatch(
        "starting_tests",
        sample_name="export_test",
        setup={},
        indents={"X Indents": 1, "Y Indents": 1},
    )
    sim.dispatch("start_test_normal")
    sim.dispatch("get_current_sample_and_test_ocr_passive")
    out = sim.dispatch(
        "process_review_file",
        sample_names=["export_test"],
        open_dir="review",
        current_sample_file_name="export_test",
        export_dir="exports",
        export_file_name="export_test.csv",
        review_changes={"poissons ratio": 0.3},
    )
    assert Path(out["export_path"]).is_file()


def test_alignment_move(sim: NanoindentSimulator) -> None:
    sim.restore_alignment(
        area_name="Area1",
        new_origin_macro=[0, 0],
        new_origin_micro=[1000, 1000],
        macro_scale_x=0.1,
        macro_scale_y=0.1,
    )
    moved = sim.move_to_aligned_point(area_name="Area1", x=100, y=50)
    assert moved["x"] == pytest.approx(1010.0)
    assert moved["y"] == pytest.approx(1005.0)


def test_flask_call_auto(sim: NanoindentSimulator) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(data_dir=tmp)
        client = app.test_client()
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json["ok"] is True

        resp = client.post(
            "/call_auto",
            json={"function_name": "move", "kwargs": {"amount": 5, "direction": "up"}},
        )
        assert resp.status_code == 200
        body = resp.json
        assert body["ok"] is True
        assert body["result"]["y"] == pytest.approx(1005.0)
