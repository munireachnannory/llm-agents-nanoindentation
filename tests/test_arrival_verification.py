"""Behaviour tests for the arrival check, driven by a synthetic stage+scene."""
import math
import pytest

from nanoindentation_agents.campaign.runner import CampaignRunner, CampaignConfig
from nanoindentation_agents.instrument.imagecal import (
    image_delta_from_stage, stage_delta_from_image)
from nanoindentation_agents.instrument.motion import StageSafetyError

FOV_W, FOV_H = 208.98, 158.98
FRAME_POS = (15871.28, 58865.31)


class FakeJournal:
    def __init__(self): self.events = []
    def append(self, ev, **kw): self.events.append((ev, kw))
    def next_sample_index(self, prefix): return 1


class FakeMotion:
    """Perfect stage: goes exactly where told."""
    def __init__(self): self.jogs = []
    def jog_to(self, dest, *, label="", **kw):
        self.jogs.append(tuple(dest))
        return tuple(dest), 0.0, True, {"attempts": [1], "inverted_axes": []}


def make_runner(*, true_gain, cfg_gain=(-1.0, -1.0), mode="enforce", corrections=1):
    """A runner whose 'instrument' has `true_gain` while it is configured for
    `cfg_gain` — the exact failure the check exists to catch."""
    r = object.__new__(CampaignRunner)
    r.c = CampaignConfig(use_simulator=False, fov_w_um=FOV_W, fov_h_um=FOV_H,
                         um_per_px=0.176056, site_gain_x=cfg_gain[0],
                         site_gain_y=cfg_gain[1], verify_arrival=mode,
                         arrival_tol_um=8.0, max_visual_corrections=corrections)
    r.crosshair_um = (104.6, 76.6)
    r.motion = FakeMotion()
    r.journal = FakeJournal()

    # The "scene": each registration is measured against the PREVIOUS frame, so
    # the fake returns the shift for that LEG of the move, per true_gain.
    seen = {"ref": FRAME_POS}

    def fake_register(pre, post):
        cur = r.motion.jogs[-1] if r.motion.jogs else r._armed
        leg = (cur[0] - seen["ref"][0], cur[1] - seen["ref"][1])
        seen["ref"] = cur
        return {"status": "ok", "score": 0.6,
                "shift_um": (-leg[0] / true_gain[0], -leg[1] / true_gain[1])}
    r._register_against = fake_register
    r._capture_now = lambda: "frame.png"
    r._snapshot = lambda src, name: "snap.png"
    r._crosshair_from = lambda p: r.crosshair_um
    return r


def plan_for(target_image_xy):
    """A plan aimed, through the CONFIGURED map, at a feature at this image point."""
    return {"sample_name": "test_01", "label": "grain_boundary",
            "frame_path": __file__, "frame_pos": FRAME_POS,
            "target": {"x_frac": target_image_xy[0] / FOV_W,
                       "y_frac": target_image_xy[1] / FOV_H}}


def aim(runner, plan, target_image_xy):
    dest = runner._site_destination(plan["target"], FRAME_POS, source="image_targets")
    plan["destination"] = dest
    runner._armed = dest
    return dest


def test_overlay_and_aiming_agree():
    """The picture must place the site where the stage is actually sent."""
    r = make_runner(true_gain=(-1.0, -1.0))
    p = plan_for((55.3, 103.8))
    dest = aim(r, p, (55.3, 103.8))
    dx, dy = image_delta_from_stage(dest[0] - FRAME_POS[0], dest[1] - FRAME_POS[1],
                                    r.c.site_gain_x, r.c.site_gain_y)
    ref = r.image_reference_um()
    assert math.isclose(ref[0] + dx, 55.3, abs_tol=0.02)
    assert math.isclose(ref[1] + dy, 103.8, abs_tol=0.02)

    # The OLD overlay formula, for contrast: mirrored, ~98 um off.
    old_ix = (dest[0] - FRAME_POS[0]) + FOV_W / 2
    assert abs(old_ix - 55.3) > 90


def test_good_map_passes_without_correcting():
    r = make_runner(true_gain=(-1.0, -1.0))
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    achieved, rep = r._verify_arrival(p, dest, 1)
    assert rep["status"] == "ok" and rep["error_um"] <= 0.1
    assert r.motion.jogs == []                      # nothing to correct


def test_gain_error_is_corrected_then_passes():
    """Right sign, 30% wrong magnitude: correct once, land inside tolerance."""
    r = make_runner(true_gain=(-1.3, -1.3))
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    achieved, rep = r._verify_arrival(p, dest, 1)
    assert len(r.motion.jogs) == 1, "should have corrected exactly once"
    assert rep["error_um"] <= r.c.arrival_tol_um, rep
    assert achieved != dest


def test_wrong_sign_refuses_to_indent():
    """The 2026-08-05 hypothesis: a flipped X map must NEVER reach an indent."""
    r = make_runner(true_gain=(+1.0, -1.0))
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    with pytest.raises(StageSafetyError, match="sign is WRONG") as exc:
        r._verify_arrival(p, dest, 1)
    # The message must show WHICH axis and that the magnitude matched — that is
    # what separates a real sign error from a failed registration.
    assert "x: scene -49.3 um vs +49.3 expected" in str(exc.value)
    assert any(e[0] == "ARRIVAL_FAILED" for e in r.journal.events)


def test_warn_mode_reports_but_proceeds():
    r = make_runner(true_gain=(+1.0, -1.0), mode="warn")
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    achieved, rep = r._verify_arrival(p, dest, 1)     # must not raise
    assert rep["status"] == "ok"


def test_registration_that_disagrees_with_the_move_changes_nothing():
    """THE 2026-08-05 FALSE ALARM, as a test.

    The registration reported that the scene had not moved in x at all while the
    stage moved 41.7 um. The old code believed it, "corrected" a stage that was
    already 0.2 um from its target, drove it 49 um away, and then refused to
    indent at the place it had just broken. A measurement that contradicts the
    move that produced it must move NOTHING and stop NOTHING.
    """
    r = make_runner(true_gain=(-1.0, -1.0))
    r._register_against = lambda a, b: {"status": "ok", "score": 0.49,
                                        "shift_um": (0.0, -24.5)}
    p = plan_for((146.3, 127.2)); dest = aim(r, p, (146.3, 127.2))
    achieved, rep = r._verify_arrival(p, dest, 1)      # must NOT raise
    assert r.motion.jogs == [], "must not correct on an untrusted measurement"
    assert achieved == dest, "the stage must not have been moved"
    assert rep["status"] == "unknown" and "inconsistent" in rep["why"]
    assert not any(e[0] == "ARRIVAL_FAILED" for e in r.journal.events)


def test_scene_moving_several_times_too_far_is_untrusted_not_corrected():
    r = make_runner(true_gain=(-0.2, -0.2))            # scene moves 5x the command
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    achieved, rep = r._verify_arrival(p, dest, 1)
    assert r.motion.jogs == [] and rep["status"] == "unknown"


def test_unregisterable_field_warns_and_proceeds_even_in_enforce():
    """A check that cannot run must not brick the campaign."""
    r = make_runner(true_gain=(-1.0, -1.0))
    r._register_against = lambda a, b: {"status": "unknown", "why": "flat field"}
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    achieved, rep = r._verify_arrival(p, dest, 1)
    assert rep["status"] == "unknown" and achieved == dest


def test_out_of_tolerance_with_no_corrections_left_refuses():
    r = make_runner(true_gain=(-2.0, -2.0), corrections=0)
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    with pytest.raises(StageSafetyError, match="from the indenter axis"):
        r._verify_arrival(p, dest, 1)


def test_simulator_skips():
    r = make_runner(true_gain=(-1.0, -1.0))
    r.c.use_simulator = True
    p = plan_for((55.3, 103.8)); dest = aim(r, p, (55.3, 103.8))
    achieved, rep = r._verify_arrival(p, dest, 1)
    assert rep == {"status": "skipped"} and achieved == dest
