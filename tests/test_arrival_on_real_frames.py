"""Regression test on the REAL frames of run 0805_130904.

That round was aborted by a false alarm: `register()` reported the scene had not
moved in x at all (0.0 um) for a stage move of -41.7 um, and the arrival check
"corrected" a stage that was in fact 0.2 um from its target to 49 um away.

The frames and the journal numbers are in the repo, so the failure is
reproducible rather than remembered. Skips cleanly if the frames are gone.
"""
import math
from pathlib import Path

import numpy as np
import pytest

FRAMES = Path(__file__).resolve().parents[1] / "notebooks/downloaded_data/arrival"
PRE, A0 = "0805_130904_round01_pre.png", "0805_130904_test_01_arrive0.png"
UMPP = 0.176056338028169

# From campaign_0805_130904.jsonl (POSITIONED / ARRIVAL_FAILED).
FRAME_POS = (15823.27, 58840.93)
ACHIEVED = (15781.55, 58790.11)
TARGET_IN_PRE = (146.29, 127.18)
STAGE_MOVE = (ACHIEVED[0] - FRAME_POS[0], ACHIEVED[1] - FRAME_POS[1])   # (-41.7, -50.8)

pytestmark = pytest.mark.skipif(
    not (FRAMES / PRE).exists() or not (FRAMES / A0).exists(),
    reason="the 0805_130904 arrival frames are not present")


def _crop(name):
    from PIL import Image
    from nanoindentation_agents.instrument import imagecal as ic
    return ic.crop_video_area(
        np.asarray(Image.open(FRAMES / name).convert("L"), dtype=float))


def test_register_recovers_the_true_scene_shift():
    """The move was (-41.7, -50.8) um; with gain -1 the scene must match it."""
    from nanoindentation_agents.instrument import imagecal as ic
    dy, dx, score = ic.register(_crop(PRE), _crop(A0))
    sx, sy = dx * UMPP, dy * UMPP
    assert abs(sx - STAGE_MOVE[0]) < 4.0, f"x: got {sx:+.1f}, want {STAGE_MOVE[0]:+.1f}"
    assert abs(sy - STAGE_MOVE[1]) < 6.0, f"y: got {sy:+.1f}, want {STAGE_MOVE[1]:+.1f}"
    assert score > 0.5


def test_the_arrival_error_that_round_was_sub_micron():
    """End to end on the real pixels: the site was ON the feature, not 49 um off."""
    from PIL import Image
    from nanoindentation_agents.instrument import imagecal as ic
    dy, dx, _ = ic.register(_crop(PRE), _crop(A0))
    now = (TARGET_IN_PRE[0] + dx * UMPP, TARGET_IN_PRE[1] + dy * UMPP)
    px = ic.find_crosshair(np.asarray(Image.open(FRAMES / A0).convert("RGB")))
    assert px is not None, "the burned-in indenter crosshair must be locatable"
    cross = (px[0] * UMPP, px[1] * UMPP)
    err = math.hypot(now[0] - cross[0], now[1] - cross[1])
    assert err < 5.0, f"arrival error {err:.1f} um — the check reported 49.2"


def test_the_old_unnormalised_peak_is_the_one_that_lied():
    """Guard the DIAGNOSIS, not just the fix: without overlap normalisation the
    correlation prefers a near-zero shift on this pair. If this ever stops being
    true, the reasoning in imagecal.register's docstring needs revisiting."""
    a, b = _crop(PRE), _crop(A0)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    H, W = a.shape[0] * 2, a.shape[1] * 2
    F = np.fft.rfft2(a, s=(H, W)) * np.conj(np.fft.rfft2(b, s=(H, W)))
    r = np.fft.fftshift(np.fft.irfft2(F, s=(H, W)))
    cy, cx = H // 2, W // 2
    m = 520
    sub = r[cy - m:cy + m, cx - m:cx + m]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    dx_old = -(ix - m) * UMPP
    assert abs(dx_old) < 10.0, (
        f"unnormalised x shift {dx_old:+.1f} um — expected it to collapse toward "
        "zero, which is the bias that caused the false alarm")


def test_trust_gate_rejects_the_measurement_that_caused_the_abort():
    from nanoindentation_agents.campaign.runner import CampaignConfig, CampaignRunner
    r = object.__new__(CampaignRunner)
    r.c = CampaignConfig(site_gain_x=-1.0, site_gain_y=-1.0)
    verdict, detail = r._trust_registration(STAGE_MOVE, (0.0, -24.5))
    assert verdict == "untrusted", detail
    # ...and accepts the true one.
    verdict, _ = r._trust_registration(STAGE_MOVE, (-41.9, -50.5))
    assert verdict == "ok"
