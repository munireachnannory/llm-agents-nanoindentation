"""Registration + trust gate against every real frame pair on disk.

Two campaigns were stopped by false alarms on 2026-08-05, both from a
registration that was confidently wrong:

  run 0805_130904  a (-41.7, -50.7) um move read as (0.0, -24.5)  -> "49 um off"
  run 0805_160842  a ( -5.8,  +6.1) um move read as (+96.1, -22.2) -> "113 um off"

Both pairs are in notebooks/downloaded_data/arrival/. Skips if they are gone.
"""
import math
from pathlib import Path

import numpy as np
import pytest

FRAMES = Path(__file__).resolve().parents[1] / "notebooks/downloaded_data/arrival"
UMPP = 0.176056338028169
SLACK = 8.0          # um; how close the measurement must land to the command

# (pre frame, post frame, stage move that separates them) — from the journals.
PAIRS = [
    ("0805_130904_round01_pre", "0805_130904_test_01_arrive0", (-41.7, -50.8)),
    ("0805_160842_test_01_arrive0", "0805_160842_test_01_arrive1", (-9.8, -5.2)),
    ("0805_160842_test_04_arrive0", "0805_160842_test_04_arrive1", (-5.8, 6.1)),
]

pytestmark = pytest.mark.skipif(
    not FRAMES.exists(), reason="no saved arrival frames")


def _crop(name):
    from PIL import Image
    from nanoindentation_agents.instrument import imagecal as ic
    p = FRAMES / f"{name}.png"
    if not p.exists():
        pytest.skip(f"{name}.png not on disk")
    return ic.crop_video_area(np.asarray(Image.open(p).convert("L"), dtype=float))


@pytest.mark.parametrize("pre,post,move", PAIRS)
def test_registration_matches_the_move_that_produced_it(pre, post, move):
    """With site_gain -1 the scene shift equals the stage move."""
    from nanoindentation_agents.instrument import imagecal as ic
    dy, dx, r = ic.register(_crop(pre), _crop(post))
    sx, sy = dx * UMPP, dy * UMPP
    assert math.hypot(sx - move[0], sy - move[1]) <= SLACK, \
        f"measured ({sx:+.1f}, {sy:+.1f}), commanded ({move[0]:+.1f}, {move[1]:+.1f})"
    # A true correlation coefficient: never above 1, and high on a real match.
    assert 0.5 < r <= 1.0, f"r={r}"


@pytest.mark.parametrize("pre,post,move", PAIRS)
def test_the_trust_gate_accepts_these_registrations(pre, post, move):
    from nanoindentation_agents.campaign.runner import CampaignConfig, CampaignRunner
    from nanoindentation_agents.instrument import imagecal as ic
    r = object.__new__(CampaignRunner)
    r.c = CampaignConfig(site_gain_x=-1.0, site_gain_y=-1.0)
    dy, dx, _ = ic.register(_crop(pre), _crop(post))
    verdict, detail = r._trust_registration(move, (dx * UMPP, dy * UMPP))
    assert verdict == "ok", detail


def test_small_leg_is_judged_not_skipped():
    """The 0805_160842 stop: a ~6 um correction read as a ~96 um shift must be
    rejected. The old gate skipped any axis commanded under 10 um, so this
    exact case went unjudged."""
    from nanoindentation_agents.campaign.runner import CampaignConfig, CampaignRunner
    r = object.__new__(CampaignRunner)
    r.c = CampaignConfig(site_gain_x=-1.0, site_gain_y=-1.0)
    verdict, detail = r._trust_registration((-5.8, 6.1), (96.1, -22.2))
    assert verdict == "untrusted", detail


def test_a_mirrored_move_is_still_reported_as_a_sign_error():
    """The absolute-tolerance rule must not swallow the case the check exists
    for: a scene that moved the right distance the wrong way."""
    from nanoindentation_agents.campaign.runner import CampaignConfig, CampaignRunner
    r = object.__new__(CampaignRunner)
    r.c = CampaignConfig(site_gain_x=-1.0, site_gain_y=-1.0)
    verdict, _ = r._trust_registration((49.2, -24.4), (-49.2, -24.4))
    assert verdict == "flipped"


def test_a_good_move_with_realistic_gain_error_is_accepted():
    """Gains are known to ~10-20%; that must not read as a failure."""
    from nanoindentation_agents.campaign.runner import CampaignConfig, CampaignRunner
    r = object.__new__(CampaignRunner)
    r.c = CampaignConfig(site_gain_x=-1.0, site_gain_y=-1.0)
    verdict, detail = r._trust_registration((30.1, 70.2), (35.9, 64.1))
    assert verdict == "ok", detail
