"""
Image-to-stage calibration — how an offset in the video panel becomes a stage move.

This is the conversion `_site_destination()` performs, and until 2026-08-04 both
its sign conventions were ASSUMPTIONS written as comments, never measured:

    image right = stage +x      # turned out to be correct (within 3%)
    image up    = stage +y      # turned out to be BACKWARDS

Measured on this instrument (60 um probe moves, frames registered by
normalised cross-correlation, sign convention proved by `self_test`):

    stage +60.0 um in X  ->  scene moved +58.3 um (RIGHT), true 0.1813 um/px
    stage +60.1 um in Y  ->  scene moved +43.5 um (DOWN),  true 0.2433 um/px

Two consequences the old code could not express:

1. X was INVERTED. A target 56 um LEFT of centre was driven so the scene moved
   further LEFT — the error grew from 56 um to ~111 um and the target left the
   field entirely. This is what the operator saw directly ("when it should move
   left it moves right"). Y's sign was right; only its magnitude was short.
2. The pixel scale is NOT isotropic. The scale bar is horizontal, so deriving
   the vertical FOV from it (`fov_h = img_h * um_per_px`) understates the true
   vertical scale by ~38%. InView's own "Field of View Cal Out of Normal Range"
   banner is consistent with this.

A NOTE ON HOW THIS WAS GOT WRONG ONCE ALREADY: the raw correlation peak has the
opposite sign to the scene's motion. Taking it at face value produced exactly
the reverse verdict (X fine, Y inverted) and would have "fixed" the working
axis while doubling the error on the broken one. `register()` now negates once,
internally, and `self_test()` verifies the sign against a known shift on every
call. Do not re-derive this by reasoning about FFT conventions.

Both are folded into one signed number per axis:

    stage_delta = site_gain * image_offset_from_centre

so a gain carries the sign AND the scale, and is directly measurable.

USE `stage_delta_from_image` / `image_delta_from_stage` FOR EVERY CONVERSION IN
EITHER DIRECTION. On 2026-08-05 the planning OVERLAY still carried the old
`image right = stage +x` assumption inline while `_site_destination` used the
measured gain, so the two disagreed by a sign: the runner drove to the site the
agent picked and the picture drew that site MIRRORED about the frame's vertical
centre-line — 2 x 49 um away from the truth. The stage was right and the plot
was wrong, which is the worst way round, because the plot is what a human
approves. Inverses derived by hand drift away from the forward map; these two
functions cannot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def stage_delta_from_image(dx_img_um: float, dy_img_um: float,
                           gain_x: float, gain_y: float) -> tuple[float, float]:
    """Image offset (RIGHT-positive, DOWN-positive) -> stage delta. THE forward map."""
    return (gain_x * float(dx_img_um), gain_y * float(dy_img_um))


def image_delta_from_stage(dx_stage_um: float, dy_stage_um: float,
                           gain_x: float, gain_y: float) -> tuple[float, float]:
    """Stage delta -> image offset. The EXACT inverse of `stage_delta_from_image`.

    Anything drawing a stage coordinate back onto a frame must come through
    here, or it will disagree with where the stage is actually sent.
    """
    if abs(gain_x) < 1e-9 or abs(gain_y) < 1e-9:
        raise ValueError(f"degenerate site gain ({gain_x}, {gain_y}) — cannot invert")
    return (float(dx_stage_um) / gain_x, float(dy_stage_um) / gain_y)


def find_crosshair(rgb: np.ndarray, *, search_frac: float = 0.15
                   ) -> Optional[tuple[float, float]]:
    """Pixel (x, y) of InView's burned-in red indenter crosshair, or None.

    THE INDENTER AXIS IS NOT EXACTLY THE CENTRE OF THE CAPTURED FRAME. Measured
    on this instrument (2026-08-05): the marker sits at (594.0, 435.0) px in a
    1187x903 panel whose geometric centre is (593.5, 451.5) — 0.1 um off in x
    but 2.9 um HIGH in y, and identical in two frames minutes apart, so it is a
    fixed overlay and that 2.9 um is a systematic aiming bias, not noise.

    Only the neighbourhood of the centre is searched: InView also burns red
    text and a red scale bar into the panel, and the indenter axis is by
    definition near the optical axis. Returns None rather than a guess when the
    blob is missing or the wrong size — a wrong crosshair is worse than the
    centre assumption it replaces.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    a = rgb[..., :3].astype(int)
    h, w = a.shape[:2]
    y0, y1 = int((0.5 - search_frac) * h), int((0.5 + search_frac) * h)
    x0, x1 = int((0.5 - search_frac) * w), int((0.5 + search_frac) * w)
    win = a[y0:y1, x0:x1]
    m = (win[..., 0] > 190) & (win[..., 1] < 90) & (win[..., 2] < 90)
    ys, xs = np.nonzero(m)
    if len(xs) < 8 or len(xs) > 400:
        return None
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    if not (6 <= bw <= 60 and 6 <= bh <= 60):     # a compact "+" , not text
        return None
    return (float(xs.mean() + x0), float(ys.mean() + y0))


def crop_video_area(img: np.ndarray, *, top: float = 0.10, bottom: float = 0.86,
                    left: float = 0.03, right: float = 0.99) -> np.ndarray:
    """Drop the InView header, toolbar and burned-in warning text.

    Those are STATIC, high-contrast (text, icons, borders) and do not move with
    the stage, so leaving them in pins a phase correlation at zero shift — the
    first attempt at this measurement reported "the scene did not move" for a
    60 um move because of exactly that.
    """
    h, w = img.shape[:2]
    return img[int(top * h):int(bottom * h), int(left * w):int(right * w)]


def register(a: np.ndarray, b: np.ndarray, *, max_shift_px: int = 700,
             min_overlap_frac: float = 0.20):
    """How far the SCENE moved between frame `a` and frame `b`.

    Returns ``(dy, dx, score)`` in the INTUITIVE convention:
        dx > 0  ->  the scene moved RIGHT   (content shifted to larger x)
        dy > 0  ->  the scene moved DOWN    (content shifted to larger y)

    The raw correlation peak comes out with the opposite sign, and reading it
    at face value is exactly how the 2026-08-04 analysis concluded the X
    convention was fine when it was in fact inverted. The negation is applied
    here, once, and `self_test()` proves it against a known shift — never
    re-derive it by reasoning.

    MASKED normalised cross-correlation (Padfield): at every candidate shift the
    mean and the variance are recomputed over THE PIXELS THAT ACTUALLY OVERLAP,
    so `score` is a true correlation coefficient in [-1, 1] and is comparable
    between shifts. Two cheaper things were tried first and BOTH produced
    confident, wrong answers on this instrument's frames:

    * Raw cross-correlation is biased towards reporting NO MOTION — the
      correlation sum over ``(W-|dx|)(H-|dy|)`` overlapping pixels shrinks with
      the shift, purely geometrically. It reported an EXACT (0.0, -24.5) um for
      a true (-41.9, -50.5) um (2026-08-05, run 0805_130904).
    * Dividing that sum by the overlap count fixes the bias but normalises with
      GLOBAL means and variances, so a small, high-contrast window can score
      above a large, correct, flatter one — scores came out above 1.0, which a
      correlation coefficient cannot be. On run 0805_160842 it turned a 5.8 um
      correction into a reported 96 um scene shift (score 1.10) and stopped the
      campaign; masked NCC returns -5.8 um (r = 0.986) for the same pair.

    Every good registration measured on this instrument scores r > 0.9 and every
    fabricated one scored above 1.0 under the old formula — but do not lean on
    the score. Judge a registration by whether it AGREES WITH THE COMMANDED MOVE
    (`CampaignRunner._trust_registration`); that is the only check that knows
    what the answer should have been.
    """
    a = a.astype(float); b = b.astype(float)
    H, W = max(a.shape[0], b.shape[0]) * 2, max(a.shape[1], b.shape[1]) * 2
    F = lambda x: np.fft.rfft2(x, s=(H, W))                      # noqa: E731
    Inv = lambda x: np.fft.fftshift(np.fft.irfft2(x, s=(H, W)))  # noqa: E731
    Fa, Fb = F(a), F(b)
    Fma, Fmb = F(np.ones_like(a)), F(np.ones_like(b))
    o = np.maximum(Inv(Fma * np.conj(Fmb)), 1e-9)   # overlapping pixel count
    Sa, Sb = Inv(Fa * np.conj(Fmb)), Inv(Fma * np.conj(Fb))
    Saa, Sbb = Inv(F(a * a) * np.conj(Fmb)), Inv(Fma * np.conj(F(b * b)))
    num = Inv(Fa * np.conj(Fb)) - Sa * Sb / o
    den = np.sqrt(np.maximum(Saa - Sa * Sa / o, 0.0)
                  * np.maximum(Sbb - Sb * Sb / o, 0.0))
    rho = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    cy, cx = H // 2, W // 2
    my = min(max_shift_px, cy); mx = min(max_shift_px, cx)
    sub = rho[cy - my:cy + my, cx - mx:cx + mx]
    ov = o[cy - my:cy + my, cx - mx:cx + mx]
    # Too little in common to judge a match, however well it correlates.
    sub = np.where(ov >= min_overlap_frac * a.size, sub, -np.inf)
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    # NEGATED so the returned sign is the direction the scene actually moved.
    return -int(iy - my), -int(ix - mx), float(sub[iy, ix])


def self_test(img: np.ndarray, shift_px: int = 341) -> tuple[bool, str]:
    """Does `register` work on THIS frame, with the right SIGN? -> (ok, message).

    Three controls: a frame against itself gives (0, 0); a copy shifted RIGHT
    must report dx = +shift; a copy shifted DOWN must report dy = +shift. A
    field with too little texture — or a sign convention that has drifted —
    fails here instead of silently producing a backwards gain.

    The controls are built with `np.roll`, which WRAPS: the wrapped strip is a
    second, spurious match at shift-minus-the-frame-width. Unnormalised
    correlation buried it (tiny overlap, tiny sum); overlap normalisation
    divides that tiny sum by a tiny overlap and can hand it the win. So each
    control searches only a window that brackets the shift it is testing.
    """
    dy, dx, s0 = register(img, img, max_shift_px=shift_px + 60)
    if (dy, dx) != (0, 0):
        return False, f"self-correlation gave ({dy}, {dx}), expected (0, 0)"
    dy, dx, s1 = register(img, np.roll(img, shift_px, axis=1),      # scene RIGHT
                          max_shift_px=shift_px + 60)
    if abs(dx - shift_px) > 3 or abs(dy) > 3:
        return False, (f"a scene shifted RIGHT by {shift_px}px registered as "
                       f"({dy}, {dx}); expected (0, +{shift_px})")
    dy, dx, s2 = register(img, np.roll(img, shift_px // 2, axis=0),  # scene DOWN
                          max_shift_px=shift_px + 60)
    if abs(dy - shift_px // 2) > 3 or abs(dx) > 3:
        return False, (f"a scene shifted DOWN by {shift_px // 2}px registered as "
                       f"({dy}, {dx}); expected (+{shift_px // 2}, 0)")
    return True, f"controls OK (self {s0:.2f}, right {s1:.2f}, down {s2:.2f})"


def gain_from_probe(stage_delta_um: float, scene_shift_px: int,
                    um_per_px: float) -> tuple[float, float]:
    """(site_gain, true_um_per_px) from one probe move.

    To centre a target sitting at +D from centre, the scene must move by -D.
    The probe measures how the scene responds to a known stage move, so:

        scene_shift = (scene_per_stage) * stage_delta
        want scene_shift = -D   ->   stage_delta = -D / scene_per_stage

    and `site_gain` is defined by stage_delta = gain * D, hence
    gain = -1 / scene_per_stage, expressed in the caller's um units.
    """
    if scene_shift_px == 0:
        raise ValueError("scene did not move — registration failed, not a gain of infinity")
    scene_um = scene_shift_px * um_per_px
    gain = -stage_delta_um / scene_um
    true_um_per_px = abs(stage_delta_um) / abs(scene_shift_px)
    return gain, true_um_per_px