"""
synthetic_microstructure.py
===========================
Deterministic two-phase virtual sample for the nanoindenter simulator.

The sample is an infinite Voronoi tessellation of grains seeded from a fixed
global seed. Each grain is either a hard bright precipitate or the softer
dark matrix; grain boundaries and occasional pores are included. The same
map drives both:

- ``render_fov`` — the microscope image returned by ``capture_video_panel``
- ``properties_at`` — the hardness/modulus returned for simulated indents

so image-guided target selection produces measurably different indentation
results by phase, closing the loop between image analysis and measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

GLOBAL_SEED = 20260707

# Field of view of the simulated video panel.
FOV_WIDTH_UM = 100.0
FOV_HEIGHT_UM = 75.0
IMG_W, IMG_H = 640, 480

GRAIN_SIZE_UM = 30.0          # mean grain diameter (Voronoi cell pitch)
BOUNDARY_WIDTH_UM = 1.1       # rendered grain-boundary half width
BOUNDARY_ZONE_UM = 2.0        # zone with boundary hardening for indents
BOUNDARY_HARDENING = 1.20     # hardness multiplier inside the boundary zone
HARD_PHASE_FRACTION = 0.34
PORE_CELL_PROBABILITY = 0.22  # chance a grain contains one small pore
PORE_RADIUS_UM = (1.5, 4.0)

PHASES: dict[str, dict[str, float]] = {
    "hard_precipitate": {"hardness": 11.8, "modulus": 214.0, "gray": 182.0},
    "matrix": {"hardness": 6.4, "modulus": 166.0, "gray": 96.0},
    "pore": {"hardness": 0.6, "modulus": 20.0, "gray": 34.0},
}

# Slight green/yellow tint so simulated frames resemble the real iMicro optics.
_TINT = (0.94, 1.0, 0.74)


def _cell_hash(i: np.ndarray | int, j: np.ndarray | int, salt: int = 0) -> np.ndarray:
    """Deterministic 32-bit hash of a Voronoi cell index."""
    h = (
        np.asarray(i, dtype=np.int64) * 73856093
        ^ np.asarray(j, dtype=np.int64) * 19349663
        ^ (GLOBAL_SEED + salt) * 83492791
    )
    h = (h ^ (h >> 13)) * 1274126177
    return np.asarray((h ^ (h >> 16)) & 0x7FFFFFFF, dtype=np.int64)


def _cell_unit(i, j, salt: int) -> np.ndarray:
    """Deterministic float in [0, 1) per cell."""
    return _cell_hash(i, j, salt) / float(0x7FFFFFFF)


def _voronoi_fields(x_um: np.ndarray, y_um: np.ndarray) -> dict[str, np.ndarray]:
    """
    For arrays of stage coordinates, return the nearest Voronoi seed cell,
    distance margin to the second-nearest seed, and pore mask.
    """
    g = GRAIN_SIZE_UM
    ci = np.floor(x_um / g).astype(np.int64)
    cj = np.floor(y_um / g).astype(np.int64)

    best_d = np.full(x_um.shape, np.inf)
    second_d = np.full(x_um.shape, np.inf)
    best_i = np.zeros(x_um.shape, dtype=np.int64)
    best_j = np.zeros(x_um.shape, dtype=np.int64)

    for di in range(-2, 3):
        for dj in range(-2, 3):
            ni, nj = ci + di, cj + dj
            sx = (ni + 0.15 + 0.7 * _cell_unit(ni, nj, 1)) * g
            sy = (nj + 0.15 + 0.7 * _cell_unit(ni, nj, 2)) * g
            d = np.hypot(x_um - sx, y_um - sy)
            closer = d < best_d
            second_d = np.where(closer, best_d, np.minimum(second_d, d))
            best_d = np.where(closer, d, best_d)
            best_i = np.where(closer, ni, best_i)
            best_j = np.where(closer, nj, best_j)

    margin = second_d - best_d  # ~0 on grain boundaries

    # Pores: one small circle per selected cell, offset inside the cell.
    has_pore = _cell_unit(best_i, best_j, 3) < PORE_CELL_PROBABILITY
    px = (best_i + 0.25 + 0.5 * _cell_unit(best_i, best_j, 4)) * g
    py = (best_j + 0.25 + 0.5 * _cell_unit(best_i, best_j, 5)) * g
    pr = PORE_RADIUS_UM[0] + (PORE_RADIUS_UM[1] - PORE_RADIUS_UM[0]) * _cell_unit(
        best_i, best_j, 6
    )
    pore = has_pore & (np.hypot(x_um - px, y_um - py) < pr)

    hard = _cell_unit(best_i, best_j, 7) < HARD_PHASE_FRACTION
    grain_shade = (_cell_unit(best_i, best_j, 8) - 0.5) * 22.0  # per-grain contrast

    return {
        "cell_i": best_i,
        "cell_j": best_j,
        "margin": margin,
        "pore": pore,
        "hard": hard,
        "grain_shade": grain_shade,
    }


def phase_at(x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
    """Phase name per point: 'hard_precipitate' | 'matrix' | 'pore'."""
    f = _voronoi_fields(np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float))
    out = np.where(f["hard"], "hard_precipitate", "matrix").astype(object)
    out[f["pore"]] = "pore"
    return out


def properties_at(x_um: float, y_um: float) -> dict[str, Any]:
    """
    Ground-truth mechanical properties at one stage coordinate.

    Includes per-grain variation and grain-boundary hardening so indent
    arrays show realistic scatter on top of the phase contrast.
    """
    x = np.asarray([float(x_um)])
    y = np.asarray([float(y_um)])
    f = _voronoi_fields(x, y)

    if bool(f["pore"][0]):
        phase = "pore"
    elif bool(f["hard"][0]):
        phase = "hard_precipitate"
    else:
        phase = "matrix"

    base = PHASES[phase]
    grain_h = 1.0 + (float(_cell_unit(f["cell_i"], f["cell_j"], 9)[0]) - 0.5) * 0.08
    grain_e = 1.0 + (float(_cell_unit(f["cell_i"], f["cell_j"], 10)[0]) - 0.5) * 0.05

    hardness = base["hardness"] * grain_h
    modulus = base["modulus"] * grain_e
    near_boundary = bool(f["margin"][0] < BOUNDARY_ZONE_UM) and phase != "pore"
    if near_boundary:
        hardness *= BOUNDARY_HARDENING

    return {
        "phase": phase,
        "hardness_gpa": float(hardness),
        "modulus_gpa": float(modulus),
        "near_boundary": near_boundary,
    }


def fov_grids(x_center_um: float, y_center_um: float, w: int = IMG_W, h: int = IMG_H):
    """Stage-coordinate meshgrid for a field of view centered on the stage position."""
    xs = x_center_um + (np.arange(w) / (w - 1) - 0.5) * FOV_WIDTH_UM
    # Image row 0 is the top of the frame = +y in stage coordinates.
    ys = y_center_um + (0.5 - np.arange(h) / (h - 1)) * FOV_HEIGHT_UM
    return np.meshgrid(xs, ys)


def ground_truth_fov(x_center_um: float, y_center_um: float, n: int = 160) -> dict[str, np.ndarray]:
    """
    Downsampled ground-truth maps for the current field of view.
    Used only for post-run validation plots in simulator mode.
    """
    xs = x_center_um + (np.arange(n) / (n - 1) - 0.5) * FOV_WIDTH_UM
    ys = y_center_um + (0.5 - np.arange(int(n * FOV_HEIGHT_UM / FOV_WIDTH_UM)) /
                        (int(n * FOV_HEIGHT_UM / FOV_WIDTH_UM) - 1)) * FOV_HEIGHT_UM
    gx, gy = np.meshgrid(xs, ys)
    f = _voronoi_fields(gx, gy)

    hardness = np.where(f["hard"], PHASES["hard_precipitate"]["hardness"], PHASES["matrix"]["hardness"])
    hardness = np.where(f["margin"] < BOUNDARY_ZONE_UM, hardness * BOUNDARY_HARDENING, hardness)
    hardness = np.where(f["pore"], PHASES["pore"]["hardness"], hardness)

    phase_id = np.where(f["hard"], 1, 0)
    phase_id = np.where(f["pore"], 2, phase_id)

    return {"x": gx, "y": gy, "phase_id": phase_id, "hardness": hardness}


def render_fov(
    x_center_um: float,
    y_center_um: float,
    extension_um: float,
    path: str | Path,
) -> None:
    """Render the simulated iMicro video panel for the current stage position."""
    gx, gy = fov_grids(x_center_um, y_center_um)
    f = _voronoi_fields(gx, gy)

    gray = np.where(
        f["hard"], PHASES["hard_precipitate"]["gray"], PHASES["matrix"]["gray"]
    ).astype(float)
    gray += f["grain_shade"]

    # Grain boundaries: dark lines where two seeds are nearly equidistant.
    boundary = f["margin"] < BOUNDARY_WIDTH_UM
    gray = np.where(boundary, 52.0, gray)
    gray = np.where(f["pore"], PHASES["pore"]["gray"], gray)

    # Camera noise + mild horizontal illumination gradient.
    rng = np.random.default_rng(
        (int(x_center_um * 7.0) ^ int(y_center_um * 13.0) ^ GLOBAL_SEED) & 0xFFFFFFFF
    )
    gray += rng.normal(0.0, 5.5, size=gray.shape)
    gray += np.linspace(-6.0, 6.0, gray.shape[1])[None, :]
    gray = np.clip(gray, 0, 255)

    rgb = np.stack([np.clip(gray * t, 0, 255) for t in _TINT], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Center crosshair (indenter axis), like the InView video panel marker.
    cx, cy = IMG_W // 2, IMG_H // 2
    draw.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), outline=(225, 60, 60), width=2)
    draw.line((cx - 42, cy, cx - 18, cy), fill=(225, 60, 60), width=2)
    draw.line((cx + 18, cy, cx + 42, cy), fill=(225, 60, 60), width=2)
    draw.line((cx, cy - 42, cx, cy - 18), fill=(225, 60, 60), width=2)
    draw.line((cx, cy + 18, cx, cy + 42), fill=(225, 60, 60), width=2)

    # Status banner.
    label = (
        f"SIM iMicro  x={x_center_um:.1f} um  y={y_center_um:.1f} um  "
        f"z={extension_um:.2f} um  FOV {FOV_WIDTH_UM:.0f}x{FOV_HEIGHT_UM:.0f} um"
    )
    draw.rectangle((0, 0, IMG_W, 22), fill=(20, 20, 20))
    draw.text((8, 5), label, fill=(255, 235, 80))

    # Scale bar: 20 µm.
    bar_px = int(20.0 / FOV_WIDTH_UM * IMG_W)
    x0, y0 = 18, IMG_H - 26
    draw.rectangle((x0, y0, x0 + bar_px, y0 + 4), fill=(255, 255, 255))
    draw.text((x0, y0 - 16), "20 um", fill=(255, 255, 255))

    img.save(path)
