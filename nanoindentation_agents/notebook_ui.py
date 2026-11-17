"""
notebook_ui.py
==============
Visual building blocks for the stage-by-stage agentic nanoindentation notebook.

Everything here is presentation + orchestration glue so notebook cells stay
short: agent banners, an agent-to-agent message log with transcript and
sequence diagram, classical CV image analysis (deterministic fallback for the
vision LLM), annotated target overlays, blocking human approval gates, the
instrument execution loop, and result/ground-truth plots.
"""

from __future__ import annotations

import html as _html
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from PIL import Image

def run_with_timeout(fn: Callable[[], Any], seconds: float, label: str = "") -> Any:
    """
    Run a blocking call (e.g. a remote LLM request) with a wall-clock timeout.
    Returns None on timeout or error so callers can fall back gracefully.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _Timeout

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    except _Timeout:
        print(f"[timeout] {label or 'call'} exceeded {seconds:.0f}s — continuing with fallback")
        return None
    except Exception as e:
        print(f"[unavailable] {label or 'call'}: {type(e).__name__}: {e} — continuing with fallback")
        return None
    finally:
        ex.shutdown(wait=False)


# ----------------------------------------------------------------------------
# Agent identity / styling
# ----------------------------------------------------------------------------

AGENT_STYLE: dict[str, dict[str, str]] = {
    "Observe Agent": {"color": "#34495e", "icon": "&#128301;"},          # telescope
    "Image Analysis Agent": {"color": "#8e44ad", "icon": "&#128269;"},   # magnifier
    "Literature Agent": {"color": "#1e8449", "icon": "&#128218;"},       # books
    "Hypothesis Agent": {"color": "#b9770e", "icon": "&#128161;"},       # bulb
    "Hypothesis Critic": {"color": "#c0392b", "icon": "&#9878;"},        # scales
    "Planning Agent": {"color": "#2471a3", "icon": "&#128203;"},         # clipboard
    "Human Operator": {"color": "#117a65", "icon": "&#129489;"},         # person
    "Instrument Executor": {"color": "#6e2c00", "icon": "&#129470;"},    # robot arm
    "Measurement Critic": {"color": "#884ea0", "icon": "&#128202;"},     # chart
    "Supervisor Agent": {"color": "#1a5276", "icon": "&#129517;"},       # compass
    "Orchestrator Agent": {"color": "#0e6655", "icon": "&#127919;"},     # target
    "Acquisition Agent (BO)": {"color": "#7d6608", "icon": "&#128200;"}, # chart up
}

_RISK_COLORS = {"low": "#1e8449", "medium": "#e67e22", "high": "#c0392b"}


def _style(agent: str) -> dict[str, str]:
    return AGENT_STYLE.get(agent, {"color": "#555", "icon": "&#9679;"})


def agent_banner(agent: str, action: str = "") -> None:
    """Display a colored banner announcing which agent is acting."""
    s = _style(agent)
    action_html = (
        f'<span style="font-weight:400;opacity:0.95;font-size:13px"> — {_html.escape(action)}</span>'
        if action
        else ""
    )
    display(
        HTML(
            f'<div style="background:{s["color"]};color:white;padding:9px 16px;'
            f'border-radius:8px;margin:10px 0 6px 0;font-family:system-ui,sans-serif;'
            f'font-size:15px;font-weight:700">{s["icon"]}&nbsp; {agent}{action_html}</div>'
        )
    )


# ----------------------------------------------------------------------------
# Agent-to-agent message log
# ----------------------------------------------------------------------------


class AgentMessageLog:
    """Records every agent → agent handoff so the collaboration is visible."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def log(self, sender: str, recipient: str, subject: str, detail: str = "") -> None:
        entry = {
            "n": len(self.entries) + 1,
            "time": datetime.now().strftime("%H:%M:%S"),
            "sender": sender,
            "recipient": recipient,
            "subject": subject,
            "detail": detail,
        }
        self.entries.append(entry)
        cs, cr = _style(sender)["color"], _style(recipient)["color"]
        detail_html = (
            f'<div style="color:#444;font-size:12px;margin:3px 0 0 26px;white-space:pre-wrap">'
            f"{_html.escape(detail[:700])}</div>"
            if detail
            else ""
        )
        display(
            HTML(
                f'<div style="border-left:4px solid {cs};background:#fafafa;padding:6px 12px;'
                f'margin:4px 0;font-family:system-ui,sans-serif;font-size:13px">'
                f'<b style="color:{cs}">{_html.escape(sender)}</b> '
                f'<span style="color:#999">&#10148;</span> '
                f'<b style="color:{cr}">{_html.escape(recipient)}</b>'
                f'<span style="color:#999;font-size:11px"> &nbsp;msg #{entry["n"]} @ {entry["time"]}</span>'
                f'<div style="margin-left:26px;color:#111">{_html.escape(subject)}</div>'
                f"{detail_html}</div>"
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.entries, columns=["n", "time", "sender", "recipient", "subject", "detail"]
        )

    def render_transcript(self) -> None:
        rows = []
        for e in self.entries:
            cs = _style(e["sender"])["color"]
            cr = _style(e["recipient"])["color"]
            detail = (
                f'<div style="color:#555;font-size:11px;margin-top:2px;white-space:pre-wrap">'
                f"{_html.escape(e['detail'][:400])}</div>"
                if e["detail"]
                else ""
            )
            rows.append(
                f'<tr><td style="padding:6px 8px;color:#999;font-size:11px;vertical-align:top">'
                f'#{e["n"]}<br>{e["time"]}</td>'
                f'<td style="padding:6px 8px;vertical-align:top;white-space:nowrap">'
                f'<b style="color:{cs}">{_html.escape(e["sender"])}</b><br>'
                f'<span style="color:#999">&#8595;</span><br>'
                f'<b style="color:{cr}">{_html.escape(e["recipient"])}</b></td>'
                f'<td style="padding:6px 8px;vertical-align:top">'
                f'<div style="color:#111;font-size:13px">{_html.escape(e["subject"])}</div>{detail}</td></tr>'
            )
        display(
            HTML(
                '<div style="border:1px solid #ddd;border-radius:8px;overflow:hidden;'
                'font-family:system-ui,sans-serif;margin:8px 0">'
                '<div style="background:#2c3e50;color:white;padding:8px 14px;font-weight:700">'
                f"Agent conversation transcript — {len(self.entries)} messages</div>"
                '<table style="border-collapse:collapse;width:100%">' + "".join(rows) + "</table></div>"
            )
        )

    def sequence_diagram(self) -> None:
        """Matplotlib sequence diagram of all agent-to-agent messages."""
        import matplotlib.pyplot as plt

        if not self.entries:
            print("No messages logged.")
            return
        agents: list[str] = []
        for e in self.entries:
            for a in (e["sender"], e["recipient"]):
                if a not in agents:
                    agents.append(a)
        xpos = {a: i for i, a in enumerate(agents)}
        n = len(self.entries)

        fig, ax = plt.subplots(figsize=(min(2.1 * len(agents), 15), max(4.5, 0.42 * n + 1.5)))
        for a, x in xpos.items():
            c = _style(a)["color"]
            ax.plot([x, x], [0, n + 1], color=c, lw=1.2, alpha=0.45, zorder=1)
            ax.text(
                x, n + 1.4, a.replace(" Agent", "\nAgent").replace(" Critic", "\nCritic"),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=c,
            )
        for i, e in enumerate(self.entries):
            y = n - i
            x0, x1 = xpos[e["sender"]], xpos[e["recipient"]]
            c = _style(e["sender"])["color"]
            if x0 == x1:
                x1 = x0 + 0.25
            ax.annotate(
                "", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6), zorder=3,
            )
            ax.text(
                (x0 + x1) / 2, y + 0.14, f'{e["n"]}. {e["subject"][:48]}',
                ha="center", va="bottom", fontsize=7.5, color="#333",
            )
        ax.set_xlim(-0.7, len(agents) - 0.3)
        ax.set_ylim(0, n + 3.2)
        ax.axis("off")
        ax.set_title("Agent interaction sequence", fontsize=12, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------
# Image display helpers
# ----------------------------------------------------------------------------


def resolve_image_path(image_info: Optional[dict[str, Any]]) -> Optional[Path]:
    """Return the first existing local path for a captured image."""
    if not image_info:
        return None
    for key in ("local_path", "image_path", "path"):
        p = image_info.get(key)
        if p and Path(p).exists():
            return Path(p)
    return None


def show_capture(
    image_info: dict[str, Any],
    title: str = "Captured iMicro video panel",
    note: str = "",
    fov_width_um: float = 100.0,
    fov_height_um: float = 75.0,
) -> None:
    """Display a captured microscope frame with stage-position caption."""
    import matplotlib.pyplot as plt

    path = resolve_image_path(image_info)
    if path is None:
        print(f"Image not found locally: {image_info}")
        return
    pos = image_info.get("position") or (image_info.get("x"), image_info.get("y"))
    img = np.asarray(Image.open(path))
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.imshow(img, extent=[0, fov_width_um, fov_height_um, 0])
    ax.set_xlabel("image x (µm)")
    ax.set_ylabel("image y (µm)")
    subtitle = f"stage position: x = {pos[0]:.1f} µm, y = {pos[1]:.1f} µm" if pos and pos[0] is not None else ""
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    if note:
        ax.text(
            0.01, -0.12, note, transform=ax.transAxes, fontsize=9, color="#555", va="top"
        )
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Classical CV analysis (deterministic fallback / cross-check for vision LLM)
# ----------------------------------------------------------------------------


def _otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=128, range=(0, 255))
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    best_t, best_var = 128.0, -1.0
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * centers)
    for k in range(1, 127):
        n0, n1 = w0[k], total - w0[k]
        if n0 == 0 or n1 == 0:
            continue
        mu0 = m0[k] / n0
        mu1 = (m0[-1] - m0[k]) / n1
        var = n0 * n1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, centers[k]
    return float(best_t)


def _ocr_scalebar_um(roi_rgb: np.ndarray) -> Optional[float]:
    """
    OCR the burned-in scale-bar label (e.g. "50 µm") from the bottom-left
    overlay ROI and return its length in microns. Returns None if pytesseract
    is unavailable or nothing parses, so the caller falls back to a default.
    """
    import re

    try:
        import pytesseract
        from PIL import Image as _Image
    except Exception:
        return None
    try:
        text = pytesseract.image_to_string(
            _Image.fromarray(roi_rgb.astype("uint8")),
            config="--psm 6 -c tessedit_char_whitelist=0123456789umµ. ",
        )
    except Exception:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[µu]?m", text)
    return float(m.group(1)) if m else None


def calibrate_um_per_px(
    image_path: str | Path,
    *,
    default_bar_um: float = 50.0,
    roi_y_frac: float = 0.88,
    roi_x_frac: float = 0.35,
    min_um_per_px: float = 0.05,
    max_um_per_px: float = 5.0,
    use_ocr: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Derive microns-per-pixel (and the true field of view) from the red scale
    bar burned into the captured iMicro video panel.

    The bar sits in the bottom-left overlay (rows >= roi_y_frac*H,
    cols <= roi_x_frac*W). Its pixel length sets the scale; its label
    ("50 µm") gives the physical length (read by OCR, else ``default_bar_um``).
    Raises RuntimeError on an implausible result so the caller can REFUSE to
    move rather than drive the indenter to the wrong location.

    Returns ``(um_per_px, info)`` where ``info`` has ``fov_w_um``,
    ``fov_h_um``, ``bar_len_px``, ``bar_um``, ``bar_um_source``,
    ``img_w`` and ``img_h``.
    """
    rgb = np.asarray(Image.open(image_path).convert("RGB")).astype(int)
    H, W = rgb.shape[:2]

    roi = rgb[int(roi_y_frac * H):H, 0:int(roi_x_frac * W)]
    R, G, B = roi[..., 0], roi[..., 1], roi[..., 2]
    red = (R > 120) & (R - G > 60) & (R - B > 60)

    # Longest horizontal run of red = the bar (label/warning text runs are short).
    best_len = 0
    for row in red:
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        if edges.size:
            best_len = max(best_len, int((edges[1::2] - edges[0::2]).max()))

    if best_len < 0.05 * roi.shape[1]:
        raise RuntimeError(
            f"Scale bar not found (longest red run {best_len}px in ROI). "
            "Refusing to calibrate — check the capture / overlay region."
        )

    ocr_um = _ocr_scalebar_um(roi) if use_ocr else None
    bar_um = ocr_um or default_bar_um
    um_per_px = bar_um / best_len

    if not (min_um_per_px < um_per_px < max_um_per_px):
        raise RuntimeError(
            f"Implausible scale {um_per_px:.4f} µm/px "
            f"(bar={best_len}px, label={bar_um}µm). Refusing to move."
        )

    info = {
        "um_per_px": um_per_px,
        "bar_len_px": best_len,
        "bar_um": bar_um,
        "bar_um_source": "ocr" if ocr_um else "default",
        "img_w": W,
        "img_h": H,
        "fov_w_um": W * um_per_px,
        "fov_h_um": H * um_per_px,
    }
    return um_per_px, info


def classical_cv_analysis(
    image_path: str | Path,
    fov_width_um: float = 100.0,
    fov_height_um: float = 75.0,
) -> dict[str, Any]:
    """
    Deterministic image analysis: segment bright (hard) phase, dark pores and
    grain boundaries, then propose indentation targets. Output matches the
    vision-LLM result schema so downstream code treats both the same way.
    """
    from scipy import ndimage

    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=float)
    h, w = gray.shape

    # Mask overlays: status banner (top), scale bar (bottom-left), crosshair disk.
    valid = np.ones_like(gray, dtype=bool)
    valid[: int(0.06 * h), :] = False
    valid[int(0.90 * h):, : int(0.30 * w)] = False
    yy, xx = np.mgrid[0:h, 0:w]
    valid &= (xx - w / 2) ** 2 + (yy - h / 2) ** 2 > (0.075 * w) ** 2

    # Interior region for target picking: keep indent sites off the image edges.
    interior = valid.copy()
    m = int(0.10 * min(h, w))
    interior[:m, :] = False
    interior[-m:, :] = False
    interior[:, :m] = False
    interior[:, -m:] = False

    vals = gray[valid]
    t_bright = _otsu_threshold(vals)
    t_pore = 42.0
    t_boundary = 64.0

    bright = (gray > t_bright) & valid
    pores = (gray < t_pore) & valid
    boundary = (gray >= t_pore) & (gray < t_boundary) & valid

    # Clean masks and label connected regions.
    bright = ndimage.binary_opening(bright, structure=np.ones((5, 5)))
    pores = ndimage.binary_opening(pores, structure=np.ones((3, 3)))
    lbl_b, n_b = ndimage.label(bright)
    lbl_p, n_p = ndimage.label(pores)

    area_valid = float(valid.sum())
    bright_frac = float(bright.sum()) / area_valid
    pore_frac = float(pores.sum()) / area_valid
    boundary_frac = float(boundary.sum()) / area_valid

    def _blob_info(lbl: np.ndarray, n: int, min_area_px: float) -> list[dict[str, float]]:
        blobs = []
        for i in range(1, n + 1):
            mask = lbl == i
            area = float(mask.sum())
            if area < min_area_px or not bool((mask & interior).any()):
                continue
            cy, cx = ndimage.center_of_mass(mask & interior)
            blobs.append(
                {"x_frac": float(cx / w), "y_frac": float(cy / h), "area_frac": area / area_valid}
            )
        blobs.sort(key=lambda b: -b["area_frac"])
        return blobs

    bright_blobs = _blob_info(lbl_b, n_b, 0.002 * area_valid)
    pore_blobs = _blob_info(lbl_p, n_p, 0.0004 * area_valid)

    # Matrix reference: deepest interior point of the matrix (away from
    # everything else, the masked overlays, and the image edges).
    matrix_mask = interior & ~bright & ~pores & ~boundary
    dist = ndimage.distance_transform_edt(matrix_mask)
    my, mx = np.unravel_index(np.argmax(dist), dist.shape)
    matrix_pt = {"x_frac": float(mx / w), "y_frac": float(my / h)}

    # Grain-boundary target: boundary pixel far from image edges and overlays.
    boundary_interior = boundary & interior
    edge_dist = ndimage.distance_transform_edt(interior)
    b_score = np.where(boundary_interior, edge_dist, -1)
    if b_score.max() > 0:
        by, bx = np.unravel_index(np.argmax(b_score), b_score.shape)
        boundary_pt = {"x_frac": float(bx / w), "y_frac": float(by / h)}
    else:
        boundary_pt = None

    targets: list[dict[str, Any]] = []
    if bright_blobs:
        b = bright_blobs[0]
        targets.append(
            {
                "label": "hard_phase_center",
                "x_frac": b["x_frac"],
                "y_frac": b["y_frac"],
                "reason": "phase_center",
                "priority": 1,
                "expected_contrast": (
                    "Bright second-phase grain — expected higher hardness/modulus than matrix"
                ),
            }
        )
    targets.append(
        {
            "label": "matrix_reference",
            "x_frac": matrix_pt["x_frac"],
            "y_frac": matrix_pt["y_frac"],
            "reason": "matrix_reference",
            "priority": 2,
            "expected_contrast": "Grain interior of the dark matrix — baseline properties",
        }
    )
    if boundary_pt is not None:
        targets.append(
            {
                "label": "grain_boundary",
                "x_frac": boundary_pt["x_frac"],
                "y_frac": boundary_pt["y_frac"],
                "reason": "grain_boundary",
                "priority": 3,
                "expected_contrast": "Grain-boundary zone — possible boundary strengthening",
            }
        )
    if len(bright_blobs) > 1:
        b = bright_blobs[1]
        targets.append(
            {
                "label": "hard_phase_2",
                "x_frac": b["x_frac"],
                "y_frac": b["y_frac"],
                "reason": "phase_center",
                "priority": 4,
                "expected_contrast": "Second bright grain — checks phase-level repeatability",
            }
        )

    interpretation = (
        f"Two-phase microstructure detected. Bright (high-reflectivity) phase covers "
        f"{bright_frac:.0%} of the field of view in {len(bright_blobs)} grain(s); dark matrix "
        f"covers {1 - bright_frac - pore_frac - boundary_frac:.0%}. Grain-boundary network "
        f"occupies {boundary_frac:.0%} and {len(pore_blobs)} pore(s) cover {pore_frac:.1%}. "
        f"Intensity threshold (Otsu) at {t_bright:.0f}/255 separates the phases cleanly, "
        f"suggesting strong compositional contrast that should correlate with mechanical properties."
    )

    return {
        "ok": True,
        "source": "classical_cv",
        "features_detected": (
            [{"type": "phase_region", "description": "bright hard phase", "x_frac": b["x_frac"],
              "y_frac": b["y_frac"], "confidence": 0.9} for b in bright_blobs[:4]]
            + [{"type": "pore", "description": "dark circular pore", "x_frac": p["x_frac"],
                "y_frac": p["y_frac"], "confidence": 0.85} for p in pore_blobs[:4]]
            + ([{"type": "grain_boundary", "description": "boundary network",
                 "x_frac": boundary_pt["x_frac"], "y_frac": boundary_pt["y_frac"],
                 "confidence": 0.8}] if boundary_pt else [])
        ),
        "phases_identified": [
            {"label": "bright_hard_phase", "area_fraction": round(bright_frac, 3),
             "center_x_frac": bright_blobs[0]["x_frac"] if bright_blobs else 0.5,
             "center_y_frac": bright_blobs[0]["y_frac"] if bright_blobs else 0.5,
             "expected_hardness": "higher"},
            {"label": "dark_matrix", "area_fraction": round(1 - bright_frac - pore_frac, 3),
             "center_x_frac": matrix_pt["x_frac"], "center_y_frac": matrix_pt["y_frac"],
             "expected_hardness": "lower"},
        ],
        "grain_boundaries": (
            [{"type": "high_angle", "x_frac": boundary_pt["x_frac"],
              "y_frac": boundary_pt["y_frac"], "confidence": 0.8}] if boundary_pt else []
        ),
        "target_locations": targets,
        "interpretation": interpretation,
        "recommendations": [
            "Indent the bright phase and the matrix to quantify the phase contrast",
            "Add a grain-boundary array to test for boundary strengthening",
            "Keep arrays clear of pores — they invalidate contact mechanics",
        ],
        "statistics": {
            "bright_area_fraction": bright_frac,
            "pore_area_fraction": pore_frac,
            "boundary_area_fraction": boundary_frac,
            "otsu_threshold": t_bright,
            "n_bright_grains": len(bright_blobs),
            "n_pores": len(pore_blobs),
        },
    }


def combine_image_analysis(
    cv_result: dict[str, Any],
    vision_result: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    """
    Merge classical CV geometry with vision-LLM interpretation (when usable).
    Returns (ImageAnalysisResult-shaped dict, provenance description).
    """
    vision_ok = bool(vision_result and vision_result.get("vision", {}).get("ok"))
    analysis: dict[str, Any] = {
        "features_detected": [f.get("type", "feature") for f in cv_result["features_detected"]],
        "grain_boundaries": cv_result["grain_boundaries"],
        "phases": cv_result["phases_identified"],
        "defects": [f for f in cv_result["features_detected"] if f.get("type") == "pore"],
        "target_locations": list(cv_result["target_locations"]),
        "statistical_summary": cv_result["statistics"],
        "recommendations": list(cv_result["recommendations"]),
        "interpretation": cv_result["interpretation"],
    }
    provenance = "classical CV segmentation"
    if vision_ok:
        v_targets = vision_result.get("target_locations") or []
        for t in v_targets:
            t = dict(t)
            t.setdefault("source", "vision_llm")
            analysis["target_locations"].append(t)
        v_interp = (vision_result.get("interpretation") or "").strip()
        if v_interp:
            analysis["interpretation"] = (
                f"[Vision LLM] {v_interp}\n\n[Classical CV] {cv_result['interpretation']}"
            )
        analysis["recommendations"] = (
            list(vision_result.get("recommendations") or []) + analysis["recommendations"]
        )
        provenance = "vision LLM + classical CV cross-check"
    return analysis, provenance


def annotate_targets(
    image_info: dict[str, Any],
    analysis: dict[str, Any],
    targets: list[dict[str, Any]],
    title: str = "Image Analysis Agent — detected features and proposed targets",
    fov_width_um: float = 100.0,
    fov_height_um: float = 75.0,
    save_path: Optional[str | Path] = None,
) -> None:
    """Overlay detected phases, boundaries, pores and numbered targets."""
    import matplotlib.pyplot as plt

    path = resolve_image_path(image_info)
    if path is None:
        print("No local image to annotate.")
        return
    img = np.asarray(Image.open(path))

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.imshow(img, extent=[0, fov_width_um, fov_height_um, 0])

    for ph in analysis.get("phases", [])[:6]:
        x = float(ph.get("center_x_frac", 0.5)) * fov_width_um
        y = float(ph.get("center_y_frac", 0.5)) * fov_height_um
        ax.plot(x, y, marker="s", ms=11, mfc="none", mec="#00d0ff", mew=2)
        ax.annotate(
            f'{ph.get("label", "phase")} ({ph.get("area_fraction", 0):.0%})',
            (x, y), xytext=(6, -10), textcoords="offset points",
            fontsize=8, color="#00d0ff", fontweight="bold",
        )
    for d in analysis.get("defects", [])[:8]:
        x = float(d.get("x_frac", 0.5)) * fov_width_um
        y = float(d.get("y_frac", 0.5)) * fov_height_um
        ax.plot(x, y, marker="o", ms=13, mfc="none", mec="#ffd400", mew=1.8)
    for gb in analysis.get("grain_boundaries", [])[:6]:
        x = float(gb.get("x_frac", 0.5)) * fov_width_um
        y = float(gb.get("y_frac", 0.5)) * fov_height_um
        ax.plot(x, y, marker="^", ms=12, mfc="none", mec="#7CFC00", mew=2)

    for i, t in enumerate(targets, start=1):
        x = float(t.get("x_frac", 0.5)) * fov_width_um
        y = float(t.get("y_frac", 0.5)) * fov_height_um
        ax.plot(x, y, marker="+", ms=20, mec="#ff2d2d", mew=2.6)
        ax.add_patch(plt.Circle((x, y), 3.4, fill=False, ec="#ff2d2d", lw=1.8))
        ax.annotate(
            f'T{i}: {t.get("label", "?")}',
            (x, y), xytext=(10, 12), textcoords="offset points",
            fontsize=9.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#c0392b", ec="none", alpha=0.92),
        )

    handles = [
        plt.Line2D([], [], marker="s", ls="", mfc="none", mec="#00d0ff", label="phase center"),
        plt.Line2D([], [], marker="^", ls="", mfc="none", mec="#7CFC00", label="grain boundary"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#ffd400", label="pore / defect"),
        plt.Line2D([], [], marker="+", ls="", mec="#ff2d2d", label="proposed indent target"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_xlabel("image x (µm)")
    ax.set_ylabel("image y (µm)")
    ax.set_title(title, fontsize=11.5)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()


def targets_table(targets: list[dict[str, Any]], fov_width_um: float = 100.0,
                  fov_height_um: float = 75.0) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(targets, start=1):
        xf, yf = float(t.get("x_frac", 0.5)), float(t.get("y_frac", 0.5))
        rows.append(
            {
                "target": f"T{i}",
                "label": t.get("label", ""),
                "image x (µm)": round(xf * fov_width_um, 1),
                "image y (µm)": round(yf * fov_height_um, 1),
                "offset from center dx (µm)": round((xf - 0.5) * fov_width_um, 1),
                "offset dy (µm)": round((0.5 - yf) * fov_height_um, 1),
                "reason": t.get("reason", ""),
                "priority": t.get("priority", ""),
                "source": t.get("source", "classical_cv"),
                "expected contrast": t.get("expected_contrast", ""),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Literature / hypothesis / critic panels
# ----------------------------------------------------------------------------


def literature_panel(state: dict[str, Any]) -> None:
    query = state.get("literature_search_query", "")
    sources = state.get("literature_sources", []) or []
    summary = state.get("literature_context", "")
    raw = state.get("literature_raw_context", "")
    meta = state.get("literature_search_meta", {}) or {}

    src_rows = "".join(
        f'<li style="margin:2px 0;color:#154360">{_html.escape(str(s))}</li>' for s in sources[:12]
    )
    snippet = _html.escape(raw[:1200])
    display(
        HTML(
            f"""
<div style="border:2px solid #1e8449;border-radius:10px;margin:10px 0;overflow:hidden;
            font-family:system-ui,sans-serif">
  <div style="background:#1e8449;color:white;padding:9px 14px;font-weight:700">
    &#128218; Literature Agent — retrieved context
    <span style="font-weight:400;font-size:12px;opacity:0.9"> &nbsp;mode: {_html.escape(str(meta.get("mode", "?")))} · {len(sources)} source(s)</span>
  </div>
  <div style="padding:12px 16px">
    <div style="font-size:12px;color:#666;margin-bottom:6px"><b>Search query:</b> {_html.escape(query[:200])}</div>
    <b style="color:#1e8449;font-size:13px">Sources</b>
    <ul style="margin:4px 0 10px 0;font-size:12.5px">{src_rows or "<li>(none)</li>"}</ul>
    <b style="color:#1e8449;font-size:13px">Summary passed to the Hypothesis Agent</b>
    <div style="background:#e8f5e9;border-radius:6px;padding:10px 12px;font-size:13px;
                margin:4px 0 10px 0;white-space:pre-wrap;color:#1b2631">{_html.escape(summary[:2500])}</div>
    <details><summary style="cursor:pointer;color:#1e8449;font-size:12.5px;font-weight:600">
      Raw retrieved excerpts</summary>
      <div style="background:#f8f9fa;border-radius:6px;padding:10px;font-size:11.5px;
                  white-space:pre-wrap;color:#333;margin-top:4px">{snippet}</div></details>
  </div>
</div>"""
        )
    )


def hypothesis_panel(state: dict[str, Any]) -> None:
    """Rich hypothesis card (reuses the driver's HTML formatter)."""
    from .notebook_driver import _format_hypothesis_html

    display(HTML(_format_hypothesis_html(state)))


def critic_panel(title: str, text: str, approved: Optional[bool] = None) -> None:
    color = "#7f8c8d" if approved is None else ("#1e8449" if approved else "#c0392b")
    verdict = "" if approved is None else (
        f'<span style="background:{color};color:white;border-radius:12px;padding:2px 10px;'
        f'font-size:11.5px;margin-left:8px">{"APPROVED" if approved else "REVISION NEEDED"}</span>'
    )
    display(
        HTML(
            f'<div style="border-left:5px solid {color};background:#fdfefe;padding:10px 14px;'
            f'margin:8px 0;border-radius:0 8px 8px 0;font-family:system-ui,sans-serif;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.08)">'
            f'<b style="color:{color};font-size:13.5px">{_html.escape(title)}</b>{verdict}'
            f'<div style="font-size:13px;color:#212f3c;margin-top:6px;white-space:pre-wrap">'
            f"{_html.escape(text[:1500])}</div></div>"
        )
    )


# ----------------------------------------------------------------------------
# Human approval gates
# ----------------------------------------------------------------------------


class ApprovalGate:
    """
    Blocking human-in-the-loop checkpoint.

    The gate displays what is about to happen (action, target, reason, expected
    outcome, risk); execution cells call ``require_approved()`` and refuse to
    run until the operator explicitly calls ``approve()`` — or stop entirely
    after ``reject()``.
    """

    registry: list["ApprovalGate"] = []

    def __init__(
        self,
        gate_id: str,
        title: str,
        description: str = "",
        action_cards: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.gate_id = gate_id
        self.title = title
        self.description = description
        self.action_cards = action_cards or []
        self.decision: Optional[str] = None  # approved | rejected
        self.decided_by = ""
        self.note = ""
        self.decided_at: Optional[str] = None
        ApprovalGate.registry.append(self)
        self.display()

    # -- rendering -----------------------------------------------------------

    def _status_badge(self) -> str:
        if self.decision == "approved":
            return ('<span style="background:#1e8449;color:white;padding:3px 12px;'
                    'border-radius:12px;font-size:12px">APPROVED</span>')
        if self.decision == "rejected":
            return ('<span style="background:#c0392b;color:white;padding:3px 12px;'
                    'border-radius:12px;font-size:12px">REJECTED</span>')
        return ('<span style="background:#e67e22;color:white;padding:3px 12px;'
                'border-radius:12px;font-size:12px">&#9888; AWAITING OPERATOR DECISION</span>')

    def _card_html(self, i: int, card: dict[str, Any]) -> str:
        risk = str(card.get("risk_level", "low")).lower()
        rc = _RISK_COLORS.get(risk, "#7f8c8d")
        # 2000, not 600: once method-plan fields (control_mode, depth/load,
        # decided_by, rationale, warnings) are on the card, a single warning
        # or two already exceeds the old 600-char cap and gets cut off mid-
        # string — found by rendering an actual card during Stage 0 review.
        args = _html.escape(json.dumps(card.get("arguments", {}), default=str)[:2000])
        rows = ""
        for label, value in [
            ("Target", card.get("target", "")),
            ("Why (agent justification)", card.get("justification", "")),
            ("Expected outcome", card.get("expected_outcome", "")),
            ("Reversible?", card.get("reversible", "")),
        ]:
            if value:
                rows += (
                    f'<tr><td style="padding:3px 10px 3px 0;color:#666;font-size:12px;'
                    f'white-space:nowrap;vertical-align:top">{label}</td>'
                    f'<td style="padding:3px 0;font-size:12.5px;color:#1b2631">{_html.escape(str(value))}</td></tr>'
                )
        return f"""
  <div style="border:1px solid #d5d8dc;border-radius:8px;margin:8px 0;padding:10px 14px;background:#fbfcfc">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <b style="font-size:13.5px;color:#1a5276">Action {i}: <code>{_html.escape(str(card.get("tool_name", "?")))}</code></b>
      <span style="background:{rc};color:white;padding:2px 10px;border-radius:10px;font-size:11px">
        risk: {risk.upper()}</span>
    </div>
    <div style="font-family:ui-monospace,monospace;font-size:11.5px;color:#4a235a;margin:5px 0">args: {args}</div>
    <table style="border-collapse:collapse">{rows}</table>
  </div>"""

    def display(self) -> None:
        cards = "".join(self._card_html(i + 1, c) for i, c in enumerate(self.action_cards))
        desc = (
            f'<div style="font-size:13px;color:#34495e;margin:6px 0">{_html.escape(self.description)}</div>'
            if self.description
            else ""
        )
        decision_line = ""
        if self.decision:
            decision_line = (
                f'<div style="font-size:12px;color:#555;margin-top:6px">Decision by '
                f"<b>{_html.escape(self.decided_by)}</b> at {self.decided_at}"
                + (f" — note: {_html.escape(self.note)}" if self.note else "")
                + "</div>"
            )
        howto = "" if self.decision else (
            '<div style="background:#fef9e7;border:1px dashed #e67e22;border-radius:6px;'
            'padding:8px 12px;margin-top:8px;font-size:12.5px;color:#6e2c00">'
            "<b>The workflow is paused.</b> Nothing will be sent to the instrument until you decide:"
            "<br>&nbsp;&nbsp;<code>gate.approve(approved_by=\"your name\", note=\"...\")</code>"
            "&nbsp;&nbsp;or&nbsp;&nbsp;<code>gate.reject(\"reason\")</code></div>"
        )
        display(
            HTML(
                f"""
<div style="border:3px solid #b03a2e;border-radius:12px;margin:12px 0;overflow:hidden;
            font-family:system-ui,sans-serif;box-shadow:0 2px 10px rgba(0,0,0,0.12)">
  <div style="background:#b03a2e;color:white;padding:10px 16px;display:flex;
              justify-content:space-between;align-items:center">
    <span style="font-weight:800;font-size:14.5px">&#128721; HUMAN APPROVAL GATE — {_html.escape(self.title)}</span>
    {self._status_badge()}
  </div>
  <div style="padding:12px 16px">{desc}{cards}{howto}{decision_line}</div>
</div>"""
            )
        )

    # -- decisions -----------------------------------------------------------

    def approve(self, approved_by: str = "operator", note: str = "") -> None:
        self.decision = "approved"
        self.decided_by = approved_by
        self.note = note
        self.decided_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.display()

    def reject(self, reason: str = "", approved_by: str = "operator") -> None:
        self.decision = "rejected"
        self.decided_by = approved_by
        self.note = reason
        self.decided_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.display()

    @property
    def is_approved(self) -> bool:
        return self.decision == "approved"

    def require_approved(self) -> None:
        if self.decision is None:
            raise RuntimeError(
                f"Approval gate '{self.gate_id}' is still PENDING — run "
                f"gate.approve(...) or gate.reject(...) before executing instrument actions."
            )
        if self.decision == "rejected":
            raise RuntimeError(
                f"Approval gate '{self.gate_id}' was REJECTED ({self.note or 'no reason given'}) "
                f"— instrument actions are blocked."
            )

    def record(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "title": self.title,
            "n_actions": len(self.action_cards),
            "decision": self.decision or "pending",
            "decided_by": self.decided_by,
            "note": self.note,
            "decided_at": self.decided_at,
        }

    @classmethod
    def audit_table(cls) -> pd.DataFrame:
        return pd.DataFrame([g.record() for g in cls.registry])


# ----------------------------------------------------------------------------
# Instrument planning + execution
# ----------------------------------------------------------------------------


def plan_target_actions(
    target: dict[str, Any],
    stage_position: tuple[float, float],
    *,
    sample_name: str,
    grid_nx: int = 3,
    grid_ny: int = 3,
    spacing_um: float = 6.0,
    rotation_deg: float = 0.0,
    setup: Optional[dict[str, Any]] = None,
    method_type: str = "eh",
    fov_width_um: float = 100.0,
    fov_height_um: float = 75.0,
    expected_outcome: str = "",
    site_gain_x: float = -1.0,
    site_gain_y: float = -1.0,
    method_plan_args: Optional[dict[str, Any]] = None,
    strategy_plan_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Convert an image-frame target into a concrete, human-reviewable action plan:
    stage moves + indent array configuration + measurement + data export.

    Uses the MEASURED image->stage gains, like everything else that converts
    between the two frames. CampaignRunner overwrites `destination` and the
    card targets with its own (identical) computation, but the justification
    text on the approval card comes from here — and until 2026-08-05 that text
    quoted a stage coordinate derived from the old `image right = stage +x`
    assumption, i.e. the mirror of where the stage would actually go. An
    approval card that misstates the destination is a safety defect even when
    the motion is correct.
    """
    from .instrument.imagecal import stage_delta_from_image

    xf = float(target.get("x_frac", 0.5))
    yf = float(target.get("y_frac", 0.5))
    dx_img = (xf - 0.5) * fov_width_um       # image RIGHT positive
    dy_img = (yf - 0.5) * fov_height_um      # image DOWN  positive
    dx, dy = stage_delta_from_image(dx_img, dy_img, site_gain_x, site_gain_y)
    x_dest = stage_position[0] + dx
    y_dest = stage_position[1] + dy

    # Informational only — CampaignRunner clears this and positions with the
    # closed-loop MotionController, which owns arrow choice and gain.
    moves: list[dict[str, Any]] = []
    if abs(dx) >= 0.05:
        moves.append({"amount": round(abs(dx), 2), "direction": "+x" if dx > 0 else "-x"})
    if abs(dy) >= 0.05:
        moves.append({"amount": round(abs(dy), 2), "direction": "+y" if dy > 0 else "-y"})

    label = target.get("label", "target")
    justification = (
        f"Image Analysis identified '{label}' ({target.get('reason', '')}) at image fraction "
        f"({xf:.2f}, {yf:.2f}) → stage ({x_dest:.1f}, {y_dest:.1f}) µm. "
        f"{target.get('expected_contrast', '')}"
    )

    cards = [
        {
            "tool_name": "indent_move",
            "arguments": {"moves": moves} if moves else {"moves": "none — already centered"},
            "target": f"{label} @ stage ({x_dest:.1f}, {y_dest:.1f}) µm",
            "justification": justification,
            "expected_outcome": "Stage centers the selected region under the indenter axis",
            "risk_level": "low",
            "reversible": "Yes — inverse move returns to the previous position",
        },
        {
            "tool_name": "indent_starting_tests + indent_start_test_normal",
            "arguments": {
                "sample_name": sample_name,
                "grid": f"{grid_nx}x{grid_ny}",
                "spacing_um": spacing_um,
                "rotation_deg": rotation_deg,
                "method_type": method_type,
                **(method_plan_args or {}),
                **(strategy_plan_args or {}),
            },
            "target": f"{label} @ stage ({x_dest:.1f}, {y_dest:.1f}) µm",
            "justification": (
                f"{grid_nx}x{grid_ny} array at {spacing_um} µm spacing gives "
                f"{grid_nx * grid_ny} indent"
                f"{'s' if grid_nx * grid_ny != 1 else ''} selected for this site"
                + (f"; {method_plan_args['control_mode']} at "
                   f"{method_plan_args['target_depth_nm']:g} nm / "
                   f"{method_plan_args['target_load_mN']:g} mN"
                   if method_plan_args else "")
            ),
            "expected_outcome": expected_outcome
            or "Hardness/modulus statistics for this region",
            "risk_level": "medium",
            "reversible": "No — indents permanently deform the surface (µm-scale plastic zone)",
        },
    ]

    return {
        "target": target,
        "label": label,
        "sample_name": sample_name,
        "moves": moves,
        "destination": (round(x_dest, 2), round(y_dest, 2)),
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "spacing_um": spacing_um,
        "rotation_deg": rotation_deg,
        "setup": setup or {},
        "method_type": method_type,
        "cards": cards,
        "justification": justification,
    }


def download_capture(
    toolkit: Any,
    cap_result: dict[str, Any],
    local_dir: str | Path,
) -> dict[str, Any]:
    """Fetch a capture from the instrument server; fall back to a same-machine path."""
    remote_path = cap_result.get("path") or cap_result.get("save_path") or ""
    directory = cap_result.get("directory") or cap_result.get("remote_dir") or ""
    filename = cap_result.get("filename") or cap_result.get("remote_name") or ""
    if not filename and remote_path:
        filename = PureWindowsPath(remote_path).name
    if not directory and remote_path:
        directory = str(PureWindowsPath(remote_path).parent)

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = ""
    try:
        res = toolkit.client.download_file(directory, filename, local_dir=str(local_dir))
        candidate = Path(res.get("local_path", ""))
        Image.open(candidate).verify()  # confirm we really got image bytes
        local_path = str(candidate)
    except Exception:
        if remote_path and Path(remote_path).exists():
            local_path = remote_path  # simulator runs on this machine

    return {
        "image_path": remote_path,
        "local_path": local_path,
        "timestamp": str(time.time()),
        "position": (cap_result.get("x"), cap_result.get("y")),
    }


def execute_target_plan(
    toolkit: Any,
    plan: dict[str, Any],
    *,
    download_dir: str | Path,
    review_changes: Optional[dict[str, Any]] = None,
    show_frames: bool = True,
    fov_width_um: float = 100.0,
    fov_height_um: float = 75.0,
    poll_interval_s: float = 2.0,
    test_timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    """
    Execute an approved plan through the same IndentToolkit path used with the
    real instrument: move → capture → configure → run → monitor → export.

    SUPERSEDED — no current notebook calls this; NB5 v2 and NB7 both position
    with the closed-loop MotionController.jog_to(). Its move step (below) picks
    the arrow with ``"right" if dx > 0``, i.e. it assumes right = stage +x. That
    is FALSE on this instrument (measured dir_x = -1: "right" DECREASES x), so
    it drives X backwards. It happened to work before 2026-08-05 only because
    plan_target_actions() then produced an X-mirrored destination and the two
    errors cancelled; that mirror has been fixed, so this function is now wrong
    on its own. Do not reach for it — use CampaignRunner, or pass the arrow map
    from MotionController if you must revive it.
    """
    print("  [warn] execute_target_plan() assumes 'right' increases stage x, which "
          "is FALSE on this instrument — its X moves go the wrong way. Position "
          "with MotionController.jog_to() instead.")
    tool_log: list[dict[str, Any]] = []

    def _call(tool: str, args: dict[str, Any]) -> Any:
        out = toolkit.call_tool(tool, args)
        tool_log.append(
            {
                "tool": tool,
                "arguments": json.dumps(args, default=str)[:220],
                "ok": out.ok,
                "elapsed_s": round(getattr(out, "elapsed_s", 0.0) or 0.0, 3),
                "summary": str(out.result if out.ok else out.error)[:180],
            }
        )
        if not out.ok:
            raise RuntimeError(f"{tool} failed: {out.error}")
        return out.result

    # 1. Stage moves — recomputed from the live position, since earlier plans
    #    in the same approved batch may already have moved the stage.
    pos0 = _call("get_xyz_positions", {})
    dest_x, dest_y = plan["destination"]
    dx = dest_x - float(pos0.get("x", 0.0))
    dy = dest_y - float(pos0.get("y", 0.0))
    if abs(dx) >= 0.05:
        _call("move", {"amount": round(abs(dx), 2), "direction": "right" if dx > 0 else "left", "t": 0})
    if abs(dy) >= 0.05:
        _call("move", {"amount": round(abs(dy), 2), "direction": "up" if dy > 0 else "down", "t": 0})
    pos = _call("get_xyz_positions", {})

    # 2. Confirmation capture at the target
    cap = _call("capture_video_panel_remote", {})
    image_info = download_capture(toolkit, cap if isinstance(cap, dict) else {}, download_dir)
    if show_frames and image_info.get("local_path"):
        show_capture(
            image_info,
            title=f"Confirmation frame — {plan['label']}",
            note="Frame captured after the approved stage move, before indenting.",
            fov_width_um=fov_width_um,
            fov_height_um=fov_height_um,
        )

    # 3. Configure + run the indent array
    nx, ny, sp = plan["grid_nx"], plan["grid_ny"], plan["spacing_um"]
    _call(
        "starting_tests",
        {
            "sample_name": plan["sample_name"],
            "setup": plan.get("setup") or {},
            "indents": {"X Indents": nx, "Y Indents": ny, "X Spacing": sp, "Y Spacing": sp},
            "method_type": plan.get("method_type", "eh"),
        },
    )
    _call("start_test_normal", {"t": 0})

    # 4. Monitor until every indent reports complete.
    #    start_test_normal() only clicks "Start" and returns immediately — the
    #    instrument then runs the whole array on its own, which takes minutes on
    #    real hardware. So we must poll on a wall-clock deadline until it finishes
    #    before touching the review/export data. (The simulator advances one indent
    #    per passive poll and reports status=="complete"; the real iMicro returns
    #    only {"test_number", "raw_text"} parsed from the "Tests: N" OCR and has no
    #    "status" key, so completion is inferred from test_number reaching nx*ny.)
    expected = nx * ny
    poll_s = float(os.environ.get("NB5_POLL_INTERVAL_S", poll_interval_s))
    timeout_s = float(
        os.environ.get("NB5_TEST_TIMEOUT_S", test_timeout_s or 90.0 * expected + 120.0)
    )
    deadline = time.monotonic() + timeout_s
    status: dict[str, Any] = {}
    stable_complete = 0
    n_done = 0
    while True:
        status = _call("get_current_sample_and_test_ocr_passive", {})
        n_done = int(status.get("test_number") or 0)
        is_complete = status.get("status") == "complete" or n_done >= expected
        # Require two consecutive "complete" reads so a single stray OCR frame
        # can't make us export a half-finished array.
        stable_complete = stable_complete + 1 if is_complete else 0
        if stable_complete >= 2:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Indent array '{plan['sample_name']}' did not report complete "
                f"within {timeout_s:.0f}s (last read {n_done}/{expected} indents). "
                "The test is likely still running on the instrument — refusing to "
                "export partial data. Increase test_timeout_s / NB5_TEST_TIMEOUT_S "
                "if the array simply needs more time to finish."
            )
        time.sleep(poll_s)

    # 5. Export + download results
    export_name = f"{plan['sample_name']}_export"
    pr = _call(
        "process_review_file",
        {
            "sample_names": [plan["sample_name"]],
            "open_dir": "",
            "current_sample_file_name": plan["sample_name"],
            "export_dir": "exports",
            "export_file_name": export_name,
            "review_changes": review_changes or {},
            "review_type": plan.get("method_type", "eh"),
            "export_format": "csv",
        },
    )
    export_file = pr.get("export_file_name", f"{export_name}.csv")
    dl = toolkit.client.download_file("exports", export_file, local_dir=str(download_dir))
    rows = pd.read_csv(dl["local_path"])

    h = rows["Hardness (GPa)"]
    e = rows["Modulus (GPa)"]
    valid = (h > 1.5) & (h < 60)
    indent_data: dict[str, Any] = {
        "test_name": plan["sample_name"],
        "sample_name": plan["sample_name"],
        "n_indents": int(len(rows)),
        "grid_nx": nx,
        "grid_ny": ny,
        "spacing_um": sp,
        "method_type": plan.get("method_type", "eh"),
        "hardness_mean": float(h[valid].mean()),
        "hardness_std": float(h[valid].std(ddof=0)),
        "modulus_mean": float(e[valid].mean()),
        "modulus_std": float(e[valid].std(ddof=0)),
        "valid_indent_count": int(valid.sum()),
        "data_path": dl["local_path"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "position": (float(pos.get("x", 0.0)), float(pos.get("y", 0.0))),
        "area_name": plan["label"],
        "results": rows.to_dict("records"),
    }
    return {
        "indent_data": indent_data,
        "rows": rows,
        "image": image_info,
        "tool_log": tool_log,
        "monitor_status": status,
    }


def plot_planned_grids(
    image_info: dict[str, Any],
    plans: list[dict[str, Any]],
    stage_position: tuple[float, float],
    title: str = "Planning Agent — proposed indent arrays on the captured frame",
    fov_width_um: float = 100.0,
    fov_height_um: float = 75.0,
    site_gain_x: float = -1.0,
    site_gain_y: float = -1.0,
    crosshair_um: Optional[tuple[float, float]] = None,
) -> None:
    """Overlay each planned indent grid footprint on the analysis frame.

    THE GAINS ARE NOT OPTIONAL DECORATION. This function draws a STAGE
    destination back onto an IMAGE, so it must invert exactly the map that
    produced the destination. Until 2026-08-05 it inverted a hard-coded
    `image right = stage +x` instead, while the runner drove on the measured
    gain (-1.0) — so every site was drawn MIRRORED about the vertical
    centre-line, 2 x its offset away from where the stage actually went. The
    operator read the picture, not the stage log, and concluded the indenter
    was missing by ~98 um when it was landing within ~4 um of the chosen
    feature. Pass the same gains the runner uses; never re-derive the inverse.
    """
    import matplotlib.pyplot as plt

    from .instrument.imagecal import image_delta_from_stage

    path = resolve_image_path(image_info)
    if path is None:
        print("No local image to annotate.")
        return
    img = np.asarray(Image.open(path))
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.imshow(img, extent=[0, fov_width_um, fov_height_um, 0])
    cmap = plt.get_cmap("Set1")

    # Where the indenter actually points in this frame: the burned-in crosshair
    # when it was located, the geometric centre otherwise.
    ref = tuple(crosshair_um) if crosshair_um else (fov_width_um / 2, fov_height_um / 2)
    ax.plot(*ref, marker="+", ms=16, mew=2.0, color="#00e5ff", zorder=5)
    ax.annotate("indenter axis", ref, xytext=(9, 9), textcoords="offset points",
                fontsize=8, color="#00e5ff", fontweight="bold")

    for k, plan in enumerate(plans):
        # Destination back to image-frame coordinates of this capture, through
        # the EXACT inverse of the map that produced it.
        dx_img, dy_img = image_delta_from_stage(
            plan["destination"][0] - stage_position[0],
            plan["destination"][1] - stage_position[1], site_gain_x, site_gain_y)
        ix, iy = ref[0] + dx_img, ref[1] + dy_img
        nx, ny, sp = plan["grid_nx"], plan["grid_ny"], plan["spacing_um"]
        theta = math.radians(float(plan.get("rotation_deg", 0.0)))
        ct, st = math.cos(theta), math.sin(theta)
        ax.annotate("", xy=(ix, iy), xytext=ref,
                    arrowprops=dict(arrowstyle="->", color="#00e5ff",
                                    lw=1.1, alpha=0.75))
        c = cmap(k % 9)
        for i in range(nx):
            for j in range(ny):
                ux = (i - (nx - 1) / 2) * sp
                uy = (j - (ny - 1) / 2) * sp
                gx = ix + ux * ct - uy * st
                gy = iy + ux * st + uy * ct
                ax.plot(gx, gy, marker="D", ms=5.5, mfc=c, mec="white", mew=0.6)
        ax.annotate(
            f'{plan["sample_name"]}\n{nx}x{ny} @ {sp} µm, '
            f'{float(plan.get("rotation_deg", 0.0)):g}°',
            (ix, iy), xytext=(14, -26), textcoords="offset points",
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=c, ec="none", alpha=0.9),
        )
    ax.set_xlabel("image x (µm)")
    ax.set_ylabel("image y (µm)")
    ax.set_title(title, fontsize=11.5)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Result visualization
# ----------------------------------------------------------------------------


def indent_results_table(indent_history: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in indent_history:
        rows.append(
            {
                "array": d.get("test_name", "?"),
                "region": d.get("area_name", ""),
                "stage x (µm)": round(d.get("position", (0, 0))[0], 1),
                "stage y (µm)": round(d.get("position", (0, 0))[1], 1),
                "grid": f'{d.get("grid_nx", "?")}x{d.get("grid_ny", "?")}',
                "valid indents": d.get("valid_indent_count", 0),
                "H mean (GPa)": round(d.get("hardness_mean", float("nan")), 2),
                "H std": round(d.get("hardness_std", float("nan")), 2),
                "E mean (GPa)": round(d.get("modulus_mean", float("nan")), 1),
                "E std": round(d.get("modulus_std", float("nan")), 1),
            }
        )
    return pd.DataFrame(rows)


def combined_rows(executions: list[dict[str, Any]]) -> pd.DataFrame:
    """Stack per-array CSV rows into one dataframe labeled by region."""
    frames = []
    for ex in executions:
        df = ex["rows"].copy()
        df["region"] = ex["indent_data"].get("area_name", ex["indent_data"].get("test_name"))
        df["array"] = ex["indent_data"].get("test_name")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_load_depth(rows: pd.DataFrame, title: str = "Reconstructed load–depth curves") -> None:
    """
    Illustrative Berkovich P–h curves reconstructed from each indent's
    hardness/modulus/contact depth (the simulator does not stream raw P–h).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    cmap = plt.get_cmap("viridis")
    h_vals = rows["Hardness (GPa)"].to_numpy()
    hmin, hmax = float(h_vals.min()), float(h_vals.max())

    for _, r in rows.iterrows():
        H = float(r["Hardness (GPa)"])
        E = float(r["Modulus (GPa)"])
        hc = float(r["Contact Depth (nm)"])
        h_max = hc * 1.18
        p_max = H * 24.5 * hc**2 * 1e-6  # mN, Berkovich area function
        hf = h_max * float(np.clip(1.0 - 5.5 * H / max(E, 1e-6), 0.35, 0.9))
        c = cmap((H - hmin) / (hmax - hmin) if hmax > hmin else 0.5)

        hl = np.linspace(0, h_max, 60)
        ax.plot(hl, p_max * (hl / h_max) ** 2, color=c, lw=1.1, alpha=0.85)
        hu = np.linspace(h_max, hf, 40)
        ax.plot(hu, p_max * ((hu - hf) / (h_max - hf)) ** 1.5, color=c, lw=1.1, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(hmin, hmax))
    fig.colorbar(sm, ax=ax, label="Hardness (GPa)")
    ax.set_xlabel("Displacement into surface (nm)")
    ax.set_ylabel("Load (mN)")
    ax.set_title(f"{title}\n(illustrative curves reconstructed from H, E, h$_c$)", fontsize=11)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_property_map(all_rows: pd.DataFrame) -> None:
    """Spatial maps of measured hardness and modulus across all arrays."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, cmap in [
        (axes[0], "Hardness (GPa)", "inferno"),
        (axes[1], "Modulus (GPa)", "viridis"),
    ]:
        sc = ax.scatter(
            all_rows["X (um)"], all_rows["Y (um)"], c=all_rows[col],
            cmap=cmap, s=95, edgecolor="k", linewidths=0.5,
        )
        fig.colorbar(sc, ax=ax, label=col)
        for array, grp in all_rows.groupby("array"):
            ax.annotate(
                str(array), (grp["X (um)"].mean(), grp["Y (um)"].max() + 4),
                ha="center", fontsize=8.5, fontweight="bold", color="#333",
            )
        ax.set_xlabel("stage x (µm)")
        ax.set_ylabel("stage y (µm)")
        ax.set_title(f"Measured {col.split(' ')[0].lower()} map")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_region_comparison(
    all_rows: pd.DataFrame,
    expectations: Optional[dict[str, tuple[float, float]]] = None,
) -> None:
    """Box plots of hardness/modulus per region, with literature bands."""
    import matplotlib.pyplot as plt

    regions = list(all_rows["region"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in [(axes[0], "Hardness (GPa)"), (axes[1], "Modulus (GPa)")]:
        data = [all_rows.loc[all_rows["region"] == r, col] for r in regions]
        names = [r.replace("_", "\n") for r in regions]
        try:
            bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        except TypeError:  # matplotlib < 3.9 uses `labels`
            bp = ax.boxplot(data, labels=names, patch_artist=True)
        for patch, r in zip(bp["boxes"], regions):
            patch.set_facecolor(plt.get_cmap("Set2")(regions.index(r) % 8))
        if expectations and col.startswith("Hardness"):
            for i, r in enumerate(regions, start=1):
                if r in expectations:
                    lo, hi = expectations[r]
                    ax.fill_between(
                        [i - 0.32, i + 0.32], lo, hi, color="#1e8449", alpha=0.15,
                        label="literature expectation" if i == 1 else None,
                    )
            if any(r in expectations for r in regions):
                ax.legend(fontsize=8.5)
        ax.set_ylabel(col)
        ax.set_title(f"{col.split(' ')[0]} by region")
        ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()


def ground_truth_validation(all_rows: pd.DataFrame) -> None:
    """
    Simulator-only: overlay measured indents on the hidden ground-truth phase
    map and compare measured vs. true hardness. Proves the agents targeted
    real microstructure, not noise.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from instrument_control.synthetic_microstructure import (
        ground_truth_fov, properties_at,
    )

    cx = float(all_rows["X (um)"].mean())
    cy = float(all_rows["Y (um)"].mean())
    gt = ground_truth_fov(cx, cy, n=220)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cmap = ListedColormap(["#4a5d3a", "#cdd39b", "#1a1a1a"])
    axes[0].pcolormesh(gt["x"], gt["y"], gt["phase_id"], cmap=cmap, shading="auto")
    sc = axes[0].scatter(
        all_rows["X (um)"], all_rows["Y (um)"], c=all_rows["Hardness (GPa)"],
        cmap="inferno", s=80, edgecolor="white", linewidths=0.8, zorder=5,
    )
    fig.colorbar(sc, ax=axes[0], label="measured H (GPa)")
    axes[0].set_title("Indents over hidden ground-truth phase map\n"
                      "(dark=matrix, light=hard phase, black=pore)", fontsize=10.5)
    axes[0].set_xlabel("stage x (µm)")
    axes[0].set_ylabel("stage y (µm)")
    axes[0].set_aspect("equal")

    truth_h = [
        properties_at(float(r["X (um)"]), float(r["Y (um)"]))["hardness_gpa"]
        for _, r in all_rows.iterrows()
    ]
    axes[1].scatter(truth_h, all_rows["Hardness (GPa)"], s=60, alpha=0.8,
                    c=all_rows["Hardness (GPa)"], cmap="inferno", edgecolor="k", linewidths=0.4)
    lim = [0, max(max(truth_h), all_rows["Hardness (GPa)"].max()) * 1.1]
    axes[1].plot(lim, lim, "k--", lw=1, label="perfect agreement")
    axes[1].set_xlabel("ground-truth hardness (GPa)")
    axes[1].set_ylabel("measured hardness (GPa)")
    axes[1].set_title("Measured vs. ground truth (per indent)", fontsize=10.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Run summary export
# ----------------------------------------------------------------------------


def save_run_report(path: str | Path, **sections: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), **sections}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
