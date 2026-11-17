"""
objective_acquisition.py
========================
Objective-driven site selection: Gaussian-process surrogates over stage
(x, y) for each measured property, a scalarized multi-objective score, and
Expected Improvement acquisition to propose the next indent sites.

Mirrors the ErrorModel pattern from notebook 4: sklearn is optional
(``HAVE_SKLEARN``), the model degrades to an inverse-distance heuristic
below ``GP_MIN_SAMPLES`` observations, and EI uses ``math.erf`` so scipy
is never required.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np

from ..objective import ResearchObjective

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, ConstantKernel,
                                                  DotProduct)
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

GP_MIN_SAMPLES = 5
IDW_LENGTH_SCALE_UM = 15.0

PROPERTY_COLUMNS = {"hardness": "Hardness (GPa)", "modulus": "Modulus (GPa)"}
PROPERTY_LIMITS = {"hardness": (0.05, 100.0), "modulus": (1.0, 1000.0)}


def default_min_site_spacing(grid_nx: int, grid_ny: int, spacing_um: float) -> float:
    """Array footprint plus one spacing of clearance between array centers."""
    return (max(int(grid_nx), int(grid_ny)) - 1) * float(spacing_um) + float(spacing_um)


def _property_norms(
    objective: ResearchObjective,
    props: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Per-property (vmin, range) with a floor so noise never dominates."""
    norms: dict[str, tuple[float, float]] = {}
    for goal in objective.property_goals:
        v = props.get(goal.name)
        if v is None or len(v) == 0:
            norms[goal.name] = (0.0, 1.0)
            continue
        vmin, vmax = float(np.min(v)), float(np.max(v))
        mean = float(np.mean(np.abs(v))) or 1.0
        rng = max(vmax - vmin, 0.5, 0.05 * mean)
        norms[goal.name] = (vmin, rng)
    return norms


def scalarize_properties(
    objective: ResearchObjective,
    props: dict[str, np.ndarray],
    norms: Optional[dict[str, tuple[float, float]]] = None,
) -> np.ndarray:
    """Weighted scalar objective score in [~0, ~1] for arrays of property values."""
    norms = norms or _property_norms(objective, props)
    weights = objective.normalized_weights()
    total = np.zeros_like(next(iter(props.values()), np.zeros(0)), dtype=float)
    for goal in objective.property_goals:
        v = np.asarray(props.get(goal.name, np.zeros_like(total)), dtype=float)
        if goal.direction == "target_range" and goal.target_range:
            lo, hi = goal.target_range
            center, width = (lo + hi) / 2.0, max((hi - lo) / 2.0, 0.25)
            score = np.exp(-(((v - center) / width) ** 2))
        else:
            vmin, rng = norms[goal.name]
            score = (v - vmin) / rng
            if goal.direction == "minimize":
                score = 1.0 - score
        total = total + weights.get(goal.name, 0.0) * score
    return total


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _extract_observations(
    indent_history: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], list[tuple[float, float]]]:
    """
    Per-indent (x, y, hardness, modulus) rows from indent history, with an
    array-level fallback when per-indent results are missing.
    Returns (X, props, row_array_labels, array_centers).
    """
    xs, hs, es, labels = [], [], [], []
    centers: list[tuple[float, float]] = []
    for d in indent_history or []:
        label = str(d.get("area_name") or d.get("test_name") or "array")
        pos = d.get("position") or (None, None)
        if pos and pos[0] is not None:
            centers.append((float(pos[0]), float(pos[1])))
        rows = d.get("results") or []
        added = 0
        for r in rows:
            try:
                x, y = float(r["X (um)"]), float(r["Y (um)"])
                h = float(r[PROPERTY_COLUMNS["hardness"]])
                e = float(r[PROPERTY_COLUMNS["modulus"]])
            except (KeyError, TypeError, ValueError):
                continue
            if not (PROPERTY_LIMITS["hardness"][0] < h <= PROPERTY_LIMITS["hardness"][1]):
                continue
            if not (PROPERTY_LIMITS["modulus"][0] < e <= PROPERTY_LIMITS["modulus"][1]):
                continue
            xs.append((x, y)); hs.append(h); es.append(e); labels.append(label)
            added += 1
        if added == 0 and pos and pos[0] is not None:
            h = float(d.get("hardness_mean") or 0.0)
            e = float(d.get("modulus_mean") or 0.0)
            if h > 0 and e > 0:
                xs.append((float(pos[0]), float(pos[1])))
                hs.append(h); es.append(e); labels.append(label)
    X = np.asarray(xs, dtype=float).reshape(-1, 2)
    props = {"hardness": np.asarray(hs, dtype=float), "modulus": np.asarray(es, dtype=float)}
    return X, props, labels, centers


class ObjectiveSurrogate:
    """GP (or IDW-fallback) model of the scalarized objective over stage (x, y)."""

    def __init__(self, objective: ResearchObjective):
        self.objective = objective
        self.mode = "untrained"
        self.n_points = 0
        self.X = np.zeros((0, 2))
        self.props: dict[str, np.ndarray] = {}
        self.row_labels: list[str] = []
        self.array_centers: list[tuple[float, float]] = []
        self.norms: dict[str, tuple[float, float]] = {}
        self.y_obs = np.zeros(0)
        self._gps: dict[str, Any] = {}
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit
    def fit(self, indent_history: list[dict[str, Any]]) -> "ObjectiveSurrogate":
        self.X, self.props, self.row_labels, self.array_centers = _extract_observations(indent_history)
        self.n_points = len(self.X)
        if self.n_points == 0:
            self.mode = "untrained"
            return self

        self.norms = _property_norms(self.objective, self.props)
        self.y_obs = scalarize_properties(self.objective, self.props, self.norms)
        self.mode = "idw-heuristic"

        if HAVE_SKLEARN and self.n_points >= GP_MIN_SAMPLES:
            self._X_mean = self.X.mean(axis=0)
            self._X_std = self.X.std(axis=0) + 1e-9
            Xs = (self.X - self._X_mean) / self._X_std
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # kernel bound hits are expected on sparse data
                for name in PROPERTY_COLUMNS:
                    # RBF bounded to the grain scale; DotProduct captures monotone
                    # compositional gradients across the film
                    kernel = (ConstantKernel(1.0, (0.01, 100)) * RBF(15.0, (2.0, 60.0))
                              + ConstantKernel(1.0, (0.01, 100)) * DotProduct(1.0, (0.01, 100))
                              + WhiteKernel(0.25, (1e-3, 10)))
                    self._gps[name] = GaussianProcessRegressor(
                        kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                    ).fit(Xs, self.props[name])
            self.mode = "gaussian-process"
        return self

    # -------------------------------------------------------------- predict
    def predict(self, XY: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-property (mean, std) at stage coordinates XY (M, 2)."""
        XY = np.asarray(XY, dtype=float).reshape(-1, 2)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.mode == "untrained":
            for name in PROPERTY_COLUMNS:
                out[name] = (np.zeros(len(XY)), np.full(len(XY), np.inf))
            return out
        if self.mode == "gaussian-process":
            Xs = (XY - self._X_mean) / self._X_std
            for name, gp in self._gps.items():
                mu, sigma = gp.predict(Xs, return_std=True)
                out[name] = (np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float))
            return out
        # idw-heuristic: inverse-distance-squared mean; uncertainty grows with
        # distance to the nearest observation so unexplored regions score high
        d = np.linalg.norm(XY[:, None, :] - self.X[None, :, :], axis=2)
        w = 1.0 / (d ** 2 + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        d_nearest = d.min(axis=1)
        scale = np.minimum(1.0, d_nearest / IDW_LENGTH_SCALE_UM)
        for name in PROPERTY_COLUMNS:
            v = self.props[name]
            mu = w @ v
            base_sigma = max(float(np.std(v)), 0.5, 0.1 * float(np.mean(np.abs(v)) or 1.0))
            out[name] = (mu, base_sigma * scale)
        return out

    def _scalar_mu_sigma(self, XY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self.predict(XY)
        mu_s = scalarize_properties(
            self.objective, {n: mu for n, (mu, _) in preds.items()}, self.norms)
        weights = self.objective.normalized_weights()
        var = np.zeros(len(np.asarray(XY).reshape(-1, 2)))
        for goal in self.objective.property_goals:
            _, sigma = preds[goal.name]
            if goal.direction == "target_range" and goal.target_range:
                scale = max((goal.target_range[1] - goal.target_range[0]) / 2.0, 0.25)
            else:
                scale = self.norms.get(goal.name, (0.0, 1.0))[1]
            var = var + (weights.get(goal.name, 0.0) * sigma / scale) ** 2
        return mu_s, np.sqrt(var)

    # ---------------------------------------------------------- acquisition
    def best_observed(self) -> dict[str, Any]:
        if self.n_points == 0:
            return {}
        i = int(np.argmax(self.y_obs))
        return {
            "x_um": float(self.X[i, 0]),
            "y_um": float(self.X[i, 1]),
            "scalar_value": float(self.y_obs[i]),
            "hardness_gpa": float(self.props["hardness"][i]),
            "modulus_gpa": float(self.props["modulus"][i]),
            "array": self.row_labels[i] if i < len(self.row_labels) else "",
        }

    def baseline(self) -> float:
        return float(np.mean(self.y_obs)) if self.n_points else 0.0

    def expected_improvement(self, XY: np.ndarray, xi: float = 0.01) -> np.ndarray:
        if self.mode == "untrained":
            return np.zeros(len(np.asarray(XY).reshape(-1, 2)))
        mu_s, sigma_s = self._scalar_mu_sigma(XY)
        y_best = float(np.max(self.y_obs))
        improve = mu_s - y_best - xi
        ei = np.where(improve > 0, improve, 0.0)
        ok = sigma_s > 1e-9
        z = np.zeros_like(mu_s)
        z[ok] = improve[ok] / sigma_s[ok]
        ei[ok] = improve[ok] * _norm_cdf(z[ok]) + sigma_s[ok] * _norm_pdf(z[ok])
        return np.maximum(ei, 0.0)

    def candidate_grid(self, pitch_um: float = 2.0) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.objective.constraints.fov_bounds
        mx = self.objective.constraints.edge_margin_frac * (xmax - xmin)
        my = self.objective.constraints.edge_margin_frac * (ymax - ymin)
        xs = np.arange(xmin + mx, xmax - mx + 1e-9, pitch_um)
        ys = np.arange(ymin + my, ymax - my + 1e-9, pitch_um)
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack([gx.ravel(), gy.ravel()])

    def propose_sites(
        self,
        n_sites: int = 1,
        *,
        existing_sites: Optional[list[tuple[float, float]]] = None,
        round_index: int = 1,
        stage_position: Optional[tuple[float, float]] = None,
        fov_width_um: float = 100.0,
        fov_height_um: float = 75.0,
        xi: float = 0.01,
    ) -> list[dict[str, Any]]:
        """
        Greedy Expected-Improvement site proposals in stage coordinates,
        keeping ``min_site_spacing_um`` clearance from existing array centers
        and from each other. Includes x_frac/y_frac for plan_target_actions
        when ``stage_position`` (the FOV center) is given.
        """
        if self.mode == "untrained":
            return []
        cands = self.candidate_grid()
        spacing = float(self.objective.constraints.min_site_spacing_um)
        keep_out = list(self.array_centers) + [tuple(s) for s in (existing_sites or [])]
        mask = np.ones(len(cands), dtype=bool)
        for ex, ey in keep_out:
            mask &= np.linalg.norm(cands - np.array([ex, ey]), axis=1) >= spacing
        if not mask.any():
            return []

        ei = self.expected_improvement(cands, xi=xi)
        mu_s, sigma_s = self._scalar_mu_sigma(cands)
        preds = self.predict(cands)
        ei = np.where(mask, ei, -1.0)

        proposals: list[dict[str, Any]] = []
        for k in range(int(n_sites)):
            i = int(np.argmax(ei))
            if ei[i] < 0:
                break
            x, y = float(cands[i, 0]), float(cands[i, 1])
            target: dict[str, Any] = {
                "x_um": round(x, 2),
                "y_um": round(y, 2),
                "label": f"bo_round{round_index}_site{k + 1}",
                "type": "bo_acquisition",
                "source": "bo_acquisition",
                "priority": k + 1,
                "reason": (
                    f"Expected Improvement {float(ei[i]):.3f} — predicted objective "
                    f"{float(mu_s[i]):.2f}±{float(sigma_s[i]):.2f} ({self.mode})"
                ),
                "acquisition": {
                    "ei": float(ei[i]),
                    "mu_scalar": float(mu_s[i]),
                    "sigma_scalar": float(sigma_s[i]),
                    "pred_hardness_gpa": float(preds["hardness"][0][i]),
                    "pred_modulus_gpa": float(preds["modulus"][0][i]),
                    "mode": self.mode,
                },
            }
            if stage_position is not None:
                target["x_frac"] = float(np.clip(0.5 + (x - stage_position[0]) / fov_width_um, 0.02, 0.98))
                target["y_frac"] = float(np.clip(0.5 - (y - stage_position[1]) / fov_height_um, 0.02, 0.98))
            proposals.append(target)
            ei = np.where(np.linalg.norm(cands - cands[i], axis=1) < spacing, -1.0, ei)
        return proposals

    # -------------------------------------------------------------- reports
    def posterior_maps(self, n: int = 70) -> dict[str, Any]:
        """Dense posterior grids for plotting (mean/σ/EI over the FOV)."""
        xmin, xmax, ymin, ymax = self.objective.constraints.fov_bounds
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        gx, gy = np.meshgrid(xs, ys)
        XY = np.column_stack([gx.ravel(), gy.ravel()])
        preds = self.predict(XY)
        mu_s, sigma_s = self._scalar_mu_sigma(XY)
        ei = self.expected_improvement(XY)
        shape = gx.shape
        return {
            "x": xs, "y": ys, "extent": (xmin, xmax, ymin, ymax),
            "hardness_mean": preds["hardness"][0].reshape(shape),
            "modulus_mean": preds["modulus"][0].reshape(shape),
            "scalar_mean": mu_s.reshape(shape),
            "scalar_sigma": sigma_s.reshape(shape),
            "ei": ei.reshape(shape),
        }

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode": self.mode,
            "n_points": int(self.n_points),
            "baseline_scalar": self.baseline(),
            "best": self.best_observed(),
        }
        if self.mode == "untrained":
            out.update({"max_ei": 0.0, "mean_scalar_sigma": float("nan")})
            return out
        cands = self.candidate_grid(pitch_um=4.0)
        _, sigma_s = self._scalar_mu_sigma(cands)
        out["max_ei"] = float(np.max(self.expected_improvement(cands)))
        out["mean_scalar_sigma"] = float(np.mean(sigma_s))
        return out


# ---------------------------------------------------------------------------
# Round-1 objective-aware ranking of image-analysis targets
# ---------------------------------------------------------------------------

_PHASE_ESTIMATES = [
    # (matcher substrings, hardness GPa, modulus GPa, note)
    (("pore", "void", "defect", "crack"), 1.0, 30.0, "pore/defect — very low properties"),
    (("boundary", "interface"), 8.5, 178.0, "grain boundary — hardening above matrix"),
    (("matrix", "dark", "lower"), 6.5, 165.0, "matrix — reference low/moderate properties"),
    (("hard", "precipitate", "bright", "second", "higher"), 11.5, 210.0,
     "bright second phase — high hardness and modulus expected"),
]


def _estimate_target_properties(target: dict[str, Any]) -> tuple[float, float, str]:
    text = " ".join(
        str(target.get(k, "")) for k in
        ("label", "type", "reason", "expected_contrast", "expected_hardness")
    ).lower()
    for keys, h, e, note in _PHASE_ESTIMATES:
        if any(k in text for k in keys):
            return h, e, note
    return 8.0, 180.0, "unclassified region — mid-range assumed"


def rank_targets_by_objective(
    targets: list[dict[str, Any]],
    objective: ResearchObjective,
    image_analysis: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Re-rank image-analysis targets by expected alignment with the objective,
    replacing the fixed priority queue. Returns new target dicts sorted
    best-first with ``objective_score`` and ``rank_reason`` attached.
    """
    if not targets:
        return []
    estimates = [_estimate_target_properties(t) for t in targets]
    props = {
        "hardness": np.array([h for h, _, _ in estimates], dtype=float),
        "modulus": np.array([e for _, e, _ in estimates], dtype=float),
    }
    scores = scalarize_properties(objective, props)
    ranked = []
    for t, score, (h, e, note) in zip(targets, scores, estimates):
        rt = dict(t)
        rt["objective_score"] = round(float(score), 3)
        rt["rank_reason"] = f"{note} (est. H~{h:g} GPa, E~{e:g} GPa)"
        ranked.append(rt)
    ranked.sort(key=lambda t: -t["objective_score"])
    for i, t in enumerate(ranked):
        t["priority"] = i + 1
    return ranked
