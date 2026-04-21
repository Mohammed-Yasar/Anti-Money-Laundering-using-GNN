"""
detection/baseline.py
---------------------
Feature standardization, risk scoring, normalization, and threshold selection.

Design guarantees (leakage-safety):
    - Standardizer.fit() is ONLY called on train_metrics.
    - The same mean/std are applied to val and test via transform().
    - RiskScorer.fit_normalizer() is ONLY called on train risk scores.
    - normalize() clips val/test scores to [min_train, max_train] before scaling.
    - select_threshold() is ONLY called on train risk scores.
"""

from __future__ import annotations

from typing import Any
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES = ["fanin", "fanout", "weighted_degree", "burst_score", "chain_score", "cycle_score"]

# Fixed weights (must sum to 1.0)
FEATURE_WEIGHTS: dict[str, float] = {
    "fanin":           0.30,
    "weighted_degree": 0.20,
    "burst_score":     0.20,
    "cycle_score":     0.15,
    "fanout":          0.10,
    "chain_score":     0.05,
}

EPSILON = 1e-9
THRESHOLD_PERCENTILE = 95.0  # top 5% → 95th percentile


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------

class Standardizer:
    """
    Fit mean/std on TRAIN metrics, apply same transform to any split.

    Attributes
    ----------
    means_ : dict[str, float]
    stds_  : dict[str, float]
    """

    def __init__(self) -> None:
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        self._fitted = False

    def fit(self, train_metrics: dict[str, dict[str, Any]]) -> "Standardizer":
        """
        Compute per-feature mean and std from TRAIN node metrics only.

        Parameters
        ----------
        train_metrics : dict
            node_id -> {feat: value, ...}  (output of metrics.compute_metrics)
        """
        for feat in FEATURES:
            vals = np.array([m[feat] for m in train_metrics.values()], dtype=float)
            self.means_[feat] = float(vals.mean())
            self.stds_[feat]  = float(vals.std())
        self._fitted = True
        return self

    def transform(
        self, metrics: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """
        Z-score each feature using TRAIN mean/std.

        Parameters
        ----------
        metrics : dict
            node_id -> {feat: value, ...}  (any split)

        Returns
        -------
        z_metrics : dict
            node_id -> {feat: z_score, ...}
        """
        if not self._fitted:
            raise RuntimeError("Standardizer must be fitted before transform().")

        z_metrics: dict[str, dict[str, float]] = {}
        for node_id, m in metrics.items():
            z_metrics[node_id] = {
                feat: (m[feat] - self.means_[feat]) / (self.stds_[feat] + EPSILON)
                for feat in FEATURES
            }
        return z_metrics


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------

class RiskScorer:
    """
    Fixed-weight weighted sum of z-scored features, normalised to [0, 1]
    using TRAIN min/max only.

    Attributes
    ----------
    min_train_ : float
    max_train_ : float
    threshold_ : float  (set after select_threshold() call)
    """

    def __init__(self) -> None:
        self.min_train_: float | None = None
        self.max_train_: float | None = None
        self.threshold_: float | None = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(z_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Compute raw weighted risk sum for each node.

        Parameters
        ----------
        z_metrics : dict
            node_id -> {feat: z_score}  (from Standardizer.transform)

        Returns
        -------
        raw_scores : dict
            node_id -> weighted sum (unbounded float)
        """
        raw: dict[str, float] = {}
        for node_id, z in z_metrics.items():
            raw[node_id] = sum(FEATURE_WEIGHTS[f] * z[f] for f in FEATURES)
        return raw

    # ------------------------------------------------------------------
    # Normalisation (fit on TRAIN only)
    # ------------------------------------------------------------------

    def fit_normalizer(self, train_scores: dict[str, float]) -> "RiskScorer":
        """
        Store min and max from TRAIN raw scores for later normalisation.
        MUST be called with train_scores only.
        """
        vals = np.array(list(train_scores.values()), dtype=float)
        self.min_train_ = float(vals.min())
        self.max_train_ = float(vals.max())
        return self

    def normalize(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Scale raw scores to [0, 1] using TRAIN min/max.
        Out-of-range values are clipped to [min_train, max_train].

        Parameters
        ----------
        scores : dict
            node_id -> raw weighted sum (any split)
        """
        if self.min_train_ is None or self.max_train_ is None:
            raise RuntimeError("fit_normalizer() must be called before normalize().")

        span = self.max_train_ - self.min_train_ + EPSILON
        normalized: dict[str, float] = {}
        for node_id, raw in scores.items():
            clipped = max(self.min_train_, min(self.max_train_, raw))
            normalized[node_id] = (clipped - self.min_train_) / span
        return normalized

    # ------------------------------------------------------------------
    # Threshold selection (TRAIN only)
    # ------------------------------------------------------------------

    def select_threshold(self, train_norm_scores: dict[str, float]) -> float:
        """
        Set threshold as the 95th-percentile of TRAIN normalised risk scores.
        MUST be called with train scores only.

        Returns
        -------
        threshold : float  (also stored as self.threshold_)
        """
        vals = np.array(list(train_norm_scores.values()), dtype=float)
        self.threshold_ = float(np.percentile(vals, THRESHOLD_PERCENTILE))
        return self.threshold_

    # ------------------------------------------------------------------
    # Apply threshold
    # ------------------------------------------------------------------

    def flag(self, norm_scores: dict[str, float]) -> dict[str, int]:
        """
        Apply stored threshold.  Returns {node_id: 0/1}.
        Raises if threshold not yet selected.
        """
        if self.threshold_ is None:
            raise RuntimeError("select_threshold() must be called before flag().")
        return {nid: int(score >= self.threshold_) for nid, score in norm_scores.items()}
