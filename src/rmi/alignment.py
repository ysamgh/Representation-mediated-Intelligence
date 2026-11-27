"""Numerically explicit Procrustes alignment for paired row-wise activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScaleMode = Literal["spread", "least_squares", "none"]
FloatArray = NDArray[np.float64]


def _as_matrix(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got ndim={array.ndim}.")
    if min(array.shape, default=0) == 0:
        raise ValueError(f"{name} must have non-zero dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


@dataclass(frozen=True)
class ProcrustesMap:
    """Fitted affine, scaled semi-orthogonal map."""

    rotation: FloatArray
    source_mean: FloatArray
    target_mean: FloatArray
    scale: float
    source_dim: int
    target_dim: int
    n_samples: int
    rank: int
    singular_values: FloatArray
    scale_mode: ScaleMode

    def transform(self, values: ArrayLike) -> FloatArray:
        """Map one activation vector or a batch from source to target space."""
        array = np.asarray(values, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.source_dim:
            raise ValueError(
                f"Expected shape (n, {self.source_dim}) or ({self.source_dim},); "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("values contains NaN or infinite values.")
        mapped = ((array - self.source_mean) @ self.rotation) * self.scale
        mapped += self.target_mean
        return mapped[0] if single else mapped

    def inverse_transform(self, values: ArrayLike) -> FloatArray:
        """Invert a square orthogonal map. Rectangular maps are not bijective."""
        if self.source_dim != self.target_dim:
            raise ValueError("inverse_transform requires a square map.")
        array = np.asarray(values, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.target_dim:
            raise ValueError(
                f"Expected shape (n, {self.target_dim}) or ({self.target_dim},); "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("values contains NaN or infinite values.")
        restored = ((array - self.target_mean) / self.scale) @ self.rotation.T
        restored += self.source_mean
        return restored[0] if single else restored

    @property
    def orthogonality_error(self) -> float:
        """Frobenius residual for the applicable semi-orthogonality constraint."""
        if self.source_dim >= self.target_dim:
            residual = self.rotation.T @ self.rotation - np.eye(self.target_dim)
        else:
            residual = self.rotation @ self.rotation.T - np.eye(self.source_dim)
        return float(np.linalg.norm(residual, ord="fro"))


def fit_procrustes(
    source: ArrayLike,
    target: ArrayLike,
    *,
    scale: ScaleMode = "spread",
    rank_tolerance: float | None = None,
) -> ProcrustesMap:
    """Fit a centered orthogonal Procrustes map using paired samples.

    Inputs use shape (n_samples, hidden_dimension). The transform is fitted
    exclusively from the supplied paired calibration rows. Callers must create
    train/validation/test splits before calling this function.

    Scale modes:
      spread: match centered Frobenius spreads (default).
      least_squares: minimize centered squared error for the fitted rotation.
      none: use unit scale.

    Rank-deficient calibration data is allowed but reported because the map
    may be non-identifiable outside the observed subspace.
    """
    x = _as_matrix(source, "source")
    y = _as_matrix(target, "target")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"source and target must have the same sample count; "
            f"got {x.shape[0]} and {y.shape[0]}."
        )
    if x.shape[0] < 2:
        raise ValueError("At least two paired samples are required.")
    if scale not in ("spread", "least_squares", "none"):
        raise ValueError(f"Unsupported scale mode: {scale!r}.")

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean

    if np.linalg.norm(x_centered) == 0.0:
        raise ValueError("source has zero centered variance.")
    if np.linalg.norm(y_centered) == 0.0:
        raise ValueError("target has zero centered variance.")

    cross_covariance = x_centered.T @ y_centered
    u, singular_values, vt = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u @ vt

    if rank_tolerance is None:
        rank = int(np.linalg.matrix_rank(cross_covariance))
    else:
        if rank_tolerance < 0:
            raise ValueError("rank_tolerance must be non-negative.")
        rank = int(np.count_nonzero(singular_values > rank_tolerance))

    if scale == "spread":
        fitted_scale = float(
            np.linalg.norm(y_centered, ord="fro")
            / np.linalg.norm(x_centered, ord="fro")
        )
    elif scale == "least_squares":
        denominator = float(np.sum(x_centered * x_centered))
        fitted_scale = float(np.sum(singular_values) / denominator)
    else:
        fitted_scale = 1.0

    if not np.isfinite(fitted_scale) or fitted_scale <= 0.0:
        raise ValueError("Fitted scale must be finite and strictly positive.")

    return ProcrustesMap(
        rotation=rotation,
        source_mean=x_mean,
        target_mean=y_mean,
        scale=fitted_scale,
        source_dim=x.shape[1],
        target_dim=y.shape[1],
        n_samples=x.shape[0],
        rank=rank,
        singular_values=singular_values,
        scale_mode=scale,
    )
