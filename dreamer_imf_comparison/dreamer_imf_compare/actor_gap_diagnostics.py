"""Dependency-light diagnostics for locating a frozen-model actor gap.

This module deliberately owns no environment or checkpoint I/O.  A study
runner may feed it retained NumPy-compatible arrays, but the statistical and
provenance rules live here so that they can be tested independently:

* coverage is an empirical score in standardized observation-action space,
  calibrated only on the declared training partition;
* one-step target error is decomposed into reward, continuation, and a
  value-projected transition term with an exactly checked Bellman closure;
* horizon summaries first average nested observations inside a world-model
  seed and use world-model seeds as the equally weighted top-level units;
* independent-noise gains and gradient cosines require disjoint noise IDs;
* candidate ranking, regret, and return calibration are emitted only for
  explicitly verified exact simulator branches sharing common reset IDs.

All public result objects are frozen dataclasses containing only immutable,
JSON-compatible scalar/tuple fields.  ``to_dict()`` returns a fresh ordinary
mapping suitable for ``json.dumps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

COVERAGE_SCHEMA = "trajectory-imf-actor-gap-coverage-v1"
RESIDUAL_SCHEMA = "trajectory-imf-actor-gap-component-residual-v1"
HORIZON_SCHEMA = "trajectory-imf-actor-gap-horizon-aggregate-v1"
NOISE_SCHEMA = "trajectory-imf-actor-gap-independent-noise-v2"
PROVENANCE_SCHEMA = "trajectory-imf-counterfactual-provenance-v1"
COUNTERFACTUAL_SCHEMA = "trajectory-imf-actor-gap-counterfactual-v1"
SCHEMA_VERSIONS = (
    COVERAGE_SCHEMA,
    RESIDUAL_SCHEMA,
    HORIZON_SCHEMA,
    NOISE_SCHEMA,
    PROVENANCE_SCHEMA,
    COUNTERFACTUAL_SCHEMA,
)

_TRAIN_ONLY = "train_only"
_STANDARDIZATION = "per_coordinate_train_mean_population_std"
_DISTANCE = "root_mean_square_euclidean"
_RESIDUAL_CONVENTION = (
    "reward + discount*(predicted_continuation-target_continuation)*"
    "target_next_value + discount*predicted_continuation*"
    "(predicted_next_value-target_next_value)"
)


class UnidentifiableCounterfactualError(ValueError):
    """Raised when exact-branch-only metrics are requested from offline data."""


def _finite_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {result.shape}")
    if result.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _string_tuple(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identifiers")
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain nonempty identifiers")
    return result


def _float_tuple(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, name=name) for value in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _matrix_tuple(value: Any, *, name: str) -> tuple[tuple[float, ...], ...]:
    matrix = _finite_array(value, ndim=2, name=name)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _sample_digest(observations: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, array in (("observations", observations), ("actions", actions)):
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _quantile(values: np.ndarray, levels: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(np.quantile(values, level)) for level in levels)


def _rms_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dimensions = left.shape[1]
    # The norm identity keeps the temporary allocation two-dimensional rather
    # than materializing [left, right, feature].  Roundoff can make an exact
    # self-distance microscopically negative, hence the explicit clamp.
    squared = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * np.matmul(left, right.T)
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared / dimensions)


def _chunk_size(value: Any) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError("distance_chunk_size must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("distance_chunk_size must be a positive integer")
    return result


def _leave_one_out_knn(points: np.ndarray, *, k: int, chunk_size: int) -> np.ndarray:
    """Compute exact train k-NN distances with bounded temporary memory."""

    result = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        distances = _rms_distances(points[start:stop], points)
        local_rows = np.arange(stop - start)
        distances[local_rows, np.arange(start, stop)] = np.inf
        result[start:stop] = np.partition(distances, kth=k - 1, axis=1)[:, k - 1]
    return result


def _leave_one_out_density(
    points: np.ndarray, *, bandwidth: float, chunk_size: int
) -> np.ndarray:
    """Compute exact leave-one-out Gaussian similarities blockwise."""

    result = np.empty(points.shape[0], dtype=np.float64)
    denominator = points.shape[0] - 1
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        distances = _rms_distances(points[start:stop], points)
        kernels = np.exp(-0.5 * np.square(distances / bandwidth))
        kernels[np.arange(stop - start), np.arange(start, stop)] = 0.0
        result[start:stop] = kernels.sum(axis=1) / denominator
    return result


def _query_knn_and_density(
    query: np.ndarray,
    train: np.ndarray,
    *,
    k: int,
    bandwidth: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    knn = np.empty(query.shape[0], dtype=np.float64)
    density = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        distances = _rms_distances(query[start:stop], train)
        knn[start:stop] = np.partition(distances, kth=k - 1, axis=1)[:, k - 1]
        density[start:stop] = np.mean(
            np.exp(-0.5 * np.square(distances / bandwidth)), axis=1
        )
    return knn, density


@dataclass(frozen=True, slots=True)
class CoverageCalibration:
    """Immutable train-only standardization and empirical coverage index."""

    train_sample_count: int
    observation_dim: int
    action_dim: int
    k: int
    bandwidth: float
    observation_mean: tuple[float, ...]
    observation_scale: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_scale: tuple[float, ...]
    constant_observation_dimensions: tuple[int, ...]
    constant_action_dimensions: tuple[int, ...]
    standardized_train_points: tuple[tuple[float, ...], ...]
    train_leave_one_out_knn_distances: tuple[float, ...]
    train_leave_one_out_kernel_densities: tuple[float, ...]
    quantile_levels: tuple[float, ...]
    distance_quantiles: tuple[float, ...]
    density_quantiles: tuple[float, ...]
    training_data_sha256: str
    schema_version: str = field(default=COVERAGE_SCHEMA, init=False)
    calibration_partition: str = field(default=_TRAIN_ONLY, init=False)
    standardization: str = field(default=_STANDARDIZATION, init=False)
    distance_metric: str = field(default=_DISTANCE, init=False)
    density_definition: str = field(
        default="mean_gaussian_kernel_similarity_without_volume_normalization",
        init=False,
    )

    def __post_init__(self) -> None:
        if self.train_sample_count < 2:
            raise ValueError("coverage calibration needs at least two train samples")
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if not 1 <= self.k < self.train_sample_count:
            raise ValueError("k must be between 1 and train_sample_count - 1")
        object.__setattr__(
            self, "bandwidth", _positive_float(self.bandwidth, name="bandwidth")
        )
        for attribute, expected in (
            ("observation_mean", self.observation_dim),
            ("observation_scale", self.observation_dim),
            ("action_mean", self.action_dim),
            ("action_scale", self.action_dim),
        ):
            values = _float_tuple(getattr(self, attribute), name=attribute)
            if len(values) != expected:
                raise ValueError(f"{attribute} must have length {expected}")
            if "scale" in attribute and any(value <= 0.0 for value in values):
                raise ValueError(f"{attribute} must be strictly positive")
            object.__setattr__(self, attribute, values)
        for attribute, upper_bound in (
            ("constant_observation_dimensions", self.observation_dim),
            ("constant_action_dimensions", self.action_dim),
        ):
            values = tuple(int(value) for value in getattr(self, attribute))
            if len(set(values)) != len(values) or any(
                value < 0 or value >= upper_bound for value in values
            ):
                raise ValueError(f"{attribute} contains an invalid dimension")
            object.__setattr__(self, attribute, values)
        points = _matrix_tuple(
            self.standardized_train_points, name="standardized_train_points"
        )
        if len(points) != self.train_sample_count:
            raise ValueError("standardized_train_points row count is inconsistent")
        if len(points[0]) != self.observation_dim + self.action_dim:
            raise ValueError("standardized_train_points feature count is inconsistent")
        object.__setattr__(self, "standardized_train_points", points)
        for attribute in (
            "train_leave_one_out_knn_distances",
            "train_leave_one_out_kernel_densities",
        ):
            values = _float_tuple(getattr(self, attribute), name=attribute)
            if len(values) != self.train_sample_count or any(
                value < 0.0 for value in values
            ):
                raise ValueError(f"{attribute} is inconsistent with train samples")
            object.__setattr__(self, attribute, values)
        if any(value > 1.0 for value in self.train_leave_one_out_kernel_densities):
            raise ValueError("train kernel densities must lie in [0, 1]")
        levels = _float_tuple(self.quantile_levels, name="quantile_levels")
        if tuple(sorted(set(levels))) != levels or any(
            level < 0.0 or level > 1.0 for level in levels
        ):
            raise ValueError("quantile_levels must be unique, sorted, and in [0, 1]")
        object.__setattr__(self, "quantile_levels", levels)
        for attribute in ("distance_quantiles", "density_quantiles"):
            values = _float_tuple(getattr(self, attribute), name=attribute)
            if (
                len(values) != len(levels)
                or any(value < 0.0 for value in values)
                or tuple(sorted(values)) != values
            ):
                raise ValueError(f"{attribute} must match quantile_levels")
            object.__setattr__(self, attribute, values)
        if any(value > 1.0 for value in self.density_quantiles):
            raise ValueError("density_quantiles must lie in [0, 1]")
        if len(self.training_data_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.training_data_sha256
        ):
            raise ValueError("training_data_sha256 must be a lowercase SHA-256 digest")

    def metadata_dict(self) -> dict[str, Any]:
        """Return report-sized calibration metadata without retained train rows."""

        return {
            "schema_version": self.schema_version,
            "calibration_partition": self.calibration_partition,
            "standardization": self.standardization,
            "distance_metric": self.distance_metric,
            "density_definition": self.density_definition,
            "train_sample_count": self.train_sample_count,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "k": self.k,
            "bandwidth": self.bandwidth,
            "observation_mean": list(self.observation_mean),
            "observation_scale": list(self.observation_scale),
            "action_mean": list(self.action_mean),
            "action_scale": list(self.action_scale),
            "constant_observation_dimensions": list(
                self.constant_observation_dimensions
            ),
            "constant_action_dimensions": list(self.constant_action_dimensions),
            "quantile_levels": list(self.quantile_levels),
            "distance_quantiles": list(self.distance_quantiles),
            "density_quantiles": list(self.density_quantiles),
            "training_data_sha256": self.training_data_sha256,
        }

    def to_dict(self, *, include_train_index: bool = True) -> dict[str, Any]:
        result = self.metadata_dict()
        if include_train_index:
            result.update(
                {
                    "standardized_train_points": [
                        list(row) for row in self.standardized_train_points
                    ],
                    "train_leave_one_out_knn_distances": list(
                        self.train_leave_one_out_knn_distances
                    ),
                    "train_leave_one_out_kernel_densities": list(
                        self.train_leave_one_out_kernel_densities
                    ),
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class CoveragePoint:
    sample_index: int
    nearest_neighbor_distance: float
    kernel_density: float
    distance_coverage_score: float
    density_coverage_score: float
    conservative_coverage_score: float
    within_train_distance_q95: bool

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        for attribute in ("nearest_neighbor_distance", "kernel_density"):
            value = _finite_float(getattr(self, attribute), name=attribute)
            if value < 0.0:
                raise ValueError(f"{attribute} must be nonnegative")
            object.__setattr__(self, attribute, value)
        for attribute in (
            "distance_coverage_score",
            "density_coverage_score",
            "conservative_coverage_score",
        ):
            value = _finite_float(getattr(self, attribute), name=attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attribute} must lie in [0, 1]")
            object.__setattr__(self, attribute, value)
        if not isinstance(self.within_train_distance_q95, (bool, np.bool_)):
            raise TypeError("within_train_distance_q95 must be boolean")
        object.__setattr__(
            self, "within_train_distance_q95", bool(self.within_train_distance_q95)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
            "kernel_density": self.kernel_density,
            "distance_coverage_score": self.distance_coverage_score,
            "density_coverage_score": self.density_coverage_score,
            "conservative_coverage_score": self.conservative_coverage_score,
            "within_train_distance_q95": self.within_train_distance_q95,
        }


@dataclass(frozen=True, slots=True)
class CoverageScores:
    calibration_training_data_sha256: str
    points: tuple[CoveragePoint, ...]
    schema_version: str = field(default=COVERAGE_SCHEMA, init=False)
    interpretation: str = field(
        default="empirical_standardized_observation_action_coverage_not_mathematical_support",
        init=False,
    )

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not points:
            raise ValueError("coverage scores must contain at least one point")
        if not all(isinstance(point, CoveragePoint) for point in points):
            raise TypeError("points must contain CoveragePoint instances")
        object.__setattr__(self, "points", points)
        if len(self.calibration_training_data_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.calibration_training_data_sha256
        ):
            raise ValueError(
                "calibration_training_data_sha256 must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, Any]:
        conservative = np.asarray(
            [point.conservative_coverage_score for point in self.points],
            dtype=np.float64,
        )
        distances = np.asarray(
            [point.nearest_neighbor_distance for point in self.points],
            dtype=np.float64,
        )
        return {
            "schema_version": self.schema_version,
            "interpretation": self.interpretation,
            "calibration_training_data_sha256": self.calibration_training_data_sha256,
            "sample_count": len(self.points),
            "mean_conservative_coverage_score": float(np.mean(conservative)),
            "minimum_conservative_coverage_score": float(np.min(conservative)),
            "mean_nearest_neighbor_distance": float(np.mean(distances)),
            "within_train_distance_q95_fraction": float(
                np.mean([point.within_train_distance_q95 for point in self.points])
            ),
            "points": [point.to_dict() for point in self.points],
        }


def fit_coverage_calibration(
    train_observations: Any,
    train_actions: Any,
    *,
    k: int = 1,
    bandwidth: float | None = None,
    quantile_levels: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    minimum_scale: float = 1e-8,
    distance_chunk_size: int = 1024,
) -> CoverageCalibration:
    """Fit standardization and empirical thresholds using train rows only.

    Constant coordinates use a neutral scale of one and are named explicitly
    in the returned metadata.  Train calibration scores use leave-one-out
    neighbors/densities, preventing trivial self-distance calibration.
    """

    observations = _finite_array(train_observations, ndim=2, name="train_observations")
    actions = _finite_array(train_actions, ndim=2, name="train_actions")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("train observations and actions must have equal row counts")
    samples = observations.shape[0]
    if samples < 2:
        raise ValueError("coverage calibration needs at least two train samples")
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    k = int(k)
    if not 1 <= k < samples:
        raise ValueError("k must be between 1 and number of train samples - 1")
    minimum_scale = _positive_float(minimum_scale, name="minimum_scale")
    distance_chunk_size = _chunk_size(distance_chunk_size)
    levels = tuple(
        _finite_float(level, name="quantile level") for level in quantile_levels
    )
    if (
        not levels
        or tuple(sorted(set(levels))) != levels
        or any(level < 0.0 or level > 1.0 for level in levels)
    ):
        raise ValueError("quantile_levels must be unique, sorted, and in [0, 1]")

    observation_mean = observations.mean(axis=0)
    raw_observation_scale = observations.std(axis=0)
    constant_observation = tuple(
        int(index) for index in np.flatnonzero(raw_observation_scale < minimum_scale)
    )
    observation_scale = np.where(
        raw_observation_scale < minimum_scale, 1.0, raw_observation_scale
    )
    action_mean = actions.mean(axis=0)
    raw_action_scale = actions.std(axis=0)
    constant_action = tuple(
        int(index) for index in np.flatnonzero(raw_action_scale < minimum_scale)
    )
    action_scale = np.where(raw_action_scale < minimum_scale, 1.0, raw_action_scale)
    points = np.concatenate(
        (
            (observations - observation_mean) / observation_scale,
            (actions - action_mean) / action_scale,
        ),
        axis=1,
    )
    knn = _leave_one_out_knn(points, k=k, chunk_size=distance_chunk_size)
    if bandwidth is None:
        positive = knn[knn > minimum_scale]
        bandwidth_value = float(np.median(positive)) if positive.size else minimum_scale
    else:
        bandwidth_value = _positive_float(bandwidth, name="bandwidth")
    bandwidth_value = max(bandwidth_value, minimum_scale)
    densities = _leave_one_out_density(
        points, bandwidth=bandwidth_value, chunk_size=distance_chunk_size
    )

    return CoverageCalibration(
        train_sample_count=samples,
        observation_dim=observations.shape[1],
        action_dim=actions.shape[1],
        k=k,
        bandwidth=bandwidth_value,
        observation_mean=tuple(float(value) for value in observation_mean),
        observation_scale=tuple(float(value) for value in observation_scale),
        action_mean=tuple(float(value) for value in action_mean),
        action_scale=tuple(float(value) for value in action_scale),
        constant_observation_dimensions=constant_observation,
        constant_action_dimensions=constant_action,
        standardized_train_points=tuple(
            tuple(float(value) for value in row) for row in points
        ),
        train_leave_one_out_knn_distances=tuple(float(value) for value in knn),
        train_leave_one_out_kernel_densities=tuple(float(value) for value in densities),
        quantile_levels=levels,
        distance_quantiles=_quantile(knn, levels),
        density_quantiles=_quantile(densities, levels),
        training_data_sha256=_sample_digest(observations, actions),
    )


def score_observation_action_coverage(
    calibration: CoverageCalibration,
    observations: Any,
    actions: Any,
    *,
    distance_chunk_size: int = 1024,
) -> CoverageScores:
    """Score held-out observation-action rows against a train-only index."""

    if not isinstance(calibration, CoverageCalibration):
        raise TypeError("calibration must be a CoverageCalibration")
    distance_chunk_size = _chunk_size(distance_chunk_size)
    observation_array = _finite_array(observations, ndim=2, name="observations")
    action_array = _finite_array(actions, ndim=2, name="actions")
    if observation_array.shape[0] != action_array.shape[0]:
        raise ValueError("observations and actions must have equal row counts")
    if observation_array.shape[1] != calibration.observation_dim:
        raise ValueError("observation feature count does not match calibration")
    if action_array.shape[1] != calibration.action_dim:
        raise ValueError("action feature count does not match calibration")
    query = np.concatenate(
        (
            (observation_array - np.asarray(calibration.observation_mean))
            / np.asarray(calibration.observation_scale),
            (action_array - np.asarray(calibration.action_mean))
            / np.asarray(calibration.action_scale),
        ),
        axis=1,
    )
    train = np.asarray(calibration.standardized_train_points, dtype=np.float64)
    knn, densities = _query_knn_and_density(
        query,
        train,
        k=calibration.k,
        bandwidth=calibration.bandwidth,
        chunk_size=distance_chunk_size,
    )
    train_knn = np.asarray(
        calibration.train_leave_one_out_knn_distances, dtype=np.float64
    )
    train_density = np.asarray(
        calibration.train_leave_one_out_kernel_densities, dtype=np.float64
    )
    q95 = float(np.quantile(train_knn, 0.95))
    rows: list[CoveragePoint] = []
    for index, (distance, density) in enumerate(zip(knn, densities, strict=True)):
        distance_score = float(np.mean(train_knn >= distance))
        density_score = float(np.mean(train_density <= density))
        rows.append(
            CoveragePoint(
                sample_index=index,
                nearest_neighbor_distance=float(distance),
                kernel_density=float(density),
                distance_coverage_score=distance_score,
                density_coverage_score=density_score,
                conservative_coverage_score=min(distance_score, density_score),
                within_train_distance_q95=bool(distance <= q95),
            )
        )
    return CoverageScores(
        calibration_training_data_sha256=calibration.training_data_sha256,
        points=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class ComponentResidualRow:
    sample_index: int
    reward_residual: float
    continuation_residual: float
    value_projected_transition_residual: float
    bellman_residual: float
    decomposition_closure_error: float

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        for attribute in (
            "reward_residual",
            "continuation_residual",
            "value_projected_transition_residual",
            "bellman_residual",
            "decomposition_closure_error",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_float(getattr(self, attribute), name=attribute),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "reward_residual": self.reward_residual,
            "continuation_residual": self.continuation_residual,
            "value_projected_transition_residual": self.value_projected_transition_residual,
            "bellman_residual": self.bellman_residual,
            "decomposition_closure_error": self.decomposition_closure_error,
        }


@dataclass(frozen=True, slots=True)
class ComponentResiduals:
    rows: tuple[ComponentResidualRow, ...]
    schema_version: str = field(default=RESIDUAL_SCHEMA, init=False)
    decomposition_convention: str = field(default=_RESIDUAL_CONVENTION, init=False)
    residual_sign: str = field(default="predicted_minus_target", init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, ComponentResidualRow) for row in rows):
            raise ValueError("rows must contain ComponentResidualRow instances")
        object.__setattr__(self, "rows", rows)

    def metric_values(self) -> dict[str, tuple[float, ...]]:
        names = (
            "reward_residual",
            "continuation_residual",
            "value_projected_transition_residual",
            "bellman_residual",
        )
        return {
            name: tuple(float(getattr(row, name)) for row in self.rows)
            for name in names
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decomposition_convention": self.decomposition_convention,
            "residual_sign": self.residual_sign,
            "sample_count": len(self.rows),
            "max_abs_decomposition_closure_error": max(
                abs(row.decomposition_closure_error) for row in self.rows
            ),
            "rows": [row.to_dict() for row in self.rows],
        }


def component_residual_decomposition(
    predicted_rewards: Any,
    target_rewards: Any,
    predicted_continuations: Any,
    target_continuations: Any,
    predicted_next_values: Any,
    target_next_values: Any,
    *,
    discount: Any,
    closure_tolerance: float = 1e-10,
) -> ComponentResiduals:
    """Decompose predicted minus target one-step Bellman targets exactly.

    The transition component is a value-space diagnostic, not raw state error.
    It uses the predicted continuation as the multiplier; the continuation term
    therefore uses the target next value and the three terms telescope exactly.
    """

    named = {
        "predicted_rewards": predicted_rewards,
        "target_rewards": target_rewards,
        "predicted_continuations": predicted_continuations,
        "target_continuations": target_continuations,
        "predicted_next_values": predicted_next_values,
        "target_next_values": target_next_values,
    }
    arrays = {
        name: _finite_array(value, ndim=1, name=name) for name, value in named.items()
    }
    shape = arrays["predicted_rewards"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError(
            "all component residual inputs must share one-dimensional shape"
        )
    for name in ("predicted_continuations", "target_continuations"):
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")
    discount_array = np.asarray(discount, dtype=np.float64)
    if discount_array.ndim == 0:
        discount_array = np.full(shape, float(discount_array), dtype=np.float64)
    elif discount_array.ndim == 1 and discount_array.shape == shape:
        discount_array = discount_array.astype(np.float64, copy=False)
    else:
        raise ValueError("discount must be a finite scalar or match the sample shape")
    if not np.isfinite(discount_array).all() or np.any(
        (discount_array < 0.0) | (discount_array > 1.0)
    ):
        raise ValueError("discount must contain finite values in [0, 1]")
    closure_tolerance = _positive_float(closure_tolerance, name="closure_tolerance")

    predicted_reward = arrays["predicted_rewards"]
    target_reward = arrays["target_rewards"]
    predicted_continuation = arrays["predicted_continuations"]
    target_continuation = arrays["target_continuations"]
    predicted_value = arrays["predicted_next_values"]
    target_value = arrays["target_next_values"]
    reward_residual = predicted_reward - target_reward
    continuation_residual = (
        discount_array * (predicted_continuation - target_continuation) * target_value
    )
    transition_residual = (
        discount_array * predicted_continuation * (predicted_value - target_value)
    )
    predicted_target = (
        predicted_reward + discount_array * predicted_continuation * predicted_value
    )
    target_target = target_reward + discount_array * target_continuation * target_value
    bellman_residual = predicted_target - target_target
    closure = bellman_residual - (
        reward_residual + continuation_residual + transition_residual
    )
    if not np.allclose(closure, 0.0, rtol=0.0, atol=closure_tolerance):
        raise ArithmeticError("component residual decomposition failed Bellman closure")
    return ComponentResiduals(
        rows=tuple(
            ComponentResidualRow(
                sample_index=index,
                reward_residual=float(reward_residual[index]),
                continuation_residual=float(continuation_residual[index]),
                value_projected_transition_residual=float(transition_residual[index]),
                bellman_residual=float(bellman_residual[index]),
                decomposition_closure_error=float(closure[index]),
            )
            for index in range(shape[0])
        )
    )


@dataclass(frozen=True, slots=True)
class WorldModelMean:
    world_model_id: str
    nested_sample_count: int
    mean: float

    def __post_init__(self) -> None:
        world_model_id = str(self.world_model_id)
        if not world_model_id:
            raise ValueError("world_model_id must be nonempty")
        if self.nested_sample_count <= 0:
            raise ValueError("nested_sample_count must be positive")
        object.__setattr__(self, "world_model_id", world_model_id)
        object.__setattr__(self, "mean", _finite_float(self.mean, name="mean"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_model_id": self.world_model_id,
            "nested_sample_count": self.nested_sample_count,
            "mean": self.mean,
        }


@dataclass(frozen=True, slots=True)
class HorizonMetricAggregate:
    horizon: int
    metric: str
    world_model_count: int
    nested_sample_count: int
    mean_of_world_model_means: float
    between_world_model_standard_deviation: float | None
    standard_error_over_world_models: float | None
    minimum_world_model_mean: float
    maximum_world_model_mean: float
    per_world_model: tuple[WorldModelMean, ...]

    def __post_init__(self) -> None:
        if self.horizon < 0 or not self.metric:
            raise ValueError("horizon must be nonnegative and metric must be nonempty")
        model_rows = tuple(self.per_world_model)
        if not model_rows or not all(
            isinstance(row, WorldModelMean) for row in model_rows
        ):
            raise ValueError("per_world_model must contain WorldModelMean instances")
        if self.world_model_count != len(model_rows):
            raise ValueError("world_model_count does not match per_world_model")
        if self.nested_sample_count != sum(
            row.nested_sample_count for row in model_rows
        ):
            raise ValueError("nested_sample_count does not match per_world_model")
        object.__setattr__(self, "per_world_model", model_rows)
        for attribute in (
            "mean_of_world_model_means",
            "minimum_world_model_mean",
            "maximum_world_model_mean",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_float(getattr(self, attribute), name=attribute),
            )
        for attribute in (
            "between_world_model_standard_deviation",
            "standard_error_over_world_models",
        ):
            value = getattr(self, attribute)
            if value is not None:
                value = _finite_float(value, name=attribute)
                if value < 0.0:
                    raise ValueError(f"{attribute} must be nonnegative")
                object.__setattr__(self, attribute, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "metric": self.metric,
            "world_model_count": self.world_model_count,
            "nested_sample_count": self.nested_sample_count,
            "mean_of_world_model_means": self.mean_of_world_model_means,
            "between_world_model_standard_deviation": self.between_world_model_standard_deviation,
            "standard_error_over_world_models": self.standard_error_over_world_models,
            "minimum_world_model_mean": self.minimum_world_model_mean,
            "maximum_world_model_mean": self.maximum_world_model_mean,
            "per_world_model": [row.to_dict() for row in self.per_world_model],
        }


@dataclass(frozen=True, slots=True)
class HorizonAggregation:
    strata: tuple[HorizonMetricAggregate, ...]
    schema_version: str = field(default=HORIZON_SCHEMA, init=False)
    top_level_unit: str = field(default="world_model", init=False)
    nested_unit_policy: str = field(
        default="mean_within_world_model_then_equal_weight_across_world_models",
        init=False,
    )

    def __post_init__(self) -> None:
        strata = tuple(self.strata)
        if not strata or not all(
            isinstance(stratum, HorizonMetricAggregate) for stratum in strata
        ):
            raise ValueError("strata must contain HorizonMetricAggregate instances")
        object.__setattr__(self, "strata", strata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "top_level_unit": self.top_level_unit,
            "nested_unit_policy": self.nested_unit_policy,
            "strata": [stratum.to_dict() for stratum in self.strata],
        }


def horizon_stratified_aggregate(
    world_model_ids: Sequence[Any],
    horizons: Any,
    metrics: Mapping[str, Any],
) -> HorizonAggregation:
    """Aggregate metrics by horizon with world models as top-level units."""

    world_models = _string_tuple(world_model_ids, name="world_model_ids")
    horizon_array = np.asarray(horizons)
    if horizon_array.ndim != 1 or horizon_array.size == 0:
        raise ValueError("horizons must be a nonempty one-dimensional array")
    if horizon_array.shape[0] != len(world_models):
        raise ValueError("horizons and world_model_ids must have equal lengths")
    if not np.issubdtype(horizon_array.dtype, np.number):
        raise ValueError("horizons must be numeric nonnegative integers")
    numeric_horizons = np.asarray(horizon_array, dtype=np.float64)
    if not np.isfinite(numeric_horizons).all() or np.any(numeric_horizons < 0.0):
        raise ValueError("horizons must be finite nonnegative integers")
    if not np.equal(numeric_horizons, np.floor(numeric_horizons)).all():
        raise ValueError("horizons must be integers")
    horizon_values = numeric_horizons.astype(np.int64)
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a nonempty mapping")
    metric_arrays: dict[str, np.ndarray] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be nonempty strings")
        array = _finite_array(value, ndim=1, name=f"metric {name!r}")
        if array.shape[0] != len(world_models):
            raise ValueError(f"metric {name!r} length does not match units")
        metric_arrays[name] = array

    rows: list[HorizonMetricAggregate] = []
    world_array = np.asarray(world_models, dtype=object)
    for horizon in sorted(int(value) for value in np.unique(horizon_values)):
        horizon_mask = horizon_values == horizon
        model_names = sorted(set(world_array[horizon_mask].tolist()))
        for metric_name in sorted(metric_arrays):
            values = metric_arrays[metric_name]
            model_rows: list[WorldModelMean] = []
            for model_name in model_names:
                mask = horizon_mask & (world_array == model_name)
                nested_values = values[mask]
                model_rows.append(
                    WorldModelMean(
                        world_model_id=model_name,
                        nested_sample_count=int(nested_values.size),
                        mean=float(np.mean(nested_values)),
                    )
                )
            means = np.asarray([row.mean for row in model_rows], dtype=np.float64)
            if means.size > 1:
                standard_deviation: float | None = float(np.std(means, ddof=1))
                standard_error: float | None = float(
                    standard_deviation / math.sqrt(means.size)
                )
            else:
                standard_deviation = None
                standard_error = None
            rows.append(
                HorizonMetricAggregate(
                    horizon=horizon,
                    metric=metric_name,
                    world_model_count=int(means.size),
                    nested_sample_count=int(np.sum(horizon_mask)),
                    mean_of_world_model_means=float(np.mean(means)),
                    between_world_model_standard_deviation=standard_deviation,
                    standard_error_over_world_models=standard_error,
                    minimum_world_model_mean=float(np.min(means)),
                    maximum_world_model_mean=float(np.max(means)),
                    per_world_model=tuple(model_rows),
                )
            )
    return HorizonAggregation(strata=tuple(rows))


@dataclass(frozen=True, slots=True)
class IndependentNoiseDiagnostics:
    proposal_noise_ids: tuple[str, ...]
    heldout_noise_ids: tuple[str, ...]
    paired_samples: int
    action_dimensions: int
    proposal_mean_improvement: float
    proposal_median_improvement: float
    proposal_positive_improvement_fraction: float
    heldout_mean_improvement: float
    heldout_median_improvement: float
    heldout_positive_improvement_fraction: float
    mean_generalization_gap: float
    median_generalization_gap: float
    generalization_gap_standard_error: float | None
    positive_generalization_gap_fraction: float
    cosine_pairs: int
    zero_proposal_gradient_count: int
    zero_heldout_gradient_count: int
    mean_gradient_cosine: float | None
    median_gradient_cosine: float | None
    schema_version: str = field(default=NOISE_SCHEMA, init=False)
    objective_input_semantics: str = field(
        default="post_update_objective_minus_pre_update_objective", init=False
    )
    generalization_gap_convention: str = field(
        default="heldout_improvement_minus_proposal_improvement", init=False
    )
    uncertainty_scope: str = field(
        default="paired_samples_within_call_not_world_model_level", init=False
    )

    def __post_init__(self) -> None:
        proposal_ids = _string_tuple(self.proposal_noise_ids, name="proposal_noise_ids")
        heldout_ids = _string_tuple(self.heldout_noise_ids, name="heldout_noise_ids")
        if set(proposal_ids) & set(heldout_ids):
            raise ValueError("proposal and held-out noise IDs must be disjoint")
        object.__setattr__(self, "proposal_noise_ids", proposal_ids)
        object.__setattr__(self, "heldout_noise_ids", heldout_ids)
        for attribute in (
            "paired_samples",
            "action_dimensions",
            "cosine_pairs",
            "zero_proposal_gradient_count",
            "zero_heldout_gradient_count",
        ):
            value = getattr(self, attribute)
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise TypeError(f"{attribute} must be an integer")
            object.__setattr__(self, attribute, int(value))
        if self.paired_samples <= 0 or self.action_dimensions <= 0:
            raise ValueError("paired_samples and action_dimensions must be positive")
        if not 0 <= self.cosine_pairs <= self.paired_samples:
            raise ValueError("cosine_pairs must lie in [0, paired_samples]")
        for attribute in (
            "zero_proposal_gradient_count",
            "zero_heldout_gradient_count",
        ):
            if not 0 <= getattr(self, attribute) <= self.paired_samples:
                raise ValueError(f"{attribute} must lie in [0, paired_samples]")
        for attribute in (
            "proposal_mean_improvement",
            "proposal_median_improvement",
            "heldout_mean_improvement",
            "heldout_median_improvement",
            "mean_generalization_gap",
            "median_generalization_gap",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_float(getattr(self, attribute), name=attribute),
            )
        if self.generalization_gap_standard_error is not None:
            standard_error = _finite_float(
                self.generalization_gap_standard_error,
                name="generalization_gap_standard_error",
            )
            if standard_error < 0.0:
                raise ValueError(
                    "generalization_gap_standard_error must be nonnegative"
                )
            object.__setattr__(
                self, "generalization_gap_standard_error", standard_error
            )
        for attribute in (
            "proposal_positive_improvement_fraction",
            "heldout_positive_improvement_fraction",
            "positive_generalization_gap_fraction",
        ):
            fraction = _finite_float(getattr(self, attribute), name=attribute)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{attribute} must lie in [0, 1]")
            object.__setattr__(self, attribute, fraction)
        for attribute in ("mean_gradient_cosine", "median_gradient_cosine"):
            value = getattr(self, attribute)
            if value is not None:
                value = _finite_float(value, name=attribute)
                if not -1.0 <= value <= 1.0:
                    raise ValueError(f"{attribute} must lie in [-1, 1]")
                object.__setattr__(self, attribute, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective_input_semantics": self.objective_input_semantics,
            "generalization_gap_convention": self.generalization_gap_convention,
            "uncertainty_scope": self.uncertainty_scope,
            "proposal_noise_ids": list(self.proposal_noise_ids),
            "heldout_noise_ids": list(self.heldout_noise_ids),
            "paired_samples": self.paired_samples,
            "action_dimensions": self.action_dimensions,
            "proposal_mean_improvement": self.proposal_mean_improvement,
            "proposal_median_improvement": self.proposal_median_improvement,
            "proposal_positive_improvement_fraction": (
                self.proposal_positive_improvement_fraction
            ),
            "heldout_mean_improvement": self.heldout_mean_improvement,
            "heldout_median_improvement": self.heldout_median_improvement,
            "heldout_positive_improvement_fraction": (
                self.heldout_positive_improvement_fraction
            ),
            "mean_generalization_gap": self.mean_generalization_gap,
            "median_generalization_gap": self.median_generalization_gap,
            "generalization_gap_standard_error": (
                self.generalization_gap_standard_error
            ),
            "positive_generalization_gap_fraction": (
                self.positive_generalization_gap_fraction
            ),
            "cosine_pairs": self.cosine_pairs,
            "zero_proposal_gradient_count": self.zero_proposal_gradient_count,
            "zero_heldout_gradient_count": self.zero_heldout_gradient_count,
            "mean_gradient_cosine": self.mean_gradient_cosine,
            "median_gradient_cosine": self.median_gradient_cosine,
        }


def independent_noise_diagnostics(
    proposal_objective_improvements: Any,
    heldout_objective_improvements: Any,
    proposal_gradients: Any,
    heldout_gradients: Any,
    *,
    proposal_noise_ids: Sequence[Any],
    heldout_noise_ids: Sequence[Any],
    nonzero_tolerance: float = 1e-12,
) -> IndependentNoiseDiagnostics:
    """Measure proposal/held-out improvements and their generalization gap.

    Both objective vectors must already be post-update minus pre-update
    improvements.  Their difference is therefore a generalization gap, not an
    absolute held-out gain.  Reporting the two improvements separately avoids
    conflating a smaller positive held-out improvement with a negative update.
    """

    proposal_ids = _string_tuple(proposal_noise_ids, name="proposal_noise_ids")
    heldout_ids = _string_tuple(heldout_noise_ids, name="heldout_noise_ids")
    overlap = sorted(set(proposal_ids) & set(heldout_ids))
    if overlap:
        raise ValueError(
            "proposal and held-out noise IDs must be disjoint; overlap: "
            + ", ".join(overlap)
        )
    proposal_values = _finite_array(
        proposal_objective_improvements,
        ndim=1,
        name="proposal_objective_improvements",
    )
    heldout_values = _finite_array(
        heldout_objective_improvements,
        ndim=1,
        name="heldout_objective_improvements",
    )
    proposal_gradient = _finite_array(
        proposal_gradients, ndim=2, name="proposal_gradients"
    )
    heldout_gradient = _finite_array(
        heldout_gradients, ndim=2, name="heldout_gradients"
    )
    if proposal_values.shape != heldout_values.shape:
        raise ValueError("proposal and held-out improvements must share shape")
    if proposal_gradient.shape != heldout_gradient.shape:
        raise ValueError("proposal and held-out gradients must share shape")
    if proposal_gradient.shape[0] != proposal_values.shape[0]:
        raise ValueError("objective and gradient sample counts must match")
    nonzero_tolerance = _positive_float(nonzero_tolerance, name="nonzero_tolerance")

    generalization_gaps = heldout_values - proposal_values
    proposal_norm = np.linalg.norm(proposal_gradient, axis=1)
    heldout_norm = np.linalg.norm(heldout_gradient, axis=1)
    valid = (proposal_norm > nonzero_tolerance) & (heldout_norm > nonzero_tolerance)
    cosines = np.sum(proposal_gradient[valid] * heldout_gradient[valid], axis=1)
    cosines = cosines / (proposal_norm[valid] * heldout_norm[valid])
    cosines = np.clip(cosines, -1.0, 1.0)
    generalization_gap_standard_error = (
        None
        if generalization_gaps.size < 2
        else float(
            np.std(generalization_gaps, ddof=1) / math.sqrt(generalization_gaps.size)
        )
    )
    return IndependentNoiseDiagnostics(
        proposal_noise_ids=proposal_ids,
        heldout_noise_ids=heldout_ids,
        paired_samples=int(generalization_gaps.size),
        action_dimensions=int(proposal_gradient.shape[1]),
        proposal_mean_improvement=float(np.mean(proposal_values)),
        proposal_median_improvement=float(np.median(proposal_values)),
        proposal_positive_improvement_fraction=float(np.mean(proposal_values > 0.0)),
        heldout_mean_improvement=float(np.mean(heldout_values)),
        heldout_median_improvement=float(np.median(heldout_values)),
        heldout_positive_improvement_fraction=float(np.mean(heldout_values > 0.0)),
        mean_generalization_gap=float(np.mean(generalization_gaps)),
        median_generalization_gap=float(np.median(generalization_gaps)),
        generalization_gap_standard_error=generalization_gap_standard_error,
        positive_generalization_gap_fraction=float(np.mean(generalization_gaps > 0.0)),
        cosine_pairs=int(cosines.size),
        zero_proposal_gradient_count=int(np.sum(proposal_norm <= nonzero_tolerance)),
        zero_heldout_gradient_count=int(np.sum(heldout_norm <= nonzero_tolerance)),
        mean_gradient_cosine=(None if not cosines.size else float(np.mean(cosines))),
        median_gradient_cosine=(
            None if not cosines.size else float(np.median(cosines))
        ),
    )


@dataclass(frozen=True, slots=True)
class CounterfactualProvenance:
    """Provenance assertion controlling exact-branch-only metrics."""

    source_kind: str
    exact_simulator_branch: bool
    snapshot_restore_verified: bool
    common_reset_ids: tuple[str, ...] = ()
    simulator_id: str | None = None
    branch_protocol: str | None = None
    schema_version: str = field(default=PROVENANCE_SCHEMA, init=False)
    reset_id_semantics: str = field(
        default="one_common_snapshot_reset_id_per_candidate_row", init=False
    )

    def __post_init__(self) -> None:
        if self.source_kind not in {"offline_only", "exact_simulator_branch"}:
            raise ValueError(
                "source_kind must be 'offline_only' or 'exact_simulator_branch'"
            )
        reset_ids = tuple(str(value) for value in self.common_reset_ids)
        if any(not value for value in reset_ids):
            raise ValueError("common_reset_ids must contain nonempty identifiers")
        object.__setattr__(self, "common_reset_ids", reset_ids)
        for attribute in ("exact_simulator_branch", "snapshot_restore_verified"):
            value = getattr(self, attribute)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{attribute} must be boolean")
            object.__setattr__(self, attribute, bool(value))
        if self.simulator_id is not None:
            simulator_id = str(self.simulator_id)
            if not simulator_id:
                raise ValueError("simulator_id must be nonempty when provided")
            object.__setattr__(self, "simulator_id", simulator_id)
        if self.branch_protocol is not None:
            branch_protocol = str(self.branch_protocol)
            if not branch_protocol:
                raise ValueError("branch_protocol must be nonempty when provided")
            object.__setattr__(self, "branch_protocol", branch_protocol)
        exact_fields = self.exact_simulator_branch and self.snapshot_restore_verified
        if self.source_kind == "exact_simulator_branch":
            if not exact_fields:
                raise ValueError(
                    "exact simulator provenance requires explicit exact branch and "
                    "snapshot/restore verification flags"
                )
            if not reset_ids:
                raise ValueError("exact simulator provenance requires common_reset_ids")
            if self.simulator_id is None or self.branch_protocol is None:
                raise ValueError(
                    "exact simulator provenance requires simulator_id and branch_protocol"
                )
        elif self.exact_simulator_branch or self.snapshot_restore_verified:
            raise ValueError(
                "offline_only provenance cannot assert simulator branching"
            )

    @classmethod
    def offline_only(cls) -> "CounterfactualProvenance":
        return cls(
            source_kind="offline_only",
            exact_simulator_branch=False,
            snapshot_restore_verified=False,
        )

    @classmethod
    def verified_exact_branch(
        cls,
        common_reset_ids: Sequence[Any],
        *,
        simulator_id: str,
        branch_protocol: str = "snapshot_restore_common_reset",
    ) -> "CounterfactualProvenance":
        return cls(
            source_kind="exact_simulator_branch",
            exact_simulator_branch=True,
            snapshot_restore_verified=True,
            common_reset_ids=tuple(str(value) for value in common_reset_ids),
            simulator_id=simulator_id,
            branch_protocol=branch_protocol,
        )

    @property
    def identifiable(self) -> bool:
        return (
            self.source_kind == "exact_simulator_branch"
            and self.exact_simulator_branch
            and self.snapshot_restore_verified
            and bool(self.common_reset_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "exact_simulator_branch": self.exact_simulator_branch,
            "snapshot_restore_verified": self.snapshot_restore_verified,
            "common_reset_ids": list(self.common_reset_ids),
            "simulator_id": self.simulator_id,
            "branch_protocol": self.branch_protocol,
            "reset_id_semantics": self.reset_id_semantics,
        }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = float(
        np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    )
    if denominator <= 1e-15:
        return None
    return float(
        np.clip(np.dot(centered_first, centered_second) / denominator, -1.0, 1.0)
    )


@dataclass(frozen=True, slots=True)
class CandidateCounterfactualMetrics:
    reset_count: int
    candidates_per_reset: int
    informative_reset_count: int
    comparable_pair_count: int
    pairwise_ranking_accuracy: float | None
    mean_within_reset_spearman: float | None
    top1_accuracy: float | None
    mean_simulator_regret: float | None
    median_simulator_regret: float | None
    maximum_simulator_regret: float | None
    predicted_minus_simulator_bias: float
    return_mae: float
    return_rmse: float
    pairwise_gain_bias: float
    pairwise_gain_rmse: float
    calibration_slope_simulator_on_predicted: float | None
    calibration_intercept_simulator_on_predicted: float | None
    return_pearson: float | None

    def __post_init__(self) -> None:
        for attribute in (
            "reset_count",
            "candidates_per_reset",
            "informative_reset_count",
            "comparable_pair_count",
        ):
            value = getattr(self, attribute)
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise TypeError(f"{attribute} must be an integer")
            object.__setattr__(self, attribute, int(value))
        if self.reset_count <= 0 or self.candidates_per_reset < 2:
            raise ValueError(
                "candidate metrics require resets and at least two candidates"
            )
        if not 0 <= self.informative_reset_count <= self.reset_count:
            raise ValueError("informative_reset_count is inconsistent")
        if self.comparable_pair_count < 0:
            raise ValueError("comparable_pair_count must be nonnegative")
        for attribute in (
            "predicted_minus_simulator_bias",
            "return_mae",
            "return_rmse",
            "pairwise_gain_bias",
            "pairwise_gain_rmse",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_float(getattr(self, attribute), name=attribute),
            )
        for attribute in ("return_mae", "return_rmse", "pairwise_gain_rmse"):
            if getattr(self, attribute) < 0.0:
                raise ValueError(f"{attribute} must be nonnegative")
        for attribute in ("pairwise_ranking_accuracy", "top1_accuracy"):
            value = getattr(self, attribute)
            if value is not None:
                value = _finite_float(value, name=attribute)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{attribute} must lie in [0, 1]")
                object.__setattr__(self, attribute, value)
        for attribute in ("mean_within_reset_spearman", "return_pearson"):
            value = getattr(self, attribute)
            if value is not None:
                value = _finite_float(value, name=attribute)
                if not -1.0 <= value <= 1.0:
                    raise ValueError(f"{attribute} must lie in [-1, 1]")
                object.__setattr__(self, attribute, value)
        for attribute in (
            "mean_simulator_regret",
            "median_simulator_regret",
            "maximum_simulator_regret",
            "calibration_slope_simulator_on_predicted",
            "calibration_intercept_simulator_on_predicted",
        ):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(
                    self, attribute, _finite_float(value, name=attribute)
                )
        for attribute in (
            "mean_simulator_regret",
            "median_simulator_regret",
            "maximum_simulator_regret",
        ):
            value = getattr(self, attribute)
            if value is not None and value < 0.0:
                raise ValueError(f"{attribute} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CounterfactualDiagnostics:
    identifiable: bool
    unidentifiable_reason: str | None
    provenance: CounterfactualProvenance | None
    metrics: CandidateCounterfactualMetrics | None
    schema_version: str = field(default=COUNTERFACTUAL_SCHEMA, init=False)
    identifiability_rule: str = field(
        default=(
            "metrics_require_explicit_exact_simulator_branch_snapshot_restore_"
            "and_common_reset_ids"
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identifiable, (bool, np.bool_)):
            raise TypeError("identifiable must be boolean")
        object.__setattr__(self, "identifiable", bool(self.identifiable))
        if self.provenance is not None and not isinstance(
            self.provenance, CounterfactualProvenance
        ):
            raise TypeError("provenance must be CounterfactualProvenance or None")
        if self.metrics is not None and not isinstance(
            self.metrics, CandidateCounterfactualMetrics
        ):
            raise TypeError("metrics must be CandidateCounterfactualMetrics or None")
        if self.identifiable:
            if self.provenance is None or not self.provenance.identifiable:
                raise ValueError("identifiable output requires exact branch provenance")
            if self.metrics is None or self.unidentifiable_reason is not None:
                raise ValueError("identifiable output requires metrics and no reason")
        elif self.metrics is not None or not self.unidentifiable_reason:
            raise ValueError(
                "unidentifiable output requires a reason and cannot contain metrics"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identifiability_rule": self.identifiability_rule,
            "identifiable": self.identifiable,
            "unidentifiable_reason": self.unidentifiable_reason,
            "provenance": (
                None if self.provenance is None else self.provenance.to_dict()
            ),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


def _unidentifiable(
    reason: str,
    provenance: CounterfactualProvenance | None,
    *,
    require_identifiable: bool,
) -> CounterfactualDiagnostics:
    if require_identifiable:
        raise UnidentifiableCounterfactualError(reason)
    return CounterfactualDiagnostics(
        identifiable=False,
        unidentifiable_reason=reason,
        provenance=provenance,
        metrics=None,
    )


def counterfactual_candidate_diagnostics(
    predicted_returns: Any | None,
    simulator_returns: Any | None,
    *,
    provenance: CounterfactualProvenance | None,
    tie_tolerance: float = 1e-8,
    require_identifiable: bool = False,
) -> CounterfactualDiagnostics:
    """Compute candidate metrics only under verified exact simulator branching.

    Passing offline tuples (or no provenance) returns an explicit
    ``identifiable=False`` object and never inspects or summarizes candidate
    arrays.  Set ``require_identifiable=True`` when the caller prefers a hard
    rejection instead of a marked result.
    """

    if provenance is not None and not isinstance(provenance, CounterfactualProvenance):
        raise TypeError("provenance must be CounterfactualProvenance or None")
    if provenance is None:
        return _unidentifiable(
            "counterfactual metrics require explicit exact simulator branch provenance",
            None,
            require_identifiable=require_identifiable,
        )
    if not provenance.identifiable:
        return _unidentifiable(
            "offline tuples cannot identify same-reset counterfactual candidate metrics",
            provenance,
            require_identifiable=require_identifiable,
        )

    predicted = _finite_array(predicted_returns, ndim=2, name="predicted_returns")
    simulator = _finite_array(simulator_returns, ndim=2, name="simulator_returns")
    if predicted.shape != simulator.shape:
        raise ValueError("predicted and simulator returns must share shape")
    if predicted.shape[1] < 2:
        raise ValueError("counterfactual diagnostics need at least two candidates")
    if len(provenance.common_reset_ids) != predicted.shape[0]:
        raise ValueError("one common_reset_id is required for every candidate row")
    if len(set(provenance.common_reset_ids)) != len(provenance.common_reset_ids):
        raise ValueError("common_reset_ids must uniquely identify candidate rows")
    tie_tolerance = _finite_float(tie_tolerance, name="tie_tolerance")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be nonnegative")

    comparable = 0
    correct = 0
    informative = 0
    top1 = 0
    regrets: list[float] = []
    correlations: list[float] = []
    predicted_gains: list[float] = []
    simulator_gains: list[float] = []
    for predicted_row, simulator_row in zip(predicted, simulator, strict=True):
        simulator_range = float(np.ptp(simulator_row))
        predicted_best = int(np.argmax(predicted_row))
        simulator_best = float(np.max(simulator_row))
        if simulator_range > tie_tolerance:
            informative += 1
            regret = simulator_best - float(simulator_row[predicted_best])
            regrets.append(regret)
            top1 += int(regret <= tie_tolerance)
        correlation = _correlation(_rankdata(predicted_row), _rankdata(simulator_row))
        if correlation is not None:
            correlations.append(correlation)
        for left in range(predicted.shape[1]):
            for right in range(left + 1, predicted.shape[1]):
                predicted_gain = float(predicted_row[left] - predicted_row[right])
                simulator_gain = float(simulator_row[left] - simulator_row[right])
                predicted_gains.append(predicted_gain)
                simulator_gains.append(simulator_gain)
                if abs(simulator_gain) <= tie_tolerance:
                    continue
                comparable += 1
                correct += int(np.sign(predicted_gain) == np.sign(simulator_gain))

    flat_predicted = predicted.reshape(-1)
    flat_simulator = simulator.reshape(-1)
    return_errors = flat_predicted - flat_simulator
    centered_predicted = flat_predicted - float(np.mean(flat_predicted))
    prediction_sum_squares = float(np.dot(centered_predicted, centered_predicted))
    if prediction_sum_squares <= 1e-15:
        slope = None
        intercept = None
    else:
        slope = float(
            np.dot(
                centered_predicted,
                flat_simulator - float(np.mean(flat_simulator)),
            )
            / prediction_sum_squares
        )
        intercept = float(np.mean(flat_simulator) - slope * np.mean(flat_predicted))
    predicted_gain_array = np.asarray(predicted_gains, dtype=np.float64)
    simulator_gain_array = np.asarray(simulator_gains, dtype=np.float64)
    gain_errors = predicted_gain_array - simulator_gain_array
    metrics = CandidateCounterfactualMetrics(
        reset_count=int(predicted.shape[0]),
        candidates_per_reset=int(predicted.shape[1]),
        informative_reset_count=informative,
        comparable_pair_count=comparable,
        pairwise_ranking_accuracy=(None if comparable == 0 else correct / comparable),
        mean_within_reset_spearman=(
            None if not correlations else float(np.mean(correlations))
        ),
        top1_accuracy=(None if informative == 0 else top1 / informative),
        mean_simulator_regret=(None if not regrets else float(np.mean(regrets))),
        median_simulator_regret=(None if not regrets else float(np.median(regrets))),
        maximum_simulator_regret=(None if not regrets else float(np.max(regrets))),
        predicted_minus_simulator_bias=float(np.mean(return_errors)),
        return_mae=float(np.mean(np.abs(return_errors))),
        return_rmse=float(np.sqrt(np.mean(np.square(return_errors)))),
        pairwise_gain_bias=float(np.mean(gain_errors)),
        pairwise_gain_rmse=float(np.sqrt(np.mean(np.square(gain_errors)))),
        calibration_slope_simulator_on_predicted=slope,
        calibration_intercept_simulator_on_predicted=intercept,
        return_pearson=_correlation(flat_predicted, flat_simulator),
    )
    return CounterfactualDiagnostics(
        identifiable=True,
        unidentifiable_reason=None,
        provenance=provenance,
        metrics=metrics,
    )


def self_test() -> None:
    """Run fast positive and negative controls used by preflight checks."""

    def check(condition: bool, message: str) -> None:
        # Remain effective under ``python -O``; this is a verification entry
        # point, so ordinary optimization must not erase its controls.
        if not condition:
            raise AssertionError(message)

    train_observations = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    train_actions = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
    calibration = fit_coverage_calibration(train_observations, train_actions)
    near = score_observation_action_coverage(calibration, [[0.1, 0.0]], [[0.1]])
    far = score_observation_action_coverage(calibration, [[100.0, 0.0]], [[100.0]])
    check(
        (
            near.points[0].nearest_neighbor_distance
            < far.points[0].nearest_neighbor_distance
        ),
        "far coverage control did not increase nearest-neighbor distance",
    )
    check(
        near.points[0].kernel_density > far.points[0].kernel_density,
        "far coverage control did not reduce empirical density",
    )
    check(
        calibration.calibration_partition == _TRAIN_ONLY,
        "coverage calibration was not labeled train-only",
    )

    residuals = component_residual_decomposition(
        [2.0], [1.0], [0.8], [0.5], [4.0], [3.0], discount=0.9
    )
    row = residuals.rows[0]
    check(
        abs(
            row.bellman_residual
            - row.reward_residual
            - row.continuation_residual
            - row.value_projected_transition_residual
        )
        < 1e-12,
        "component residuals did not close to the Bellman residual",
    )

    aggregate = horizon_stratified_aggregate(
        ["wm-a", "wm-a", "wm-a", "wm-b"],
        [1, 1, 1, 1],
        {"metric": [0.0, 0.0, 0.0, 10.0]},
    )
    check(
        aggregate.strata[0].mean_of_world_model_means == 5.0,
        "world models were not equally weighted top-level units",
    )

    noise = independent_noise_diagnostics(
        [0.0, 0.0],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        proposal_noise_ids=["fit-noise"],
        heldout_noise_ids=["heldout-noise"],
    )
    check(
        noise.proposal_mean_improvement == 0.0
        and noise.heldout_mean_improvement == 1.0
        and noise.mean_generalization_gap == 1.0
        and noise.mean_gradient_cosine == 1.0,
        "aligned independent-noise control was not recovered",
    )
    try:
        independent_noise_diagnostics(
            [0.0],
            [1.0],
            [[1.0]],
            [[1.0]],
            proposal_noise_ids=["reused"],
            heldout_noise_ids=["reused"],
        )
    except ValueError:
        pass
    else:  # pragma: no cover - a deliberate fail-closed self-test
        raise AssertionError("overlapping noise IDs were accepted")

    offline = counterfactual_candidate_diagnostics(
        [[0.0, 1.0]],
        [[0.0, 1.0]],
        provenance=CounterfactualProvenance.offline_only(),
    )
    check(
        not offline.identifiable and offline.metrics is None,
        "offline tuples produced counterfactual metrics",
    )
    try:
        counterfactual_candidate_diagnostics(
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            provenance=CounterfactualProvenance.offline_only(),
            require_identifiable=True,
        )
    except UnidentifiableCounterfactualError:
        pass
    else:  # pragma: no cover - a deliberate fail-closed self-test
        raise AssertionError("offline counterfactual claim was accepted")

    exact = counterfactual_candidate_diagnostics(
        [[0.0, 1.0, 2.0]],
        [[0.0, 1.0, 2.0]],
        provenance=CounterfactualProvenance.verified_exact_branch(
            ["reset-0"], simulator_id="self-test-simulator"
        ),
    )
    check(
        exact.identifiable and exact.metrics is not None,
        "verified exact branch did not produce candidate metrics",
    )
    if exact.metrics is None:  # Keeps static narrowing independent of check().
        raise AssertionError("exact metrics unexpectedly absent")
    check(
        exact.metrics.pairwise_ranking_accuracy == 1.0,
        "perfect candidate ranking control was not recovered",
    )
    check(
        exact.metrics.mean_simulator_regret == 0.0,
        "perfect candidate regret control was not recovered",
    )

    # Every top-level schema projection must remain directly serializable.
    json.dumps(calibration.to_dict())
    json.dumps(near.to_dict())
    json.dumps(residuals.to_dict())
    json.dumps(aggregate.to_dict())
    json.dumps(noise.to_dict())
    json.dumps(offline.to_dict())
    json.dumps(exact.to_dict())


__all__ = [
    "COUNTERFACTUAL_SCHEMA",
    "COVERAGE_SCHEMA",
    "HORIZON_SCHEMA",
    "NOISE_SCHEMA",
    "PROVENANCE_SCHEMA",
    "RESIDUAL_SCHEMA",
    "SCHEMA_VERSIONS",
    "CandidateCounterfactualMetrics",
    "ComponentResidualRow",
    "ComponentResiduals",
    "CounterfactualDiagnostics",
    "CounterfactualProvenance",
    "CoverageCalibration",
    "CoveragePoint",
    "CoverageScores",
    "HorizonAggregation",
    "HorizonMetricAggregate",
    "IndependentNoiseDiagnostics",
    "UnidentifiableCounterfactualError",
    "WorldModelMean",
    "component_residual_decomposition",
    "counterfactual_candidate_diagnostics",
    "fit_coverage_calibration",
    "horizon_stratified_aggregate",
    "independent_noise_diagnostics",
    "score_observation_action_coverage",
    "self_test",
]
