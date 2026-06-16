from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from v2_config import ActionBounds, KNNConfig


@dataclass
class FilterResult:
    filtered_actions: np.ndarray
    replaced_count: int
    mean_predicted_mprr: float
    max_predicted_mprr: float
    predicted_mprr: np.ndarray


class KNNSafetyFilter:
    """
    Lightweight k-NN acceptance filter using a static CSV dataset.
    """

    def __init__(
        self,
        csv_path: str,
        knn_config: KNNConfig,
        action_columns: Sequence[str] = ("ID1_prev", "ID2", "SOI2"),
    ):
        self.csv_path = Path(csv_path).expanduser()
        self.knn_config = knn_config
        self.action_columns = tuple(action_columns)
        self.examples, self.example_mprr, self.example_safety = self._load_csv_examples(
            self.csv_path, self.action_columns, self.knn_config.safety_threshold
        )
        if self.examples.size == 0:
            raise ValueError("Safety CSV loaded zero valid rows.")
        self.safe_examples = self.examples[self.example_safety == 1]
        if self.safe_examples.size == 0:
            raise ValueError("Safety CSV has no safe examples at the configured threshold.")
        self._cache_low: np.ndarray | None = None
        self._cache_high: np.ndarray | None = None
        self._cache_span: np.ndarray | None = None
        self._norm_examples: np.ndarray | None = None
        self._norm_examples_sq: np.ndarray | None = None
        self._norm_safe_examples: np.ndarray | None = None

    def _ensure_normalized_cache(self, bounds: ActionBounds) -> None:
        low = np.asarray(bounds.low, dtype=np.float32)
        high = np.asarray(bounds.high, dtype=np.float32)
        if (
            self._cache_low is not None
            and self._cache_high is not None
            and np.allclose(self._cache_low, low)
            and np.allclose(self._cache_high, high)
        ):
            return

        span = np.maximum(1e-6, high - low).astype(np.float32)
        self._cache_low = low.copy()
        self._cache_high = high.copy()
        self._cache_span = span
        self._norm_examples = ((self.examples - low) / span).astype(np.float32, copy=False)
        self._norm_examples_sq = np.sum(self._norm_examples * self._norm_examples, axis=1).astype(
            np.float32, copy=False
        )
        self._norm_safe_examples = ((self.safe_examples - low) / span).astype(np.float32, copy=False)

    @staticmethod
    def _load_csv_examples(
        csv_path: Path, columns: Sequence[str], safety_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Safety CSV not found: {csv_path}")
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required to load safety CSV efficiently. Install it with 'pip install pandas'."
            ) from exc

        header_df = pd.read_csv(csv_path, nrows=0)
        fieldnames = list(header_df.columns)
        if not fieldnames:
            raise ValueError("Safety CSV has no header row.")

        field_lookup = {field.lower(): field for field in fieldnames}
        resolved_cols: List[str] = []
        for col in columns:
            key = col.lower()
            if key not in field_lookup:
                raise ValueError(
                    f"Missing required CSV column '{col}'. "
                    f"Found: {', '.join(fieldnames)}"
                )
            resolved_cols.append(field_lookup[key])
        if "mprr" not in field_lookup:
            raise ValueError(
                "Missing required CSV column 'MPRR'. "
                f"Found: {', '.join(fieldnames)}"
            )
        mprr_col = field_lookup["mprr"]

        selected_cols = [*resolved_cols, mprr_col]
        df = pd.read_csv(csv_path, usecols=selected_cols)
        numeric_df = df[selected_cols].apply(pd.to_numeric, errors="coerce")
        valid_mask = numeric_df.notna().all(axis=1)
        if not bool(valid_mask.any()):
            return (
                np.zeros((0, len(columns)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        valid_numeric = numeric_df.loc[valid_mask]
        actions = valid_numeric[resolved_cols].to_numpy(dtype=np.float32, copy=True)
        mprr = valid_numeric[mprr_col].to_numpy(dtype=np.float32, copy=True)
        safety = (mprr <= float(safety_threshold)).astype(np.float32, copy=False)
        return (
            actions,
            mprr,
            safety,
        )

    def _normalized_distances(self, action: np.ndarray, bounds: ActionBounds) -> np.ndarray:
        self._ensure_normalized_cache(bounds)
        assert self._cache_low is not None and self._cache_span is not None and self._norm_examples is not None
        norm_action = (action - self._cache_low) / self._cache_span
        diffs = self._norm_examples - norm_action
        return np.linalg.norm(diffs, axis=1)

    def _predict_mprr_batch(self, actions: np.ndarray, bounds: ActionBounds) -> np.ndarray:
        actions_arr = np.asarray(actions, dtype=np.float32).reshape(-1, 3)
        self._ensure_normalized_cache(bounds)
        assert (
            self._cache_low is not None
            and self._cache_span is not None
            and self._norm_examples is not None
            and self._norm_examples_sq is not None
        )
        if self._norm_examples.shape[0] == 0:
            return np.full((actions_arr.shape[0],), np.inf, dtype=np.float32)

        norm_actions = ((actions_arr - self._cache_low) / self._cache_span).astype(np.float32, copy=False)
        action_sq = np.sum(norm_actions * norm_actions, axis=1, keepdims=True)
        dot = norm_actions @ self._norm_examples.T
        distances_sq = action_sq + self._norm_examples_sq[None, :] - (2.0 * dot)
        np.maximum(distances_sq, 0.0, out=distances_sq)

        k = max(1, min(self.knn_config.k_neighbors, distances_sq.shape[1]))
        nearest_idx = np.argpartition(distances_sq, k - 1, axis=1)[:, :k]
        nearest_mprr = self.example_mprr[nearest_idx]
        nearest_dist_sq = np.take_along_axis(distances_sq, nearest_idx, axis=1)

        # Use inverse-distance weighting; exact matches fall back to the mean of zero-distance neighbors.
        zero_mask = nearest_dist_sq <= 1e-12
        has_zero = np.any(zero_mask, axis=1)

        weighted_pred = np.empty((actions_arr.shape[0],), dtype=np.float32)
        if np.any(~has_zero):
            non_zero_rows = ~has_zero
            nearest_dist = np.sqrt(nearest_dist_sq[non_zero_rows], dtype=np.float32)
            weights = 1.0 / np.maximum(nearest_dist, 1e-6)
            numer = np.sum(weights * nearest_mprr[non_zero_rows], axis=1)
            denom = np.sum(weights, axis=1)
            weighted_pred[non_zero_rows] = (numer / np.maximum(denom, 1e-12)).astype(
                np.float32, copy=False
            )
        if np.any(has_zero):
            zero_rows = has_zero
            zero_weights = zero_mask[zero_rows].astype(np.float32, copy=False)
            numer = np.sum(zero_weights * nearest_mprr[zero_rows], axis=1)
            denom = np.sum(zero_weights, axis=1)
            weighted_pred[zero_rows] = (numer / np.maximum(denom, 1e-12)).astype(
                np.float32, copy=False
            )
        return weighted_pred

    def predict_mprr_batch(self, actions: np.ndarray, bounds: ActionBounds) -> np.ndarray:
        return self._predict_mprr_batch(actions, bounds)

    def _predict_mprr(self, action: np.ndarray, bounds: ActionBounds) -> float:
        return float(self._predict_mprr_batch(np.asarray(action, dtype=np.float32).reshape(1, 3), bounds)[0])

    def is_safe(self, action: np.ndarray, bounds: ActionBounds) -> Tuple[bool, float]:
        predicted_mprr = self._predict_mprr(action, bounds)
        safe = predicted_mprr <= float(self.knn_config.safety_threshold)
        return safe, predicted_mprr

    def filter_sequence(
        self,
        actions: np.ndarray,
        bounds: ActionBounds,
    ) -> FilterResult:
        filtered = np.asarray(actions, dtype=np.float32).copy()
        if filtered.size == 0:
            return FilterResult(
                filtered_actions=filtered,
                replaced_count=0,
                mean_predicted_mprr=0.0,
                max_predicted_mprr=0.0,
                predicted_mprr=np.zeros((0,), dtype=np.float32),
            )

        predicted_mprr = self._predict_mprr_batch(filtered, bounds)
        safe_mask = predicted_mprr <= float(self.knn_config.safety_threshold)
        unsafe_idx = np.where(~safe_mask)[0]
        replaced_count = 0

        for idx in unsafe_idx:
            original = filtered[idx].copy()
            candidate = self._interpolate_to_nearest_safe(original, bounds)
            filtered[idx] = candidate.astype(np.float32)
            replaced_count += int(np.linalg.norm(candidate - original) > 1e-7)

        final_predicted_mprr = predicted_mprr.astype(np.float32, copy=True)
        if unsafe_idx.size > 0:
            final_predicted_mprr[unsafe_idx] = self._predict_mprr_batch(filtered[unsafe_idx], bounds)
        return FilterResult(
            filtered_actions=filtered,
            replaced_count=replaced_count,
            mean_predicted_mprr=float(np.mean(final_predicted_mprr)),
            max_predicted_mprr=float(np.max(final_predicted_mprr)),
            predicted_mprr=final_predicted_mprr.astype(np.float32, copy=False),
        )

    def _normalized_distances_to_reference(
        self, action: np.ndarray, reference_actions: np.ndarray, bounds: ActionBounds
    ) -> np.ndarray:
        self._ensure_normalized_cache(bounds)
        assert self._cache_low is not None and self._cache_span is not None and self._norm_safe_examples is not None
        norm_action = (action - self._cache_low) / self._cache_span
        if reference_actions is self.safe_examples:
            norm_reference = self._norm_safe_examples
        else:
            norm_reference = (reference_actions - self._cache_low) / self._cache_span
        diffs = norm_reference - norm_action
        return np.linalg.norm(diffs, axis=1)

    def _interpolate_to_nearest_safe(self, action: np.ndarray, bounds: ActionBounds) -> np.ndarray:
        distances = self._normalized_distances_to_reference(
            action.astype(np.float32), self.safe_examples, bounds
        )
        nearest_idx = int(np.argmin(distances))
        nearest_safe = self.safe_examples[nearest_idx].astype(np.float32).copy()

        direction = nearest_safe - action.astype(np.float32)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            interpolated = action.astype(np.float32) + (direction * float(fraction))
            clipped = np.clip(interpolated, bounds.low, bounds.high).astype(np.float32)
            safe, _ = self.is_safe(clipped, bounds)
            if safe:
                return clipped

        return np.clip(nearest_safe, bounds.low, bounds.high).astype(np.float32)

