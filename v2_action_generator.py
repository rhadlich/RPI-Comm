from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

import numpy as np

from v2_config import ActionBounds, PayloadConfig, SequenceConfig


@dataclass
class CycleSpec:
    mode_name: str
    filter_enabled: bool
    random_mask: np.ndarray
    fixed_values: np.ndarray


@dataclass
class SequenceBundle:
    actions: np.ndarray
    specs: List[CycleSpec]
    summary: Dict[str, float]
    predicted_mprr: np.ndarray | None = None


class SafetyFilterResult(Protocol):
    filtered_actions: np.ndarray
    replaced_count: int
    mean_predicted_mprr: float
    max_predicted_mprr: float
    predicted_mprr: np.ndarray


class SequenceSafetyFilter(Protocol):
    def filter_sequence(self, actions: np.ndarray, bounds: ActionBounds) -> SafetyFilterResult:
        ...


class RandomSequenceGenerator:
    """
    Generates a sequence from a 260-cycle base block:
      - Base block: anchor hold + structured EXPLORE excitation
      - Full/partial no-injection families are sampled and interleaved into
        a new sequence while preserving all base cycles in order
    """

    def __init__(
        self,
        sequence_config: SequenceConfig,
        payload_config: PayloadConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.sequence_config = sequence_config
        self.payload_config = payload_config or PayloadConfig()
        self.rng = rng or np.random.default_rng()

    def suggest_anchor(self, bounds: ActionBounds) -> np.ndarray:
        bounds.validate()
        low = np.asarray(bounds.low, dtype=np.float32)
        high = np.asarray(bounds.high, dtype=np.float32)
        center = (low + high) * 0.5
        span = np.maximum(high - low, np.float32(1e-6))
        std = span / 6.0  # ~99.7% inside bounds for an unconstrained Gaussian

        for _ in range(32):
            candidate = self.rng.normal(loc=center, scale=std).astype(np.float32)
            if np.all(candidate >= low) and np.all(candidate <= high):
                return candidate

        # Fallback if rejection sampling misses too often.
        return np.clip(candidate, low, high).astype(np.float32)

    def normalize_anchor(self, anchor: np.ndarray, bounds: ActionBounds) -> np.ndarray:
        bounds.validate()
        anchor_vec = np.asarray(anchor, dtype=np.float32).reshape(3)
        return np.clip(anchor_vec, bounds.low, bounds.high).astype(np.float32)

    def generate_sequence(
        self,
        anchor: np.ndarray,
        bounds: ActionBounds,
        safety_filter: SequenceSafetyFilter | None = None,
    ) -> SequenceBundle:
        anchor_vec = self.normalize_anchor(anchor, bounds)
        base_actions, base_specs = self._generate_base_block(anchor_vec, bounds)
        if safety_filter is not None:
            base_filter_result = safety_filter.filter_sequence(actions=base_actions, bounds=bounds)
            base_actions = np.asarray(base_filter_result.filtered_actions, dtype=np.float32)
            base_predicted_mprr = np.asarray(base_filter_result.predicted_mprr, dtype=np.float32)
        else:
            base_filter_result = None
            base_predicted_mprr = None
        final_actions, final_specs, event_stats = self._interleave_zero_event_family(
            base_actions, base_specs
        )
        final_predicted_mprr: np.ndarray | None = None
        if base_predicted_mprr is not None:
            final_predicted_mprr = np.full((len(final_specs),), np.nan, dtype=np.float32)
            base_idx = 0
            for idx, spec in enumerate(final_specs):
                if spec.filter_enabled:
                    if base_idx >= len(base_predicted_mprr):
                        raise RuntimeError("Prediction mapping mismatch between base and final sequence.")
                    final_predicted_mprr[idx] = base_predicted_mprr[base_idx]
                    base_idx += 1
            if base_idx != len(base_predicted_mprr):
                raise RuntimeError("Unused base predictions after sequence interleaving.")
        summary = {
            "base_cycles": float(len(base_actions)),
            "final_cycles": float(len(final_actions)),
            "full_family_ratio": float(event_stats["full_family_ratio"]),
            "partial_family_ratio": float(event_stats["partial_family_ratio"]),
            "full_family_cycles": float(event_stats["full_family_cycles"]),
            "partial_family_cycles": float(event_stats["partial_family_cycles"]),
            "partial_id1_cycles": float(event_stats["partial_id1_cycles"]),
            "partial_id2_cycles": float(event_stats["partial_id2_cycles"]),
            "replaced_base_cycles": float(base_filter_result.replaced_count if base_filter_result is not None else 0),
            "mean_predicted_mprr": float(
                base_filter_result.mean_predicted_mprr if base_filter_result is not None else 0.0
            ),
            "max_predicted_mprr": float(
                base_filter_result.max_predicted_mprr if base_filter_result is not None else 0.0
            ),
        }
        return SequenceBundle(
            actions=final_actions,
            specs=final_specs,
            summary=summary,
            predicted_mprr=final_predicted_mprr,
        )

    def build_payload_from_action(self, action_values_3d: np.ndarray) -> np.ndarray:
        payload = self.payload_config.default_payload_12.copy().astype(np.float32)
        action = np.asarray(action_values_3d, dtype=np.float32).reshape(3)
        # Action ordering: [ID1, ID2, SOI2] -> payload positions [D1, D2, T2]
        payload[1] = action[0]
        payload[3] = action[1]
        payload[2] = action[2]
        return payload

    def _generate_base_block(
        self, anchor: np.ndarray, bounds: ActionBounds
    ) -> Tuple[np.ndarray, List[CycleSpec]]:
        actions: List[np.ndarray] = []
        specs: List[CycleSpec] = []

        def append_cycle(action: np.ndarray, mode: str, random_mask: np.ndarray) -> None:
            fixed_values = anchor.copy()
            fixed_values[random_mask.astype(bool)] = np.nan
            actions.append(action.astype(np.float32))
            specs.append(
                CycleSpec(
                    mode_name=mode,
                    filter_enabled=True,
                    random_mask=random_mask.astype(bool),
                    fixed_values=fixed_values.astype(np.float32),
                )
            )

        # 1) Anchor hold: 10 cycles
        for _ in range(self.sequence_config.base_anchor_hold_cycles):
            append_cycle(anchor.copy(), "HOLD", np.array([False, False, False]))

        # 2) Single-action random excitation: 3 x 25
        for axis in range(3):
            random_mask = np.array([False, False, False], dtype=bool)
            random_mask[axis] = True
            for _ in range(self.sequence_config.single_action_cycles):
                action = anchor.copy()
                action[axis] = self.rng.uniform(bounds.low[axis], bounds.high[axis])
                append_cycle(action, "EXPLORE", random_mask)

        # 3) Pair-action random excitation: 3 x 25
        pairs = [(0, 1), (0, 2), (1, 2)]
        for a_idx, b_idx in pairs:
            random_mask = np.array([False, False, False], dtype=bool)
            random_mask[a_idx] = True
            random_mask[b_idx] = True
            for _ in range(self.sequence_config.pair_action_cycles):
                action = anchor.copy()
                action[a_idx] = self.rng.uniform(bounds.low[a_idx], bounds.high[a_idx])
                action[b_idx] = self.rng.uniform(bounds.low[b_idx], bounds.high[b_idx])
                append_cycle(action, "EXPLORE", random_mask)

        # 4) Full 3D random excitation: 100
        random_mask = np.array([True, True, True], dtype=bool)
        for _ in range(self.sequence_config.full_3d_cycles):
            action = self.rng.uniform(bounds.low, bounds.high).astype(np.float32)
            append_cycle(action, "EXPLORE", random_mask)

        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.shape[0] != self.sequence_config.base_block_cycles:
            raise RuntimeError(
                f"Base sequence size mismatch: got {actions_array.shape[0]}, "
                f"expected {self.sequence_config.base_block_cycles}."
            )
        return actions_array, specs

    def _interleave_zero_event_family(
        self,
        base_actions: np.ndarray,
        base_specs: List[CycleSpec],
    ) -> Tuple[np.ndarray, List[CycleSpec], Dict[str, float]]:
        base_actions_arr = np.asarray(base_actions, dtype=np.float32)
        base_specs_list = list(base_specs)
        base_len = len(base_actions_arr)
        first_explore_idx = self.sequence_config.base_anchor_hold_cycles
        min_gap = self.sequence_config.min_normal_cycles_between_event_starts

        # Event plan fields: (event_type, hold_cycles, recovery_cycles, use_smooth)
        event_plans: List[Tuple[str, int, int, bool]] = []
        full_family_cycles = 0
        partial_family_cycles = 0

        def ratios_ok_with_total(total_len: int) -> bool:
            full_ratio = full_family_cycles / total_len
            partial_ratio = partial_family_cycles / total_len
            return (
                full_ratio
                >= (self.sequence_config.full_no_injection_ratio - self.sequence_config.event_ratio_tolerance)
                and partial_ratio
                >= (self.sequence_config.partial_no_injection_ratio - self.sequence_config.event_ratio_tolerance)
            )

        def choose_event_type(projected_total_len: int) -> str:
            full_ratio = full_family_cycles / projected_total_len
            partial_ratio = partial_family_cycles / projected_total_len
            full_gap = self.sequence_config.full_no_injection_ratio - full_ratio
            partial_gap = self.sequence_config.partial_no_injection_ratio - partial_ratio
            if full_gap >= partial_gap:
                return "full"
            return "partial"

        def sample_event_shape() -> Tuple[int, int, bool]:
            hold_cycles = int(
                self.rng.integers(
                    self.sequence_config.zero_hold_min_cycles,
                    self.sequence_config.zero_hold_max_cycles + 1,
                )
            )
            use_smooth = bool(self.rng.integers(0, 2))
            recovery_cycles = 1
            if use_smooth:
                recovery_cycles = int(
                    self.rng.integers(
                        self.sequence_config.smooth_recovery_min_cycles,
                        self.sequence_config.smooth_recovery_max_cycles + 1,
                    )
                )
            return hold_cycles, recovery_cycles, use_smooth

        explore_cycles = max(0, base_len - first_explore_idx)
        max_event_count = 0
        if explore_cycles > 0:
            max_event_count = 1 + (explore_cycles - 1) // max(1, min_gap)

        while True:
            projected_total = base_len + full_family_cycles + partial_family_cycles
            if ratios_ok_with_total(projected_total):
                break
            if len(event_plans) >= max_event_count:
                break

            event_type = choose_event_type(projected_total)
            hold_cycles, recovery_cycles, use_smooth = sample_event_shape()
            family_added = hold_cycles + recovery_cycles
            event_plans.append((event_type, hold_cycles, recovery_cycles, use_smooth))
            if event_type == "full":
                full_family_cycles += family_added
            else:
                partial_family_cycles += family_added

        event_count = len(event_plans)
        event_starts: List[int] = []
        if event_count > 0:
            min_last_start = first_explore_idx + (event_count - 1) * min_gap
            if min_last_start > (base_len - 1):
                max_placeable = 1 + max(0, (base_len - 1 - first_explore_idx)) // max(1, min_gap)
                event_plans = event_plans[:max_placeable]
                event_count = len(event_plans)
                full_family_cycles = 0
                partial_family_cycles = 0
                for event_type, hold_cycles, recovery_cycles, _ in event_plans:
                    family_added = hold_cycles + recovery_cycles
                    if event_type == "full":
                        full_family_cycles += family_added
                    else:
                        partial_family_cycles += family_added

            prev_start = -10_000
            for event_idx in range(event_count):
                min_start = first_explore_idx + (event_idx * min_gap)
                if event_idx > 0:
                    min_start = max(min_start, prev_start + min_gap)
                remaining = event_count - event_idx - 1
                max_start = (base_len - 1) - (remaining * min_gap)
                start = int(self.rng.integers(min_start, max_start + 1))
                event_starts.append(start)
                prev_start = start

        final_actions: List[np.ndarray] = []
        final_specs: List[CycleSpec] = []
        partial_id1_cycles = 0
        partial_id2_cycles = 0

        def choose_zero_mask(event_type: str) -> np.ndarray:
            zero_mask = np.array([False, False], dtype=bool)
            if event_type == "full":
                zero_mask[:] = True
            elif partial_id1_cycles < partial_id2_cycles:
                zero_mask[0] = True
            elif partial_id2_cycles < partial_id1_cycles:
                zero_mask[1] = True
            else:
                zero_mask[int(self.rng.integers(0, 2))] = True
            return zero_mask

        def append_fixed_cycle(action: np.ndarray, mode_name: str) -> None:
            action_vec = np.asarray(action, dtype=np.float32).copy()
            final_actions.append(action_vec)
            final_specs.append(
                CycleSpec(
                    mode_name=mode_name,
                    filter_enabled=False,
                    random_mask=np.array([False, False, False], dtype=bool),
                    fixed_values=action_vec.copy(),
                )
            )

        def append_event_family(event_type: str, hold_cycles: int, recovery_cycles: int, use_smooth: bool) -> None:
            nonlocal partial_id1_cycles
            nonlocal partial_id2_cycles

            pre_event_action = final_actions[-1].copy()
            zero_mask = choose_zero_mask(event_type)

            for _ in range(hold_cycles):
                action = pre_event_action.copy()
                if zero_mask[0]:
                    action[0] = 0.0
                if zero_mask[1]:
                    action[1] = 0.0
                append_fixed_cycle(action, "NO_INJECTION")

            if use_smooth:
                start_action = final_actions[-1].copy()
                for step in range(1, recovery_cycles + 1):
                    frac = float(step / recovery_cycles)
                    action = start_action.copy()
                    if zero_mask[0]:
                        action[0] = start_action[0] + (pre_event_action[0] - start_action[0]) * frac
                    if zero_mask[1]:
                        action[1] = start_action[1] + (pre_event_action[1] - start_action[1]) * frac
                    append_fixed_cycle(action, "RECOVER")
            else:
                append_fixed_cycle(pre_event_action, "RECOVER")

            family_added = hold_cycles + recovery_cycles
            if event_type == "partial":
                if zero_mask[0]:
                    partial_id1_cycles += family_added
                if zero_mask[1]:
                    partial_id2_cycles += family_added

        event_ptr = 0
        for base_idx in range(base_len):
            while event_ptr < event_count and event_starts[event_ptr] == base_idx:
                if not final_actions:
                    # Do not start events before there is a pre-event cycle.
                    break
                event_type, hold_cycles, recovery_cycles, use_smooth = event_plans[event_ptr]
                append_event_family(event_type, hold_cycles, recovery_cycles, use_smooth)
                event_ptr += 1

            final_actions.append(base_actions_arr[base_idx].copy())
            final_specs.append(base_specs_list[base_idx])

        final_len = max(1, len(final_actions))
        event_stats = {
            "full_family_cycles": float(full_family_cycles),
            "partial_family_cycles": float(partial_family_cycles),
            "full_family_ratio": float(full_family_cycles / final_len),
            "partial_family_ratio": float(partial_family_cycles / final_len),
            "partial_id1_cycles": float(partial_id1_cycles),
            "partial_id2_cycles": float(partial_id2_cycles),
        }
        return np.asarray(final_actions, dtype=np.float32), final_specs, event_stats

    def resample_action_for_spec(self, spec: CycleSpec, bounds: ActionBounds, anchor: np.ndarray) -> np.ndarray:
        if not spec.filter_enabled:
            return spec.fixed_values.copy()

        random_mask = np.asarray(spec.random_mask, dtype=bool)
        action = np.asarray(anchor, dtype=np.float32).copy()
        fixed_values = np.asarray(spec.fixed_values, dtype=np.float32)

        for idx in range(3):
            if random_mask[idx]:
                action[idx] = self.rng.uniform(bounds.low[idx], bounds.high[idx])
            else:
                value = fixed_values[idx]
                if np.isnan(value):
                    value = anchor[idx]
                action[idx] = value
        return np.clip(action, bounds.low, bounds.high).astype(np.float32)

