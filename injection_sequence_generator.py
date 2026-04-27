import numpy as np

OUTGOING_FLOAT_COUNT = 12
DEFAULT_ACTION_LOW = np.array([0.0, 0.0, -360.0], dtype=np.float32)   # [D1, D2, SOI2]
DEFAULT_ACTION_HIGH = np.array([5.0, 5.0, -200.0], dtype=np.float32)  # [D1, D2, SOI2]


class InjectionSequenceGenerator:
    """
    Stateful 3D action-space trajectory generator.

    Action space is [inj_duration1, inj_duration2, SOI2].
    Payload order is fixed:
    [timing1, duration1, timing2, duration2, ..., timing6, duration6]
    """

    def __init__(self):
        self.rng = np.random.default_rng()
        self.action_low = DEFAULT_ACTION_LOW.copy()
        self.action_high = DEFAULT_ACTION_HIGH.copy()
        self.default_payload = np.zeros(OUTGOING_FLOAT_COUNT, dtype=np.float32)

        self.target_action = None
        self.anchor_action = np.zeros(3, dtype=np.float32)
        self.vector_unit = None
        self.vector_magnitude = 0.0
        self.travel_distance = 0.0
        self.target_end_action = np.zeros(3, dtype=np.float32)
        self.step_delta = np.zeros(3, dtype=np.float32)
        self.current_mode = "smooth_ramp"
        self.current_length = 50
        self.current_cycle = 0
        self.vector_confirmed = False
        self.vector_unit_normalized = None
        self.planned_actions = None
        self.plan_events = {
            "hold": {"enabled": False, "start_idx": -1, "cycles": 0},
            "zero_drop": {"enabled": False, "start_idx": -1, "cycles": 0},
        }

    def _get_span(self):
        return self.action_high - self.action_low

    def _to_normalized(self, action_values):
        span = self._get_span()
        return (action_values - self.action_low) / span

    def _from_normalized(self, normalized_values):
        span = self._get_span()
        return self.action_low + (normalized_values * span)

    def reset(self):
        self.current_cycle = 0
        self.vector_confirmed = False
        self.planned_actions = None
        self.plan_events = {
            "hold": {"enabled": False, "start_idx": -1, "cycles": 0},
            "zero_drop": {"enabled": False, "start_idx": -1, "cycles": 0},
        }

    def configure_bounds(self, d1_low, d1_high, d2_low, d2_high, soi2_low, soi2_high):
        lows = np.array([d1_low, d2_low, soi2_low], dtype=np.float32)
        highs = np.array([d1_high, d2_high, soi2_high], dtype=np.float32)
        if np.any(highs <= lows):
            raise ValueError("Each upper bound must be greater than lower bound.")
        self.action_low = lows
        self.action_high = highs

    def set_default_payload(self, payload_12):
        values = np.asarray(payload_12, dtype=np.float32)
        if values.shape[0] != OUTGOING_FLOAT_COUNT:
            raise ValueError("Default payload must have 12 values.")
        self.default_payload = values.copy()

    def sample_random_target(self):
        # Sample uniformly in normalized [0,1]^3, then scale each axis by its own bounds.
        normalized_sample = self.rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        self.target_action = self._from_normalized(normalized_sample).astype(np.float32)
        return self.target_action.copy()

    def set_anchor_from_payload(self, payload_12):
        values = np.asarray(payload_12, dtype=np.float32)
        if values.shape[0] != OUTGOING_FLOAT_COUNT:
            raise ValueError("Anchor payload must have 12 values.")
        self.anchor_action = np.array([values[1], values[3], values[2]], dtype=np.float32)
        return self.anchor_action.copy()

    def set_anchor_action(self, anchor_action_3):
        anchor_action = np.asarray(anchor_action_3, dtype=np.float32)
        if anchor_action.shape[0] != 3:
            raise ValueError("Anchor action must have 3 values [D1, D2, SOI2].")
        self.anchor_action = np.clip(anchor_action, self.action_low, self.action_high).astype(np.float32)
        return self.anchor_action.copy()

    def sample_random_unit_vector(self):
        vec_norm_space = self.rng.normal(size=3).astype(np.float32)
        norm = float(np.linalg.norm(vec_norm_space))
        if norm < 1e-8:
            vec_norm_space = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            norm = 1.0
        # Unit vector in normalized action space (equalized axis importance).
        self.vector_unit_normalized = (vec_norm_space / norm).astype(np.float32)

        # Keep a physical-space representation for display/debug convenience.
        physical_vec = self.vector_unit_normalized * self._get_span()
        physical_norm = float(np.linalg.norm(physical_vec))
        if physical_norm < 1e-8:
            physical_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            physical_norm = 1.0
        self.vector_unit = (physical_vec / physical_norm).astype(np.float32)
        self.vector_magnitude = 1.0
        self.vector_confirmed = False
        return self.vector_unit.copy(), float(self.vector_magnitude)

    def _max_travel_to_bounds(self):
        if self.vector_unit_normalized is None:
            return 0.0
        anchor_norm = self._to_normalized(self.anchor_action.astype(np.float32))
        max_candidates = []
        for idx in range(3):
            comp = float(self.vector_unit_normalized[idx])
            anchor = float(anchor_norm[idx])
            if abs(comp) < 1e-8:
                continue
            if comp > 0.0:
                bound_dist = (1.0 - anchor) / comp
            else:
                bound_dist = (0.0 - anchor) / comp
            if bound_dist > 0.0:
                max_candidates.append(bound_dist)
        if not max_candidates:
            return 0.0
        return float(min(max_candidates))

    def confirm_vector(self):
        if self.vector_unit is None:
            raise ValueError("Vector is not generated yet.")
        self.vector_confirmed = True

    def _sample_mode_and_length(self, mode_probabilities, min_length, max_length, include_no_injection=True):
        if min_length <= 0 or max_length < min_length:
            raise ValueError("Invalid trajectory length range.")
        mode_names = ["smooth_ramp", "step_hold", "boundary_probe"]
        if include_no_injection:
            mode_names.append("no_injection")
        weights = np.array([mode_probabilities[name] for name in mode_names], dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("Sampling weights must sum to > 0.")
        probs = weights / weight_sum
        mode_idx = int(self.rng.choice(len(mode_names), p=probs))
        self.current_mode = mode_names[mode_idx]
        self.current_length = int(self.rng.integers(min_length, max_length + 1))
        self.current_cycle = 0

    def _set_vector_from_anchor_to_end(self, end_action):
        end_action = np.asarray(end_action, dtype=np.float32)
        if end_action.shape[0] != 3:
            raise ValueError("End action must have 3 values [D1, D2, SOI2].")
        self.target_end_action = np.clip(end_action, self.action_low, self.action_high).astype(np.float32)

        anchor_norm = self._to_normalized(self.anchor_action.astype(np.float32))
        target_norm = self._to_normalized(self.target_end_action.astype(np.float32))
        direction_norm = target_norm - anchor_norm
        travel_distance = float(np.linalg.norm(direction_norm))
        self.travel_distance = travel_distance

        if travel_distance < 1e-8:
            self.vector_unit_normalized = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.vector_unit = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.vector_magnitude = 0.0
            return

        self.vector_unit_normalized = (direction_norm / travel_distance).astype(np.float32)
        direction_phys = self.target_end_action - self.anchor_action
        direction_phys_norm = float(np.linalg.norm(direction_phys))
        if direction_phys_norm < 1e-8:
            self.vector_unit = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.vector_magnitude = 0.0
            return
        self.vector_unit = (direction_phys / direction_phys_norm).astype(np.float32)
        self.vector_magnitude = 1.0

    def sample_trajectory_plan(self, mode_probabilities, min_length, max_length):
        self.planned_actions = None
        self.plan_events = {
            "hold": {"enabled": False, "start_idx": -1, "cycles": 0},
            "zero_drop": {"enabled": False, "start_idx": -1, "cycles": 0},
        }
        self._sample_mode_and_length(
            mode_probabilities,
            min_length,
            max_length,
            include_no_injection=True,
        )
        if self.current_mode == "no_injection":
            end_action = self.anchor_action.copy()
            end_action[0] = 0.0
            end_action[1] = 0.0
            self.target_end_action = np.clip(end_action, self.action_low, self.action_high).astype(np.float32)
        else:
            max_travel = self._max_travel_to_bounds()  # normalized-distance limit
            travel_scale = float(self.rng.uniform(0.25, 0.95))
            self.travel_distance = max(0.0, max_travel * travel_scale)
            anchor_norm = self._to_normalized(self.anchor_action.astype(np.float32))
            target_norm = anchor_norm + (self.vector_unit_normalized * self.travel_distance)
            target_norm = np.clip(target_norm, 0.0, 1.0)
            self.target_end_action = self._from_normalized(target_norm).astype(np.float32)
        denom = max(1, self.current_length - 1)
        self.step_delta = ((self.target_end_action - self.anchor_action) / denom).astype(np.float32)
        return self.current_mode, self.current_length

    def sample_trajectory_plan_to_endpoint(self, end_action, mode_probabilities, min_length, max_length):
        if min_length <= 0 or max_length < min_length:
            raise ValueError("Invalid trajectory length range.")
        if sum(max(0.0, float(v)) for v in mode_probabilities.values()) <= 0.0:
            raise ValueError("Sampling weights must sum to > 0.")
        self.current_length = int(self.rng.integers(min_length, max_length + 1))
        self.current_cycle = 0
        self.current_mode = "eventful_ramp"
        self._set_vector_from_anchor_to_end(end_action)
        self.plan_events = self._sample_plan_events(mode_probabilities, self.current_length)
        self.planned_actions = self._build_endpoint_plan_with_events()
        denom = max(1, self.current_length - 1)
        self.step_delta = ((self.target_end_action - self.anchor_action) / denom).astype(np.float32)
        self.vector_confirmed = False
        return self.current_mode, self.current_length

    def _sample_plan_events(self, mode_probabilities, length):
        total = sum(max(0.0, float(v)) for v in mode_probabilities.values())
        if total <= 0.0:
            return {
                "hold": {"enabled": False, "start_idx": -1, "cycles": 0},
                "zero_drop": {"enabled": False, "start_idx": -1, "cycles": 0},
            }
        hold_prob = max(0.0, float(mode_probabilities.get("step_hold", 0.0))) / total
        drop_prob = max(0.0, float(mode_probabilities.get("no_injection", 0.0))) / total
        hold = self._sample_event_window(length, hold_prob)
        zero_drop = self._sample_event_window(length, drop_prob)

        if hold["enabled"] and zero_drop["enabled"]:
            hold_start = hold["start_idx"]
            hold_end = hold_start + hold["cycles"]
            drop_start = zero_drop["start_idx"]
            drop_end = drop_start + zero_drop["cycles"]
            overlap = (hold_start < drop_end) and (drop_start < hold_end)
            if overlap:
                zero_drop = {"enabled": False, "start_idx": -1, "cycles": 0}
        return {"hold": hold, "zero_drop": zero_drop}

    def _sample_event_window(self, length, event_probability):
        if length < 8:
            return {"enabled": False, "start_idx": -1, "cycles": 0}
        if float(self.rng.random()) >= float(min(max(event_probability, 0.0), 1.0)):
            return {"enabled": False, "start_idx": -1, "cycles": 0}

        max_cycles = max(2, min(12, length // 4))
        cycles = int(self.rng.integers(2, max_cycles + 1))
        start_max = length - cycles - 2
        if start_max < 1:
            return {"enabled": False, "start_idx": -1, "cycles": 0}
        start_idx = int(self.rng.integers(1, start_max + 1))
        return {"enabled": True, "start_idx": start_idx, "cycles": cycles}

    def _build_endpoint_plan_with_events(self):
        if self.current_length <= 0:
            return None
        if self.current_length == 1:
            return self.target_end_action.reshape(1, 3).astype(np.float32)

        steps = np.linspace(0.0, 1.0, self.current_length, dtype=np.float32).reshape(-1, 1)
        actions = self.anchor_action + ((self.target_end_action - self.anchor_action) * steps)
        actions = actions.astype(np.float32)

        hold = self.plan_events["hold"]
        if hold["enabled"]:
            hold_start = hold["start_idx"]
            hold_end = hold_start + hold["cycles"]
            held_action = actions[hold_start - 1].copy()
            actions[hold_start:hold_end] = held_action
            remainder = self.current_length - hold_end
            if remainder >= 1:
                actions[hold_end] = held_action
                if remainder > 1:
                    tail_steps = np.linspace(0.0, 1.0, remainder - 1, dtype=np.float32).reshape(-1, 1)
                    actions[hold_end + 1 :] = held_action + (
                        (self.target_end_action - held_action) * tail_steps
                    )

        zero_drop = self.plan_events["zero_drop"]
        if zero_drop["enabled"]:
            drop_start = zero_drop["start_idx"]
            drop_end = drop_start + zero_drop["cycles"]
            pre_drop_fueling = actions[drop_start - 1, :2].copy()
            actions[drop_start:drop_end, :2] = 0.0
            remainder = self.current_length - drop_end
            if remainder >= 1:
                actions[drop_end, :2] = pre_drop_fueling
                if remainder > 1:
                    tail_steps = np.linspace(0.0, 1.0, remainder - 1, dtype=np.float32).reshape(-1, 1)
                    actions[drop_end + 1 :, :2] = pre_drop_fueling + (
                        (self.target_end_action[:2] - pre_drop_fueling) * tail_steps
                    )

        actions = np.clip(actions, self.action_low, self.action_high).astype(np.float32)
        actions[-1] = self.target_end_action.copy()
        return actions

    def _get_multiplier(self, cycle_idx):
        denom = max(1, self.current_length - 1)
        progress = cycle_idx / denom
        if self.current_mode == "smooth_ramp":
            return progress
        if self.current_mode == "step_hold":
            if progress < 0.33:
                return 0.35
            if progress < 0.66:
                return 0.7
            return 1.0
        if self.current_mode == "boundary_probe":
            return min(1.1 * progress, 1.0)
        if self.current_mode == "no_injection":
            return 0.0
        return 0.0

    def _vector_to_action(self, multiplier):
        if self.current_mode == "no_injection":
            action = self.anchor_action.copy()
            action[0] = 0.0
            action[1] = 0.0
            return action
        if self.vector_unit is None:
            return self.anchor_action.copy()
        action = self.anchor_action + ((self.target_end_action - self.anchor_action) * multiplier)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def build_payload_from_action(self, action_values_3d):
        action = np.asarray(action_values_3d, dtype=np.float32)
        payload = self.default_payload.copy()
        payload[1] = action[0]  # D1
        payload[3] = action[1]  # D2
        payload[2] = action[2]  # SOI2
        return payload.astype(np.float32)

    def next_trajectory_values(self, derate_factor=1.0):
        if not self.vector_confirmed:
            raise ValueError("Vector must be confirmed before starting trajectory.")
        if self.planned_actions is not None and self.current_cycle < len(self.planned_actions):
            action = self.planned_actions[self.current_cycle].copy()
            if float(derate_factor) < 1.0:
                action[:2] = self.anchor_action[:2] + (
                    (action[:2] - self.anchor_action[:2]) * float(max(0.0, derate_factor))
                )
                action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
            denom = max(1, self.current_length - 1)
            multiplier = float(min(1.0, self.current_cycle / denom))
        else:
            multiplier = self._get_multiplier(self.current_cycle) * float(max(0.0, derate_factor))
            action = self._vector_to_action(multiplier)
        payload = self.build_payload_from_action(action)
        self.current_cycle += 1
        return payload, action, float(multiplier), self.current_cycle >= self.current_length

    def get_abort_payload(self):
        abort_action = self.anchor_action.copy()
        abort_action[0] = 0.0
        abort_action[1] = 0.0
        return self.build_payload_from_action(abort_action), abort_action

    def get_plan_details(self):
        return {
            "anchor": self.anchor_action.copy(),
            "unit_vector": None if self.vector_unit is None else self.vector_unit.copy(),
            "target_end_action": self.target_end_action.copy(),
            "length": int(self.current_length),
            "step_delta": self.step_delta.copy(),
            "bounds_low": self.action_low.copy(),
            "bounds_high": self.action_high.copy(),
            "travel_distance": float(self.travel_distance),
            "mode": self.current_mode,
            "plan_events": {
                "hold": self.plan_events["hold"].copy(),
                "zero_drop": self.plan_events["zero_drop"].copy(),
            },
        }
