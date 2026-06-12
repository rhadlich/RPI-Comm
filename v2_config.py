from dataclasses import dataclass, field
from typing import Dict

import numpy as np


BYTE_ORDER = ">"
INCOMING_FLOAT_COUNT = 2
INJECTION_PAYLOAD_FLOAT_COUNT = 12
TCP_PAYLOAD_FLOAT_COUNT = 13
SOCKET_TIMEOUT_SECONDS = 1.0
COMM_ENABLE_REALTIME_PRIORITY = True
COMM_REALTIME_PRIORITY = 20
COMM_PROGRESS_EMIT_EVERY = 4
COMM_LATENCY_SAMPLE_EVERY = 25

PREP_WORKER_NICE_INCREMENT = 10
ALLOW_PREPARE_DURING_COLLECTION = False

DEFAULT_SERVER_IP = "192.168.1.5"
DEFAULT_SERVER_PORT = 54545

MODE_CODE_MAP: Dict[str, float] = {
    "WARMUP": 0.0,
    "EXPLORE": 1.0,
    "DERATE": 2.0,
    "ABORT": 3.0,
    "RECOVER": 4.0,
    "HOLD": 5.0,
    "NO_INJECTION": 6.0,
}


@dataclass(frozen=True)
class SequenceConfig:
    base_anchor_hold_cycles: int = 10
    single_action_cycles: int = 25
    pair_action_cycles: int = 25
    full_3d_cycles: int = 100
    full_no_injection_ratio: float = 0.10
    partial_no_injection_ratio: float = 0.05
    zero_hold_min_cycles: int = 3
    zero_hold_max_cycles: int = 5
    smooth_recovery_min_cycles: int = 5
    smooth_recovery_max_cycles: int = 10
    min_normal_cycles_between_event_starts: int = 20
    event_ratio_tolerance: float = 0.01

    @property
    def base_block_cycles(self) -> int:
        return (
            self.base_anchor_hold_cycles
            + (3 * self.single_action_cycles)
            + (3 * self.pair_action_cycles)
            + self.full_3d_cycles
        )


@dataclass(frozen=True)
class KNNConfig:
    k_neighbors: int = 5
    safety_threshold: float = 9.0
    max_normalized_mean_distance: float = 0.22
    max_resample_attempts_per_cycle: int = 120


@dataclass(frozen=True)
class ActionBounds:
    low: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.0, -140.0], dtype=np.float32)
    )
    high: np.ndarray = field(
        default_factory=lambda: np.array([0.8, 1.0, -10.0], dtype=np.float32)
    )

    def validate(self) -> None:
        if self.low.shape != (3,) or self.high.shape != (3,):
            raise ValueError("Bounds must be 3D vectors.")
        if np.any(self.high <= self.low):
            raise ValueError("Each upper bound must be greater than lower bound.")


@dataclass(frozen=True)
class PayloadConfig:
    # Payload order: T1,D1,T2,D2,T3,D3,T4,D4,T5,D5,T6,D6
    default_payload_12: np.ndarray = field(
        default_factory=lambda: np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
    )

