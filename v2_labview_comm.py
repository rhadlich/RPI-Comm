from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from v2_config import (
    BYTE_ORDER,
    COMM_ENABLE_REALTIME_PRIORITY,
    COMM_LATENCY_SAMPLE_EVERY,
    COMM_PROGRESS_EMIT_EVERY,
    COMM_REALTIME_PRIORITY,
    INCOMING_FLOAT_COUNT,
    INJECTION_PAYLOAD_FLOAT_COUNT,
    MODE_CODE_MAP,
    SOCKET_TIMEOUT_SECONDS,
    TCP_PAYLOAD_FLOAT_COUNT,
)


def recv_exact(sock: socket.socket, expected_bytes: int, stop_event: threading.Event) -> bytes | None:
    buffer = b""
    while len(buffer) < expected_bytes and not stop_event.is_set():
        try:
            chunk = sock.recv(expected_bytes - len(buffer))
        except socket.timeout:
            continue
        if not chunk:
            return None
        buffer += chunk
    if len(buffer) != expected_bytes:
        return None
    return buffer


@dataclass
class CommProgress:
    sent_cycles: int
    total_cycles: int
    cycles_remaining: int
    sequence_version: int = 0


@dataclass
class CommIncoming:
    value_1: float
    value_2: float


class LabVIEWCommunicator:
    """
    Communication-only TCP module. It does not generate actions and does not
    perform safety decisions. It only sends prepared payloads.
    """

    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = int(server_port)

        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None

        self.idle_payload = np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        self.idle_mode_name = "WARMUP"
        self.idle_tcp_bytes = self._build_tcp_bytes(self.idle_payload, self.idle_mode_name)

        self.prepared_payloads: List[np.ndarray] = []
        self.prepared_modes: List[str] = []
        self.prepared_tcp_bytes: List[bytes] = []
        self.sequence_version = 0
        self.collecting = False
        self.sequence_idx = 0
        self.collection_completed_once = False

        self.on_status: Optional[Callable[[str], None]] = None
        self.on_progress: Optional[Callable[[CommProgress], None]] = None
        self.on_complete: Optional[Callable[[], None]] = None
        self.on_incoming: Optional[Callable[[CommIncoming], None]] = None

        self._progress_emit_every = max(1, int(COMM_PROGRESS_EMIT_EVERY))
        self._latency_sample_every = max(1, int(COMM_LATENCY_SAMPLE_EVERY))
        self._latency_samples_ms: List[float] = []

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    def set_idle_payload(self, payload_12: np.ndarray, mode_name: str = "WARMUP") -> None:
        payload = np.asarray(payload_12, dtype=np.float32).reshape(INJECTION_PAYLOAD_FLOAT_COUNT)
        if mode_name not in MODE_CODE_MAP:
            raise ValueError(f"Unknown mode name: {mode_name}")
        tcp_bytes = self._build_tcp_bytes(payload, mode_name)
        with self.state_lock:
            self.idle_payload = payload
            self.idle_mode_name = mode_name
            self.idle_tcp_bytes = tcp_bytes

    def set_prepared_sequence(self, payloads_12: Sequence[np.ndarray], mode_names: Sequence[str]) -> None:
        if len(payloads_12) != len(mode_names):
            raise ValueError("Payload/mode lengths must match.")
        clean_payloads: List[np.ndarray] = []
        clean_modes: List[str] = []
        clean_tcp_bytes: List[bytes] = []
        for payload, mode in zip(payloads_12, mode_names):
            arr = np.asarray(payload, dtype=np.float32).reshape(INJECTION_PAYLOAD_FLOAT_COUNT)
            if mode not in MODE_CODE_MAP:
                raise ValueError(f"Unknown mode name in sequence: {mode}")
            clean_payloads.append(arr)
            clean_modes.append(mode)
            clean_tcp_bytes.append(self._build_tcp_bytes(arr, mode))
        with self.state_lock:
            self.prepared_payloads = clean_payloads
            self.prepared_modes = clean_modes
            self.prepared_tcp_bytes = clean_tcp_bytes
            self.sequence_idx = 0
            self.collection_completed_once = False
            self.sequence_version += 1

    def start_collection(self) -> None:
        with self.state_lock:
            if not self.prepared_payloads:
                raise RuntimeError("No prepared sequence loaded.")
            self.collecting = True
            self.sequence_idx = 0
            self.collection_completed_once = False

    def stop_collection(self) -> None:
        with self.state_lock:
            self.collecting = False

    def is_collecting(self) -> bool:
        with self.state_lock:
            return bool(self.collecting)

    def _status(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)

    def _emit_progress(self, sent_cycles: int, total_cycles: int) -> None:
        if self.on_progress is None:
            return
        remaining = max(0, total_cycles - sent_cycles)
        with self.state_lock:
            sequence_version = self.sequence_version
        self.on_progress(
            CommProgress(
                sent_cycles=sent_cycles,
                total_cycles=total_cycles,
                cycles_remaining=remaining,
                sequence_version=sequence_version,
            )
        )

    def _emit_incoming(self, incoming_values: np.ndarray) -> None:
        if self.on_incoming is None:
            return
        if incoming_values.shape[0] < 2:
            return
        self.on_incoming(
            CommIncoming(
                value_1=float(incoming_values[0]),
                value_2=float(incoming_values[1]),
            )
        )

    def _compose_tcp_payload(self, payload_12: np.ndarray, mode_name: str) -> np.ndarray:
        if mode_name not in MODE_CODE_MAP:
            raise ValueError(f"Unknown mode for TCP payload: {mode_name}")
        payload = np.zeros(TCP_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        payload[:INJECTION_PAYLOAD_FLOAT_COUNT] = payload_12
        payload[INJECTION_PAYLOAD_FLOAT_COUNT] = np.float32(MODE_CODE_MAP[mode_name])
        return payload

    def _build_tcp_bytes(self, payload_12: np.ndarray, mode_name: str) -> bytes:
        return self._compose_tcp_payload(payload_12, mode_name).astype(f"{BYTE_ORDER}f4").tobytes()

    def _next_payload(self) -> Tuple[bytes, int, int]:
        with self.state_lock:
            collecting = self.collecting
            total = len(self.prepared_tcp_bytes)
            idx = self.sequence_idx

            if collecting and idx < total:
                payload_bytes = self.prepared_tcp_bytes[idx]
                self.sequence_idx += 1
                sent_cycles = self.sequence_idx
                if self.sequence_idx >= total:
                    self.collecting = False
                return payload_bytes, sent_cycles, total

            sent_cycles = min(idx, total)
            return self.idle_tcp_bytes, sent_cycles, total

    def _try_raise_thread_priority(self) -> None:
        if not COMM_ENABLE_REALTIME_PRIORITY:
            return
        try:
            import os

            if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_FIFO"):
                self._status("Real-time scheduler unsupported on this platform; using default priority.")
                return
            params = os.sched_param(int(COMM_REALTIME_PRIORITY))
            os.sched_setscheduler(0, os.SCHED_FIFO, params)
            self._status(f"Communication thread using SCHED_FIFO priority {int(COMM_REALTIME_PRIORITY)}.")
        except PermissionError:
            self._status("Real-time priority denied (insufficient privileges); using default priority.")
        except OSError as exc:
            self._status(f"Unable to set real-time thread priority ({exc}); using default priority.")

    def _record_latency_sample(self, latency_ms: float) -> None:
        if latency_ms < 0.0:
            return
        self._latency_samples_ms.append(latency_ms)

    def _emit_latency_summary(self) -> None:
        if not self._latency_samples_ms:
            return
        samples = np.asarray(self._latency_samples_ms, dtype=np.float32)
        avg_ms = float(np.mean(samples))
        p95_ms = float(np.percentile(samples, 95))
        worst_ms = float(np.max(samples))
        self._status(
            f"Send latency stats (recv->send): avg={avg_ms:.3f}ms p95={p95_ms:.3f}ms max={worst_ms:.3f}ms "
            f"n={samples.size}"
        )

    def _loop(self) -> None:
        expected_bytes = INCOMING_FLOAT_COUNT * 4
        sock: Optional[socket.socket] = None
        self._latency_samples_ms = []
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(SOCKET_TIMEOUT_SECONDS)
            sock.connect((self.server_ip, self.server_port))
            self._try_raise_thread_priority()
            self._status(f"Connected to LabVIEW server at {self.server_ip}:{self.server_port}.")

            while not self.stop_event.is_set():
                incoming = recv_exact(sock, expected_bytes, self.stop_event)
                if incoming is None:
                    break
                incoming_values = np.frombuffer(incoming, dtype=f"{BYTE_ORDER}f4", count=INCOMING_FLOAT_COUNT)
                self._emit_incoming(incoming_values)

                send_start_ns = time.perf_counter_ns()
                payload_bytes, sent_cycles, total_cycles = self._next_payload()
                sock.sendall(payload_bytes)
                latency_ms = (time.perf_counter_ns() - send_start_ns) / 1_000_000.0

                if sent_cycles % self._latency_sample_every == 0:
                    self._record_latency_sample(latency_ms)

                should_emit_progress = (
                    total_cycles > 0
                    and (sent_cycles == total_cycles or sent_cycles % self._progress_emit_every == 0)
                )
                if should_emit_progress:
                    self._emit_progress(sent_cycles=sent_cycles, total_cycles=total_cycles)

                should_complete = total_cycles > 0 and sent_cycles >= total_cycles
                if should_complete:
                    with self.state_lock:
                        first_completion = not self.collection_completed_once
                        self.collection_completed_once = True
                    if first_completion:
                        self._status("Data collection complete.")
                        if self.on_complete is not None:
                            self.on_complete()

        except Exception as exc:
            self._status(f"TCP communication error: {exc}")
        finally:
            self._emit_latency_summary()
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            self._status("TCP worker stopped.")

