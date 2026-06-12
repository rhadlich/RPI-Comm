import threading
import time
from typing import Optional

import numpy as np

from v2_config import COMM_LATENCY_SAMPLE_EVERY, COMM_PROGRESS_EMIT_EVERY
from v2_labview_comm import LabVIEWCommunicator

# Simulated LabVIEW request interval (seconds per control cycle).
DUMMY_CYCLE_INTERVAL_S = 0.08


class DummyLabVIEWCommunicator(LabVIEWCommunicator):
    """
    Drop-in replacement for LabVIEWCommunicator that simulates the recv/send
    loop without opening a TCP socket.
    """

    def __init__(
        self,
        server_ip: str,
        server_port: int,
        cycle_interval_s: float = DUMMY_CYCLE_INTERVAL_S,
    ):
        super().__init__(server_ip=server_ip, server_port=server_port)
        self.cycle_interval_s = max(0.01, float(cycle_interval_s))

    def _loop(self) -> None:
        self._latency_samples_ms = []
        phase = 0.0
        try:
            self._status(
                f"Dummy mode: simulating LabVIEW at {self.server_ip}:{self.server_port} "
                f"(no TCP, ~{1.0 / self.cycle_interval_s:.1f} Hz)."
            )
            while not self.stop_event.is_set():
                if self.stop_event.wait(timeout=self.cycle_interval_s):
                    break

                phase += self.cycle_interval_s
                incoming_values = np.asarray(
                    [0.5 + 0.5 * np.sin(phase), 0.5 + 0.5 * np.cos(phase * 0.7)],
                    dtype=np.float32,
                )
                self._emit_incoming(incoming_values)

                send_start_ns = time.perf_counter_ns()
                _payload_bytes, sent_cycles, total_cycles = self._next_payload()
                latency_ms = (time.perf_counter_ns() - send_start_ns) / 1_000_000.0

                if sent_cycles % max(1, int(COMM_LATENCY_SAMPLE_EVERY)) == 0:
                    self._record_latency_sample(latency_ms)

                progress_emit_every = max(1, int(COMM_PROGRESS_EMIT_EVERY))
                should_emit_progress = total_cycles > 0 and (
                    sent_cycles == total_cycles or sent_cycles % progress_emit_every == 0
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
            self._status(f"Dummy communication error: {exc}")
        finally:
            self._emit_latency_summary()
            self._status("Dummy communication worker stopped.")
