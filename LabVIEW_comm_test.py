import socket
import threading
import tkinter as tk
from queue import Empty, Queue
from tkinter import messagebox

import numpy as np
from injection_sequence_generator import InjectionSequenceGenerator

SERVER_IP = "192.168.1.5"
SERVER_PORT = 54545
BYTE_ORDER = ">"
INCOMING_FLOAT_COUNT = 2
OUTGOING_FLOAT_COUNT = 12
SOCKET_TIMEOUT_SECONDS = 1.0


def recv_exact(sock, expected_bytes, stop_event):
    buffer = b""
    while len(buffer) < expected_bytes and not stop_event.is_set():
        try:
            chunk = sock.recv(expected_bytes - len(buffer))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("LabVIEW server closed connection")
        buffer += chunk
    return buffer if len(buffer) == expected_bytes else None


class TCPResponderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LabVIEW TCP Responder")

        self.stop_event = threading.Event()
        self.values_lock = threading.Lock()
        self.mode_lock = threading.Lock()
        self.sequence_lock = threading.Lock()
        self.latest_values = np.zeros(OUTGOING_FLOAT_COUNT, dtype=np.float32)
        self.output_mode = "manual"
        self.sequence_mode = "indefinite"
        self.sequence_length = 10
        self.sequence_samples_sent = 0
        self.sequence_generator = InjectionSequenceGenerator()
        self.status_queue = Queue()
        self.worker_thread = None

        self.entry_vars = []
        self._build_ui()
        self._start_worker()
        self._poll_status_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Value source:", anchor="w").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.mode_var = tk.StringVar(value="manual")
        tk.Radiobutton(
            frame,
            text="Manual",
            variable=self.mode_var,
            value="manual",
            command=self.on_mode_change,
        ).grid(row=0, column=1, sticky="w", pady=(0, 8), padx=(0, 10))
        tk.Radiobutton(
            frame,
            text="Sequence",
            variable=self.mode_var,
            value="sequence",
            command=self.on_mode_change,
        ).grid(row=0, column=2, sticky="w", pady=(0, 8))

        manual_frame = tk.LabelFrame(frame, text="Manual Values", padx=10, pady=10)
        manual_frame.grid(row=1, column=0, columnspan=4, sticky="nw")

        sequence_frame = tk.LabelFrame(frame, text="Sequence Options", padx=10, pady=10)
        sequence_frame.grid(row=1, column=5, columnspan=4, sticky="ne", padx=(30, 0))

        tk.Label(
            manual_frame,
            text="Set 12 output values (6 injections: timing and duration):",
            anchor="w",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        for injection_idx in range(6):
            timing_idx = injection_idx * 2
            duration_idx = timing_idx + 1

            tk.Label(manual_frame, text=f"Inj {injection_idx + 1} timing").grid(
                row=injection_idx + 1, column=0, sticky="w", padx=(0, 6), pady=2
            )
            timing_var = tk.StringVar(value="0.0")
            tk.Entry(manual_frame, width=12, textvariable=timing_var).grid(
                row=injection_idx + 1, column=1, sticky="w", padx=(0, 12), pady=2
            )
            self.entry_vars.append((timing_idx, timing_var))

            tk.Label(manual_frame, text=f"Inj {injection_idx + 1} duration").grid(
                row=injection_idx + 1, column=2, sticky="w", padx=(0, 6), pady=2
            )
            duration_var = tk.StringVar(value="0.0")
            tk.Entry(manual_frame, width=12, textvariable=duration_var).grid(
                row=injection_idx + 1, column=3, sticky="w", pady=2
            )
            self.entry_vars.append((duration_idx, duration_var))

        self.apply_button = tk.Button(manual_frame, text="Apply", command=self.apply_values)
        self.apply_button.grid(row=8, column=0, sticky="w", pady=(12, 0))

        tk.Label(sequence_frame, text="Sequence sampling:").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.sequence_mode_var = tk.StringVar(value="indefinite")
        tk.Radiobutton(
            sequence_frame,
            text="Indefinite",
            variable=self.sequence_mode_var,
            value="indefinite",
            command=self.on_sequence_mode_change,
        ).grid(row=1, column=0, sticky="w", pady=(0, 4))
        tk.Radiobutton(
            sequence_frame,
            text="Fixed length",
            variable=self.sequence_mode_var,
            value="fixed",
            command=self.on_sequence_mode_change,
        ).grid(row=2, column=0, sticky="w", pady=(0, 8))

        tk.Label(sequence_frame, text="Sequence length:").grid(
            row=3, column=0, sticky="w", pady=(0, 4)
        )
        self.sequence_length_var = tk.StringVar(value="10")
        self.sequence_length_entry = tk.Entry(
            sequence_frame, width=12, textvariable=self.sequence_length_var
        )
        self.sequence_length_entry.grid(row=4, column=0, sticky="w", pady=(0, 8))
        self.sequence_length_var.trace_add("write", self.on_sequence_length_change)

        self.zero_button = tk.Button(frame, text="ZERO", command=self.zero_and_apply_values)
        self.zero_button.grid(row=2, column=1, sticky="w", pady=(12, 0), padx=(8, 0))

        self.stop_button = tk.Button(frame, text="Stop", command=self.stop_communication)
        self.stop_button.grid(row=2, column=0, sticky="w", pady=(12, 0))

        self.status_var = tk.StringVar(value="Starting TCP worker...")
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=3, column=0, columnspan=9, sticky="w", pady=(12, 0)
        )

        self.last_received_var = tk.StringVar(value="Last received: []")
        tk.Label(frame, textvariable=self.last_received_var, anchor="w", justify="left").grid(
            row=4, column=0, columnspan=9, sticky="w", pady=(6, 0)
        )

        self.last_sent_var = tk.StringVar(value="Last sent: []")
        tk.Label(frame, textvariable=self.last_sent_var, anchor="w", justify="left").grid(
            row=5, column=0, columnspan=9, sticky="w", pady=(4, 0)
        )

    def apply_values(self):
        try:
            updated_values = np.zeros(OUTGOING_FLOAT_COUNT, dtype=np.float32)
            for idx, entry_var in self.entry_vars:
                updated_values[idx] = float(entry_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "All 12 fields must be valid numbers.")
            return

        with self.values_lock:
            self.latest_values = updated_values
        self.status_var.set("Values applied. Next LabVIEW trigger will use these values.")

    def zero_and_apply_values(self):
        zeros = np.zeros(OUTGOING_FLOAT_COUNT, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")

        with self.values_lock:
            self.latest_values = zeros

        with self.mode_lock:
            self.output_mode = "manual"
        self.mode_var.set("manual")
        self.status_var.set("ZERO applied: all values set to 0.0 and mode set to manual.")

    def on_mode_change(self):
        selected_mode = self.mode_var.get()
        if selected_mode == "sequence":
            self.apply_sequence_settings(show_status=False)
        with self.mode_lock:
            self.output_mode = selected_mode
            if selected_mode == "sequence":
                self.sequence_samples_sent = 0
                with self.sequence_lock:
                    self.sequence_generator.reset()
        self.status_var.set(f"Source mode set to: {selected_mode}")

    def on_sequence_mode_change(self):
        selected_sequence_mode = self.sequence_mode_var.get()
        self.apply_sequence_settings(show_status=False)
        with self.mode_lock:
            self.sequence_mode = selected_sequence_mode
            self.sequence_samples_sent = 0
        with self.sequence_lock:
            self.sequence_generator.reset()
        self.status_var.set(f"Sequence sampling mode set to: {selected_sequence_mode}")

    def on_sequence_length_change(self, *_):
        self.apply_sequence_settings(show_status=False)

    def apply_sequence_settings(self, show_status=True):
        raw_text = self.sequence_length_var.get().strip()
        try:
            parsed_value = int(raw_text)
            if parsed_value <= 0:
                raise ValueError
        except ValueError:
            if show_status:
                self.status_var.set("Sequence length must be a positive integer.")
            return
        with self.mode_lock:
            self.sequence_length = parsed_value
            self.sequence_mode = self.sequence_mode_var.get()
            self.sequence_samples_sent = 0
        with self.sequence_lock:
            self.sequence_generator.reset()
        if show_status:
            self.status_var.set(
                f"Sequence settings applied: {self.sequence_mode} / length {self.sequence_length}"
            )

    def _start_worker(self):
        self.worker_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        self.worker_thread.start()

    def _tcp_loop(self):
        expected_bytes = INCOMING_FLOAT_COUNT * 4
        packet_count = 0
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(SOCKET_TIMEOUT_SECONDS)
            sock.connect((SERVER_IP, SERVER_PORT))
            self.status_queue.put(("status", "Connected to LabVIEW server."))

            while not self.stop_event.is_set():
                incoming = recv_exact(sock, expected_bytes, self.stop_event)
                if incoming is None:
                    break
                incoming_values = np.frombuffer(incoming, dtype=f"{BYTE_ORDER}f4")
                self.status_queue.put(("received", incoming_values.tolist()))

                with self.mode_lock:
                    mode = self.output_mode
                if mode == "sequence":
                    with self.mode_lock:
                        sequence_mode = self.sequence_mode
                        sequence_length = self.sequence_length
                        samples_sent = self.sequence_samples_sent

                    if sequence_mode == "fixed" and samples_sent >= sequence_length:
                        with self.mode_lock:
                            self.output_mode = "manual"
                        self.status_queue.put(
                            (
                                "set_manual_mode",
                                f"Fixed sequence complete ({sequence_length} samples). Returning to manual mode.",
                            )
                        )
                        with self.values_lock:
                            response_values = self.latest_values.copy()
                        mode = "manual"
                    else:
                        with self.sequence_lock:
                            response_values = self.sequence_generator.next_values()
                        with self.mode_lock:
                            self.sequence_samples_sent += 1
                else:
                    with self.values_lock:
                        response_values = self.latest_values.copy()
                payload = response_values.astype(f"{BYTE_ORDER}f4").tobytes(order="C")
                sock.sendall(payload)
                self.status_queue.put(("sent", response_values.tolist()))

                packet_count += 1
                self.status_queue.put(
                    (
                        "status",
                        f"Packet {packet_count} ({mode}): recv {incoming_values.tolist()} | sent {response_values.tolist()}",
                    )
                )
        except Exception as exc:
            self.status_queue.put(("status", f"Communication error: {exc}"))
        finally:
            if sock is not None:
                sock.close()
            self.status_queue.put(("status", "Communication stopped."))

    def _poll_status_queue(self):
        try:
            while True:
                event_type, payload = self.status_queue.get_nowait()
                if event_type == "received":
                    self.last_received_var.set(f"Last received: {payload}")
                elif event_type == "sent":
                    self.last_sent_var.set(f"Last sent: {payload}")
                elif event_type == "set_manual_mode":
                    self.mode_var.set("manual")
                    self.status_var.set(payload)
                else:
                    print(payload)
                    self.status_var.set(payload)
        except Empty:
            pass
        self.root.after(100, self._poll_status_queue)

    def stop_communication(self):
        self.stop_event.set()
        self.apply_button.config(state=tk.DISABLED)
        self.zero_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.sequence_length_entry.config(state=tk.DISABLED)
        self.status_var.set("Stopping communication...")

    def on_close(self):
        self.stop_communication()
        self.root.destroy()


def main():
    root = tk.Tk()
    TCPResponderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
