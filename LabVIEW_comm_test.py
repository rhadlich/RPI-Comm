import os
import socket
import threading
import tkinter as tk
import time
from queue import Empty, Queue
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from injection_sequence_generator import InjectionSequenceGenerator

RT_PRIORITY = 80

SERVER_IP = "192.168.1.5"
SERVER_PORT = 54545
BYTE_ORDER = ">"
INCOMING_FLOAT_COUNT = 2
INJECTION_PAYLOAD_FLOAT_COUNT = 12
TCP_PAYLOAD_FLOAT_COUNT = 13
SOCKET_TIMEOUT_SECONDS = 1.0
INJECTION_EVENT_COUNT = 6
MODE_CODE_MAP = {
    "WARMUP": 0.0,
    "EXPLORE": 1.0,
    "DERATE": 2.0,
    "ABORT": 3.0,
    "RECOVER": 4.0,
    "HOLD": 5.0,
    "NO_INJECTION": 6.0,
}
SOFT_LIMIT_AVG_MPRR = 9.0
SOFT_LIMIT_DERATE_ENTRY_MPRR = 10.5
HARD_LIMIT_AVG_MPRR = 11.0
SINGLE_CYCLE_MPRR_LIMIT = 12.0
BOUNDARY_MAX_CYCLES = 30
SOFT_LIMIT_CYCLE_CAP_DEFAULT = 50


def _set_thread_realtime_priority(priority=RT_PRIORITY):
    """
    Attempt SCHED_FIFO real-time scheduling for the calling thread.

    On Linux this requires CAP_SYS_NICE or root. Grants the privilege before
    calling via: sudo setcap cap_sys_nice+ep python3
    On macOS or other platforms without sched_setscheduler the call is a no-op.

    Returns (success: bool, message: str).
    """
    if not hasattr(os, "sched_setscheduler"):
        return False, "Real-time scheduling not supported on this platform (skipped)."
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        return True, f"RT scheduler: SCHED_FIFO priority={priority}."
    except PermissionError:
        return (
            False,
            f"Permission denied setting RT priority {priority}. "
            "Run as root or grant CAP_SYS_NICE: "
            "sudo setcap cap_sys_nice+ep $(which python3)",
        )
    except OSError as exc:
        return False, f"Failed to set RT priority: {exc}"


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
        self._configure_scaling()

        self.stop_event = threading.Event()
        self.values_lock = threading.Lock()
        self.mode_lock = threading.Lock()
        self.sequence_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        self.latest_values = np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        self.sequence_generator = InjectionSequenceGenerator()
        self.output_mode = "MANUAL"
        self.execution_mode_text = "manual"
        self.recover_cycles_remaining = 0
        self.soft_limit_cycles_over = 0
        self.derate_active = False
        self.active_anchor_action = np.zeros(3, dtype=np.float32)
        self.last_trajectory_action = np.zeros(3, dtype=np.float32)
        self.boundary_control_active = False
        self.boundary_explore_cycles = 0
        self.boundary_action = None
        self.abort_latch_cycles = 0
        self.trajectory_start_time_s = None
        self.trajectory_warmup_seconds = 0.0
        self.mprr_window_n = 10
        self.soft_limit_cycle_cap = SOFT_LIMIT_CYCLE_CAP_DEFAULT
        self.mprr_history = []

        self.status_queue = Queue()
        self.worker_thread = None
        self.entry_vars = []
        self.popup = None
        self.popup_content_frame = None
        self.popup_stage = ""
        self.popup_vars = {}

        self._build_ui()
        self.root.update_idletasks()
        self.root.resizable(False, False)
        self._start_worker()
        self._poll_status_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_scaling(self):
        default_font = tkfont.nametofont("TkDefaultFont")
        text_font = tkfont.nametofont("TkTextFont")
        fixed_font = tkfont.nametofont("TkFixedFont")
        default_font.configure(size=13)
        text_font.configure(size=13)
        fixed_font.configure(size=13)
        self.section_title_font = tkfont.Font(
            family=default_font.cget("family"), size=14, weight="bold"
        )

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        manual_frame = tk.LabelFrame(
            frame, text="Manual Values", font=self.section_title_font, padx=10, pady=10
        )
        manual_frame.grid(row=0, column=0, columnspan=3, sticky="nw")

        settings_frame = tk.LabelFrame(
            frame, text="Trajectory Settings", font=self.section_title_font, padx=10, pady=10
        )
        settings_frame.grid(row=0, column=3, columnspan=3, sticky="ne", padx=(20, 0))

        tk.Label(manual_frame, text="Payload order: T1,D1,T2,D2,...,T6,D6").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )
        for injection_idx in range(INJECTION_EVENT_COUNT):
            timing_idx = injection_idx * 2
            duration_idx = timing_idx + 1
            tk.Label(manual_frame, text=f"Inj {injection_idx + 1} timing").grid(
                row=injection_idx + 1, column=0, sticky="w", padx=(0, 6), pady=2
            )
            timing_var = tk.StringVar(value="0.0")
            tk.Entry(manual_frame, width=12, textvariable=timing_var).grid(
                row=injection_idx + 1, column=1, sticky="w", padx=(0, 10), pady=2
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

        self.zero_button = tk.Button(manual_frame, text="ZERO", command=self.zero_and_apply_values)
        self.zero_button.grid(row=8, column=1, sticky="w", pady=(12, 0), padx=(8, 0))

        self.generate_button = tk.Button(
            frame, text="Generate Random Target", command=self.open_target_popup
        )
        self.generate_button.grid(row=1, column=0, sticky="w", pady=(12, 0))

        self.stop_button = tk.Button(frame, text="Stop", command=self.stop_communication)
        self.stop_button.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(12, 0))
        self.estop_button = tk.Button(
            frame,
            text="E-STOP",
            command=self._e_stop,
            fg="red",
            activeforeground="red",
        )
        self.estop_button.grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(12, 0))

        self.mode_var = tk.StringVar(value="Mode: MANUAL")
        tk.Label(frame, textvariable=self.mode_var, anchor="w").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )
        self.status_var = tk.StringVar(value="Starting TCP worker...")
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self.exec_mode_var = tk.StringVar(value="Trajectory mode: manual")
        tk.Label(frame, textvariable=self.exec_mode_var, anchor="w").grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self.latest_mprr_var = tk.StringVar(value="Latest MPRR: --")
        tk.Label(frame, textvariable=self.latest_mprr_var, anchor="w").grid(
            row=5, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self.avg_mprr_var = tk.StringVar(value=f"Avg MPRR ({self.mprr_window_n}-cycle): --")
        tk.Label(frame, textvariable=self.avg_mprr_var, anchor="w").grid(
            row=6, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self.last_received_var = tk.StringVar(value="Last received: []")
        tk.Label(frame, textvariable=self.last_received_var, anchor="w", justify="left").grid(
            row=7, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self.last_sent_var = tk.StringVar(value="Last sent: []")
        tk.Label(frame, textvariable=self.last_sent_var, anchor="w", justify="left").grid(
            row=8, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )

        self._build_settings_ui(settings_frame)

    def _build_settings_ui(self, frame):
        self.bound_vars = {
            "d1_low": tk.StringVar(value="0.0"),
            "d1_high": tk.StringVar(value="0.9"),
            "d2_low": tk.StringVar(value="0.0"),
            "d2_high": tk.StringVar(value="1.4"),
            "soi2_low": tk.StringVar(value="-140.0"),
            "soi2_high": tk.StringVar(value="-10.0"),
        }
        self.mode_prob_vars = {
            "smooth_ramp": tk.StringVar(value="0.45"),
            "step_hold": tk.StringVar(value="0.20"),
            "no_injection": tk.StringVar(value="0.20"),
        }
        self.length_min_var = tk.StringVar(value="200")
        self.length_max_var = tk.StringVar(value="400")
        self.warmup_seconds_var = tk.StringVar(value="5.0")
        self.mprr_window_var = tk.StringVar(value=str(self.mprr_window_n))
        self.soft_limit_cycle_cap_var = tk.StringVar(value=str(self.soft_limit_cycle_cap))

        row = 0
        tk.Label(frame, text="Global action bounds").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        for key, label in [
            ("d1_low", "D1 low"), ("d1_high", "D1 high"),
            ("d2_low", "D2 low"), ("d2_high", "D2 high"),
            ("soi2_low", "SOI2 low"), ("soi2_high", "SOI2 high"),
        ]:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, width=10, textvariable=self.bound_vars[key]).grid(
                row=row, column=1, sticky="w", pady=1
            )
            row += 1

        tk.Label(frame, text="Sampling balance").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        for key, label in [
            ("smooth_ramp", "P smooth"), ("step_hold", "P hold"),
            ("no_injection", "P zero-drop"),
        ]:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, width=10, textvariable=self.mode_prob_vars[key]).grid(
                row=row, column=1, sticky="w", pady=1
            )
            row += 1

        tk.Label(frame, text="Len min/max").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, width=10, textvariable=self.length_min_var).grid(row=row, column=1, sticky="w")
        tk.Entry(frame, width=10, textvariable=self.length_max_var).grid(row=row, column=2, sticky="w")
        row += 1
        tk.Label(frame, text="Warmup (s)").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, width=10, textvariable=self.warmup_seconds_var).grid(row=row, column=1, sticky="w")
        row += 1
        tk.Label(frame, text="MPRR avg window N").grid(row=row, column=0, sticky="w")
        self.mprr_window_entry = tk.Entry(frame, width=10, textvariable=self.mprr_window_var)
        self.mprr_window_entry.grid(row=row, column=1, sticky="w")
        self.mprr_window_var.trace_add("write", self._on_mprr_window_change)
        row += 1
        tk.Label(frame, text="Soft-limit cycle cap").grid(row=row, column=0, sticky="w")
        self.soft_limit_cycle_cap_entry = tk.Entry(
            frame, width=10, textvariable=self.soft_limit_cycle_cap_var
        )
        self.soft_limit_cycle_cap_entry.grid(row=row, column=1, sticky="w")
        self.soft_limit_cycle_cap_var.trace_add("write", self._on_soft_limit_cycle_cap_change)

    def _get_manual_values(self):
        values = np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        for idx, entry_var in self.entry_vars:
            values[idx] = float(entry_var.get())
        return values

    def _read_generator_config(self):
        d1_low = float(self.bound_vars["d1_low"].get())
        d1_high = float(self.bound_vars["d1_high"].get())
        d2_low = float(self.bound_vars["d2_low"].get())
        d2_high = float(self.bound_vars["d2_high"].get())
        soi2_low = float(self.bound_vars["soi2_low"].get())
        soi2_high = float(self.bound_vars["soi2_high"].get())
        if not (d1_high > d1_low and d2_high > d2_low and soi2_high > soi2_low):
            raise ValueError("Each high bound must be greater than its low bound.")

        mode_probs = {
            "smooth_ramp": float(self.mode_prob_vars["smooth_ramp"].get()),
            "step_hold": float(self.mode_prob_vars["step_hold"].get()),
            "boundary_probe": 0.0,
            "no_injection": float(self.mode_prob_vars["no_injection"].get()),
        }
        if any(v < 0.0 for v in mode_probs.values()):
            raise ValueError("Mode probabilities must be >= 0.")
        if sum(mode_probs.values()) <= 0.0:
            raise ValueError("At least one mode probability must be > 0.")

        min_len = int(self.length_min_var.get())
        max_len = int(self.length_max_var.get())
        if min_len <= 0 or max_len < min_len:
            raise ValueError("Length bounds must satisfy: min > 0 and max >= min.")

        warmup = float(self.warmup_seconds_var.get())
        if warmup < 0.0:
            raise ValueError("Warmup must be >= 0.")
        return (d1_low, d1_high, d2_low, d2_high, soi2_low, soi2_high, mode_probs, min_len, max_len, warmup)

    def _on_mprr_window_change(self, *_):
        text = self.mprr_window_var.get().strip()
        if not text:
            return
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            return
        with self.stats_lock:
            self.mprr_window_n = value
        self.avg_mprr_var.set(f"Avg MPRR ({value}-cycle): --")

    def _on_soft_limit_cycle_cap_change(self, *_):
        text = self.soft_limit_cycle_cap_var.get().strip()
        if not text:
            return
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            return
        with self.mode_lock:
            self.soft_limit_cycle_cap = value

    def apply_values(self):
        with self.mode_lock:
            if self.output_mode == "TRAJECTORY":
                self.status_var.set("Cannot apply manual values while trajectory is running.")
                return
        try:
            values = self._get_manual_values()
        except ValueError:
            messagebox.showerror("Invalid input", "Manual fields must be numeric.")
            return
        with self.values_lock:
            self.latest_values = values
        self.status_var.set("Manual values applied.")

    def zero_and_apply_values(self):
        with self.mode_lock:
            if self.output_mode == "TRAJECTORY":
                self.status_var.set("Cannot zero values while trajectory is running.")
                return
        self.latest_values = np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        with self.values_lock:
            self.latest_values = np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        with self.mode_lock:
            self.output_mode = "MANUAL"
        self.mode_var.set("Mode: MANUAL")
        self.exec_mode_var.set("Trajectory mode: manual")
        self.status_var.set("Manual values zeroed.")

    def open_target_popup(self):
        with self.mode_lock:
            if self.output_mode == "TRAJECTORY":
                self.status_var.set("Trajectory is running; wait until it returns to manual.")
                return
        self.status_var.set("Generating random anchor target...")
        try:
            config = self._read_generator_config()
            latest_values = self._get_manual_values()
            with self.values_lock:
                self.latest_values = latest_values.copy()
            with self.sequence_lock:
                self.sequence_generator.configure_bounds(*config[:6])
                self.sequence_generator.set_default_payload(latest_values)
                anchor_target = self.sequence_generator.sample_random_target()
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return

        if self.popup is None or not self.popup.winfo_exists():
            self.popup = tk.Toplevel(self.root)
            self.popup.title("Trajectory Workflow")
            self.popup.geometry("900x620")
            self.popup.protocol("WM_DELETE_WINDOW", self._close_popup)
            self.popup_content_frame = tk.Frame(self.popup, padx=12, pady=12)
            self.popup_content_frame.pack(fill="both", expand=True)
        else:
            self.popup.deiconify()
            self.popup.lift()
            self.popup.focus_force()

        self.popup_vars = {
            "anchor_target": anchor_target,
            "anchor": np.zeros(3, dtype=np.float32),
            "target_end_action": np.zeros(3, dtype=np.float32),
            "step_delta": np.zeros(3, dtype=np.float32),
            "length": 0,
            "plan_events": {
                "hold": {"enabled": False, "start_idx": -1, "cycles": 0},
                "zero_drop": {"enabled": False, "start_idx": -1, "cycles": 0},
            },
            "warmup_seconds": config[9],
            "cycle_idx": 0,
        }
        self.popup_stage = "TARGET_STAGE"
        self._render_popup_stage()
        self.status_var.set("Anchor target generated. Match manual values, then click Confirm Anchor.")

    def _close_popup(self):
        with self.mode_lock:
            if self.output_mode == "TRAJECTORY":
                self.status_var.set("Trajectory running. Wait until completion.")
                return
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None
        self.root.focus_force()

    def _force_close_popup(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None
        self.root.focus_force()

    def _clear_popup(self):
        for child in self.popup_content_frame.winfo_children():
            child.destroy()

    def _render_popup_stage(self):
        if self.popup is None or not self.popup.winfo_exists():
            return
        self._clear_popup()
        tk.Button(
            self.popup_content_frame,
            text="E-STOP",
            command=self._e_stop,
            fg="red",
            activeforeground="red",
        ).pack(anchor="w", pady=(0, 8))
        if self.popup_stage == "TARGET_STAGE":
            self._render_target_stage()
        elif self.popup_stage == "PLAN_REVIEW_STAGE":
            self._render_plan_stage()
        elif self.popup_stage == "READY_STAGE":
            self._render_ready_stage()
        elif self.popup_stage == "SUMMARY_STAGE":
            self._render_summary_stage()

    def _render_target_stage(self):
        anchor_target = self.popup_vars["anchor_target"]
        tk.Label(self.popup_content_frame, text="Stage 1: Random Anchor Target", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Anchor target: D1={anchor_target[0]:.3f}, D2={anchor_target[1]:.3f}, SOI2={anchor_target[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text="Manually adjust values to match this target as closely as possible, then confirm anchor.",
        ).pack(anchor="w", pady=(0, 10))
        tk.Button(self.popup_content_frame, text="Confirm Anchor", command=self._confirm_anchor).pack(anchor="w")

    def _confirm_anchor(self):
        self.status_var.set("Anchor confirmed. Building endpoint-driven trajectory plan...")
        try:
            config = self._read_generator_config()
            latest_values = self._get_manual_values()
            with self.values_lock:
                self.latest_values = latest_values.copy()
            with self.sequence_lock:
                self.sequence_generator.configure_bounds(*config[:6])
                self.sequence_generator.set_default_payload(latest_values)
                anchor = self.sequence_generator.set_anchor_from_payload(latest_values)
            self.popup_vars["anchor"] = anchor.copy()
            self._generate_new_plan(config)
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            self.status_var.set("Failed to confirm anchor. Check numeric fields and bounds.")
            return
        self.status_var.set("Anchor confirmed. New trajectory plan sampled.")

    def _generate_new_plan(self, config=None):
        if config is None:
            config = self._read_generator_config()
        anchor = self.popup_vars["anchor"]
        sampled_end_action = self._sample_random_endpoint(anchor)
        with self.sequence_lock:
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            self.sequence_generator.set_anchor_action(anchor)
            self.sequence_generator.sample_trajectory_plan_to_endpoint(
                sampled_end_action, config[6], config[7], config[8]
            )
            details = self.sequence_generator.get_plan_details()

        self.popup_vars["target_end_action"] = details["target_end_action"]
        self.popup_vars["step_delta"] = details["step_delta"]
        self.popup_vars["length"] = details["length"]
        self.popup_vars["plan_events"] = details["plan_events"]
        self.popup_vars["warmup_seconds"] = config[9]
        self.popup_vars["cycle_idx"] = 0
        self.popup_stage = "PLAN_REVIEW_STAGE"
        self._render_popup_stage()

    def _sample_random_endpoint(self, anchor_action):
        with self.sequence_lock:
            for _ in range(60):
                endpoint = self.sequence_generator.sample_random_target()
                if float(np.linalg.norm(endpoint - anchor_action)) > 1e-6:
                    return endpoint
        raise ValueError("Failed to sample endpoint distinct from anchor. Check action bounds.")

    def _format_event_text(self, event_cfg, label):
        if not event_cfg["enabled"]:
            return f"{label}: no"
        start_cycle = event_cfg["start_idx"] + 1
        end_cycle = event_cfg["start_idx"] + event_cfg["cycles"]
        return f"{label}: yes | {event_cfg['cycles']} cycles (cycles {start_cycle}-{end_cycle})"

    def _render_plan_stage(self):
        anchor = self.popup_vars["anchor"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        hold_text = self._format_event_text(self.popup_vars["plan_events"]["hold"], "Flat hold")
        zero_drop_text = self._format_event_text(
            self.popup_vars["plan_events"]["zero_drop"], "Fueling drop to 0 (D1/D2)"
        )
        tk.Label(self.popup_content_frame, text="Stage 2: Trajectory Plan Review", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Start (Anchor): D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"End target: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=f"Length: {self.popup_vars['length']} cycles").pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Per-cycle delta: dD1={delta[0]:.5f}, dD2={delta[1]:.5f}, dSOI2={delta[2]:.5f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=hold_text).pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=zero_drop_text).pack(anchor="w", pady=(0, 12))
        tk.Button(self.popup_content_frame, text="Confirm Trajectory Plan", command=self._confirm_plan).pack(anchor="w", pady=(0, 12))
        tk.Button(self.popup_content_frame, text="Generate New Plan", command=self._regenerate_plan).pack(anchor="w")

    def _confirm_plan(self):
        try:
            with self.sequence_lock:
                self.sequence_generator.confirm_vector()
        except Exception as exc:
            messagebox.showerror("Plan confirmation failed", str(exc))
            self.status_var.set("Failed to confirm trajectory plan.")
            return
        self.popup_stage = "READY_STAGE"
        self._render_popup_stage()
        self.status_var.set("Trajectory plan confirmed. Ready to start.")

    def _regenerate_plan(self):
        self.status_var.set("Resampling trajectory plan from confirmed anchor...")
        try:
            config = self._read_generator_config()
            self._generate_new_plan(config)
        except Exception as exc:
            messagebox.showerror("Plan resampling failed", str(exc))
            self.status_var.set("Failed to generate a new plan.")
            return
        self.status_var.set("New trajectory plan sampled. Review and confirm.")

    def _render_ready_stage(self):
        self.ready_info_var = tk.StringVar(
            value=f"Length={self.popup_vars['length']} cycles | Warmup={self.popup_vars['warmup_seconds']:.1f}s"
        )
        self.ready_progress_var = tk.StringVar(value="Status: waiting to start")
        tk.Label(self.popup_content_frame, text="Stage 3: Ready", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, textvariable=self.ready_info_var).pack(anchor="w", pady=(8, 4))
        tk.Label(self.popup_content_frame, textvariable=self.ready_progress_var).pack(anchor="w", pady=(0, 12))
        tk.Button(self.popup_content_frame, text="Start Trajectory", command=self._start_trajectory).pack(anchor="w")

    def _start_trajectory(self):
        with self.mode_lock:
            if self.output_mode == "TRAJECTORY":
                return
            self.output_mode = "TRAJECTORY"
            self.execution_mode_text = "warmup"
            self.soft_limit_cycles_over = 0
            self.derate_active = False
            self.active_anchor_action = self.popup_vars["anchor"].copy()
            self.last_trajectory_action = self.active_anchor_action.copy()
            self.boundary_control_active = False
            self.boundary_explore_cycles = 0
            self.boundary_action = None
            self.abort_latch_cycles = 0
            self.trajectory_start_time_s = time.monotonic()
            self.trajectory_warmup_seconds = float(self.popup_vars.get("warmup_seconds", 0.0))
        self.mode_var.set("Mode: TRAJECTORY")
        self.exec_mode_var.set("Trajectory mode: warmup")
        self.generate_button.config(state=tk.DISABLED)
        self.status_var.set("Trajectory started. Waiting for trigger cycles...")
        self.popup_vars["cycle_idx"] = 0
        self.popup_stage = "SUMMARY_STAGE"
        self._render_popup_stage()

    def _classify_execution_mode(self, cycle_zero_idx, action):
        events = self.popup_vars.get("plan_events", {})
        zero_drop = events.get("zero_drop", {"enabled": False, "start_idx": -1, "cycles": 0})
        hold = events.get("hold", {"enabled": False, "start_idx": -1, "cycles": 0})
        if zero_drop["enabled"]:
            start = zero_drop["start_idx"]
            end = start + zero_drop["cycles"]
            if start <= cycle_zero_idx < end:
                return "drop_to_0"
        if hold["enabled"]:
            start = hold["start_idx"]
            end = start + hold["cycles"]
            if start <= cycle_zero_idx < end:
                return "random_hold_constant"
        with self.sequence_lock:
            low = self.sequence_generator.action_low.copy()
            high = self.sequence_generator.action_high.copy()
        if (
            abs(float(action[0] - low[0])) < 1e-4
            or abs(float(action[0] - high[0])) < 1e-4
            or abs(float(action[1] - low[1])) < 1e-4
            or abs(float(action[1] - high[1])) < 1e-4
        ):
            return "boundary_hold"
        return "explore"

    def _map_transport_mode(self, mode, exec_mode, cycle_zero_idx):
        if mode == "TRAJECTORY":
            if exec_mode == "drop_to_0":
                return "NO_INJECTION"
            if exec_mode == "random_hold_constant":
                return "HOLD"
            if exec_mode in ("exploration_derate", "boundary_hold"):
                return "DERATE"
            if exec_mode == "boundary_backoff":
                return "RECOVER"
            return "EXPLORE"
        if self.abort_latch_cycles > 0:
            return "ABORT"
        if self.recover_cycles_remaining > 0:
            return "RECOVER"
        return "WARMUP"

    def _compose_tcp_payload(self, injection_values, mode_name):
        values = np.asarray(injection_values, dtype=np.float32)
        if values.shape[0] != INJECTION_PAYLOAD_FLOAT_COUNT:
            raise ValueError(
                f"Injection payload must contain {INJECTION_PAYLOAD_FLOAT_COUNT} floats."
            )
        if mode_name not in MODE_CODE_MAP:
            raise ValueError(f"Unknown mode for TCP payload: {mode_name}")
        payload = np.zeros(TCP_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        payload[:INJECTION_PAYLOAD_FLOAT_COUNT] = values
        payload[INJECTION_PAYLOAD_FLOAT_COUNT] = np.float32(MODE_CODE_MAP[mode_name])
        return payload

    def _compute_derate_factor(self, avg_mprr):
        with self.mode_lock:
            derate_active = self.derate_active

        if not derate_active:
            return 1.0, False

        span = max(1e-6, SOFT_LIMIT_DERATE_ENTRY_MPRR - SOFT_LIMIT_AVG_MPRR)
        frac = min(1.0, max(0.0, (avg_mprr - SOFT_LIMIT_AVG_MPRR) / span))
        # Mild derate near 9.0 and near-zero movement near 10.5.
        factor = 0.85 - (0.80 * frac)
        return max(0.03, min(1.0, factor)), True

    def _update_soft_limit_state(self, avg_mprr):
        with self.mode_lock:
            inside_soft_band = (
                SOFT_LIMIT_AVG_MPRR <= avg_mprr <= SOFT_LIMIT_DERATE_ENTRY_MPRR
            )
            if inside_soft_band:
                self.soft_limit_cycles_over += 1
            else:
                self.soft_limit_cycles_over = 0

            self.derate_active = avg_mprr >= SOFT_LIMIT_AVG_MPRR
            self.boundary_control_active = avg_mprr > SOFT_LIMIT_DERATE_ENTRY_MPRR
            cap = max(1, int(self.soft_limit_cycle_cap))
            soft_limit_timeout = inside_soft_band and self.soft_limit_cycles_over > cap

            return self.derate_active, self.boundary_control_active, soft_limit_timeout

    def _next_boundary_action(self, avg_mprr):
        step_delta = np.asarray(
            self.popup_vars.get("step_delta", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        )
        with self.mode_lock:
            if self.boundary_action is None:
                self.boundary_action = self.last_trajectory_action.copy()
            action = self.boundary_action.copy()

        if avg_mprr > SOFT_LIMIT_DERATE_ENTRY_MPRR:
            action = action - step_delta
            control_mode = "boundary_backoff"
        elif avg_mprr >= SOFT_LIMIT_AVG_MPRR:
            control_mode = "boundary_hold"
        else:
            action = action + (0.5 * step_delta)
            control_mode = "boundary_reapproach"

        with self.sequence_lock:
            action = np.clip(action, self.sequence_generator.action_low, self.sequence_generator.action_high)
            payload = self.sequence_generator.build_payload_from_action(action)

        with self.mode_lock:
            self.boundary_action = action.astype(np.float32)
            self.last_trajectory_action = self.boundary_action.copy()
            self.boundary_explore_cycles += 1
            max_cycles_hit = self.boundary_explore_cycles >= BOUNDARY_MAX_CYCLES

        return payload.astype(np.float32), action.astype(np.float32), control_mode, max_cycles_hit

    def _build_anchor_return_payload(self):
        with self.sequence_lock:
            anchor = self.active_anchor_action.copy()
            payload = self.sequence_generator.build_payload_from_action(anchor)
        return payload.astype(np.float32), anchor

    def _start_worker(self):
        self.worker_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        self.worker_thread.start()

    def _tcp_loop(self):
        rt_ok, rt_msg = _set_thread_realtime_priority(RT_PRIORITY)
        self.status_queue.put(("status", rt_msg))

        expected_bytes = INCOMING_FLOAT_COUNT * 4
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

                latest_mprr = float(incoming_values[1]) if incoming_values.size > 1 else 0.0
                with self.stats_lock:
                    self.mprr_history.append(latest_mprr)
                    max_keep = max(3000, self.mprr_window_n * 3)
                    if len(self.mprr_history) > max_keep:
                        self.mprr_history = self.mprr_history[-max_keep:]
                    window = min(self.mprr_window_n, len(self.mprr_history))
                    avg = float(np.mean(self.mprr_history[-window:])) if window > 0 else 0.0
                    n_value = self.mprr_window_n
                self.status_queue.put(("mprr_update", (latest_mprr, avg, n_value)))

                with self.mode_lock:
                    mode = self.output_mode

                exec_mode = "manual"
                cycle_zero_idx = -1
                transport_mode = "WARMUP"
                if mode == "TRAJECTORY":
                    with self.mode_lock:
                        start_time = self.trajectory_start_time_s
                        warmup_seconds = self.trajectory_warmup_seconds
                    warmup_active = False
                    if start_time is not None:
                        warmup_active = (time.monotonic() - start_time) < warmup_seconds

                    if warmup_active:
                        response_values, action = self._build_anchor_return_payload()
                        exec_mode = "warmup"
                        transport_mode = "WARMUP"
                        self.status_queue.put(
                            (
                                "trajectory_progress",
                                (
                                    0,
                                    self.popup_vars.get("length", 0),
                                    action.tolist(),
                                    exec_mode,
                                ),
                            )
                        )
                    else:
                        with self.mode_lock:
                            self.trajectory_start_time_s = None
                        hard_limit_hit = avg > HARD_LIMIT_AVG_MPRR
                        single_cycle_hit = latest_mprr > SINGLE_CYCLE_MPRR_LIMIT
                        if hard_limit_hit or single_cycle_hit:
                            response_values, anchor = self._build_anchor_return_payload()
                            transport_mode = "ABORT"
                            exec_mode = "abort_return_anchor"
                            with self.values_lock:
                                self.latest_values[1] = float(anchor[0])
                                self.latest_values[3] = float(anchor[1])
                                self.latest_values[2] = float(anchor[2])
                            with self.mode_lock:
                                self.output_mode = "MANUAL"
                                self.execution_mode_text = "manual"
                                self.recover_cycles_remaining = 0
                                self.derate_active = False
                                self.soft_limit_cycles_over = 0
                                self.boundary_control_active = False
                                self.boundary_explore_cycles = 0
                                self.boundary_action = None
                                self.trajectory_start_time_s = None
                                self.trajectory_warmup_seconds = 0.0
                            reason = (
                                f"Abort: avg MPRR={avg:.3f} > {HARD_LIMIT_AVG_MPRR:.1f}"
                                if hard_limit_hit
                                else f"Abort: single-cycle MPRR={latest_mprr:.3f} > {SINGLE_CYCLE_MPRR_LIMIT:.1f}"
                            )
                            self.status_queue.put(("trajectory_abort", (anchor.copy(), reason)))
                        else:
                            derate_active, boundary_active, soft_limit_timeout = self._update_soft_limit_state(avg)
                            if soft_limit_timeout:
                                response_values, anchor = self._build_anchor_return_payload()
                                transport_mode = "ABORT"
                                exec_mode = "abort_return_anchor"
                                with self.values_lock:
                                    self.latest_values[1] = float(anchor[0])
                                    self.latest_values[3] = float(anchor[1])
                                    self.latest_values[2] = float(anchor[2])
                                with self.mode_lock:
                                    self.output_mode = "MANUAL"
                                    self.execution_mode_text = "manual"
                                    self.recover_cycles_remaining = 3
                                    self.derate_active = False
                                    self.soft_limit_cycles_over = 0
                                    self.boundary_control_active = False
                                    self.boundary_explore_cycles = 0
                                    self.boundary_action = None
                                    self.trajectory_start_time_s = None
                                    self.trajectory_warmup_seconds = 0.0
                                self.status_queue.put(
                                    (
                                        "trajectory_abort",
                                        (
                                            anchor.copy(),
                                            (
                                                "Abort: avg MPRR remained within "
                                                f"{SOFT_LIMIT_AVG_MPRR:.1f}-{SOFT_LIMIT_DERATE_ENTRY_MPRR:.1f} "
                                                f"for more than {self.soft_limit_cycle_cap} cycles"
                                            ),
                                        ),
                                    )
                                )
                            elif boundary_active:
                                response_values, action, boundary_mode, max_cycles_hit = self._next_boundary_action(avg)
                                exec_mode = boundary_mode
                                transport_mode = self._map_transport_mode(mode, exec_mode, cycle_zero_idx)
                                with self.mode_lock:
                                    cycle_idx = self.boundary_explore_cycles
                                self.status_queue.put(
                                    (
                                        "trajectory_progress",
                                        (
                                            cycle_idx,
                                            BOUNDARY_MAX_CYCLES,
                                            action.tolist(),
                                            exec_mode,
                                        ),
                                    )
                                )
                                if max_cycles_hit:
                                    response_values, anchor = self._build_anchor_return_payload()
                                    with self.values_lock:
                                        self.latest_values[1] = float(anchor[0])
                                        self.latest_values[3] = float(anchor[1])
                                        self.latest_values[2] = float(anchor[2])
                                    with self.mode_lock:
                                        self.output_mode = "MANUAL"
                                        self.execution_mode_text = "manual"
                                        self.recover_cycles_remaining = 3
                                        self.derate_active = False
                                        self.soft_limit_cycles_over = 0
                                        self.boundary_control_active = False
                                        self.boundary_explore_cycles = 0
                                        self.boundary_action = None
                                        self.trajectory_start_time_s = None
                                        self.trajectory_warmup_seconds = 0.0
                                    self.status_queue.put(
                                        (
                                            "trajectory_abort",
                                            (
                                                anchor.copy(),
                                                f"Boundary exploration capped at {BOUNDARY_MAX_CYCLES} cycles",
                                            ),
                                        )
                                    )
                            else:
                                derate_factor, _ = self._compute_derate_factor(avg)
                                with self.sequence_lock:
                                    payload, action, _, done = self.sequence_generator.next_trajectory_values(
                                        derate_factor=derate_factor
                                    )
                                    cycle_zero_idx = self.sequence_generator.current_cycle - 1
                                response_values = payload.astype(np.float32)
                                with self.mode_lock:
                                    self.last_trajectory_action = action.astype(np.float32)
                                exec_mode = self._classify_execution_mode(cycle_zero_idx, action)
                                if derate_active:
                                    exec_mode = "exploration_derate"
                                self.status_queue.put(
                                    (
                                        "trajectory_progress",
                                        (
                                            cycle_zero_idx + 1,
                                            self.popup_vars.get("length", 0),
                                            action.tolist(),
                                            exec_mode,
                                        ),
                                    )
                                )
                                transport_mode = self._map_transport_mode(mode, exec_mode, cycle_zero_idx)
                                if derate_active and transport_mode == "EXPLORE":
                                    transport_mode = "DERATE"
                                if done:
                                    anchor = self.popup_vars.get("anchor", np.zeros(3, dtype=np.float32))
                                    with self.values_lock:
                                        self.latest_values[1] = float(anchor[0])
                                        self.latest_values[3] = float(anchor[1])
                                        self.latest_values[2] = float(anchor[2])
                                    with self.mode_lock:
                                        self.output_mode = "MANUAL"
                                        self.execution_mode_text = "manual"
                                        self.recover_cycles_remaining = 3
                                        self.derate_active = False
                                        self.soft_limit_cycles_over = 0
                                        self.boundary_control_active = False
                                        self.boundary_explore_cycles = 0
                                        self.boundary_action = None
                                        self.trajectory_start_time_s = None
                                        self.trajectory_warmup_seconds = 0.0
                                    self.status_queue.put(("trajectory_complete", anchor.copy()))
                else:
                    with self.values_lock:
                        response_values = self.latest_values.copy()
                    with self.mode_lock:
                        if self.abort_latch_cycles > 0:
                            self.abort_latch_cycles -= 1
                            transport_mode = "ABORT"
                        elif self.recover_cycles_remaining > 0:
                            self.recover_cycles_remaining -= 1
                            transport_mode = "RECOVER"
                        else:
                            transport_mode = "WARMUP"
                tcp_payload_values = self._compose_tcp_payload(response_values, transport_mode)
                self.status_queue.put(("tcp_mode", transport_mode))

                payload_bytes = tcp_payload_values.astype(f"{BYTE_ORDER}f4").tobytes(order="C")
                sock.sendall(payload_bytes)
                self.status_queue.put(("sent", tcp_payload_values.tolist()))
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
                elif event_type == "mprr_update":
                    latest, avg, n_value = payload
                    self.latest_mprr_var.set(f"Latest MPRR: {latest:.4f}")
                    self.avg_mprr_var.set(f"Avg MPRR ({n_value}-cycle): {avg:.4f}")
                elif event_type == "trajectory_progress":
                    cycle_idx, length, action, exec_mode = payload
                    self.exec_mode_var.set(f"Trajectory mode: {exec_mode}")
                    if hasattr(self, "ready_progress_var"):
                        self.ready_progress_var.set(
                            f"Cycle {cycle_idx}/{length} | D1={action[0]:.3f}, D2={action[1]:.3f}, SOI2={action[2]:.3f}"
                        )
                    if hasattr(self, "derate_indicator_var"):
                        self.derate_indicator_var.set(
                            "DERATE" if exec_mode == "exploration_derate" else ""
                        )
                    if hasattr(self, "boundary_indicator_var"):
                        self.boundary_indicator_var.set(
                            "BOUNDARY CONTROL"
                            if exec_mode in ("boundary_backoff", "boundary_hold", "boundary_reapproach")
                            else ""
                        )
                elif event_type == "trajectory_abort":
                    anchor, reason = payload
                    if anchor is not None:
                        for idx, entry_var in self.entry_vars:
                            if idx == 1:
                                entry_var.set(f"{float(anchor[0]):.6g}")
                            elif idx == 3:
                                entry_var.set(f"{float(anchor[1]):.6g}")
                            elif idx == 2:
                                entry_var.set(f"{float(anchor[2]):.6g}")
                    self.mode_var.set("Mode: MANUAL")
                    self.exec_mode_var.set("Trajectory mode: ABORT")
                    self.generate_button.config(state=tk.NORMAL)
                    self._force_close_popup()
                    self.status_var.set(f"{reason}. Returned to initial anchor in MANUAL mode.")
                elif event_type == "tcp_mode":
                    self.status_var.set(
                        f"TCP mode={payload} (code={MODE_CODE_MAP[payload]:.0f})"
                    )
                elif event_type == "trajectory_complete":
                    if payload is not None:
                        for idx, entry_var in self.entry_vars:
                            if idx == 1:
                                entry_var.set(f"{float(payload[0]):.6g}")
                            elif idx == 3:
                                entry_var.set(f"{float(payload[1]):.6g}")
                            elif idx == 2:
                                entry_var.set(f"{float(payload[2]):.6g}")
                    self.mode_var.set("Mode: MANUAL")
                    self.exec_mode_var.set("Trajectory mode: manual")
                    self.generate_button.config(state=tk.NORMAL)
                    self.status_var.set("Trajectory complete. Returned to MANUAL mode.")
                    if hasattr(self, "derate_indicator_var"):
                        self.derate_indicator_var.set("")
                    if hasattr(self, "boundary_indicator_var"):
                        self.boundary_indicator_var.set("")
                else:
                    self.status_var.set(payload)
        except Empty:
            pass
        self.root.after(100, self._poll_status_queue)

    def _render_summary_stage(self):
        anchor = self.popup_vars.get("anchor", np.zeros(3, dtype=np.float32))
        end_action = self.popup_vars.get("target_end_action", np.zeros(3, dtype=np.float32))
        delta = self.popup_vars.get("step_delta", np.zeros(3, dtype=np.float32))
        hold_text = self._format_event_text(self.popup_vars["plan_events"]["hold"], "Flat hold")
        zero_drop_text = self._format_event_text(
            self.popup_vars["plan_events"]["zero_drop"], "Fueling drop to 0 (D1/D2)"
        )
        tk.Label(self.popup_content_frame, text="Stage 4: Trajectory Running/Complete", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Start (Anchor): D1={anchor[0]:.4f}, D2={anchor[1]:.4f}, SOI2={anchor[2]:.4f}").pack(anchor="w", pady=(8, 0))
        tk.Label(self.popup_content_frame, text=f"End target: D1={end_action[0]:.4f}, D2={end_action[1]:.4f}, SOI2={end_action[2]:.4f}").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Length: {self.popup_vars.get('length', 0)} cycles").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Per-cycle delta: dD1={delta[0]:.6f}, dD2={delta[1]:.6f}, dSOI2={delta[2]:.6f}").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=hold_text).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=zero_drop_text).pack(anchor="w", pady=(0, 8))
        self.ready_progress_var = tk.StringVar(value="Status: waiting for trajectory cycles...")
        tk.Label(self.popup_content_frame, textvariable=self.ready_progress_var).pack(anchor="w", pady=(0, 10))
        self.derate_indicator_var = tk.StringVar(value="")
        self.boundary_indicator_var = tk.StringVar(value="")
        tk.Label(
            self.popup_content_frame,
            textvariable=self.derate_indicator_var,
            fg="red",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            textvariable=self.boundary_indicator_var,
            fg="red",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        tk.Button(self.popup_content_frame, text="Close Popup", command=self._close_popup).pack(anchor="w")

    def _e_stop(self):
        with self.mode_lock:
            self.output_mode = "MANUAL"
            self.execution_mode_text = "manual"
            self.recover_cycles_remaining = 0
            self.abort_latch_cycles = 2
            self.derate_active = False
            self.soft_limit_cycles_over = 0
            self.boundary_control_active = False
            self.boundary_explore_cycles = 0
            self.boundary_action = None
            self.trajectory_start_time_s = None
            self.trajectory_warmup_seconds = 0.0
        zeros = np.zeros(INJECTION_PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        with self.values_lock:
            self.latest_values = zeros.copy()
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.mode_var.set("Mode: MANUAL")
        self.exec_mode_var.set("Trajectory mode: ABORT")
        self.generate_button.config(state=tk.NORMAL)
        self._force_close_popup()
        self.status_var.set("E-STOP triggered: trajectory stopped, outputs zeroed, MANUAL mode restored.")

    def stop_communication(self):
        self.stop_event.set()
        self.apply_button.config(state=tk.DISABLED)
        self.zero_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.estop_button.config(state=tk.DISABLED)
        self.mprr_window_entry.config(state=tk.DISABLED)
        self.soft_limit_cycle_cap_entry.config(state=tk.DISABLED)
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
