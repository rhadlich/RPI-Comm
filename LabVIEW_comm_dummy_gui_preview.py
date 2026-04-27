import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from injection_sequence_generator import InjectionSequenceGenerator

INJECTION_EVENT_COUNT = 6


class DummyTrajectoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trajectory GUI Preview (Dummy)")
        self._configure_scaling()

        self.sequence_generator = InjectionSequenceGenerator()
        self.latest_values = np.zeros(12, dtype=np.float32)
        self.entry_vars = []
        self.popup = None
        self.popup_content_frame = None
        self.popup_stage = ""
        self.popup_vars = {}
        self.current_mode = "MANUAL"
        self.trajectory_running = False
        self.preview_job_id = None
        self.preview_cycle_idx = 0

        self._build_ui()
        self.root.resizable(False, False)

    def _configure_scaling(self):
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=13)
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

        tk.Button(manual_frame, text="Apply", command=self.apply_values).grid(
            row=8, column=0, sticky="w", pady=(12, 0)
        )
        tk.Button(manual_frame, text="ZERO", command=self.zero_and_apply_values).grid(
            row=8, column=1, sticky="w", pady=(12, 0), padx=(8, 0)
        )
        self.generate_button = tk.Button(
            frame, text="Generate Random Target", command=self.open_target_popup
        )
        self.generate_button.grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        self.estop_button = tk.Button(
            frame,
            text="E-STOP",
            command=self._e_stop,
            fg="red",
            activeforeground="red",
        )
        self.estop_button.grid(row=1, column=1, sticky="w", pady=(12, 0), padx=(8, 0))

        self.mode_var = tk.StringVar(value="Mode: MANUAL")
        tk.Label(frame, textvariable=self.mode_var, anchor="w").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )
        self.status_var = tk.StringVar(value="Dummy mode: no data is sent.")
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self._build_settings_ui(settings_frame)

    def _build_settings_ui(self, frame):
        self.bound_vars = {
            "d1_low": tk.StringVar(value="0.0"),
            "d1_high": tk.StringVar(value="4.0"),
            "d2_low": tk.StringVar(value="0.0"),
            "d2_high": tk.StringVar(value="3.0"),
            "soi2_low": tk.StringVar(value="-330.0"),
            "soi2_high": tk.StringVar(value="-220.0"),
        }
        self.mode_prob_vars = {
            "smooth_ramp": tk.StringVar(value="0.45"),
            "step_hold": tk.StringVar(value="0.20"),
            "boundary_probe": tk.StringVar(value="0.15"),
            "no_injection": tk.StringVar(value="0.20"),
        }
        self.length_min_var = tk.StringVar(value="40")
        self.length_max_var = tk.StringVar(value="120")
        self.warmup_seconds_var = tk.StringVar(value="3.0")

        row = 0
        tk.Label(frame, text="Global action bounds").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        for key, label in [
            ("d1_low", "D1 low"), ("d1_high", "D1 high"),
            ("d2_low", "D2 low"), ("d2_high", "D2 high"),
            ("soi2_low", "SOI2 low"), ("soi2_high", "SOI2 high"),
        ]:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, width=10, textvariable=self.bound_vars[key]).grid(row=row, column=1, sticky="w", pady=1)
            row += 1
        tk.Label(frame, text="Sampling balance").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        for key, label in [
            ("smooth_ramp", "P smooth"), ("step_hold", "P step"),
            ("boundary_probe", "P boundary"), ("no_injection", "P no-inj"),
        ]:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, width=10, textvariable=self.mode_prob_vars[key]).grid(row=row, column=1, sticky="w", pady=1)
            row += 1
        tk.Label(frame, text="Len min/max").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, width=10, textvariable=self.length_min_var).grid(row=row, column=1, sticky="w")
        tk.Entry(frame, width=10, textvariable=self.length_max_var).grid(row=row, column=2, sticky="w")
        row += 1
        tk.Label(frame, text="Warmup (s)").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, width=10, textvariable=self.warmup_seconds_var).grid(row=row, column=1, sticky="w")

    def _get_manual_values(self):
        values = np.zeros(12, dtype=np.float32)
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
            "boundary_probe": float(self.mode_prob_vars["boundary_probe"].get()),
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

        return (
            d1_low,
            d1_high,
            d2_low,
            d2_high,
            soi2_low,
            soi2_high,
            mode_probs,
            min_len,
            max_len,
            warmup,
        )

    def apply_values(self):
        if self.trajectory_running:
            self.status_var.set("Cannot apply manual values while trajectory is running.")
            return
        try:
            self.latest_values = self._get_manual_values()
        except ValueError:
            messagebox.showerror("Invalid input", "Manual fields must be numeric.")
            return
        self.status_var.set("Manual values applied.")

    def zero_and_apply_values(self):
        if self.trajectory_running:
            self.status_var.set("Cannot zero values while trajectory is running.")
            return
        self.latest_values = np.zeros(12, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.status_var.set("Manual values zeroed.")

    def open_target_popup(self):
        if self.trajectory_running:
            self.status_var.set("Trajectory is running; wait until it returns to manual.")
            return
        self.status_var.set("Generating random anchor target...")
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            anchor_target = self.sequence_generator.sample_random_target()
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return
        if self.popup is None or not self.popup.winfo_exists():
            self.popup = tk.Toplevel(self.root)
            self.popup.title("Trajectory Workflow Preview")
            self.popup.geometry("820x560")
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
            "mode": "",
            "travel_distance": 0.0,
            "plan_events": {
                "hold": {"enabled": False, "start_idx": -1, "cycles": 0},
                "zero_drop": {"enabled": False, "start_idx": -1, "cycles": 0},
            },
            "warmup_seconds": config[9],
        }
        self.popup_stage = "TARGET_STAGE"
        self._render_popup_stage()
        self.root.update_idletasks()
        self.status_var.set("Anchor target generated. Match manual values, then click Confirm Anchor.")

    def _close_popup(self):
        if self.trajectory_running:
            self.status_var.set("Trajectory preview running. Wait until completion.")
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
        elif self.popup_stage == "VECTOR_REVIEW_STAGE":
            self._render_vector_stage()
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
            text="Now manually adjust the live values to match this anchor target as closely as possible.",
        ).pack(anchor="w", pady=(0, 10))
        tk.Button(self.popup_content_frame, text="Confirm Anchor", command=self._confirm_anchor).pack(anchor="w")

    def _confirm_anchor(self):
        self.status_var.set("Anchor confirmed. Building endpoint-driven trajectory plan...")
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            anchor = self.sequence_generator.set_anchor_from_payload(self.latest_values)
            self.popup_vars["anchor"] = anchor.copy()
            self._generate_new_plan(config)
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            self.status_var.set("Failed to confirm anchor. Check numeric fields and bounds.")
            return
        self.status_var.set("Anchor confirmed. New target endpoint sampled and trajectory plan computed.")

    def _generate_new_plan(self, config=None):
        if config is None:
            config = self._read_generator_config()
        anchor = self.popup_vars["anchor"]
        self.sequence_generator.configure_bounds(*config[:6])
        self.sequence_generator.set_default_payload(self.latest_values)
        self.sequence_generator.set_anchor_action(anchor)

        sampled_end_action = self._sample_random_endpoint(anchor)
        self.sequence_generator.sample_trajectory_plan_to_endpoint(
            sampled_end_action,
            config[6],
            config[7],
            config[8],
        )
        details = self.sequence_generator.get_plan_details()
        self.popup_vars["target_end_action"] = details["target_end_action"]
        self.popup_vars["step_delta"] = details["step_delta"]
        self.popup_vars["length"] = details["length"]
        self.popup_vars["mode"] = details["mode"]
        self.popup_vars["travel_distance"] = details["travel_distance"]
        self.popup_vars["plan_events"] = details["plan_events"]
        self.popup_stage = "VECTOR_REVIEW_STAGE"
        self._render_popup_stage()

    def _sample_random_endpoint(self, anchor_action):
        for _ in range(50):
            endpoint = self.sequence_generator.sample_random_target()
            if float(np.linalg.norm(endpoint - anchor_action)) > 1e-6:
                return endpoint
        raise ValueError("Failed to sample endpoint distinct from anchor. Check action bounds.")

    def _render_vector_stage(self):
        anchor = self.popup_vars["anchor"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        hold_text = self._format_event_text(self.popup_vars["plan_events"]["hold"], "Flat hold")
        zero_drop_text = self._format_event_text(
            self.popup_vars["plan_events"]["zero_drop"],
            "Fueling drop to 0 (D1/D2)",
        )
        tk.Label(self.popup_content_frame, text="Stage 2: Trajectory Plan Review", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Start (Anchor): D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}").pack(anchor="w", pady=(8, 4))
        tk.Label(self.popup_content_frame, text=f"End target: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}").pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=f"Length={self.popup_vars['length']} cycles").pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=f"Per-cycle delta: dD1={delta[0]:.5f}, dD2={delta[1]:.5f}, dSOI2={delta[2]:.5f}").pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=hold_text).pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=zero_drop_text).pack(anchor="w", pady=(0, 8))
        tk.Button(self.popup_content_frame, text="Confirm Trajectory Plan", command=self._confirm_vector).pack(anchor="w", pady=(0, 12))
        tk.Button(self.popup_content_frame, text="Generate New Plan", command=self._regenerate_plan).pack(anchor="w")

    def _format_event_text(self, event_cfg, label):
        if not event_cfg["enabled"]:
            return f"{label}: no"
        start_cycle = event_cfg["start_idx"] + 1
        end_cycle = event_cfg["start_idx"] + event_cfg["cycles"]
        return f"{label}: yes | {event_cfg['cycles']} cycles (cycles {start_cycle}-{end_cycle})"

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

    def _confirm_vector(self):
        self.status_var.set("Trajectory plan confirmed. Preview is ready.")
        try:
            self.sequence_generator.confirm_vector()
        except Exception as exc:
            messagebox.showerror("Plan confirmation failed", str(exc))
            self.status_var.set("Failed to confirm trajectory plan.")
            return
        self.popup_stage = "READY_STAGE"
        self._render_popup_stage()

    def _render_ready_stage(self):
        self.ready_info_var = tk.StringVar(
            value=f"Length={self.popup_vars['length']} cycles | Warmup={self.popup_vars['warmup_seconds']:.1f}s"
        )
        self.ready_progress_var = tk.StringVar(value="Status: waiting to start")
        tk.Label(self.popup_content_frame, text="Stage 3: Ready", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, textvariable=self.ready_info_var).pack(anchor="w", pady=(8, 4))
        tk.Label(self.popup_content_frame, textvariable=self.ready_progress_var).pack(anchor="w", pady=(0, 12))
        tk.Button(
            self.popup_content_frame,
            text="Start Trajectory (Preview)",
            command=self._start_trajectory_preview,
        ).pack(anchor="w")

    def _start_trajectory_preview(self):
        if self.trajectory_running:
            return
        self.current_mode = "TRAJECTORY"
        self.mode_var.set("Mode: TRAJECTORY")
        self.trajectory_running = True
        self.preview_cycle_idx = 0
        self.generate_button.config(state=tk.DISABLED)
        self.status_var.set("Trajectory preview started.")
        self.root.update_idletasks()
        self.ready_progress_var.set(
            f"Warmup running... {self.popup_vars['warmup_seconds']:.1f}s remaining"
        )
        warmup_ms = max(0, int(self.popup_vars["warmup_seconds"] * 1000))
        self.preview_job_id = self.root.after(warmup_ms, self._run_preview_cycle)

    def _classify_execution_mode(self, cycle_zero_idx, action):
        zero_drop = self.popup_vars["plan_events"]["zero_drop"]
        if zero_drop["enabled"]:
            start = zero_drop["start_idx"]
            end = start + zero_drop["cycles"]
            if start <= cycle_zero_idx < end:
                return "drop_to_0"

        hold = self.popup_vars["plan_events"]["hold"]
        if hold["enabled"]:
            start = hold["start_idx"]
            end = start + hold["cycles"]
            if start <= cycle_zero_idx < end:
                return "random_hold_constant"

        low = self.sequence_generator.action_low
        high = self.sequence_generator.action_high
        if (
            abs(float(action[0] - low[0])) < 1e-4
            or abs(float(action[0] - high[0])) < 1e-4
            or abs(float(action[1] - low[1])) < 1e-4
            or abs(float(action[1] - high[1])) < 1e-4
        ):
            return "boundary_hold"
        return "exploration_derate"

    def _run_preview_cycle(self):
        if not self.trajectory_running:
            return
        try:
            _, action, _, done = self.sequence_generator.next_trajectory_values()
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))
            self._finish_preview(return_to_anchor=False)
            return

        self.preview_cycle_idx += 1
        exec_mode = self._classify_execution_mode(self.preview_cycle_idx - 1, action)
        self.ready_progress_var.set(
            f"Trajectory cycle {self.preview_cycle_idx}/{self.popup_vars['length']} | "
            f"D1={action[0]:.3f}, D2={action[1]:.3f}, SOI2={action[2]:.3f} | "
            f"mode={exec_mode}"
        )
        if done:
            self._finish_preview(return_to_anchor=True)
            return
        self.preview_job_id = self.root.after(120, self._run_preview_cycle)

    def _e_stop(self):
        self.trajectory_running = False
        self.current_mode = "MANUAL"
        self.mode_var.set("Mode: MANUAL")
        self.generate_button.config(state=tk.NORMAL)
        if self.preview_job_id is not None:
            try:
                self.root.after_cancel(self.preview_job_id)
            except Exception:
                pass
            self.preview_job_id = None
        self.latest_values = np.zeros(12, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.sequence_generator.reset()
        self._force_close_popup()
        self.status_var.set("E-STOP triggered: trajectory stopped, outputs zeroed, MANUAL mode restored.")

    def _finish_preview(self, return_to_anchor):
        self.trajectory_running = False
        self.current_mode = "MANUAL"
        self.mode_var.set("Mode: MANUAL")
        self.generate_button.config(state=tk.NORMAL)
        if self.preview_job_id is not None:
            try:
                self.root.after_cancel(self.preview_job_id)
            except Exception:
                pass
            self.preview_job_id = None

        if return_to_anchor and "anchor" in self.popup_vars:
            anchor = self.popup_vars["anchor"]
            self.latest_values[1] = anchor[0]
            self.latest_values[3] = anchor[1]
            self.latest_values[2] = anchor[2]
            for idx, entry_var in self.entry_vars:
                entry_var.set(f"{self.latest_values[idx]:.6g}")
            self.status_var.set("Trajectory complete. Returned to anchor in MANUAL mode.")
        else:
            self.status_var.set("Trajectory preview ended and returned to MANUAL mode.")

        self.popup_stage = "SUMMARY_STAGE"
        self._render_popup_stage()

    def _render_summary_stage(self):
        anchor = self.popup_vars["anchor"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        hold_text = self._format_event_text(self.popup_vars["plan_events"]["hold"], "Flat hold")
        zero_drop_text = self._format_event_text(
            self.popup_vars["plan_events"]["zero_drop"],
            "Fueling drop to 0 (D1/D2)",
        )
        tk.Label(self.popup_content_frame, text="Stage 4: Trajectory Summary", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text="Preview complete. Returned to anchor setpoint in MANUAL mode.",
        ).pack(anchor="w", pady=(8, 8))
        tk.Label(self.popup_content_frame, text=f"Start (Anchor): D1={anchor[0]:.4f}, D2={anchor[1]:.4f}, SOI2={anchor[2]:.4f}").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"End target: D1={end_action[0]:.4f}, D2={end_action[1]:.4f}, SOI2={end_action[2]:.4f}").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Length: {self.popup_vars['length']} cycles").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Per-cycle delta: dD1={delta[0]:.6f}, dD2={delta[1]:.6f}, dSOI2={delta[2]:.6f}").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=hold_text).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=zero_drop_text).pack(anchor="w")
        tk.Button(self.popup_content_frame, text="Close And Return To Main UI", command=self._close_popup).pack(anchor="w", pady=(14, 0))


def main():
    root = tk.Tk()
    DummyTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
