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
        tk.Button(frame, text="Generate Random Target", command=self.open_target_popup).grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )

        self.status_var = tk.StringVar(value="Dummy mode: no data is sent.")
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(10, 0)
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
        return (
            float(self.bound_vars["d1_low"].get()),
            float(self.bound_vars["d1_high"].get()),
            float(self.bound_vars["d2_low"].get()),
            float(self.bound_vars["d2_high"].get()),
            float(self.bound_vars["soi2_low"].get()),
            float(self.bound_vars["soi2_high"].get()),
            {
                "smooth_ramp": float(self.mode_prob_vars["smooth_ramp"].get()),
                "step_hold": float(self.mode_prob_vars["step_hold"].get()),
                "boundary_probe": float(self.mode_prob_vars["boundary_probe"].get()),
                "no_injection": float(self.mode_prob_vars["no_injection"].get()),
            },
            int(self.length_min_var.get()),
            int(self.length_max_var.get()),
            float(self.warmup_seconds_var.get()),
        )

    def apply_values(self):
        try:
            self.latest_values = self._get_manual_values()
        except ValueError:
            messagebox.showerror("Invalid input", "Manual fields must be numeric.")
            return
        self.status_var.set("Manual values applied.")

    def zero_and_apply_values(self):
        self.latest_values = np.zeros(12, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.status_var.set("Manual values zeroed.")

    def open_target_popup(self):
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            target = self.sequence_generator.sample_random_target()
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
        self.popup_vars = {
            "target": target,
            "anchor": np.zeros(3, dtype=np.float32),
            "vector": np.zeros(3, dtype=np.float32),
            "target_end_action": np.zeros(3, dtype=np.float32),
            "step_delta": np.zeros(3, dtype=np.float32),
            "length": 0,
            "mode": "",
            "travel_distance": 0.0,
            "warmup_seconds": config[9],
        }
        self.popup_stage = "TARGET_STAGE"
        self._render_popup_stage()

    def _close_popup(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None

    def _clear_popup(self):
        for child in self.popup_content_frame.winfo_children():
            child.destroy()

    def _render_popup_stage(self):
        if self.popup is None or not self.popup.winfo_exists():
            return
        self._clear_popup()
        if self.popup_stage == "TARGET_STAGE":
            self._render_target_stage()
        elif self.popup_stage == "VECTOR_REVIEW_STAGE":
            self._render_vector_stage()
        elif self.popup_stage == "READY_STAGE":
            self._render_ready_stage()
        elif self.popup_stage == "SUMMARY_STAGE":
            self._render_summary_stage()

    def _render_target_stage(self):
        target = self.popup_vars["target"]
        tk.Label(self.popup_content_frame, text="Stage 1: Random Target", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Target: D1={target[0]:.3f}, D2={target[1]:.3f}, SOI2={target[2]:.3f}").pack(anchor="w", pady=(8, 12))
        tk.Button(self.popup_content_frame, text="Confirm Anchor and Generate Random Vector", command=self._confirm_anchor_and_generate_vector).pack(anchor="w")

    def _confirm_anchor_and_generate_vector(self):
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            self.sequence_generator.set_anchor_from_payload(self.latest_values)
            self.sequence_generator.sample_random_unit_vector()
            self.sequence_generator.sample_trajectory_plan(config[6], config[7], config[8])
            details = self.sequence_generator.get_plan_details()
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return
        self.popup_vars["anchor"] = details["anchor"]
        self.popup_vars["vector"] = details["unit_vector"]
        self.popup_vars["target_end_action"] = details["target_end_action"]
        self.popup_vars["step_delta"] = details["step_delta"]
        self.popup_vars["length"] = details["length"]
        self.popup_vars["mode"] = details["mode"]
        self.popup_vars["travel_distance"] = details["travel_distance"]
        self.popup_stage = "VECTOR_REVIEW_STAGE"
        self._render_popup_stage()

    def _render_vector_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        tk.Label(self.popup_content_frame, text="Stage 2: Vector Review", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Anchor: D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}").pack(anchor="w", pady=(8, 4))
        tk.Label(self.popup_content_frame, text=f"Unit vector: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]").pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=f"Target end action: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}").pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=f"Length={self.popup_vars['length']} cycles | Mode={self.popup_vars['mode']}").pack(anchor="w", pady=(0, 4))
        tk.Label(self.popup_content_frame, text=f"Per-cycle delta: dD1={delta[0]:.5f}, dD2={delta[1]:.5f}, dSOI2={delta[2]:.5f}").pack(anchor="w", pady=(0, 4))
        tk.Button(self.popup_content_frame, text="Confirm Random Vector", command=self._confirm_vector).pack(anchor="w", pady=(0, 12))

    def _confirm_vector(self):
        try:
            self.sequence_generator.confirm_vector()
        except Exception as exc:
            messagebox.showerror("Vector confirmation failed", str(exc))
            return
        self.popup_stage = "READY_STAGE"
        self._render_popup_stage()

    def _render_ready_stage(self):
        tk.Label(self.popup_content_frame, text="Stage 3: Ready", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Length={self.popup_vars['length']} cycles | Warmup={self.popup_vars['warmup_seconds']:.1f}s").pack(anchor="w", pady=(8, 12))
        tk.Button(self.popup_content_frame, text="Start Trajectory (Preview)", command=self._show_trajectory_summary).pack(anchor="w")

    def _show_trajectory_summary(self):
        self.popup_stage = "SUMMARY_STAGE"
        self._render_popup_stage()

    def _render_summary_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        tk.Label(self.popup_content_frame, text="Stage 4: Trajectory Summary", font=self.section_title_font).pack(anchor="w")
        tk.Label(self.popup_content_frame, text="Preview complete. No data sending in dummy mode.").pack(anchor="w", pady=(8, 8))
        tk.Label(self.popup_content_frame, text=f"Anchor: D1={anchor[0]:.4f}, D2={anchor[1]:.4f}, SOI2={anchor[2]:.4f}").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Vector: [{vector[0]:.5f}, {vector[1]:.5f}, {vector[2]:.5f}]").pack(anchor="w")
        tk.Label(self.popup_content_frame, text=f"Target end: D1={end_action[0]:.4f}, D2={end_action[1]:.4f}, SOI2={end_action[2]:.4f}").pack(anchor="w")
        tk.Button(self.popup_content_frame, text="Close And Return To Main UI", command=self._close_popup).pack(anchor="w", pady=(14, 0))


def main():
    root = tk.Tk()
    DummyTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from injection_sequence_generator import InjectionSequenceGenerator

INJECTION_EVENT_COUNT = 6


class DummyTrajectoryGUI:
    """Local GUI preview with no LabVIEW/TCP communication."""

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

        tk.Button(frame, text="Generate Random Target", command=self.open_target_popup).grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )

        self.status_var = tk.StringVar(
            value="Dummy mode: no data is sent. Use popup to preview full trajectory flow."
        )
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(10, 0)
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
            ("d1_low", "D1 low"),
            ("d1_high", "D1 high"),
            ("d2_low", "D2 low"),
            ("d2_high", "D2 high"),
            ("soi2_low", "SOI2 low"),
            ("soi2_high", "SOI2 high"),
        ]:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, width=10, textvariable=self.bound_vars[key]).grid(
                row=row, column=1, sticky="w", pady=1
            )
            row += 1

        tk.Label(frame, text="Sampling balance").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        for key, label in [
            ("smooth_ramp", "P smooth"),
            ("step_hold", "P step"),
            ("boundary_probe", "P boundary"),
            ("no_injection", "P no-inj"),
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
        mode_probabilities = {
            "smooth_ramp": float(self.mode_prob_vars["smooth_ramp"].get()),
            "step_hold": float(self.mode_prob_vars["step_hold"].get()),
            "boundary_probe": float(self.mode_prob_vars["boundary_probe"].get()),
            "no_injection": float(self.mode_prob_vars["no_injection"].get()),
        }
        min_length = int(self.length_min_var.get())
        max_length = int(self.length_max_var.get())
        warmup_seconds = float(self.warmup_seconds_var.get())
        if warmup_seconds < 0.0:
            raise ValueError("Warmup seconds must be >= 0.")
        return (
            d1_low,
            d1_high,
            d2_low,
            d2_high,
            soi2_low,
            soi2_high,
            mode_probabilities,
            min_length,
            max_length,
            warmup_seconds,
        )

    def apply_values(self):
        try:
            self.latest_values = self._get_manual_values()
        except ValueError:
            messagebox.showerror("Invalid input", "Manual fields must be numeric.")
            return
        self.status_var.set("Manual values applied.")

    def zero_and_apply_values(self):
        self.latest_values = np.zeros(12, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.status_var.set("Manual values zeroed.")

    def open_target_popup(self):
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            target = self.sequence_generator.sample_random_target()
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

        self.popup_vars = {
            "target": target,
            "anchor": np.zeros(3, dtype=np.float32),
            "vector": np.zeros(3, dtype=np.float32),
            "target_end_action": np.zeros(3, dtype=np.float32),
            "step_delta": np.zeros(3, dtype=np.float32),
            "length": 0,
            "mode": "",
            "travel_distance": 0.0,
            "warmup_seconds": config[9],
            "bounds_low": np.array(config[:6:2], dtype=np.float32),
            "bounds_high": np.array(config[1:6:2], dtype=np.float32),
        }
        self.popup_stage = "TARGET_STAGE"
        self._render_popup_stage()

    def _close_popup(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None

    def _clear_popup(self):
        for child in self.popup_content_frame.winfo_children():
            child.destroy()

    def _render_popup_stage(self):
        if self.popup is None or not self.popup.winfo_exists():
            return
        self._clear_popup()
        if self.popup_stage == "TARGET_STAGE":
            self._render_target_stage()
        elif self.popup_stage == "VECTOR_REVIEW_STAGE":
            self._render_vector_stage()
        elif self.popup_stage == "READY_STAGE":
            self._render_ready_stage()
        elif self.popup_stage == "SUMMARY_STAGE":
            self._render_summary_stage()

    def _render_target_stage(self):
        target = self.popup_vars["target"]
        low = self.popup_vars["bounds_low"]
        high = self.popup_vars["bounds_high"]
        tk.Label(self.popup_content_frame, text="Stage 1: Random Target", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Target: D1={target[0]:.3f}, D2={target[1]:.3f}, SOI2={target[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=(
                f"Global bounds: D1[{low[0]:.3f},{high[0]:.3f}]  "
                f"D2[{low[1]:.3f},{high[1]:.3f}]  "
                f"SOI2[{low[2]:.3f},{high[2]:.3f}]"
            ),
        ).pack(anchor="w", pady=(0, 10))
        tk.Button(
            self.popup_content_frame,
            text="Confirm Anchor and Generate Random Vector",
            command=self._confirm_anchor_and_generate_vector,
        ).pack(anchor="w")

    def _confirm_anchor_and_generate_vector(self):
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            self.sequence_generator.set_anchor_from_payload(self.latest_values)
            self.sequence_generator.sample_random_unit_vector()
            self.sequence_generator.sample_trajectory_plan(config[6], config[7], config[8])
            details = self.sequence_generator.get_plan_details()
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return

        self.popup_vars["anchor"] = details["anchor"]
        self.popup_vars["vector"] = details["unit_vector"]
        self.popup_vars["target_end_action"] = details["target_end_action"]
        self.popup_vars["step_delta"] = details["step_delta"]
        self.popup_vars["length"] = details["length"]
        self.popup_vars["mode"] = details["mode"]
        self.popup_vars["travel_distance"] = details["travel_distance"]
        self.popup_stage = "VECTOR_REVIEW_STAGE"
        self._render_popup_stage()

    def _render_vector_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        tk.Label(self.popup_content_frame, text="Stage 2: Vector Review", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Anchor: D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Unit vector: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Target end action: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Length={self.popup_vars['length']} cycles | Mode={self.popup_vars['mode']}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Per-cycle delta: dD1={delta[0]:.5f}, dD2={delta[1]:.5f}, dSOI2={delta[2]:.5f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Travel distance: {self.popup_vars['travel_distance']:.4f}",
        ).pack(anchor="w", pady=(0, 12))
        tk.Button(
            self.popup_content_frame, text="Confirm Random Vector", command=self._confirm_vector
        ).pack(anchor="w")

    def _confirm_vector(self):
        try:
            self.sequence_generator.confirm_vector()
        except Exception as exc:
            messagebox.showerror("Vector confirmation failed", str(exc))
            return
        self.popup_stage = "READY_STAGE"
        self._render_popup_stage()

    def _render_ready_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        tk.Label(self.popup_content_frame, text="Stage 3: Ready", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Anchor: D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Vector: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Target end: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Warmup={self.popup_vars['warmup_seconds']:.1f}s | Length={self.popup_vars['length']} cycles",
        ).pack(anchor="w", pady=(0, 12))
        tk.Button(
            self.popup_content_frame,
            text="Start Trajectory (Preview)",
            command=self._show_trajectory_summary,
        ).pack(anchor="w")

    def _show_trajectory_summary(self):
        self.popup_stage = "SUMMARY_STAGE"
        self._render_popup_stage()

    def _render_summary_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        tk.Label(self.popup_content_frame, text="Stage 4: Trajectory Summary", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text="Preview complete. In dummy mode this is where curve generation would start.",
        ).pack(anchor="w", pady=(8, 8))
        summary_lines = [
            f"Mode: {self.popup_vars['mode']}",
            f"Warmup: {self.popup_vars['warmup_seconds']:.2f} s",
            f"Length: {self.popup_vars['length']} cycles",
            f"Anchor: D1={anchor[0]:.4f}, D2={anchor[1]:.4f}, SOI2={anchor[2]:.4f}",
            f"Unit vector: [{vector[0]:.5f}, {vector[1]:.5f}, {vector[2]:.5f}]",
            f"Target end: D1={end_action[0]:.4f}, D2={end_action[1]:.4f}, SOI2={end_action[2]:.4f}",
            f"Per-cycle delta: dD1={delta[0]:.6f}, dD2={delta[1]:.6f}, dSOI2={delta[2]:.6f}",
            f"Travel distance: {self.popup_vars['travel_distance']:.6f}",
        ]
        for line in summary_lines:
            tk.Label(self.popup_content_frame, text=line, anchor="w", justify="left").pack(
                anchor="w", pady=(2, 0)
            )
        tk.Button(
            self.popup_content_frame,
            text="Close And Return To Main UI",
            command=self._close_popup,
        ).pack(anchor="w", pady=(14, 0))


def main():
    root = tk.Tk()
    DummyTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from injection_sequence_generator import InjectionSequenceGenerator

INJECTION_EVENT_COUNT = 6


class DummyTrajectoryGUI:
    """Local GUI preview with no LabVIEW/TCP communication."""

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

        tk.Button(frame, text="Generate Random Target", command=self.open_target_popup).grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )

        self.status_var = tk.StringVar(
            value="Dummy mode: no data is sent. Use popup to preview full trajectory flow."
        )
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(10, 0)
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
            ("d1_low", "D1 low"),
            ("d1_high", "D1 high"),
            ("d2_low", "D2 low"),
            ("d2_high", "D2 high"),
            ("soi2_low", "SOI2 low"),
            ("soi2_high", "SOI2 high"),
        ]:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, width=10, textvariable=self.bound_vars[key]).grid(
                row=row, column=1, sticky="w", pady=1
            )
            row += 1

        tk.Label(frame, text="Sampling balance").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        for key, label in [
            ("smooth_ramp", "P smooth"),
            ("step_hold", "P step"),
            ("boundary_probe", "P boundary"),
            ("no_injection", "P no-inj"),
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
        mode_probabilities = {
            "smooth_ramp": float(self.mode_prob_vars["smooth_ramp"].get()),
            "step_hold": float(self.mode_prob_vars["step_hold"].get()),
            "boundary_probe": float(self.mode_prob_vars["boundary_probe"].get()),
            "no_injection": float(self.mode_prob_vars["no_injection"].get()),
        }
        min_length = int(self.length_min_var.get())
        max_length = int(self.length_max_var.get())
        warmup_seconds = float(self.warmup_seconds_var.get())
        if warmup_seconds < 0.0:
            raise ValueError("Warmup seconds must be >= 0.")
        return (
            d1_low,
            d1_high,
            d2_low,
            d2_high,
            soi2_low,
            soi2_high,
            mode_probabilities,
            min_length,
            max_length,
            warmup_seconds,
        )

    def apply_values(self):
        try:
            self.latest_values = self._get_manual_values()
        except ValueError:
            messagebox.showerror("Invalid input", "Manual fields must be numeric.")
            return
        self.status_var.set("Manual values applied.")

    def zero_and_apply_values(self):
        self.latest_values = np.zeros(12, dtype=np.float32)
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.status_var.set("Manual values zeroed.")

    def open_target_popup(self):
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            target = self.sequence_generator.sample_random_target()
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

        self.popup_vars = {
            "target": target,
            "anchor": np.zeros(3, dtype=np.float32),
            "vector": np.zeros(3, dtype=np.float32),
            "target_end_action": np.zeros(3, dtype=np.float32),
            "step_delta": np.zeros(3, dtype=np.float32),
            "length": 0,
            "mode": "",
            "travel_distance": 0.0,
            "warmup_seconds": config[9],
            "bounds_low": np.array(config[:6:2], dtype=np.float32),
            "bounds_high": np.array(config[1:6:2], dtype=np.float32),
        }
        self.popup_stage = "TARGET_STAGE"
        self._render_popup_stage()

    def _close_popup(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None

    def _clear_popup(self):
        for child in self.popup_content_frame.winfo_children():
            child.destroy()

    def _render_popup_stage(self):
        if self.popup is None or not self.popup.winfo_exists():
            return
        self._clear_popup()
        if self.popup_stage == "TARGET_STAGE":
            self._render_target_stage()
        elif self.popup_stage == "VECTOR_REVIEW_STAGE":
            self._render_vector_stage()
        elif self.popup_stage == "READY_STAGE":
            self._render_ready_stage()
        elif self.popup_stage == "SUMMARY_STAGE":
            self._render_summary_stage()

    def _render_target_stage(self):
        target = self.popup_vars["target"]
        low = self.popup_vars["bounds_low"]
        high = self.popup_vars["bounds_high"]
        tk.Label(self.popup_content_frame, text="Stage 1: Random Target", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Target: D1={target[0]:.3f}, D2={target[1]:.3f}, SOI2={target[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=(
                f"Global bounds: D1[{low[0]:.3f},{high[0]:.3f}]  "
                f"D2[{low[1]:.3f},{high[1]:.3f}]  "
                f"SOI2[{low[2]:.3f},{high[2]:.3f}]"
            ),
        ).pack(anchor="w", pady=(0, 10))
        tk.Button(
            self.popup_content_frame,
            text="Confirm Anchor and Generate Random Vector",
            command=self._confirm_anchor_and_generate_vector,
        ).pack(anchor="w")

    def _confirm_anchor_and_generate_vector(self):
        try:
            config = self._read_generator_config()
            self.latest_values = self._get_manual_values()
            self.sequence_generator.configure_bounds(*config[:6])
            self.sequence_generator.set_default_payload(self.latest_values)
            self.sequence_generator.set_anchor_from_payload(self.latest_values)
            self.sequence_generator.sample_random_unit_vector()
            self.sequence_generator.sample_trajectory_plan(config[6], config[7], config[8])
            details = self.sequence_generator.get_plan_details()
        except Exception as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return

        self.popup_vars["anchor"] = details["anchor"]
        self.popup_vars["vector"] = details["unit_vector"]
        self.popup_vars["target_end_action"] = details["target_end_action"]
        self.popup_vars["step_delta"] = details["step_delta"]
        self.popup_vars["length"] = details["length"]
        self.popup_vars["mode"] = details["mode"]
        self.popup_vars["travel_distance"] = details["travel_distance"]
        self.popup_stage = "VECTOR_REVIEW_STAGE"
        self._render_popup_stage()

    def _render_vector_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        tk.Label(self.popup_content_frame, text="Stage 2: Vector Review", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Anchor: D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Unit vector: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Target end action: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Length={self.popup_vars['length']} cycles | Mode={self.popup_vars['mode']}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Per-cycle delta: dD1={delta[0]:.5f}, dD2={delta[1]:.5f}, dSOI2={delta[2]:.5f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Travel distance: {self.popup_vars['travel_distance']:.4f}",
        ).pack(anchor="w", pady=(0, 12))
        tk.Button(
            self.popup_content_frame, text="Confirm Random Vector", command=self._confirm_vector
        ).pack(anchor="w")

    def _confirm_vector(self):
        try:
            self.sequence_generator.confirm_vector()
        except Exception as exc:
            messagebox.showerror("Vector confirmation failed", str(exc))
            return
        self.popup_stage = "READY_STAGE"
        self._render_popup_stage()

    def _render_ready_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        tk.Label(self.popup_content_frame, text="Stage 3: Ready", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text=f"Anchor: D1={anchor[0]:.3f}, D2={anchor[1]:.3f}, SOI2={anchor[2]:.3f}",
        ).pack(anchor="w", pady=(8, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Vector: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Target end: D1={end_action[0]:.3f}, D2={end_action[1]:.3f}, SOI2={end_action[2]:.3f}",
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            self.popup_content_frame,
            text=f"Warmup={self.popup_vars['warmup_seconds']:.1f}s | Length={self.popup_vars['length']} cycles",
        ).pack(anchor="w", pady=(0, 12))
        tk.Button(
            self.popup_content_frame,
            text="Start Trajectory (Preview)",
            command=self._show_trajectory_summary,
        ).pack(anchor="w")

    def _show_trajectory_summary(self):
        self.popup_stage = "SUMMARY_STAGE"
        self._render_popup_stage()

    def _render_summary_stage(self):
        anchor = self.popup_vars["anchor"]
        vector = self.popup_vars["vector"]
        end_action = self.popup_vars["target_end_action"]
        delta = self.popup_vars["step_delta"]
        tk.Label(self.popup_content_frame, text="Stage 4: Trajectory Summary", font=self.section_title_font).pack(anchor="w")
        tk.Label(
            self.popup_content_frame,
            text="Preview complete. In dummy mode this is where curve generation would start.",
        ).pack(anchor="w", pady=(8, 8))
        summary_lines = [
            f"Mode: {self.popup_vars['mode']}",
            f"Warmup: {self.popup_vars['warmup_seconds']:.2f} s",
            f"Length: {self.popup_vars['length']} cycles",
            f"Anchor: D1={anchor[0]:.4f}, D2={anchor[1]:.4f}, SOI2={anchor[2]:.4f}",
            f"Unit vector: [{vector[0]:.5f}, {vector[1]:.5f}, {vector[2]:.5f}]",
            f"Target end: D1={end_action[0]:.4f}, D2={end_action[1]:.4f}, SOI2={end_action[2]:.4f}",
            f"Per-cycle delta: dD1={delta[0]:.6f}, dD2={delta[1]:.6f}, dSOI2={delta[2]:.6f}",
            f"Travel distance: {self.popup_vars['travel_distance']:.6f}",
        ]
        for line in summary_lines:
            tk.Label(self.popup_content_frame, text=line, anchor="w", justify="left").pack(
                anchor="w", pady=(2, 0)
            )
        tk.Button(
            self.popup_content_frame,
            text="Close And Return To Main UI",
            command=self._close_popup,
        ).pack(anchor="w", pady=(14, 0))


def main():
    root = tk.Tk()
    DummyTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox


class DummyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LabVIEW TCP Responder (Dummy UI)")

        self.entry_vars = []
        self.sequence_mode = "indefinite"
        self.sequence_length = 10
        self._build_ui()
        self.root.update_idletasks()
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_scaling(self):
        default_font = tkfont.nametofont("TkDefaultFont")
        text_font = tkfont.nametofont("TkTextFont")
        fixed_font = tkfont.nametofont("TkFixedFont")
        default_font.configure(size=14)
        text_font.configure(size=14)
        fixed_font.configure(size=14)
        self.root.option_add("*Button.Padx", 10)
        self.root.option_add("*Button.Pady", 6)
        self.root.option_add("*Entry.Font", default_font)
        self.section_title_font = tkfont.Font(
            family=default_font.cget("family"), size=15, weight="bold"
        )

    def _build_ui(self):
        self._configure_scaling()
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        self.mode_var = tk.StringVar(value="manual")
        tk.Label(frame, text="Value source:", font=self.section_title_font).grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 8)
        )
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

        manual_frame = tk.LabelFrame(
            frame, text="Manual Values", font=self.section_title_font, padx=10, pady=10
        )
        manual_frame.grid(row=1, column=0, columnspan=4, sticky="nw")

        sequence_frame = tk.LabelFrame(
            frame, text="Sequence Options", font=self.section_title_font, padx=10, pady=10
        )
        sequence_frame.grid(row=1, column=5, columnspan=4, sticky="ne", padx=(30, 0))

        tk.Label(
            manual_frame,
            text="Set 12 output values (payload order: T1,D1,T2,D2,...,T6,D6):",
            anchor="w",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

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
        self.sequence_length_entry.bind("<KeyRelease>", self.on_sequence_length_change)
        self.sequence_length_entry.bind("<FocusOut>", self.on_sequence_length_change)

        for injection_idx in range(6):
            timing_idx = injection_idx * 2
            duration_idx = timing_idx + 1

            tk.Label(manual_frame, text=f"Inj {injection_idx + 1} timing (deg aTDC)").grid(
                row=injection_idx + 1, column=0, sticky="w", padx=(0, 6), pady=2
            )
            timing_var = tk.StringVar(value="0.0")
            tk.Entry(manual_frame, width=16, textvariable=timing_var).grid(
                row=injection_idx + 1, column=1, sticky="w", padx=(0, 12), pady=2
            )
            self.entry_vars.append((timing_idx, timing_var))

            tk.Label(manual_frame, text=f"Inj {injection_idx + 1} duration (ms)").grid(
                row=injection_idx + 1, column=2, sticky="w", padx=(0, 6), pady=2
            )
            duration_var = tk.StringVar(value="0.0")
            tk.Entry(manual_frame, width=16, textvariable=duration_var).grid(
                row=injection_idx + 1, column=3, sticky="w", pady=2
            )
            self.entry_vars.append((duration_idx, duration_var))

        self.apply_button = tk.Button(manual_frame, text="Apply", command=self.apply_values)
        self.apply_button.grid(row=8, column=0, sticky="w", pady=(12, 0))

        self.zero_button = tk.Button(frame, text="ZERO", command=self.zero_and_apply_values)
        self.zero_button.grid(row=2, column=1, sticky="w", pady=(12, 0), padx=(8, 0))

        self.stop_button = tk.Button(frame, text="Stop", command=self.stop_pressed)
        self.stop_button.grid(row=2, column=0, sticky="w", pady=(12, 0))

        self.status_var = tk.StringVar(value="Dummy mode: no communication active.")
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
            parsed_values = [0.0] * 12
            for idx, entry_var in self.entry_vars:
                parsed_values[idx] = float(entry_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "All 12 fields must be valid numbers.")
            return

        mode = self.mode_var.get()
        if mode == "manual":
            self.last_sent_var.set(f"Last sent: {parsed_values}")
            self.status_var.set(f"Manual values accepted (dummy): {parsed_values}")
        else:
            self.status_var.set(
                f"Sequence mode (dummy): {self.sequence_mode}, length={self.sequence_length}"
            )

    def zero_and_apply_values(self):
        for _, entry_var in self.entry_vars:
            entry_var.set("0.0")
        self.mode_var.set("manual")
        zeros = [0.0] * 12
        self.last_sent_var.set(f"Last sent: {zeros}")
        self.status_var.set("ZERO applied: all values set to 0.0 and mode set to manual.")

    def on_mode_change(self):
        if self.mode_var.get() == "sequence":
            self.apply_sequence_settings(show_status=False)
            self.status_var.set(
                f"Source mode set to: sequence (dummy, active: {self.sequence_mode}, length={self.sequence_length})"
            )
        else:
            self.status_var.set("Source mode set to: manual (dummy)")

    def on_sequence_mode_change(self):
        self.apply_sequence_settings(show_status=False)
        self.status_var.set(f"Sequence sampling mode set to: {self.sequence_mode_var.get()} (dummy)")

    def on_sequence_length_change(self, _event=None):
        self.apply_sequence_settings(show_status=False)

    def apply_sequence_settings(self, show_status=True):
        raw_text = self.sequence_length_var.get().strip()
        try:
            parsed_value = int(raw_text)
            if parsed_value <= 0:
                raise ValueError
        except ValueError:
            if show_status:
                self.status_var.set("Sequence length must be a positive integer (dummy).")
            return
        self.sequence_mode = self.sequence_mode_var.get()
        self.sequence_length = parsed_value
        if show_status:
            self.status_var.set(
                f"Sequence settings applied (dummy): {self.sequence_mode} / length {self.sequence_length}"
            )

    def stop_pressed(self):
        self.status_var.set("Dummy Stop pressed (no communication to stop).")

    def on_close(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    DummyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
