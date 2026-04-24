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
            text="Set 12 output values (6 injections: timing and duration):",
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
