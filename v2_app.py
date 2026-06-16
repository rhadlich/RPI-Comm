from __future__ import annotations

import tkinter as tk
import threading
from dataclasses import fields
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk

import numpy as np

from v2_action_generator import RandomSequenceGenerator, SequenceBundle
from v2_config import (
    ALLOW_PREPARE_DURING_COLLECTION,
    ActionBounds,
    DEFAULT_SERVER_IP,
    DEFAULT_SERVER_PORT,
    KNNConfig,
    PayloadConfig,
    PREP_WORKER_NICE_INCREMENT,
    SequenceConfig,
)
from v2_labview_comm import CommIncoming, CommProgress, LabVIEWCommunicator
from v2_safety_filter import KNNSafetyFilter


class V2App:
    def __init__(
        self,
        root: tk.Tk,
        *,
        communicator_cls: type[LabVIEWCommunicator] = LabVIEWCommunicator,
        window_title: str = "LabVIEW Collector v2",
    ):
        self.root = root
        self.root.title(window_title)
        self.root.resizable(False, False)
        self.communicator_cls = communicator_cls

        self.sequence_config = SequenceConfig()
        self.knn_config = KNNConfig()
        self.payload_config = PayloadConfig()
        self.sequence_config_vars: dict[str, tk.StringVar] = {}
        self.knn_config_vars: dict[str, tk.StringVar] = {}

        self.generator = RandomSequenceGenerator(
            sequence_config=self.sequence_config,
            payload_config=self.payload_config,
        )
        self.safety_filter: KNNSafetyFilter | None = None
        self.sequence_bundle: SequenceBundle | None = None
        self.filtered_actions: np.ndarray | None = None
        self.predicted_mprr: np.ndarray | None = None
        self.prepared_payloads: list[np.ndarray] = []
        self.prepared_modes: list[str] = []
        self.prepared_idle_payload: np.ndarray | None = None
        self.preparing = False
        self._prepare_job_id = 0
        self._prepare_thread: threading.Thread | None = None

        self.ui_queue: Queue = Queue()

        self.server_ip_var = tk.StringVar(value=DEFAULT_SERVER_IP)
        self.server_port_var = tk.StringVar(value=str(DEFAULT_SERVER_PORT))
        self.csv_path_var = tk.StringVar(value=str(Path.cwd() / "safety_examples.csv"))

        self.bounds_vars = {
            "id1_low": tk.StringVar(value="0.6"),
            "id1_high": tk.StringVar(value="0.8"),
            "id2_low": tk.StringVar(value="0.0"),
            "id2_high": tk.StringVar(value="1.0"),
            "soi2_low": tk.StringVar(value="-140.0"),
            "soi2_high": tk.StringVar(value="-10.0"),
        }
        self.anchor_vars = {
            "id1": tk.StringVar(value=""),
            "id2": tk.StringVar(value=""),
            "soi2": tk.StringVar(value=""),
        }
        self.manual_vars = {
            "id1": tk.StringVar(value=""),
            "id2": tk.StringVar(value=""),
            "soi2": tk.StringVar(value=""),
        }
        self.manual_idle_override_enabled = False
        self.manual_idle_action: np.ndarray | None = None
        self._manual_apply_job: str | None = None

        self.status_var = tk.StringVar(value="Initialize by loading safety CSV.")
        self.cycles_remaining_var = tk.StringVar(value="Cycles remaining: --")
        self.prepare_summary_var = tk.StringVar(value="Sequence not prepared.")
        self.incoming_values_var = tk.StringVar(value="LabVIEW incoming: --, --")

        self.collector_ready = False
        self._build_ui()

        self._init_communicator()

        self._attempt_startup_csv_load()
        self._poll_ui_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)
        frame.grid_columnconfigure(0, weight=1)

        net_frame = tk.LabelFrame(frame, text="Communication", padx=8, pady=8)
        net_frame.grid(row=0, column=0, sticky="ew")
        tk.Label(net_frame, text="Server IP").grid(row=0, column=0, sticky="w")
        tk.Entry(net_frame, width=16, textvariable=self.server_ip_var).grid(row=0, column=1, sticky="w", padx=(6, 12))
        tk.Label(net_frame, text="Port").grid(row=0, column=2, sticky="w")
        tk.Entry(net_frame, width=8, textvariable=self.server_port_var).grid(row=0, column=3, sticky="w", padx=(6, 0))
        tk.Button(net_frame, text="Reconnect", command=self.reconnect_communicator).grid(
            row=0, column=4, sticky="w", padx=(10, 0)
        )

        csv_frame = tk.LabelFrame(frame, text="Safety Dataset (CSV)", padx=8, pady=8)
        csv_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        tk.Entry(csv_frame, width=52, textvariable=self.csv_path_var).grid(row=0, column=0, sticky="w")
        tk.Button(csv_frame, text="Browse", command=self.browse_csv).grid(row=0, column=1, padx=(8, 0), sticky="w")
        tk.Button(csv_frame, text="Load CSV", command=self.load_safety_csv).grid(row=1, column=1, padx=(8, 0), pady=(6, 0), sticky="w")

        bounds_frame = tk.LabelFrame(frame, text="Local Bounds", padx=8, pady=8)
        bounds_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self._bound_row(bounds_frame, 0, "ID1", "id1_low", "id1_high")
        self._bound_row(bounds_frame, 1, "ID2", "id2_low", "id2_high")
        self._bound_row(bounds_frame, 2, "SOI2", "soi2_low", "soi2_high")

        anchor_frame = tk.LabelFrame(frame, text="Anchor", padx=8, pady=8)
        anchor_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        tk.Label(anchor_frame, text="ID1").grid(row=0, column=0, sticky="w")
        tk.Entry(anchor_frame, width=10, textvariable=self.anchor_vars["id1"]).grid(row=0, column=1, sticky="w", padx=(6, 12))
        tk.Label(anchor_frame, text="ID2").grid(row=0, column=2, sticky="w")
        tk.Entry(anchor_frame, width=10, textvariable=self.anchor_vars["id2"]).grid(row=0, column=3, sticky="w", padx=(6, 12))
        tk.Label(anchor_frame, text="SOI2").grid(row=0, column=4, sticky="w")
        tk.Entry(anchor_frame, width=10, textvariable=self.anchor_vars["soi2"]).grid(row=0, column=5, sticky="w", padx=(6, 12))
        tk.Button(anchor_frame, text="Suggest Anchor", command=self.suggest_anchor).grid(row=0, column=6, sticky="w")

        manual_frame = tk.LabelFrame(frame, text="Current Injection Settings (Anchor)", padx=8, pady=8)
        manual_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        tk.Label(manual_frame, text="ID1").grid(row=0, column=0, sticky="w")
        tk.Entry(manual_frame, width=10, textvariable=self.manual_vars["id1"]).grid(row=0, column=1, sticky="w", padx=(6, 12))
        tk.Label(manual_frame, text="ID2").grid(row=0, column=2, sticky="w")
        tk.Entry(manual_frame, width=10, textvariable=self.manual_vars["id2"]).grid(row=0, column=3, sticky="w", padx=(6, 12))
        tk.Label(manual_frame, text="SOI2").grid(row=0, column=4, sticky="w")
        tk.Entry(manual_frame, width=10, textvariable=self.manual_vars["soi2"]).grid(row=0, column=5, sticky="w", padx=(6, 12))
        tk.Button(manual_frame, text="Accept Anchor", command=self.accept_anchor_as_manual).grid(
            row=0, column=6, sticky="w", padx=(8, 0)
        )
        tk.Button(manual_frame, text="Apply Injection Settings", command=self.apply_manual_injection_settings).grid(
            row=1, column=6, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        for var in self.manual_vars.values():
            var.trace_add("write", self._on_manual_value_changed)

        self._init_config_vars()
        config_frame = tk.LabelFrame(frame, text="Config Overrides", padx=8, pady=8)
        config_frame.grid(row=0, column=1, rowspan=10, sticky="n", padx=(12, 0))
        self._build_config_ui(config_frame)

        action_frame = tk.Frame(frame)
        action_frame.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        tk.Button(action_frame, text="Prepare Sequence", command=self.prepare_sequence).grid(row=0, column=0, sticky="w")
        tk.Button(action_frame, text="Start Data Collection", command=self.start_collection).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        tk.Button(action_frame, text="Stop Data Collection", command=self.stop_collection).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )
        tk.Button(action_frame, text="Preview Sequence", command=self.show_sequence_preview).grid(
            row=0, column=3, sticky="w", padx=(8, 0)
        )

        tk.Label(frame, textvariable=self.incoming_values_var, anchor="w", justify="left").grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )
        tk.Label(frame, textvariable=self.status_var, anchor="w", justify="left").grid(
            row=7, column=0, sticky="w", pady=(2, 0)
        )
        tk.Label(frame, textvariable=self.prepare_summary_var, anchor="w", justify="left").grid(
            row=8, column=0, sticky="w", pady=(2, 0)
        )
        tk.Label(frame, textvariable=self.cycles_remaining_var, anchor="w").grid(
            row=9, column=0, sticky="w", pady=(8, 0)
        )
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())

    def _init_config_vars(self) -> None:
        self.sequence_config_vars = {
            f.name: tk.StringVar(value=str(getattr(self.sequence_config, f.name)))
            for f in fields(SequenceConfig)
        }
        self.knn_config_vars = {
            f.name: tk.StringVar(value=str(getattr(self.knn_config, f.name)))
            for f in fields(KNNConfig)
        }

    def _build_config_ui(self, parent: tk.Widget) -> None:
        seq_frame = tk.LabelFrame(parent, text="SequenceConfig", padx=6, pady=6)
        seq_frame.grid(row=0, column=0, sticky="nw")
        knn_frame = tk.LabelFrame(parent, text="KNNConfig", padx=6, pady=6)
        knn_frame.grid(row=0, column=1, sticky="nw", padx=(10, 0))

        seq_fields = [
            "base_anchor_hold_cycles",
            "single_action_cycles",
            "pair_action_cycles",
            "full_3d_cycles",
            "full_no_injection_ratio",
            "partial_no_injection_ratio",
            "zero_hold_min_cycles",
            "zero_hold_max_cycles",
            "smooth_recovery_min_cycles",
            "smooth_recovery_max_cycles",
            "min_normal_cycles_between_event_starts",
            "event_ratio_tolerance",
        ]
        for row, key in enumerate(seq_fields):
            tk.Label(seq_frame, text=key).grid(row=row, column=0, sticky="w")
            tk.Entry(seq_frame, width=12, textvariable=self.sequence_config_vars[key]).grid(
                row=row, column=1, sticky="w", padx=(8, 0), pady=1
            )

        knn_fields = [
            "k_neighbors",
            "safety_threshold",
            "max_normalized_mean_distance",
            "max_resample_attempts_per_cycle",
        ]
        for row, key in enumerate(knn_fields):
            tk.Label(knn_frame, text=key).grid(row=row, column=0, sticky="w")
            tk.Entry(knn_frame, width=12, textvariable=self.knn_config_vars[key]).grid(
                row=row, column=1, sticky="w", padx=(8, 0), pady=1
            )

        tk.Button(parent, text="Apply Config Overrides", command=self.apply_config_overrides).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

    def _bound_row(self, parent: tk.Widget, row: int, label: str, low_key: str, high_key: str) -> None:
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        tk.Label(parent, text="low").grid(row=row, column=1, sticky="w", padx=(12, 4))
        tk.Entry(parent, width=10, textvariable=self.bounds_vars[low_key]).grid(row=row, column=2, sticky="w")
        tk.Label(parent, text="high").grid(row=row, column=3, sticky="w", padx=(12, 4))
        tk.Entry(parent, width=10, textvariable=self.bounds_vars[high_key]).grid(row=row, column=4, sticky="w")

    def _attempt_startup_csv_load(self) -> None:
        try:
            self.load_safety_csv(show_message=False)
        except Exception:
            # Keep startup lightweight; user can load manually.
            pass

    def browse_csv(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select safety examples CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if chosen:
            self.csv_path_var.set(chosen)

    def _read_bounds(self) -> ActionBounds:
        low = np.array(
            [
                float(self.bounds_vars["id1_low"].get()),
                float(self.bounds_vars["id2_low"].get()),
                float(self.bounds_vars["soi2_low"].get()),
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                float(self.bounds_vars["id1_high"].get()),
                float(self.bounds_vars["id2_high"].get()),
                float(self.bounds_vars["soi2_high"].get()),
            ],
            dtype=np.float32,
        )
        bounds = ActionBounds(low=low, high=high)
        bounds.validate()
        return bounds

    def _read_anchor_from_manual(self, bounds: ActionBounds) -> np.ndarray:
        texts = [
            self.manual_vars["id1"].get().strip(),
            self.manual_vars["id2"].get().strip(),
            self.manual_vars["soi2"].get().strip(),
        ]
        if any(not txt for txt in texts):
            raise ValueError("Fill ID1, ID2, and SOI2 in Current Injection Settings (Anchor).")
        anchor = np.array([float(v) for v in texts], dtype=np.float32)
        anchor = self.generator.normalize_anchor(anchor, bounds)
        self.manual_vars["id1"].set(f"{float(anchor[0]):.5g}")
        self.manual_vars["id2"].set(f"{float(anchor[1]):.5g}")
        self.manual_vars["soi2"].set(f"{float(anchor[2]):.5g}")
        return anchor

    def _set_anchor_fields(self, anchor: np.ndarray) -> None:
        self.anchor_vars["id1"].set(f"{float(anchor[0]):.5g}")
        self.anchor_vars["id2"].set(f"{float(anchor[1]):.5g}")
        self.anchor_vars["soi2"].set(f"{float(anchor[2]):.5g}")

    def suggest_anchor(self) -> None:
        try:
            bounds = self._read_bounds()
            anchor = self.generator.suggest_anchor(bounds)
            self._set_anchor_fields(anchor)
            self.status_var.set(
                "Anchor suggestion updated. Use Accept Anchor or edit Current Injection Settings (Anchor)."
            )
        except Exception as exc:
            messagebox.showerror("Anchor Error", str(exc))

    def _on_manual_value_changed(self, *_args) -> None:
        if self._manual_apply_job is not None:
            try:
                self.root.after_cancel(self._manual_apply_job)
            except Exception:
                pass
            self._manual_apply_job = None

    def _read_manual_action(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.manual_vars["id1"].get().strip()),
                float(self.manual_vars["id2"].get().strip()),
                float(self.manual_vars["soi2"].get().strip()),
            ],
            dtype=np.float32,
        )

    def _sync_idle_payload_to_communicator(self) -> None:
        if self.manual_idle_override_enabled and self.manual_idle_action is not None:
            idle_payload = self.generator.build_payload_from_action(self.manual_idle_action)
            self.communicator.set_idle_payload(idle_payload)
            return
        if self.prepared_idle_payload is not None:
            self.communicator.set_idle_payload(self.prepared_idle_payload)

    def _try_apply_manual_idle_override(self) -> bool:
        self._manual_apply_job = None
        try:
            action = self._read_manual_action()
        except Exception:
            return False
        self.manual_idle_action = action
        self.manual_idle_override_enabled = True
        self._sync_idle_payload_to_communicator()
        self.status_var.set(
            f"Manual idle override applied: ID1={action[0]:.3f}, ID2={action[1]:.3f}, SOI2={action[2]:.3f}"
        )
        return True

    def accept_anchor_as_manual(self) -> None:
        try:
            anchor = np.asarray(
                [
                    float(self.anchor_vars["id1"].get().strip()),
                    float(self.anchor_vars["id2"].get().strip()),
                    float(self.anchor_vars["soi2"].get().strip()),
                ],
                dtype=np.float32,
            )
        except Exception:
            messagebox.showerror("Accept Anchor Error", "Anchor values must be valid numeric values.")
            return
        self.manual_vars["id1"].set(f"{anchor[0]:.5g}")
        self.manual_vars["id2"].set(f"{anchor[1]:.5g}")
        self.manual_vars["soi2"].set(f"{anchor[2]:.5g}")
        self.status_var.set("Accepted anchor into Current Injection Settings. Click Apply Injection Settings to send.")

    def apply_manual_injection_settings(self) -> None:
        try:
            applied = self._try_apply_manual_idle_override()
            if not applied:
                raise ValueError("Current Injection Settings must all be valid numeric values.")
        except Exception as exc:
            messagebox.showerror("Apply Injection Error", str(exc))

    def load_safety_csv(self, show_message: bool = True) -> None:
        self.apply_config_overrides(show_message=False)
        csv_path = self.csv_path_var.get().strip()
        self.safety_filter = KNNSafetyFilter(
            csv_path=csv_path,
            knn_config=self.knn_config,
            action_columns=("ID1_prev", "ID2", "SOI2"),
        )
        if show_message:
            self.status_var.set(
                f"Safety CSV loaded: {csv_path} ({self.safety_filter.examples.shape[0]} rows)"
            )
        else:
            self.status_var.set(f"Safety CSV ready: {csv_path}")

    def prepare_sequence(self) -> None:
        try:
            self.apply_config_overrides(show_message=False)
            if self.preparing:
                messagebox.showwarning("Prepare In Progress", "Sequence preparation is already running.")
                return
            if self.communicator.is_collecting() and not ALLOW_PREPARE_DURING_COLLECTION:
                messagebox.showwarning(
                    "Collection Active",
                    "Cannot prepare a new sequence while collection is active. Stop collection first.",
                )
                return
            bounds = self._read_bounds()
            anchor = self._read_anchor_from_manual(bounds)
            self.manual_idle_action = anchor.copy()
            self.manual_idle_override_enabled = True
            csv_path = self.csv_path_var.get().strip()

            self.collector_ready = False
            self.preparing = True
            self._prepare_job_id += 1
            job_id = self._prepare_job_id
            self.status_var.set("Preparing sequence in background (lower priority worker)...")
            self.prepare_summary_var.set("Preparing sequence; applying safety filter...")
            self.cycles_remaining_var.set("Cycles remaining: --")

            self._prepare_thread = threading.Thread(
                target=self._prepare_sequence_worker,
                args=(job_id, csv_path, bounds.low.copy(), bounds.high.copy(), anchor.copy()),
                daemon=True,
            )
            self._prepare_thread.start()
        except Exception as exc:
            self.collector_ready = False
            messagebox.showerror("Prepare Error", str(exc))

    def _prepare_sequence_worker(
        self,
        job_id: int,
        csv_path: str,
        bounds_low: np.ndarray,
        bounds_high: np.ndarray,
        anchor: np.ndarray,
    ) -> None:
        try:
            import os

            os.nice(int(PREP_WORKER_NICE_INCREMENT))
        except Exception:
            pass
        try:
            action_bounds = ActionBounds(
                low=np.asarray(bounds_low, dtype=np.float32),
                high=np.asarray(bounds_high, dtype=np.float32),
            )
            action_bounds.validate()

            local_generator = RandomSequenceGenerator(
                sequence_config=self.sequence_config,
                payload_config=self.payload_config,
            )
            local_filter = KNNSafetyFilter(
                csv_path=csv_path,
                knn_config=self.knn_config,
                action_columns=("ID1_prev", "ID2", "SOI2"),
            )
            bundle = local_generator.generate_sequence(
                anchor=np.asarray(anchor, dtype=np.float32),
                bounds=action_bounds,
                safety_filter=local_filter,
            )
            if bundle.predicted_mprr is None:
                predicted_mprr = np.full((bundle.actions.shape[0],), np.nan, dtype=np.float32)
            else:
                predicted_mprr = np.asarray(bundle.predicted_mprr, dtype=np.float32).reshape(-1)
            if predicted_mprr.shape[0] != bundle.actions.shape[0]:
                predicted_mprr = np.full((bundle.actions.shape[0],), np.nan, dtype=np.float32)
            missing_mask = ~np.isfinite(predicted_mprr)
            if np.any(missing_mask):
                predicted_mprr[missing_mask] = local_filter.predict_mprr_batch(
                    bundle.actions[missing_mask], action_bounds
                )
            modes = [spec.mode_name for spec in bundle.specs]
            payloads = [local_generator.build_payload_from_action(action) for action in bundle.actions]
            idle_payload = local_generator.build_payload_from_action(np.asarray(anchor, dtype=np.float32))
            self.ui_queue.put(
                (
                    "prepare_done",
                    {
                        "job_id": job_id,
                        "bundle": bundle,
                        "payloads": payloads,
                        "modes": modes,
                        "predicted_mprr": predicted_mprr,
                        "idle_payload": idle_payload,
                        "safety_filter": local_filter,
                    },
                )
            )
        except Exception as exc:
            self.ui_queue.put(("prepare_error", {"job_id": job_id, "error": str(exc)}))

    def start_collection(self) -> None:
        if self.preparing:
            messagebox.showwarning("Not Ready", "Sequence is still preparing in the background.")
            return
        if not self.collector_ready:
            messagebox.showwarning("Not Ready", "Prepare the sequence first.")
            return
        try:
            self.communicator.start_collection()
            self.status_var.set("Data collection started.")
        except Exception as exc:
            messagebox.showerror("Start Error", str(exc))

    def stop_collection(self) -> None:
        self.communicator.stop_collection()
        self.status_var.set("Data collection stopped.")

    def _init_communicator(self) -> None:
        self.communicator = self.communicator_cls(
            server_ip=self.server_ip_var.get().strip(),
            server_port=int(self.server_port_var.get().strip()),
        )
        self.communicator.on_status = self._on_comm_status
        self.communicator.on_progress = self._on_comm_progress
        self.communicator.on_complete = self._on_comm_complete
        self.communicator.on_incoming = self._on_comm_incoming
        self.communicator.start()

    def reconnect_communicator(self) -> None:
        try:
            self.communicator.stop()
            self._init_communicator()
            self._sync_idle_payload_to_communicator()
            if self.prepared_payloads and self.prepared_modes:
                self.communicator.set_prepared_sequence(
                    payloads_12=self.prepared_payloads,
                    mode_names=self.prepared_modes,
                )
            self.status_var.set("Reconnected communication worker.")
        except Exception as exc:
            messagebox.showerror("Reconnect Error", str(exc))

    def _on_comm_status(self, message: str) -> None:
        self.ui_queue.put(("status", message))

    def _on_comm_progress(self, progress: CommProgress) -> None:
        self.ui_queue.put(("progress", progress))

    def _on_comm_complete(self) -> None:
        self.ui_queue.put(("complete", None))

    def _on_comm_incoming(self, incoming: CommIncoming) -> None:
        self.ui_queue.put(("incoming", incoming))

    def apply_config_overrides(self, show_message: bool = True) -> None:
        try:
            self.sequence_config = SequenceConfig(
                base_anchor_hold_cycles=int(self.sequence_config_vars["base_anchor_hold_cycles"].get()),
                single_action_cycles=int(self.sequence_config_vars["single_action_cycles"].get()),
                pair_action_cycles=int(self.sequence_config_vars["pair_action_cycles"].get()),
                full_3d_cycles=int(self.sequence_config_vars["full_3d_cycles"].get()),
                full_no_injection_ratio=float(self.sequence_config_vars["full_no_injection_ratio"].get()),
                partial_no_injection_ratio=float(self.sequence_config_vars["partial_no_injection_ratio"].get()),
                zero_hold_min_cycles=int(self.sequence_config_vars["zero_hold_min_cycles"].get()),
                zero_hold_max_cycles=int(self.sequence_config_vars["zero_hold_max_cycles"].get()),
                smooth_recovery_min_cycles=int(self.sequence_config_vars["smooth_recovery_min_cycles"].get()),
                smooth_recovery_max_cycles=int(self.sequence_config_vars["smooth_recovery_max_cycles"].get()),
                min_normal_cycles_between_event_starts=int(
                    self.sequence_config_vars["min_normal_cycles_between_event_starts"].get()
                ),
                event_ratio_tolerance=float(self.sequence_config_vars["event_ratio_tolerance"].get()),
            )
            self.knn_config = KNNConfig(
                k_neighbors=int(self.knn_config_vars["k_neighbors"].get()),
                safety_threshold=float(self.knn_config_vars["safety_threshold"].get()),
                max_normalized_mean_distance=float(self.knn_config_vars["max_normalized_mean_distance"].get()),
                max_resample_attempts_per_cycle=int(self.knn_config_vars["max_resample_attempts_per_cycle"].get()),
            )
            self.generator = RandomSequenceGenerator(
                sequence_config=self.sequence_config,
                payload_config=self.payload_config,
            )
            if show_message:
                self.status_var.set("Config overrides applied.")
        except Exception as exc:
            if show_message:
                messagebox.showerror("Config Error", str(exc))
            raise

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                event_type, payload = self.ui_queue.get_nowait()
                if event_type == "status":
                    self.status_var.set(str(payload))
                elif event_type == "progress":
                    progress: CommProgress = payload
                    self.cycles_remaining_var.set(f"Cycles remaining: {progress.cycles_remaining}")
                elif event_type == "complete":
                    self.status_var.set("Data collection complete.")
                    self.cycles_remaining_var.set("Cycles remaining: 0")
                elif event_type == "incoming":
                    incoming: CommIncoming = payload
                    self.incoming_values_var.set(
                        f"LabVIEW incoming: {incoming.value_1:.2f}, {incoming.value_2:.2f}"
                    )
                elif event_type == "prepare_done":
                    data = payload
                    job_id = int(data["job_id"])
                    if job_id != self._prepare_job_id:
                        continue

                    bundle: SequenceBundle = data["bundle"]
                    payloads: list[np.ndarray] = data["payloads"]
                    modes: list[str] = data["modes"]
                    predicted_mprr: np.ndarray = data["predicted_mprr"]
                    idle_payload: np.ndarray = data["idle_payload"]
                    safety_filter: KNNSafetyFilter = data["safety_filter"]

                    self.sequence_bundle = bundle
                    self.filtered_actions = bundle.actions
                    self.predicted_mprr = np.asarray(predicted_mprr, dtype=np.float32)
                    self.prepared_payloads = payloads
                    self.prepared_modes = modes
                    self.prepared_idle_payload = idle_payload
                    self.safety_filter = safety_filter
                    self.collector_ready = True
                    self.preparing = False

                    self._sync_idle_payload_to_communicator()
                    self.communicator.set_prepared_sequence(payloads_12=payloads, mode_names=modes)

                    total_cycles = len(payloads)
                    summary = (
                        f"Prepared cycles: {total_cycles} | "
                        f"Base: {int(bundle.summary['base_cycles'])} | "
                        f"Full-family: {bundle.summary['full_family_ratio']*100:.1f}% | "
                        f"Partial-family: {bundle.summary['partial_family_ratio']*100:.1f}% | "
                        f"Base replaced: {int(bundle.summary['replaced_base_cycles'])}"
                    )
                    self.prepare_summary_var.set(summary)
                    self.cycles_remaining_var.set(f"Cycles remaining: {total_cycles}")
                    self.status_var.set("Sequence prepared and safety-filtered. Ready to start data collection.")
                elif event_type == "prepare_error":
                    data = payload
                    job_id = int(data["job_id"])
                    if job_id != self._prepare_job_id:
                        continue
                    self.preparing = False
                    self.collector_ready = False
                    self.status_var.set("Sequence preparation failed.")
                    messagebox.showerror("Prepare Error", str(data["error"]))
        except Empty:
            pass
        self.root.after(120, self._poll_ui_queue)

    def show_sequence_preview(self) -> None:
        if self.preparing:
            messagebox.showinfo("Preview Not Ready", "Sequence is still preparing. Please wait.")
            return
        if self.filtered_actions is None or self.sequence_bundle is None or self.predicted_mprr is None:
            messagebox.showwarning("Preview Not Ready", "Prepare the sequence first.")
            return

        actions = np.asarray(self.filtered_actions, dtype=np.float32)
        mprr = np.asarray(self.predicted_mprr, dtype=np.float32)
        specs = list(self.sequence_bundle.specs)
        if len(actions) != len(specs) or len(actions) != len(mprr):
            messagebox.showerror(
                "Preview Error",
                "Prepared sequence data is inconsistent. Please prepare sequence again.",
            )
            return

        popup = tk.Toplevel(self.root)
        popup.title("Prepared Sequence Preview")
        popup.geometry("980x520")
        popup.transient(self.root)
        popup.grab_set()

        top_frame = tk.Frame(popup, padx=10, pady=10)
        top_frame.pack(fill="both", expand=True)

        threshold = float(self.knn_config.safety_threshold)
        summary_text = (
            f"Rows: {len(actions)}    "
            f"Mean predicted MPRR: {float(np.mean(mprr)):.3f}    "
            f"Max predicted MPRR: {float(np.max(mprr)):.3f}    "
            f"Safety threshold: {threshold:.3f}"
        )
        tk.Label(top_frame, text=summary_text, anchor="w", justify="left").pack(fill="x", pady=(0, 8))

        columns = ("cycle", "mode", "id1", "id2", "soi2", "mprr", "safe")
        tree = ttk.Treeview(top_frame, columns=columns, show="headings", height=18)
        tree.heading("cycle", text="Cycle")
        tree.heading("mode", text="Mode")
        tree.heading("id1", text="ID1")
        tree.heading("id2", text="ID2")
        tree.heading("soi2", text="SOI2")
        tree.heading("mprr", text="Predicted MPRR")
        tree.heading("safe", text="Safe?")

        tree.column("cycle", width=80, anchor="center")
        tree.column("mode", width=140, anchor="center")
        tree.column("id1", width=120, anchor="e")
        tree.column("id2", width=120, anchor="e")
        tree.column("soi2", width=120, anchor="e")
        tree.column("mprr", width=140, anchor="e")
        tree.column("safe", width=100, anchor="center")

        yscroll = ttk.Scrollbar(top_frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(top_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")

        for idx, (action, spec, pred) in enumerate(zip(actions, specs, mprr), start=1):
            pred_val = float(pred)
            tree.insert(
                "",
                "end",
                values=(
                    idx,
                    spec.mode_name,
                    f"{float(action[0]):.5f}",
                    f"{float(action[1]):.5f}",
                    f"{float(action[2]):.5f}",
                    f"{pred_val:.5f}",
                    "YES" if pred_val <= threshold else "NO",
                ),
            )

    def on_close(self) -> None:
        self.communicator.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = V2App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

