# Raspberry Pi LabVIEW Communication

Python TCP client and control GUI for closed-loop communication with a LabVIEW test cell. Each engine cycle, the application receives feedback from LabVIEW, computes injection setpoints (manual or trajectory-based), and returns a binary payload with timing/duration values and an execution mode code.

## Features

- TCP client that connects to a LabVIEW server on a fixed IP and port
- Manual control of six injection events (timing and duration pairs)
- Automated 3D trajectory exploration over injection duration 1, injection duration 2, and SOI2
- MPRR-based soft limits, derate, boundary control, and hard aborts
- E-stop to zero outputs and return to manual mode
- Optional offline GUI preview without a network connection

## Requirements

- Python 3.8 or newer
- [NumPy](https://numpy.org/)
- Tkinter (included with most Python installs on Raspberry Pi OS and desktop distributions)

## Installation

```bash
git clone https://github.com/rhadlich/RPI-Comm.git
cd RPI-Comm
pip install numpy
```

## Usage

### Main application (TCP + GUI)

Edit `SERVER_IP` and `SERVER_PORT` at the top of `LabVIEW_comm_test.py` to match your LabVIEW host, then run:

```bash
python3 LabVIEW_comm_test.py
```

1. Set manual injection values or configure trajectory bounds and mode probabilities in the GUI.
2. Click **Start Communication** to connect to the LabVIEW server.
3. Use **Generate Trajectory** to plan and run an automated exploration sequence, or stay in manual mode.
4. Use **E-STOP** to halt the trajectory, zero outputs, and restore manual control.
5. Click **Stop Communication** before closing the window.

### Trajectory GUI preview (no TCP)

To exercise trajectory planning and the popup workflow without LabVIEW:

```bash
python3 LabVIEW_comm_dummy_gui_preview.py
```

## TCP protocol

Communication uses a simple request/response loop per cycle over TCP.

| Direction | Content |
|-----------|---------|
| LabVIEW to Pi | 2 big-endian 32-bit floats (8 bytes). Index 1 is treated as the latest MPRR. |
| Pi to LabVIEW | 13 big-endian 32-bit floats (52 bytes). |

**Outgoing payload (13 floats)**

- Floats 0–11: six injection events as `[T1, D1, T2, D2, …, T6, D6]`
- Float 12: mode code

| Mode | Code |
|------|------|
| WARMUP | 0 |
| EXPLORE | 1 |
| DERATE | 2 |
| ABORT | 3 |
| RECOVER | 4 |
| HOLD | 5 |
| NO_INJECTION | 6 |

## Project layout

| File | Description |
|------|-------------|
| `LabVIEW_comm_test.py` | Main Tkinter GUI and TCP communication loop |
| `injection_sequence_generator.py` | Trajectory planning in action space `[D1, D2, SOI2]` |
| `LabVIEW_comm_dummy_gui_preview.py` | Standalone GUI preview without network I/O |

## Platform notes

On Linux (including Raspberry Pi), the TCP worker thread attempts `SCHED_FIFO` real-time scheduling for lower jitter. That requires root or:

```bash
sudo setcap cap_sys_nice+ep $(which python3)
```

On macOS and other systems without `sched_setscheduler`, scheduling falls back to normal priority with a status message in the GUI.

## License

No license file is included in this repository. Contact the maintainer for terms of use before redistribution.
