import tkinter as tk

from v2_app import V2App
from v2_labview_comm_dummy import DummyLabVIEWCommunicator


class V2AppDummy(V2App):
    def __init__(self, root: tk.Tk):
        super().__init__(
            root,
            communicator_cls=DummyLabVIEWCommunicator,
            window_title="LabVIEW Collector v2 (Dummy Preview)",
        )
        self.status_var.set(
            "Dummy preview mode: no TCP connection. Configure SOI1 in Config Overrides -> PayloadConfig -> soi1_t1."
        )


def main() -> None:
    root = tk.Tk()
    V2AppDummy(root)
    root.mainloop()


if __name__ == "__main__":
    main()
