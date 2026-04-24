import numpy as np

OUTGOING_FLOAT_COUNT = 12


class InjectionSequenceGenerator:
    """
    Simple example generator that yields a different 12-value vector
    each time `next_values()` is called.

    Expected payload order:
    [timing1, duration1, timing2, duration2, ..., timing6, duration6]
    """

    def __init__(self):
        self.step = 0

    def next_values(self):
        base = np.arange(OUTGOING_FLOAT_COUNT, dtype=np.float32)
        # Example evolving sequence; replace with your real strategy.
        values = (base * 0.1) + (self.step * 0.05)
        self.step += 1
        return values.astype(np.float32)

    def reset(self):
        self.step = 0
