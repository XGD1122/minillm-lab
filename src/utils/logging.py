"""Training metric logging — TensorBoard with CSV fallback."""

import os
import csv

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    SummaryWriter = None
    _has_tensorboard = False


class CSVLogger:
    """Minimal CSV-based metric logger as tensorboard fallback."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "metrics.csv")
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "tag", "value"])
        self._file.flush()

    def add_scalar(self, tag: str, value: float, step: int):
        self._writer.writerow([step, tag, value])
        self._file.flush()

    def close(self):
        self._file.close()


class TrainingLogger:
    """Logger that uses TensorBoard if available, otherwise CSV."""

    def __init__(self, log_dir: str):
        if _has_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
            self._csv = None
        else:
            print("  [info] tensorboard not installed, using CSV logger")
            self.writer = CSVLogger(log_dir)
            self._csv = self.writer
        self._step = 0

    def log_scalar(self, tag: str, value: float, step: int | None = None):
        if step is None:
            step = self._step
        self.writer.add_scalar(tag, value, step)

    def set_step(self, step: int):
        self._step = step

    def close(self):
        self.writer.close()
