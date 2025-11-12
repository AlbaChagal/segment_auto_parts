import os
import json
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter

from data_structures import SegmentationMetrics
from config import Config
from logger import Logger


class TensorBoardLogger(object):
    """
    Independent writer per split (e.g., 'train', 'val').
    Logs:
      - scalar loss
      - per-class precision/recall/IoU/Dice
      - macro averages
    """
    def __init__(self, config: Config, log_dir: str, split: str, model_id: str):
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=config.tensorboard_logger_logging_level)
        run_dir = os.path.join(log_dir, f'{split}')
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(run_dir)
        self.split = split
        self.logger.info(f'Initialized TensorBoardLogger writing to path: {run_dir}')
        self.class_names: List[str] = sorted(config.class_names_to_labels.keys(),
                                             key=lambda name: config.class_names_to_labels[name])

    def log_loss(self, loss_value: float, step: int):
        self.writer.add_scalar(f"_{self.split}/loss", loss_value, step)

    def log_metrics(self, metrics: SegmentationMetrics, step: int):
        per_cls = metrics.per_class
        macro   = metrics.macro

        # macro
        for k, v in macro.__dict__.items():
            self.logger.debug(f'log_metrics - logged macro {k} for step {step} on split {self.split}')
            self.writer.add_scalar(f"_{self.split}/{k}", float(v), step)

        # per-class
        for i, name in enumerate(self.class_names):
            self.logger.debug(f'log_metrics - logged {name} class metrics for step {step} on split {self.split}')
            self.writer.add_scalar(f"{self.split}_classes/{name}/precision",
                                   float(per_cls.precision[i]), step)
            self.writer.add_scalar(f"{self.split}_classes/{name}/recall",
                                   float(per_cls.recall[i]), step)
            self.writer.add_scalar(f"{self.split}_classes/{name}/iou",
                                   float(per_cls.iou[i]), step)
            self.writer.add_scalar(f"{self.split}_classes/{name}/dice",
                                   float(per_cls.dice[i]), step)
        self.logger.debug(f'log_metrics - logged metrics for step {step} on split {self.split}')

    def flush(self):
        self.logger.debug(f'flush - flushing TensorBoard writer for: {self.split}')
        self.writer.flush()

    def close(self):
        self.logger.debug(f'close - closing TensorBoard writer for: {self.split}')
        self.writer.close()


if __name__ == "__main__":
    os.makedirs("outputs/test/tensorboard/test", exist_ok=True)
    w = SummaryWriter("outputs/test/tensorboard/test")
    w.add_scalar("test/x", 1.0, 0)
    w.flush()
    w.close()
    print(f'Wrote test TensorBoard data to outputs/test/tensorboard/test to open with '
          f'`tensorboard --logdir outputs/test/tensorboard` just a sanity check that tensorboard works.')