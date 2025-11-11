from typing import Optional

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg

from config import Config
from logger import Logger

class SegModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=config.model_logging_level)
        self.device: torch.device = self._get_device(self.logger)
        self.model = seg.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, config.num_classes, kernel_size=1)

    @staticmethod
    def _get_device(logger: Optional[Logger] = None):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if logger:
            logger.info(f'_get_device - found {device} as device')

        return torch.device(device)

    def forward(self, x):
        return self.model(x)["out"]
