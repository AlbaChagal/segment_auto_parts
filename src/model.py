from typing import Optional

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg

from config import Config
from logger import Logger

class SegModel(nn.Module):
    """
    A semantic segmentation model based on DeepLabV3 with a ResNet-50 backbone.
    """
    def __init__(self, config: Config):
        """
        Initialize the SegModel.
        :param config: The configuration object.
        :return: None
        """
        super().__init__()
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=config.model_logging_level)
        self.device: torch.device = self._get_device(self.logger)
        self.model: seg.DeepLabV3 = seg.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, config.num_classes, kernel_size=1)

    @staticmethod
    def _get_device(logger: Optional[Logger] = None):
        """
        Get the available device (hierarchy: 1. GPU, 2. MPS, 3. CPU).
        :param logger:
        :return:
        """
        device: str
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
        """
        Forward pass through the segmentation model.
         1. Input: tensor of shape [N, 3, H, W]
         2. Output: tensor of shape [N, num_classes, H, W]
        :param x: Input tensor to infer
        :return: Output tensor
        """
        return self.model(x)["out"]
