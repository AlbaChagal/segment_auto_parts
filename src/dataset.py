from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

from preprocessors import ImagePreprocessor, MaskPreprocessor
from config import Config
from logger import Logger


class CarPartsDataset(Dataset):
    def __init__(self, config: Config, images_dir: str, masks_dir: str, size: Tuple[int, int], augment: bool = False):
        self.config: Config = config
        self.logger: Logger = Logger(name=self.__class__.__name__,
                                     logging_level=config.dataset_logging_level)

        self.images: List[str] = sorted(os.listdir(images_dir))
        self.images_dir: str = images_dir
        self.masks_dir: str = masks_dir
        self.size: Tuple[int, int] = size
        self.augment = augment
        self.img_preprocessor = ImagePreprocessor(self.config)
        self.mask_preprocessor = MaskPreprocessor(self.config)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.images[idx]))

        mask_name = os.path.splitext(self.images[idx])[0] + ".png"
        mask = Image.open(os.path.join(self.masks_dir, mask_name))

        img = self.img_preprocessor(img)
        mask = self.mask_preprocessor(mask)

        # remap raw pixel values -> contiguous class indices [0..5]
        lut = torch.full((256,), -1, dtype=torch.long)
        for idx, val in enumerate([0, 32, 64, 96, 128, 160]):
            lut[val] = idx
        mask = lut[mask]  # [H,W] in [0..5], -1 if unknown

        # if any -1 slipped in, clamp to background (0) to keep CE happy; metrics will also guard
        if (mask < 0).any():
            mask = mask.clamp_min(0)

        return img, mask
