import os
from typing import List, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset

from augmenter import Augmenter
from config import Config
from data_structures import AugmentationProbabilities
from preprocessors import ImagePreprocessor, MaskPreprocessor
from logger import Logger


class CarPartsDataset(Dataset):
    def __init__(self,
                 config: Config,
                 images_dir: str,
                 masks_dir: str,
                 size: Tuple[int, int],
                 is_augment: bool = False):
        self.config: Config = config
        self.logger: Logger = Logger(name=self.__class__.__name__,
                                     logging_level=config.dataset_logging_level)
        self.images: List[str] = sorted(os.listdir(images_dir))
        self.images_dir: str = images_dir
        self.masks_dir: str = masks_dir
        self.size: Tuple[int, int] = size
        self.is_augment: bool = is_augment

        # Modules
        self.img_preprocessor: ImagePreprocessor = \
            ImagePreprocessor(self.config)
        self.mask_preprocessor: MaskPreprocessor = \
            MaskPreprocessor(self.config)
        self.augmenter: Augmenter = Augmenter(
            config=self.config,
            augmentation_probabilities=AugmentationProbabilities(),
            out_size=size
        ) if is_augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img: Union[torch.Tensor, Image] = \
            Image.open(os.path.join(self.images_dir, self.images[idx]))

        mask_name: str = os.path.splitext(self.images[idx])[0] + ".png"
        mask: Union[torch.Tensor, Image] = \
            Image.open(os.path.join(self.masks_dir, mask_name))

        if self.is_augment:
            img, mask = self.augmenter(img, mask)

        img = self.img_preprocessor(img)
        mask = self.mask_preprocessor(mask)

        # remap raw pixel values -> contiguous class indices [0..5]
        lut: torch.Tensor = torch.full((256,), -1, dtype=torch.long)
        for idx, val in enumerate(list(self.config.class_names_to_labels.values())):
            lut[val] = idx
        mask = lut[mask]  # [H,W] in [0..5], -1 if unknown

        # if any -1 slipped in, clamp to background (0) to keep CE happy; metrics will also guard
        if (mask < 0).any():
            mask = mask.clamp_min(0)

        return img, mask


    def set_is_augment(self, is_augment: bool):
        self.is_augment: bool = is_augment