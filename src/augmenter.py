import random
from typing import Tuple, List, Union
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as F
from config import Config
from logger import Logger
from data_structures import AugmentationProbabilities


class Augmenter:
    """
    Paired, object-preserving augmentations for (image, mask).
    Use in training only. Call with (PIL.Image RGB, PIL.Image L) -> (PIL.Image RGB, PIL.Image L).
    """
    def __init__(
        self,
        config: Config,
        augmentation_probabilities: AugmentationProbabilities,
        out_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initializes the Augmenter with the given configuration and augmentation probabilities.
        1. Sets up resize transforms for images and masks.
        2. Initializes random states for reproducibility.
        3. Stores augmentation probabilities for various transformations.
        :param config: Configuration object containing augmentation settings.
        :param augmentation_probabilities: Object containing probabilities for each augmentation type.
        :param out_size: Desired output size (height, width) for images and masks.
        :return: None
        """
        self.logger: Logger = Logger(self.__class__.__name__,
                                     config.augmenter_logging_level)
        self.img_resize_transform: T.Resize = T.Resize(out_size)
        self.mask_resize_transform: T.Resize = T.Resize(out_size,
                                                        interpolation=F.InterpolationMode.NEAREST)
        self.config: Config = config
        self.random_state: random.Random = \
            random.Random(self.config.random_seed)
        self.np_random_state: np.random.RandomState = \
            np.random.RandomState(self.config.random_seed)
        self.out_h: float
        self.out_w: float
        self.out_h, self.out_w = out_size
        self.crop_prob: float = augmentation_probabilities.crop
        self.hflip_prob: float = augmentation_probabilities.horizontal_flip
        self.vflip_prob: float = augmentation_probabilities.vertical_flip
        self.perspective_prob: float = augmentation_probabilities.perspective
        self.brightness_prob: float = augmentation_probabilities.brightness
        self.contrast_prob: float = augmentation_probabilities.contrast
        self.saturation_prob: float = augmentation_probabilities.saturation
        self.blur_prob: float = augmentation_probabilities.blur
        self.noise_prob: float = augmentation_probabilities.noise
        self.noise_sigma_prob: float = augmentation_probabilities.noise_sigma
        self.min_pos_frac_prob: float = augmentation_probabilities.min_pos_frac
        self.crop_attempts_prob: float = augmentation_probabilities.crop_attempts

    def __call__(self, img: Union[Image.Image, torch.Tensor], mask: Union[Image.Image, torch.Tensor]):
        """
        Applies a series of random augmentations to the input image and mask.
        The augmentations include resizing, flipping, affine transformations,
        perspective transformations, object-preserving cropping, and photometric
        adjustments. The function ensures that the mask values remain valid
        class labels after transformations.
        :param img: The input image.
        :param mask: The corresponding segmentation mask.
        :return: A tuple of augmented (image, mask).
        """
        self.logger.debug(f'Starting augmentation pipeline, got image with '
                          f'size {img.size} and mask with size {mask.size}')
        img: Union[Image.Image, torch.Tensor] = self.img_resize_transform(img)
        mask: Union[Image.Image, torch.Tensor] = self.mask_resize_transform(mask)

        # random subset of ops each call; order is fixed for stability
        if self.random_state.random() < self.hflip_prob:
            self.logger.debug(f'Applying horizontal flip augmentation')
            img, mask = F.hflip(img), F.hflip(mask)
        if self.random_state.random() < self.vflip_prob:
            self.logger.debug(f'Applying vertical flip augmentation')
            img, mask = F.vflip(img), F.vflip(mask)

        # light affine (always-on but with small jitter)
        img, mask = self._transform_affine(img=img, mask=mask)

        # perspective (occasionally)
        if self.random_state.random() < self.perspective_prob:
            self.logger.debug(f'Applying perspective transform augmentation')
            img, mask = self._transform_prespective(img=img, mask=mask)
        # object-preserving crop to target
        if self.random_state.random() < self.crop_prob:
            self.logger.debug(f'Applying object-preserving crop augmentation')
            img, mask = self._object_preserving_crop(img, mask, (self.out_h, self.out_w))

        # photometric image-only
        if self.random_state.random() < self.brightness_prob:
            self.logger.debug(f'Applying brightness adjustment augmentation')
            img, mask = self._adjust_brightness(img, mask)
        if self.random_state.random() < self.contrast_prob:
            self.logger.debug(f'Applying contrast adjustment augmentation')
            img, mask = self._adjust_contrast(img, mask)
        if self.random_state.random() < self.saturation_prob:
            self.logger.debug(f'Applying saturation adjustment augmentation')
            img, mask = self._adjust_saturation(img, mask)
        if self.random_state.random() < self.blur_prob:
            self.logger.debug(f'Applying gaussian blur augmentation')
            img, mask = self._gaussian_blur(img, mask)
        if self.random_state.random() < self.noise_prob:
            self.logger.debug(f'Applying gaussian noise augmentation')
            img, mask = self._gaussian_noise(img, mask)

        # sanitize mask labels if any interpolation artifacts remain
        mask = self._snap_mask_to_known_values(mask)
        return img, mask

    def _adjust_brightness(
            self,
            img: Union[Image.Image, torch.Tensor],
            mask: Union[Image.Image, torch.Tensor]
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        img: Union[Image.Image, torch.Tensor] = \
            F.adjust_brightness(
                img,
                self.random_state.uniform(1 - self.brightness_prob, 1 + self.brightness_prob)
            )
        return img, mask

    def _adjust_contrast(
            self,
            img: Union[Image.Image, torch.Tensor],
            mask: Union[Image.Image, torch.Tensor]
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        img: Union[Image.Image, torch.Tensor] = \
            F.adjust_contrast(
                img,
                self.random_state.uniform(1 - self.contrast_prob, 1 + self.contrast_prob)
            )
        return img, mask

    def _adjust_saturation(
            self,
            img: Union[Image.Image, torch.Tensor],
            mask: Union[Image.Image, torch.Tensor]
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        img: Union[Image.Image, torch.Tensor] = \
            F.adjust_saturation(
                img,
                self.random_state.uniform(1 - self.saturation_prob, 1 + self.saturation_prob)
            )
        return img, mask

    @staticmethod
    def _gaussian_blur(
            img: Union[Image.Image, torch.Tensor],
            mask: Union[Image.Image, torch.Tensor]
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        k: int = max(3, int(0.01 * min(img.size)) | 1)
        img: Union[Image.Image, torch.Tensor] = F.gaussian_blur(img, kernel_size=k)
        return img, mask

    def _gaussian_noise(
            self,
            img: Union[Image.Image, torch.Tensor],
            mask: Union[Image.Image, torch.Tensor]
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        arr: np.ndarray = np.asarray(img, dtype=np.float32) / 255.0
        noise: np.ndarray = \
            self.np_random_state.randn(*arr.shape).astype(np.float32) * self.noise_sigma_prob
        arr = np.clip(arr + noise, 0.0, 1.0)
        img: Union[torch.Tensor, Image.Image] = \
            Image.fromarray((arr * 255.0).astype(np.uint8))
        return img, mask

    def _transform_affine(self, img: Union[torch.Tensor, Image.Image], mask: Union[torch.Tensor, Image.Image]):
        angle: float = self.random_state.uniform(-10, 10)
        tx: float = self.random_state.uniform(-0.08, 0.08)
        ty: float = self.random_state.uniform(-0.08, 0.08)
        scale: float = self.random_state.uniform(0.8, 1.15)
        shear: float = self.random_state.uniform(-10, 10)

        img_h: int = img.height if isinstance(img, Image.Image) else img.shape[1]
        img_w: int = img.width if isinstance(img, Image.Image) else img.shape[2]
        mask_h: int = mask.height if isinstance(mask, Image.Image) else mask.shape[0]
        mask_w: int = mask.width if isinstance(mask, Image.Image) else mask.shape[1]
        img: Union[torch.Tensor, Image.Image] = F.affine(
            img,
            angle,
            translate=[int(tx * img_w), int(ty * img_h)],
            scale=scale,
            shear=[shear, shear],
            interpolation=F.InterpolationMode.BILINEAR,
            fill=0
        )
        mask: Union[torch.Tensor, Image.Image] = F.affine(
            mask,
            angle,
            translate=[int(tx * mask_w), int(ty * mask_h)],
            scale=scale,
            shear=[shear, shear],
            interpolation=F.InterpolationMode.NEAREST,
            fill=0
        )
        return img, mask

    def _transform_prespective(
            self,
            img: Union[torch.Tensor, Image.Image],
            mask: Union[torch.Tensor, Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a random perspective transformation to the image and mask.
        :param img: The input image.
        :param mask: The corresponding segmentation mask.
        :return: A tuple of transformed (image, mask).
        """
        h: int
        w: int
        dx: int
        dy: int
        w, h = img.size
        dx, dy = int(0.2 * w * 0.5), int(0.2 * h * 0.5)
        start: List[List[int]] = [[0, 0], [w, 0], [w, h], [0, h]]
        end: List[List[int]] = [
            [self.random_state.randint(0, dx), self.random_state.randint(0, dy)],
            [self.random_state.randint(w - dx, w), self.random_state.randint(0, dy)],
            [self.random_state.randint(w - dx, w), self.random_state.randint(h - dy, h)],
            [self.random_state.randint(0, dx), self.random_state.randint(h - dy, h)]
        ]
        img: Union[torch.Tensor, Image.Image] = \
            F.perspective(img, start, end, interpolation=F.InterpolationMode.BILINEAR, fill=0)
        mask: Union[torch.Tensor, Image.Image] = \
            F.perspective(mask, start, end, interpolation=F.InterpolationMode.NEAREST, fill=0)

        return img, mask

    def _object_preserving_crop(
            self,
            img: Image.Image,
            mask: Image.Image,
            size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Object-preserving random crop: attempts N times to find a crop of given size
        that contains at least min_pos_frac_prob fraction of positive pixels in the mask.
        :param img: The input image.
        :param mask: The corresponding segmentation mask.
        :param size: The desired output size (height, width).
        :return: A tuple of cropped (image, mask).
        """
        th: int
        tw: int
        h: int
        w: int
        th, tw = size
        w, h = img.size
        if w < tw or h < th:
            pad_h: int
            pad_w: int
            pad_w, pad_h = max(0, tw - w), max(0, th - h)
            pad: Tuple[int, int, int, int] = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = F.pad(img, pad, fill=0)
            mask = F.pad(mask, pad, fill=0)
            w, h = img.size
        area_thresh = self.min_pos_frac_prob * th * tw

        i: int
        j: int
        m_crop: torch.Tensor
        for _ in range(self.crop_attempts_prob):
            i = self.random_state.randint(0, h - th)
            j = self.random_state.randint(0, w - tw)
            m_crop = F.crop(mask, i, j, th, tw)
            if np.count_nonzero(np.asarray(m_crop, dtype=np.uint8)) >= area_thresh:
                return F.crop(img, i, j, th, tw), m_crop
        # fallback center/resize
        img: torch.Tensor = F.center_crop(img, (th, tw))
        mask: torch.Tensor = F.center_crop(mask, (th, tw))
        return img, mask

    def _snap_mask_to_known_values(self, mask: Image.Image) -> Image.Image:
        """
        Snaps mask pixel values to the nearest known class label values to
        correct any interpolation artifacts.
        :param mask: The input segmentation mask.
        :return: The corrected segmentation mask.
        """
        arr: np.ndarray = np.asarray(mask, dtype=np.uint8)
        if np.isin(arr, list(self.config.class_names_to_labels.values())).all():
            return mask
        arr16: np.ndarray = arr.astype(np.int16)
        _vals_np: np.ndarray = np.array(list(self.config.class_names_to_labels.values()), dtype=np.int16)
        d: np.ndarray = np.abs(arr16[..., None] - _vals_np[None, None, :])   # [H,W,K]
        idx: np.ndarray = d.argmin(axis=2).astype(np.int32)
        snapped: np.ndarray = _vals_np[idx].astype(np.uint8)
        return Image.fromarray(snapped)



if __name__ == "__main__":
    def test_random_augmentations(image_path: Path,
                                  mask_path: Path,
                                  out_dir: Path,
                                  augmentation_probabilities: AugmentationProbabilities,
                                  config: Config = Config(),
                                  n_samples: int = 20):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running random augmentation test -> {out_dir}")

        # Load and resize to training size
        img0 = Image.open(image_path).convert("RGB")
        msk0 = Image.open(mask_path).convert("L")
        img0 = F.resize(img0, size=list(config.image_size), interpolation=F.InterpolationMode.BILINEAR)
        msk0 = F.resize(msk0, size=list(config.image_size), interpolation=F.InterpolationMode.NEAREST)

        # Initialize augmenter with normal parameters
        aug = Augmenter(out_size=config.image_size,
                        config=config,
                        augmentation_probabilities=augmentation_probabilities)

        # Generate augmented samples
        for i in range(n_samples):
            img_aug, mask_aug = aug(img0.copy(), msk0.copy())
            img_aug.save(out_dir / f"img_aug_{i:02d}.jpg", quality=95)
            mask_aug.save(out_dir / f"mask_aug_{i:02d}.png")

        print(f"Saved {n_samples} augmented samples to {out_dir}")

    # Set these to a real sample in your dataset
    image_path_main            = Path("data/train/images/000.jpg")   # path to an existing training image
    mask_path_main             = Path("data/train/masks/000.png")    # path to the matching mask
    out_dir_main               = Path("outputs/augmenter_test/000")
    random_state_main          = random.Random(42)
    np_random_state_main       = np.random.RandomState(42)
    augmentation_probabilities = AugmentationProbabilities()
    test_random_augmentations(image_path=image_path_main,
                              mask_path=mask_path_main,
                              out_dir=out_dir_main,
                              augmentation_probabilities=augmentation_probabilities)
