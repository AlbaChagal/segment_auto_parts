from PIL import Image
import torch.nn
import torchvision.transforms as T

from config import Config


class ImagePreprocessor(torch.nn.Module):
    """
    Preprocesses input images: resizes, converts to tensor, normalizes.
    1. Resize to config.image_size
    2. Convert to tensor
    3. Normalize using ImageNet mean and std
    4. Returns preprocessed tensor
    5. Input: PIL Image
    6. Output: torch.Tensor
    7. Usage: preprocessor = ImagePreprocessor(config); tensor = preprocessor(image
    """
    def __init__(self, config: Config):
        """
        Initializes the ImagePreprocessor with the given configuration.
        :param config: Configuration object containing image_size.
        """
        super().__init__()
        self.config: Config = config
        self.resize: T.Resize = T.Resize(self.config.image_size)
        self.to_tensor: T.ToTensor = T.ToTensor()
        # Original ImageNet normalization values
        self.normalize: T.Normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        self.transform: T.Compose = T.Compose([
            self.resize,
            self.to_tensor,
            self.normalize
        ])

    def forward(self, img: Image) -> torch.Tensor:
        """
        Preprocesses the input image.
        1. Resize to config.image_size
        2. Convert to tensor
        3. Normalize using ImageNet mean and std
        :param img: Input PIL Image.
        :return: Preprocessed image tensor.
        """
        img: torch.Tensor = self.transform(img)
        return img

class MaskPreprocessor(torch.nn.Module):
    """
    Preprocesses segmentation masks: resizes, converts to tensor, converts to long.
    1. Resize to config.image_size using nearest neighbor
    2. Convert to tensor
    3. Convert to long dtype
    4. Returns preprocessed mask tensor
    5. Input: PIL Image
    6. Output: torch.Tensor
    7. Usage: preprocessor = MaskPreprocessor(config); tensor = preprocessor(mask)
    """
    def __init__(self, config: Config):
        """
        Initializes the MaskPreprocessor with the given configuration.
        :param config: Configuration object containing image_size.
        """
        super().__init__()
        self.config: Config = config
        self.resize: T.Resize = T.Resize(self.config.image_size,
                                         interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor: T.PILToTensor = T.PILToTensor()

        self.transform: T.Compose = T.Compose([
            self.resize,
            self.to_tensor,
        ])

    def forward(self, mask: Image) -> torch.Tensor:
        """
        Preprocesses the input segmentation mask.
        1. Resize to config.image_size using nearest neighbor
        2. Convert to tensor
        3. Convert to long dtype
        :param mask: Input PIL Image.
        :return: Preprocessed mask tensor.
        """
        mask: torch.Tensor = self.transform(mask).squeeze(0).long()
        return mask
