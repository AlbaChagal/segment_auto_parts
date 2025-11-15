from PIL import Image
import torch.nn
import torchvision.transforms as T

from config import Config
from logger import Logger


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
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=config.preprocessor_logging_level)
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
        self.logger.debug(f'forward - preprocessing image with size {img.size}')
        img: torch.Tensor = self.transform(img)
        self.logger.debug(f'forward - preprocessed image tensor shape {img.shape}')
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
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=config.preprocessor_logging_level)
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
        self.logger.debug(f'forward - preprocessing mask with size {mask.size}')
        mask: torch.Tensor = self.transform(mask).squeeze(0).long()
        self.logger.debug(f'forward - preprocessed mask tensor shape {mask.shape}')
        return mask


if __name__ == '__main__':
    config: Config = Config()
    img_preprocessor: ImagePreprocessor = ImagePreprocessor(config=config)
    mask_preprocessor: MaskPreprocessor = MaskPreprocessor(config=config)

    dummy_image: Image = Image.new('RGB', (500, 500))
    dummy_mask: Image = Image.new('L', (500, 500))

    preprocessed_img: torch.Tensor = img_preprocessor(dummy_image)
    preprocessed_mask: torch.Tensor = mask_preprocessor(dummy_mask)

    print(f'Preprocessed image shape: {preprocessed_img.shape}')
    print(f'Preprocessed mask shape: {preprocessed_mask.shape}')