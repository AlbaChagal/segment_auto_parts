from PIL import Image
import torch.nn
import torchvision.transforms as T

from config import Config


class ImagePreprocessor(torch.nn.Module):
    def __init__(self, config: Config):
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
        img: torch.Tensor = self.transform(img)
        return img

class MaskPreprocessor(torch.nn.Module):
    def __init__(self, config: Config):
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
        mask: torch.Tensor = self.transform(mask).squeeze(0).long()
        return mask
