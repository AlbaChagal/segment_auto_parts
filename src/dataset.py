from typing import List, Tuple

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class CarPartsDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, size: Tuple[int, int], augment: bool = False):
        self.images: List[str] = sorted(os.listdir(images_dir))
        self.images_dir: str = images_dir
        self.masks_dir: str = masks_dir
        self.size: Tuple[int, int] = size
        self.augment = augment
        self.transform_img = T.Compose([
            T.Resize(size), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.transform_mask = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.NEAREST), T.PILToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.images[idx]))

        mask_name = os.path.splitext(self.images[idx])[0] + ".png"
        mask = Image.open(os.path.join(self.masks_dir, mask_name))

        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze(0).long()
        return img, mask
