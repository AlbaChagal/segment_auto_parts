from typing import Tuple


class Config:

    num_classes: int = 6
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4
    lr: float = 1e-4
    num_epochs: int = 30
    model_path: str = "weights/model.pth"
    data_dir: str = "data/train/"

    # Logging
    model_logging_level = 'info'
    trainer_logging_level = 'info'