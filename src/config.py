from typing import Tuple, List, Dict


class Config:

    # General params
    random_seed: int = 42

    # Training params
    batch_size: int = 4
    val_batch_size: int = 1
    lr: float = 1e-4
    num_epochs: int = 30
    # Class weights are computed using median frequency
    # balancing with the script 'count_label_distribution.py'
    class_weights: List[float] = \
        [0.130084, 0.722525, 0.835331, 3.642700, 1.245533, 21.490470]


    # Path params
    outputs_folder_name: str = "outputs"
    weight_folder_name: str = "weights"
    tensorboard_folder_name: str = "tensorboard"
    data_dir: str = "data/train/"

    # Data params
    image_size: Tuple[int, int] = (512, 512)
    num_classes: int = 6
    val_percentage: float = 0.1
    class_names_to_labels: Dict[str, int] = {"background": 0, "front_door": 32, "rear_door": 64,
                                             "front_fender": 96, "rear_fender": 128, "door_handle": 160}

    # Logging
    model_logging_level = 'info'
    trainer_logging_level = 'info'
    metrics_logging_level = 'info'
    tensorboard_logger_logging_level = 'info'
    train_logging_freq: int = 50
    val_logging_freq: int = 200