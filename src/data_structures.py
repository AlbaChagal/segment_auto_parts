from dataclasses import dataclass
from typing import List


@dataclass
class PerClassMetrics:
    """
    A dataclass to hold per-class segmentation metrics.
    """
    precision: List[float]
    recall: List[float]
    iou: List[float]
    dice: List[float]
    support: List[int]

@dataclass
class MacroMetrics:
    """
    A dataclass to hold macro-averaged segmentation metrics.
    """
    precision: float
    recall: float
    iou: float
    dice: float

    def __str__(self):
        return (f'precision={self.precision:.4f}, '
                f'recall={self.recall:.4f}, '
                f'iou={self.iou:.4f}, '
                f'dice={self.dice:.4f}')

@dataclass
class SegmentationMetrics:
    """
    A dataclass to hold segmentation metrics.
    """
    per_class: PerClassMetrics
    macro: MacroMetrics

@dataclass
class AugmentationProbabilities:
    """
    A dataclass to hold augmentation probabilities for different transformations.
    """
    crop: float = 0.
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.3
    perspective: float = 0.
    brightness: float = 0.
    contrast: float = 0.
    saturation: float = 0.
    blur: float = 0.
    noise: float = 0.
    noise_sigma: float = 0.
    min_pos_frac: float = 0.
    crop_attempts: int = 0

@dataclass
class TimeMetrics:
    """
    A dataclass to hold time metrics for various stages of training.
    """
    total: float = 0.0
    data: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    optimizer_step: float = 0.0
    batch: float = 0.0
    metrics: float = 0.0

    def __add__(self, other):
        return TimeMetrics(
            total=self.total + other.total,
            data=self.data + other.data,
            forward=self.forward + other.forward,
            backward=self.backward + other.backward,
            optimizer_step=self.optimizer_step + other.optimizer_step,
            batch=self.batch + other.batch,
            metrics=self.metrics + other.metrics,
        )

    def __truediv__(self, scalar: float):
        return TimeMetrics(
            total=self.total / scalar,
            data=self.data / scalar,
            forward=self.forward / scalar,
            backward=self.backward / scalar,
            optimizer_step=self.optimizer_step / scalar,
            batch=self.batch / scalar,
            metrics=self.metrics / scalar,
        )

    def __str__(self):
        return f'total: {self.total: .4f}, data: {self.data:.4f}, ' \
        f'forward: {self.forward:.4f}, step: {self.optimizer_step:.4f}, ' \
        f'batch: {self.batch:.4f}, metrics: {self.metrics:.4f}'