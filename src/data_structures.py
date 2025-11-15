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


if __name__ == "__main__":
    # Example usage of the dataclasses
    per_class_metrics = PerClassMetrics(
        precision=[0.9, 0.8, 0.85],
        recall=[0.88, 0.75, 0.9],
        iou=[0.8, 0.7, 0.75],
        dice=[0.85, 0.78, 0.82],
        support=[100, 150, 120]
    )

    macro_metrics = MacroMetrics(
        precision=0.85,
        recall=0.81,
        iou=0.75,
        dice=0.82
    )

    segmentation_metrics = SegmentationMetrics(
        per_class=per_class_metrics,
        macro=macro_metrics
    )

    print("Segmentation Metrics:")
    print(segmentation_metrics.macro)
    for i, class_name in enumerate(['class_1', 'class_2', 'class_3']):
        print(f"{class_name} - Precision: {segmentation_metrics.per_class.precision[i]:.4f}, "
              f"Recall: {segmentation_metrics.per_class.recall[i]:.4f}, "
              f"IoU: {segmentation_metrics.per_class.iou[i]:.4f}, "
              f"Dice: {segmentation_metrics.per_class.dice[i]:.4f}, "
              f"Support: {segmentation_metrics.per_class.support[i]}")
    aug_probs = AugmentationProbabilities(
        crop=0.2,
        horizontal_flip=0.5,
        vertical_flip=0.3
    )
    print("\nAugmentation Probabilities:")
    print(aug_probs)
    time_metrics = TimeMetrics(
        total=1.234,
        data=0.123,
        forward=0.456,
        backward=0.234,
        optimizer_step=0.098,
        batch=1.234,
        metrics=0.045
    )
    print("\nTime Metrics:")
    print(time_metrics)

    avg_time_metrics = time_metrics / 2
    print("\nAverage Time Metrics (divided by 2):")
    print(avg_time_metrics)

    print("\nAdded Time Metrics (time_metrics + time_metrics):")
    print(time_metrics + time_metrics)
