from dataclasses import dataclass
from typing import List


@dataclass
class PerClassMetrics:
    precision: List[float]
    recall: List[float]
    iou: List[float]
    dice: List[float]
    support: List[int]

@dataclass
class MacroMetrics:
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
    per_class: PerClassMetrics
    macro: MacroMetrics

@dataclass
class TimeMetrics:
    total: float = 0.0
    data: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    step: float = 0.0
    batch: float = 0.0
    metrics: float = 0.0

    def __add__(self, other):
        return TimeMetrics(
            total=self.total + other.total,
            data=self.data + other.data,
            forward=self.forward + other.forward,
            backward=self.backward + other.backward,
            step=self.step + other.step,
            batch=self.batch + other.batch,
            metrics=self.metrics + other.metrics,
        )

    def __truediv__(self, scalar: float):
        return TimeMetrics(
            total=self.total / scalar,
            data=self.data / scalar,
            forward=self.forward / scalar,
            backward=self.backward / scalar,
            step=self.step / scalar,
            batch=self.batch / scalar,
            metrics=self.metrics / scalar,
        )

    def __str__(self):
        return f'total: {self.total: .4f}, data: {self.data:.4f}, ' \
        f'forward: {self.forward:.4f}, step: {self.step:.4f}, ' \
        f'batch: {self.batch:.4f}, metrics: {self.metrics:.4f}'