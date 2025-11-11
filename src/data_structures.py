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

@dataclass
class SegmentationMetrics:
    per_class: PerClassMetrics
    macro: MacroMetrics