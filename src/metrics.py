import torch

from config import Config
from data_structures import PerClassMetrics, MacroMetrics, SegmentationMetrics
from logger import Logger
from model import SegModel


class StreamingSegMetrics:
    """
    Online confusion-matrix metrics for K classes.
    Computes per-class: TP, FP, FN, precision, recall, IoU, Dice(F1).
    """
    def __init__(self, config: Config, ignore_index: int = -1):
        self.config: Config = config
        self.device: torch.device = torch.device('cpu')
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=config.metrics_logging_level)
        self.num_classes: int = self.config.num_classes
        self.ignore_index: int = ignore_index
        self.confusion_matrix: torch.Tensor = torch.zeros((self.num_classes, self.num_classes),
                                                          dtype=torch.long).to(self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        A single update step for the confusion matrix.
        :param preds:  [N,C,H,W] logits OR [N,H,W] class indices
        :param targets:[N,H,W]   class indices in [0..K-1]
        """

        self.logger.debug(f'update - updating confusion matrix with preds shape {preds.shape} '
                         f'and targets shape {targets.shape}')
        if preds.dim() == 4:
            preds = preds.argmax(1)
        preds = preds.reshape(-1).to(torch.int64, non_blocking=False).cpu()
        targets = targets.reshape(-1).to(torch.int64, non_blocking=False).cpu()

        keep = (targets >= 0) & (targets < self.num_classes)
        preds = preds[keep].clamp_(0, self.num_classes - 1)
        targets = targets[keep]

        k: int = self.num_classes
        inds: torch.Tensor = targets * k + preds
        self.logger.debug(f'update - computed inds with shape {inds.shape}, '
                          f'target shape: {targets.shape}, '
                          f'preds shape: {preds.shape}, '
                          f'k={k}')

        cm: torch.Tensor = torch.bincount(inds, minlength=k*k).reshape(k, k)
        self.logger.debug(f'update - computed batch confusion matrix with shape {cm.shape}')
        self.confusion_matrix += cm.to(self.confusion_matrix.dtype)

    @torch.no_grad()
    def compute(self) -> SegmentationMetrics:
        self.logger.debug(f'compute - computing segmentation metrics from confusion matrix')
        cm: torch.Tensor = self.confusion_matrix.to(torch.float32)
        tp: torch.Tensor = cm.diag()
        fp: torch.Tensor = cm.sum(0) - tp
        fn: torch.Tensor = cm.sum(1) - tp
        denominator_precision: torch.Tensor = tp + fp
        denominator_recall: torch.Tensor  = tp + fn
        denominator_iou: torch.Tensor  = denominator_precision + fn  # tp + fp + fn

        eps: float = 1e-8
        precision: torch.Tensor = tp / (denominator_precision + eps)
        recall: torch.Tensor    = tp / (denominator_recall  + eps)
        iou: torch.Tensor       = tp / (denominator_iou  + eps)
        dice: torch.Tensor      = (2 * tp) / ((2 * tp) + fp + fn + eps)

        self.logger.debug(f'compute - computed per-class metrics: '
                          f'tp: {tp}, fp: {fp}, fn: {fn}, precision: {precision}, '
                          f'recall: {recall}, iou: {iou}, dice: {dice}')

        per_class: PerClassMetrics = PerClassMetrics(
            precision=precision.tolist(),
            recall=   recall.tolist(),
            iou=      iou.tolist(),
            dice=     dice.tolist(),
            support=  cm.sum(1).tolist(),
        )

        macro: MacroMetrics = MacroMetrics(
            precision=precision.mean().item(),
            recall=   recall.mean().item(),
            iou=      iou.mean().item(),
            dice=     dice.mean().item(),
        )

        seg_metrics: SegmentationMetrics = SegmentationMetrics(
            per_class=per_class,
            macro=macro,
        )
        return seg_metrics
