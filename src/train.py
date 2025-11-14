import datetime
import os
from random import Random
from time import perf_counter
from typing import Tuple
from numpy.random import RandomState
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import Config
from dataset import CarPartsDataset
from logger import Logger
from metrics import StreamingSegMetrics
from model import SegModel
from data_structures import TimeMetrics, SegmentationMetrics
from tensorboard_logger import TensorBoardLogger


class Trainer(object):
    """
    Trainer class for semantic segmentation model training
    1. Initializes all components: data loaders, model, optimizer, loggers,
         metrics
    2. Implements the training loop with periodic validation and logging
    3. Saves model checkpoints
    4. Supports reproducibility via random seed management
    5. Logs detailed timing metrics for performance monitoring
    6. Uses cross-entropy loss with class weights for training
    7. Input: Config parameters, training and validation datasets
    8. Output: Trained model weights, TensorBoard logs
    9. Usage: trainer = Trainer(); trainer.train()
    """
    def __init__(self):
        """
        Initialize the Trainer with configuration, data loaders, model,
        optimizer, loggers, and metrics.
        1. Load configuration
        2. Set up random states for reproducibility
        3. Create data loaders for training and validation
        4. Initialize model and optimizer
        5. Set up loggers for console and TensorBoard
        6. Initialize streaming metrics for training and validation
        7. Create necessary directories for outputs
        8. Log initialization details
        """
        self.cfg: Config = Config()
        self.model_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Randomness
        self.random_state: Random = Random(self.cfg.random_seed)
        self.np_random_state: RandomState = RandomState(self.cfg.random_seed)
        self.torch_random_state: torch.Generator = torch.Generator().manual_seed(self.cfg.random_seed)

        # Paths
        self.model_path: str = os.path.join(self.cfg.outputs_folder_name, self.model_id)
        self.weights_path: str = os.path.join(self.model_path, self.cfg.weight_folder_name)
        self.tensorboard_path: str = os.path.join(self.model_path, self.cfg.tensorboard_folder_name)
        os.makedirs(self.weights_path, exist_ok=True)

        # Loggers
        self.logger: Logger = Logger(name=self.__class__.__name__,
                                     logging_level=self.cfg.trainer_logging_level)
        self.tb_train = TensorBoardLogger(self.cfg,
                                          log_dir=self.tensorboard_path,
                                          split="train")
        self.tb_val = TensorBoardLogger(self.cfg,
                                        log_dir=self.tensorboard_path,
                                        split="val")

        # Modules
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.train_loader, self.val_loader = self._make_loaders()
        self.model: SegModel = SegModel(self.cfg)
        self.model.to(self.model.device)
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(self.model.parameters(),
                                                              lr=self.cfg.lr)
        self.meter_train: StreamingSegMetrics = StreamingSegMetrics(self.cfg)
        self.meter_val: StreamingSegMetrics = StreamingSegMetrics(self.cfg)
        self.logger.info(f'initialized with model ID: {self.model_id}')

    def reset_random_states(self) -> None:
        """
        Reset all random states to the configured random seed
        :return:
        """
        seed: int = self.cfg.random_seed
        self.random_state.seed(seed)
        self.np_random_state.seed(seed)
        self.torch_random_state.manual_seed(seed)
        self.logger.info(f'Reset random states with seed: {seed}')

    def _make_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders
        :return: Tuple[DataLoader, DataLoader]: train and val data loaders
        """
        self.reset_random_states()
        base_ds: CarPartsDataset = CarPartsDataset(
            config=self.cfg,
            images_dir=os.path.join(self.cfg.data_dir, "images"),
            masks_dir=os.path.join(self.cfg.data_dir, "masks"),
            size=self.cfg.image_size,
            is_augment=False,
        )
        n_total: int = len(base_ds)
        n_val: int = max(1, int(self.cfg.val_percentage * n_total))
        n_train: int = n_total - n_val

        gsplit: torch.Generator = torch.Generator().manual_seed(self.cfg.random_seed)
        perm: torch.Tensor = torch.randperm(n_total, generator=gsplit)
        train_idx: torch.Tensor
        val_idx: torch.Tensor
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        assert set(train_idx).isdisjoint(set(val_idx)), \
            f"Train and validation indices overlap! Train idx: {train_idx}, Val idx: {val_idx} " \
            f"overlap: {set(train_idx).intersection(set(val_idx))}"

        train_ds: CarPartsDataset = CarPartsDataset(
            config=self.cfg,
            images_dir=os.path.join(self.cfg.data_dir, "images"),
            masks_dir=os.path.join(self.cfg.data_dir, "masks"),
            size=self.cfg.image_size,
            is_augment=self.cfg.is_augment_training_data
        )
        val_ds: CarPartsDataset = CarPartsDataset(
            config=self.cfg,
            images_dir=os.path.join(self.cfg.data_dir, "images"),
            masks_dir=os.path.join(self.cfg.data_dir, "masks"),
            size=self.cfg.image_size,
            is_augment=False
        )

        train_subset: Subset = Subset(train_ds, train_idx)
        val_subset: Subset = Subset(val_ds, val_idx)

        gloader: torch.Generator = torch.Generator().manual_seed(self.cfg.random_seed)

        train_loader: DataLoader = DataLoader(
            train_subset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=gloader,
            persistent_workers=False,
            prefetch_factor=None,
            timeout=0
        )

        val_loader: DataLoader = DataLoader(
            val_subset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            timeout=0
        )

        self.logger.info(
            f"Created data loaders - train samples: {n_train}, val samples: {n_val}, "
            f"train batches: {len(train_loader)}, val batches: {len(val_loader)}"
        )
        return train_loader, val_loader

    def _training_step(
            self,
            imgs: torch.Tensor,
            masks: torch.Tensor,
            class_weights: torch.Tensor,
            running_loss: float
    ) -> Tuple[TimeMetrics, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Single training step
        :param imgs: The input images (batch)
        :param masks: The ground truth masks (batch)
        :param class_weights: The class weights tensor
        :param running_loss: The running loss value
        :return: time metrics, loss, logits, masks, updated running loss
        """
        t_data_start: float = perf_counter()
        imgs: torch.Tensor = imgs.to(self.model.device, non_blocking=True)
        masks: torch.Tensor = masks.to(self.model.device, non_blocking=True)
        t_fwd_start: float = perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        logits: torch.Tensor = self.model(imgs)
        t_loss_start: float = perf_counter()
        loss: torch.Tensor = F.cross_entropy(
            logits,
            masks,
            weight=class_weights,
            ignore_index=-1
        )
        t_bwd_start: float = perf_counter()
        loss.backward()
        t_step_start: float = perf_counter()
        self.optimizer.step()
        running_loss += float(loss.item())
        t_metrics_start = perf_counter()
        self.meter_train.update(logits.detach(), masks.detach())
        t_end: float = perf_counter()

        batch_time_metrics: TimeMetrics = TimeMetrics(
            data=t_fwd_start - t_data_start,
            forward=t_loss_start - t_fwd_start,
            backward=t_bwd_start - t_loss_start,
            optimizer_step=t_step_start - t_bwd_start,
            metrics=t_end - t_metrics_start,
            batch=t_end - t_data_start,
            total=t_end - t_data_start
        )
        return batch_time_metrics, loss, logits, masks, running_loss

    def _val_step(
            self,
            imgs: torch.Tensor,
            masks: torch.Tensor,
            class_weights: torch.Tensor,
            val_loss: float,
            val_time_metrics: TimeMetrics
    ) -> Tuple[float, TimeMetrics]:
        """
        Single validation step
        :param imgs: The input images (batch)
        :param masks: The ground truth masks (batch)
        :param class_weights: The class weights tensor
        :param val_loss: The accumulated validation loss
        :param val_time_metrics: The accumulated validation time metrics
        :return: updated val_loss, updated val_time_metrics
        """
        t_batch_start: float = perf_counter()
        imgs: torch.Tensor = imgs.to(self.model.device, non_blocking=True)
        masks: torch.Tensor = masks.to(self.model.device, non_blocking=True)
        t_fwd_start: float = perf_counter()
        logits: torch.Tensor = self.model(imgs)
        t_loss_start: float = perf_counter()
        _val_loss: torch.Tensor = F.cross_entropy(
            logits,
            masks,
            weight=class_weights,
            ignore_index=-1
        )
        val_loss += float(_val_loss.item())
        t_metrics_start: float = perf_counter()
        self.meter_val.update(logits, masks)
        t_batch_end: float = perf_counter()

        val_time_metrics += TimeMetrics(
            total=t_batch_end - t_batch_start,
            data=t_fwd_start - t_batch_start,
            forward=t_loss_start - t_fwd_start,
            optimizer_step=t_metrics_start - t_loss_start,
            batch=t_batch_end - t_batch_start,
            metrics=t_batch_end - t_metrics_start
        )
        return val_loss, val_time_metrics

    def _log_train_metrics(self,
                           global_step: int,
                           loss_value: float,
                           step_in_epoch: int,
                           train_time_metrics: TimeMetrics,
                           running_loss: float) -> None:
        """
        Log training metrics to TensorBoard and console
        :param global_step: The current global training step
        :param loss: The loss tensor
        :param step_in_epoch: The current step in the epoch
        :param train_time_metrics: The accumulated training time metrics
        :param running_loss: The running loss value
        :return:
        """
        self.tb_train.log_loss(loss_value, global_step)
        if global_step % self.cfg.train_logging_freq == 0:
            self.logger.info(f'Logging training metrics at step {global_step}')
            train_time_metrics /= self.cfg.train_logging_freq
            train_metrics: SegmentationMetrics = self.meter_train.compute()

            self.logger.info(
                f'Training step {global_step} - '
                f'Avg loss - {running_loss / max(1, len(self.train_loader)):.4f} '
                f'Avg result - {train_metrics.macro} '
                f'Avg times (s) - {train_time_metrics}'
            )

            self.tb_train.log_metrics(train_metrics, step=global_step)
            self.tb_train.flush()

    def _log_val_metrics(self,
                         global_step: int,
                         avg_val_loss: float,
                         val_metrics: SegmentationMetrics,
                         val_time_metrics: TimeMetrics) -> None:
        """
        Log validation metrics to TensorBoard and console
        :param global_step: The current global training step
        :param avg_val_loss: The average validation loss
        :param val_metrics: The validation segmentation metrics
        :param val_time_metrics: The validation time metrics
        :return:
        """
        self.logger.info(f'Val for Step {global_step} - '
                         f'Avg loss - {avg_val_loss:.4f} '
                         f'Avg result - {val_metrics.macro} '
                         f'Avg times - {val_time_metrics}')
        self.tb_val.log_loss(avg_val_loss, global_step)
        self.tb_val.log_metrics(val_metrics, step=global_step)
        self.tb_val.flush()

    def _save_checkpoint(self, global_step: int) -> None:
        """
        Save model checkpoint
        :param global_step: The current global training step
        :return:
        """
        checkpoint_file_path: str = os.path.join(
            self.weights_path,
            f"checkpoint_step_{global_step}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_file_path)
        self.logger.info(f'Saved model checkpoint at step '
                         f'{global_step} to {self.weights_path}')

    def train(self):
        """
        Main training loop
        1. Iterate over epochs
        2. For each epoch, iterate over training batches
        3. Perform training step and log metrics
        4. Periodically perform validation and log metrics
        5. Save model checkpoints
        6. Manage global step count
        7. Log detailed timing metrics
        8. Use class weights in loss computation
        9. Ensure reproducibility via random seed management
        :return: None. Trains the model and saves checkpoints and logs tensorboards.
        """
        global_step: int = 0
        class_weights: torch.Tensor = torch.tensor(
            self.cfg.class_weights,
            dtype=torch.float32,
            device=self.model.device
        )
        step_in_epoch: int
        train_metrics: SegmentationMetrics
        val_metrics: SegmentationMetrics
        train_time_metrics: TimeMetrics
        val_time_metrics: TimeMetrics
        loss: torch.Tensor
        imgs: torch.Tensor
        masks: torch.Tensor
        logits: torch.Tensor
        running_loss: float
        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            self.meter_train.reset()
            running_loss = 0.0
            step_in_epoch = 0
            train_time_metrics = TimeMetrics()
            self.logger.info(f'New epoch {epoch + 1}/{self.cfg.num_epochs} at global step {global_step}')
            for imgs, masks in self.train_loader:

                batch_time_metrics, loss, logits, masks, running_loss = \
                    self._training_step(
                        imgs=imgs,
                        masks=masks,
                        class_weights=class_weights,
                        running_loss=running_loss
                    )
                step_in_epoch += 1
                global_step += 1
                self.logger.info(f'Global step: {global_step}, Loss: {loss.item():.4f}')

                # Log results
                train_time_metrics += batch_time_metrics
                self._log_train_metrics(
                    global_step=global_step,
                    loss_value=loss.item(),
                    step_in_epoch=step_in_epoch,
                    train_time_metrics=train_time_metrics,
                    running_loss=running_loss
                )

                # ---- validation ----
                if global_step % self.cfg.val_logging_freq == 0:
                    self.logger.info(f'Starting validation at step {global_step}')
                    self.model.eval()
                    self.meter_val.reset()

                    val_loss = 0.0
                    val_step = 0

                    with torch.no_grad():
                        val_time_metrics = TimeMetrics()
                        for imgs, masks in self.val_loader:
                            val_loss, val_time_metrics = self._val_step(
                                imgs=imgs,
                                masks=masks,
                                class_weights=class_weights,
                                val_loss=val_loss,
                                val_time_metrics=val_time_metrics
                            )

                            val_step += 1

                        val_time_metrics /= val_step + 1
                        val_metrics = self.meter_val.compute()
                        avg_val_loss = val_loss / max(1, len(self.val_loader))
                        # Log results
                        self._log_val_metrics(
                            global_step=global_step,
                            avg_val_loss=avg_val_loss,
                            val_metrics=val_metrics,
                            val_time_metrics=val_time_metrics
                        )

                    # ---- checkpoint ----
                    self._save_checkpoint(global_step=global_step)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
