import datetime
import os
from time import perf_counter
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import Config
from dataset import CarPartsDataset
from logger import Logger
from metrics import StreamingSegMetrics
from model import SegModel
from data_structures import TimeMetrics
from tensorboard_logger import TensorBoardLogger


class Trainer(object):
    def __init__(self):
        self.cfg: Config = Config()
        self.model_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
                                          split="train",
                                          model_id=self.model_id)
        self.tb_val = TensorBoardLogger(self.cfg,
                                        log_dir=self.tensorboard_path,
                                        split="val",
                                        model_id=self.model_id)

        # Modules
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.train_loader, self.val_loader = self.make_loaders()
        self.model: SegModel = SegModel(self.cfg)
        self.model.to(self.model.device)
        self.opt: torch.optim.AdamW = torch.optim.AdamW(self.model.parameters(),
                                                        lr=self.cfg.lr)
        self.meter_train: StreamingSegMetrics = StreamingSegMetrics(self.cfg)
        self.meter_val: StreamingSegMetrics = StreamingSegMetrics(self.cfg)
        self.logger.info(f'initialized with model ID: {self.model_id}')

    def make_loaders(self) -> Tuple[DataLoader, DataLoader]:
        dataset: CarPartsDataset = CarPartsDataset(
            config=self.cfg,
            images_dir=os.path.join(self.cfg.data_dir, "images"),
            masks_dir=os.path.join(self.cfg.data_dir, "masks"),
            size=self.cfg.image_size,
            augment=False,
        )
        n_val: int = max(1, int(self.cfg.val_percentage * len(dataset)))
        n_train: int = len(dataset) - n_val

        train_dataset: CarPartsDataset
        val_dataset: CarPartsDataset
        train_dataset, val_dataset = random_split(dataset,
                                        lengths=[n_train, n_val],
                                        generator=torch.Generator().manual_seed(self.cfg.random_seed))

        train_loader: DataLoader = DataLoader(train_dataset,
                                              batch_size=self.cfg.batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)

        val_loader: DataLoader   = DataLoader(val_dataset,
                                              batch_size=self.cfg.val_batch_size,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)
        self.logger.info(
            f'Created data loaders - train samples: {n_train}, val samples: {n_val}, '
            f'train batches: {len(train_loader)}, val batches: {len(val_loader)}'
        )

        return train_loader, val_loader

    def training_step(self,
                      imgs: torch.Tensor,
                      masks: torch.Tensor,
                      class_weights: torch.Tensor,
                      running_loss: float):
        t_data_start = perf_counter()
        imgs = imgs.to(self.model.device, non_blocking=True)
        masks = masks.to(self.model.device, non_blocking=True)
        t_fwd_start = perf_counter()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(imgs)
        t_loss_start = perf_counter()
        loss = F.cross_entropy(logits,
                               masks,
                               weight=class_weights,
                               ignore_index=-1)
        t_bwd_start = perf_counter()
        loss.backward()
        t_step_start = perf_counter()
        self.opt.step()
        running_loss += float(loss.item())
        t_metrics_start = perf_counter()
        self.meter_train.update(logits.detach(), masks.detach())
        t_end = perf_counter()

        batch_time_metrics = TimeMetrics(
            data=t_fwd_start - t_data_start,
            forward=t_loss_start - t_fwd_start,
            backward=t_bwd_start - t_loss_start,
            optimizer_step=t_step_start - t_bwd_start,
            metrics=t_end - t_metrics_start,
            batch=t_end - t_data_start,
            total=t_end - t_data_start
        )
        return batch_time_metrics, loss, logits, masks, running_loss

    def val_step(
            self,
            imgs: torch.Tensor,
            masks: torch.Tensor,
            class_weights: torch.Tensor,
            val_loss: float,
            val_time_metrics: TimeMetrics
    ):
        t_batch_start = t_data_start = perf_counter()
        imgs = imgs.to(self.model.device, non_blocking=True)
        masks = masks.to(self.model.device, non_blocking=True)
        t_fwd_start = perf_counter()
        logits = self.model(imgs)
        t_loss_start = perf_counter()
        _val_loss = F.cross_entropy(logits,
                                    masks,
                                    weight=class_weights,
                                    ignore_index=-1)
        val_loss += float(_val_loss.item())
        t_metrics_start = perf_counter()
        self.meter_val.update(logits, masks)
        t_batch_end = perf_counter()

        val_time_metrics += TimeMetrics(
            total=t_batch_end - t_batch_start,
            data=t_fwd_start - t_data_start,
            forward=t_loss_start - t_fwd_start,
            optimizer_step=t_metrics_start - t_loss_start,
            batch=t_batch_end - t_batch_start,
            metrics=t_batch_end - t_metrics_start
        )
        return val_loss, val_time_metrics

    def train(self):

        global_step: int = 0
        class_weights: torch.Tensor = torch.tensor(
            self.cfg.class_weights,
            dtype=torch.float32,
            device=self.model.device
        )
        step_in_epoch: int
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
                    self.training_step(
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
                self.tb_train.log_loss(float(loss.item()), global_step)
                if global_step % self.cfg.train_logging_freq == 0:
                    self.logger.info(f'Logging training metrics at step {global_step}')
                    train_time_metrics /= (step_in_epoch + 1)
                    train_metrics = self.meter_train.compute()


                    self.logger.info(
                        f'Training step {global_step} - '
                        f'Avg loss (s) - {running_loss / max(1, len(self.train_loader)):.4f} '
                        f'Avg result (s) - {train_metrics.macro} '
                        f'Avg times (s) - {train_time_metrics}'
                    )

                    self.tb_train.log_metrics(train_metrics, step=global_step)
                    self.tb_train.flush()

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
                            val_loss, val_time_metrics = self.val_step(
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
                        self.logger.info(f'Val for Step {epoch + 1} - '
                                         f'Avg loss - {avg_val_loss:.4f} '
                                         f'Avg result - {val_metrics.macro} '
                                         f'Avg times - {val_time_metrics}')
                        self.tb_val.log_loss(avg_val_loss, global_step)
                        self.tb_val.log_metrics(val_metrics, step=global_step)
                        self.tb_val.flush()

                    # ---- checkpoint ----
                    checkpoint_file_path: str = os.path.join(
                        self.weights_path,
                        f"checkpoint_step_{global_step}.pth"
                    )
                    torch.save(self.model.state_dict(), checkpoint_file_path)
                    self.logger.info(f'Saved model checkpoint at step {global_step} to {self.weights_path}')

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
