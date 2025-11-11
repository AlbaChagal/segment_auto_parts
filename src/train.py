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
        self.train_loader, self.val_loader = self.make_loaders(self.cfg)
        self.model: SegModel = SegModel(self.cfg)
        self.model.to(self.model.device)
        self.opt: torch.optim.AdamW = torch.optim.AdamW(self.model.parameters(),
                                                        lr=self.cfg.lr)
        self.meter_train: StreamingSegMetrics = StreamingSegMetrics(self.cfg)
        self.meter_val: StreamingSegMetrics = StreamingSegMetrics(self.cfg)
        self.logger.info(f'initialized with model ID: {self.model_id}')

    @staticmethod
    def make_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
        dataset: CarPartsDataset = CarPartsDataset(
            config=cfg,
            images_dir=os.path.join(cfg.data_dir, "images"),
            masks_dir=os.path.join(cfg.data_dir, "masks"),
            size=cfg.image_size,
            augment=False,
        )
        n_val: int = max(1, int(cfg.val_percentage * len(dataset)))
        n_train: int = len(dataset) - n_val

        train_dataset: CarPartsDataset
        val_dataset: CarPartsDataset
        train_dataset, val_dataset = random_split(dataset,
                                        lengths=[n_train, n_val],
                                        generator=torch.Generator().manual_seed(cfg.random_seed))

        train_loader: DataLoader = DataLoader(train_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)

        val_loader: DataLoader   = DataLoader(val_dataset,
                                              batch_size=cfg.val_batch_size,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

        return train_loader, val_loader

    def train(self):

        global_step: int = 0
        step_in_epoch: int = 0
        class_weights: torch.Tensor = torch.tensor(
            self.cfg.class_weights,
            dtype=torch.float32,
            device=self.model.device
        )
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
            train_time_metrics = TimeMetrics()
            for step_in_epoch, (imgs, masks) in tqdm(enumerate(self.train_loader),
                                                     desc=f"Epoch {epoch+1}/{self.cfg.num_epochs} [train]"):
                t_batch_start = t_data_start = perf_counter()

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

                self.tb_train.log_loss(float(loss.item()), global_step)
                t_batch_end = perf_counter()
                global_step += 1

                batch_time_metrics = TimeMetrics(
                    total=t_batch_end - t_batch_start,
                    data=t_fwd_start - t_data_start,
                    forward=t_loss_start - t_fwd_start,
                    backward=t_bwd_start - t_loss_start,
                    step=t_step_start - t_bwd_start,
                    batch=t_batch_end - t_batch_start,
                    metrics=t_metrics_start - t_step_start
                )
                train_time_metrics += batch_time_metrics

                # if global_step > 3:
                #     break

            if global_step % self.cfg.train_logging_freq == 0:
                train_time_metrics /= (step_in_epoch + 1)
                train_metrics = self.meter_train.compute()

                # Log results
                self.logger.info(f'Val Epoch {epoch + 1}/{self.cfg.num_epochs} - '
                                 f'Avg loss (s) - {running_loss / max(1, len(self.train_loader)):.4f} '
                                 f'Avg result (s) - {train_metrics.macro} '
                                 f'Avg times (s) - {train_time_metrics}')

                self.tb_train.log_metrics(train_metrics, step=global_step)
                self.tb_train.flush()

            # ---- validation ----
            if global_step % self.cfg.val_logging_freq == 0:
                self.model.eval()
                self.meter_val.reset()

                val_loss = 0.0
                val_step = 0

                with torch.no_grad():
                    val_time_metrics = TimeMetrics()
                    for imgs, masks in tqdm(self.val_loader,
                                            desc=f"Epoch {epoch+1}/{self.cfg.num_epochs} [val]"):
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
                            step=t_metrics_start - t_loss_start,
                            batch=t_batch_end - t_batch_start,
                            metrics=t_batch_end - t_metrics_start
                        )

                        val_step += 1

                val_time_metrics /= val_step + 1
                val_metrics = self.meter_val.compute()
                avg_val_loss = val_loss / max(1, len(self.val_loader))
                # Log results
                self.logger.info(f'Val Epoch {epoch + 1}/{self.cfg.num_epochs} - '
                                 f'Avg loss - {avg_val_loss:.4f} '
                                 f'Avg result - {val_metrics.macro} '
                                 f'Avg times - {val_time_metrics}')
                self.tb_val.log_loss(avg_val_loss, epoch)
                self.tb_val.log_metrics(val_metrics, step=global_step)
                self.tb_val.flush()

            # ---- checkpoint ----
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, f"{epoch}.pth"))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
