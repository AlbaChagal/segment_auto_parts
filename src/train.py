import datetime
import os
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
        train_loader: DataLoader
        val_loader: DataLoader
        self.train_loader, self.val_loader = self.make_loaders(self.cfg)
        self.model = SegModel(self.cfg)
        self.model.to(self.model.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.logger.info(f'initialized with model ID: {self.model_id}')

    @staticmethod
    def make_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
        ds: CarPartsDataset = CarPartsDataset(
            images_dir=os.path.join(cfg.data_dir, "images"),
            masks_dir=os.path.join(cfg.data_dir, "masks"),
            size=cfg.image_size,
            augment=False,
        )
        n_val: int = max(1, int(cfg.val_percentage * len(ds)))
        n_train: int = len(ds) - n_val

        train_ds: CarPartsDataset
        val_ds: CarPartsDataset
        train_ds, val_ds = random_split(ds,
                                        lengths=[n_train, n_val],
                                        generator=torch.Generator().manual_seed(cfg.random_seed))

        train_loader: DataLoader = DataLoader(train_ds,
                                              batch_size=cfg.batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)

        val_loader: DataLoader   = DataLoader(val_ds,
                                              batch_size=cfg.val_batch_size,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

        return train_loader, val_loader

    def train(self):

        global_step: int = 0
        loss: torch.Tensor
        imgs: torch.Tensor
        masks: torch.Tensor
        logits: torch.Tensor
        meter_train: StreamingSegMetrics
        meter_val: StreamingSegMetrics
        running_loss: float

        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            meter_train = StreamingSegMetrics(self.cfg)
            running_loss = 0.0

            for imgs, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs} [train]"):
                imgs = imgs.to(self.model.device, non_blocking=True)
                masks = masks.to(self.model.device, non_blocking=True)

                self.opt.zero_grad(set_to_none=True)
                logits = self.model(imgs)
                loss = F.cross_entropy(logits, masks)
                loss.backward()
                self.opt.step()

                running_loss += float(loss.item())
                meter_train.update(logits.detach(), masks.detach())

                self.tb_train.log_loss(float(loss.item()), global_step)
                global_step += 1
                if global_step > 3:
                    break

            train_metrics = meter_train.compute()
            self.tb_train.log_metrics(train_metrics, step=global_step)
            self.tb_train.flush()

            # ---- val ----
            self.model.eval()
            meter_val = StreamingSegMetrics(self.cfg)
            val_loss = 0.0
            val_step = 0
            with torch.no_grad():
                for imgs, masks in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs} [val]"):

                    imgs = imgs.to(self.model.device, non_blocking=True)
                    masks = masks.to(self.model.device, non_blocking=True)
                    logits = self.model(imgs)
                    _val_loss = F.cross_entropy(logits, masks)
                    val_loss += float(_val_loss.item())
                    meter_val.update(logits, masks)
                    val_step += 1

            val_metrics = meter_val.compute()
            self.tb_val.log_loss(val_loss / max(1, len(self.val_loader)), epoch)
            self.tb_val.log_metrics(val_metrics, step=global_step)
            self.tb_val.flush()

            # ---- checkpoint ----
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, f"{epoch}.pth"))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
