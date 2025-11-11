import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataset import CarPartsDataset
from model import SegModel
from config import Config
import os

from src.logger import Logger


class Trainer(object):
    def __init__(self):
        self.cfg = Config()
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level=self.cfg.trainer_logging_level)

    def train(self):
        data_set = CarPartsDataset(os.path.join(self.cfg.data_dir,"images"),
                                   os.path.join(self.cfg.data_dir,"masks"),
                                   self.cfg.image_size)

        data_loader = DataLoader(data_set,
                                 batch_size=self.cfg.batch_size,
                                 shuffle=True,
                                 num_workers=4)
        model = SegModel(self.cfg)
        model.to(model.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

        for epoch in range(self.cfg.num_epochs):
            model.train()
            total_loss = 0
            for imgs, masks in tqdm(data_loader):
                imgs, masks = imgs.to(model.device), masks.to(model.device)
                opt.zero_grad()
                out = model(imgs)
                loss = F.cross_entropy(out, masks)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.cfg.num_epochs}: Loss={total_loss/len(data_loader):.4f}")
            os.makedirs(os.path.dirname(self.cfg.model_path), exist_ok=True)
            torch.save(model.state_dict(), self.cfg.model_path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
