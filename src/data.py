import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):

    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.rand(3, 224, 224), torch.rand(22)


class DummyDataModule(pl.LightningDataModule):

    def __init__(self, size=16, batch_size=2):
        super().__init__()
        self.size = size
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(DummyDataset(size=self.size),
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(DummyDataset(size=self.size),
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(DummyDataset(size=self.size),
                          batch_size=self.batch_size)
