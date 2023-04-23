import json
import lightning as L
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import torch.nn.functional as F

class PetDataModule(L.LightningDataModule):
    def __init__(self, data_dir="../datasets", batch_size=64,  height_width=(64, 64), split_size=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.height_width = height_width
        self.val_size = split_size

    def prepare_data(self):
        # download (will run once to download the data)
        datasets.OxfordIIITPet(self.data_dir, target_types="category", split="trainval", download=True)
        datasets.OxfordIIITPet(self.data_dir, target_types="category", split="test", download=True)

        # transformations
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str):
        self.data_test = datasets.OxfordIIITPet(
            self.data_dir, 
            transform=self.test_transform,
            download=False,
            split="test"
        )

        data_full = datasets.OxfordIIITPet(
            self.data_dir, 
            transform=self.train_transform, 
            download=False,
            split="trainval"
        )

        num_samples = len(data_full)
        self.data_train, self.data_val = random_split(data_full, [int(num_samples*(1-self.val_size)), int(num_samples*self.val_size)])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False)
    
class LightningModel(L.LightningModule):
    def __init__(
            self, 
            model, 
            learning_rate:float = 0.01,
            betas: tuple = (0.9, 0.999),
            eps: float = 1e-6,):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.betas = betas
        self.eps = eps
        self.save_hyperparameters() #ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=37)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=37)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=37)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            betas=self.betas,
            ps=self.eps)
        return optimizer
    
def get_secret_keys(path):
    with open(path) as f:
        return json.load(f)