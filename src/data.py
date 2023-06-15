import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

class PetDataModule(L.LightningDataModule):
    """
    DataModule for creating OxfordIIITPet dataset
    """
    def __init__(
        self, 
        data_dir:str="./datasets", 
        batch_size:int=64, 
        num_workers:int=2,
        split_size:float=0.80,
        seed:int=42,
        test_transform=None,   
        train_transform=None,
        ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = split_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_workers = num_workers
        self.seed=seed

    def prepare_data(self):
        # download (will run once to download the data and save it to the data_dir)
        datasets.OxfordIIITPet(root=self.data_dir,target_types="category", download=True)

        # transformations
        if self.train_transform is None:
            self.train_transform =  self._transform_dataset()

        if self.test_transform is None:
            self.test_transform = self._transform_dataset()

    def setup(self, stage: str):
        # Get data from OxfordIIITPet dataset
        self.data_test = datasets.OxfordIIITPet(
            self.data_dir, 
            transform=self.test_transform,
            download=False,
            split="test",
        )

        # Split the data into training and validation parts
        data_full = datasets.OxfordIIITPet(
            self.data_dir, 
            transform=self.train_transform, 
            download=False,
            split="trainval",
        )

        # Split the training data into training and validation parts
        split_len = int(self.train_size*len(data_full))
        self.data_train, self.data_val = random_split(data_full, 
                                                      [split_len, len(data_full)-split_len], 
                                                      generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.data_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.data_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.data_test,
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=self.num_workers,
        )
        return test_loader
    
    def _transform_dataset(self):
        return transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
