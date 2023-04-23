import lightning as L
import torch
import wandb
from utils import LightningModel, PetDataModule

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())
    dm = PetDataModule()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cpu", 
        devices="auto", 
        deterministic=True
    )

    trainer.fit(model=lightning_model, datamodule=dm)