#import sys
#import os
import wandb
import torch
#from watermark import watermark
from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.callbacks import ModelCheckpoint
#from torchvision.models import ResNet18_Weights
from src.utils_helpers import get_secret_keys
from src.models import LightningModel 
from src.data import PetDataModule

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.SGD)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)
        parser.set_defaults({
            "trainer.logger": {
                "class_path":"lightning.pytorch.loggers.WandbLogger",
                "init_args": {
                "job_type": "Train",
                }
            }
         })
        

def cli_main(args: ArgsType = None):    
    cli = MyLightningCLI(
        model_class=LightningModel,
        datamodule_class=PetDataModule,
        run=False,
        seed_everything_default=42,
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc", save_last=False)],
            "deterministic": True,
            "devices": "auto",
            "precision": "16",
            "log_every_n_steps": 5,
            "val_check_interval": 0.25,
        },
        args=args
        ) #customise run_dev_run, accelerator

    #api_key = get_secret_keys("secrets.json")["wandb"]
    # try:
    #     wandb.login(key=cli.datamodule.api_key)
    #     print("wandb Logged in Successfully")
    # except TypeError:
    #     print("wandb Login Failed. Please verify your wandb API key")

    if not cli.trainer.fast_dev_run:
        ckpt_path = "best"
    else:
        ckpt_path = None
        cli.trainer.logger = None

    #print(cli.datamodule.batch_size)
    #datasets
    # weights = ResNet18_Weights.IMAGENET1K_V1
    # preprocess_transform = weights.transforms()
    # dm = PetDataModule(
    #     batch_size=cli.datamodule.batch_size, 
    #     num_workers=os.cpu_count(),
    #     train_transform=preprocess_transform,
    #     test_transform=preprocess_transform,
    # )

    #model
    # pytorch_model= torch.hub.load('pytorch/vision:v0.13.0', cli.model_name, weights="DEFAULT")
    # pytorch_model.fc = torch.nn.Linear(pytorch_model.fc.in_features, 37)

    # pytorch_model = PetModel(
    #     model=configurations.model, 
    #     weights=configurations.pretrained_weights, 
    #     num_classes=configurations.num_classes
    # )

    #training steps (optimizer, loss, metrics)
    lightning_model = LightningModel(
        model=None, 
        model_name=cli.model.model_name,
        learning_rate=cli.model.learning_rate,
    )

    cli.trainer.fit(lightning_model, datamodule=cli.datamodule)
    cli.trainer.test(lightning_model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    wandb.finish()
    
# if __name__ == "__main__":
#     print(watermark(packages="torch,lightning", python=True))
#     print(f"The provided arguments are {sys.argv[1:]}")
#     cli_main()
    
    
    
