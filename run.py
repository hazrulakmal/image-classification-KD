#import sys
#import os
import wandb
#import torch
from watermark import watermark
from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.callbacks import ModelCheckpoint
#from torchvision.models import ResNet18_Weights
from src.utils_helpers import get_secret_keys
from src.training import LightningTraining, DistilledTraining
from src.data import PetDataModule

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        #parser.add_optimizer_args(torch.optim.SGD)
        #parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)
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
        #model_class=LightningTraining,
        datamodule_class=PetDataModule,
        run=False,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc", save_last=False)],
            "deterministic": True,
            "devices": "auto",
            "precision": "32-true",
            "log_every_n_steps": 5,
            "val_check_interval": 0.25,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
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

    #training steps (optimizer, loss, metrics)
    # lightning_model = LightningTraining(
    #     model=None, 
    #     model_name=cli.model.model_name,
    #     learning_rate=cli.model.hparams.learning_rate,
    #     momentum=cli.model.hparams.momentum,
    #     nesterov=cli.model.hparams.nesterov,
    #     weight_decay=cli.model.hparams.weight_decay,
    #     T_max=cli.model.hparams.T_max,
    # )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    wandb.finish()
    
# if __name__ == "__main__":
#     print(watermark(packages="torch,lightning", python=True))
#     cli_main()
    
    
    
