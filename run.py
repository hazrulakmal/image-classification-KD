#import sys
#import os
import wandb
from watermark import watermark
import lightning
from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.callbacks import ModelCheckpoint
from src.training import LightningTraining, DistilledTraining
from src.data import PetDataModule

### code to integrate CLI into training and evaluation pipeline ### 

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        #parser.add_optimizer_args(torch.optim.SGD)
        #parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)
        parser.set_defaults({
            "trainer.logger": {
                "class_path":"lightning.pytorch.loggers.WandbLogger",
                "init_args": {
                "job_type": "Train",
                "log_model": True,
                }
            }
         })
        parser.set_defaults({
            "trainer.callbacks": {
              "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
              "init_args": {
                "save_top_k":1,
                "mode":"max",
                "monitor":"val_acc",
                "save_last":False,
                },
            },
        })
        

def cli_main(args: ArgsType = None):    
    cli = MyLightningCLI(
        model_class=None,
        datamodule_class=PetDataModule,
        run=False,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "max_epochs": 10,
            "deterministic": True,
            "devices": "auto",
            "precision": "16",
            "log_every_n_steps": 5,
            "val_check_interval": 0.25,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        },
        args=args
        ) 

    if not cli.trainer.fast_dev_run:
        ckpt_path = "best"
    else:
        ckpt_path = None
        cli.trainer.logger = None

    #execute training steps (optimizer, loss, metrics)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    #execute testing steps (metrics)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    wandb.finish()
    
# if __name__ == "__main__":
#     print(watermark(packages="torch,lightning,wandb,torchvision,torchmetrics,jsonargparse", python=True))
#     cli_main()
    
    
    
