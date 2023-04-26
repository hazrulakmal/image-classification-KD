from run import cli_main
from watermark import watermark
import os

args = {
    "trainer": {
        "accelerator": "cpu",
        "fast_dev_run": True,
        "logger":{
            "name": "resnet18-vanilla-test",
            "project": "image-classification-KD"
        },
        "precision": "32-true",
        "log_every_n_steps": 5,
        #"max_epochs": 2,
    },
    
    "model": {
        "class_path": "src.training.LightningTraining",
        "init_args": {
            "model_name": "resnet18",
            "dropout_rates": 0.4,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay":1e-3,
            "T_max": 10
        }
    },

    "data": {
        "batch_size": 16,
        "num_workers": os.cpu_count(),
    },

    # "model" : {
    #     "class_path": "src.training.DistilledTraining",
    #     "init_args": {
    #         "model_name": "resnet18",
    #         "teacher_model_name": "resnet50",
    #         "artifact_path": "resnet50-vanilla:latest",
    #         "alpha": 0.5,
    #         "temperature": 2.0,
    #         "learning_rate": 0.01,
    #         "momentum": 0.9,
    #         "nesterov": True,
    #         "weight_decay":1e-3,
    #         "T_max": 10
    #     }
    # },
}

if __name__ == "__main__":
    print(watermark(packages="torch,lightning,wandb,torchvision,torchmetrics,jsonargparse", python=True))
    cli_main(args)