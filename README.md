# Knowledge Distillation for Image Classification
Using knowledge distillation to improve lightweight vision model performance for image classification.

## Introduction
Deploying large deep models with billions of model parameters to devices with limited resources, such as mobile phones and embedded devices, is challenging due to high computational complexity and large storage requirements. Therefore, we aim to explore knowledge distillation techniques as a means of mitigating this problem by producing lightweight models while maximising their predictive performance for vision classification tasks

## Objective 
- To demonstrate that the quality of student model predictive performance is influenced by the quality of the corresponding teacher model.
- Perform cross-architecture distillation between the teacher and student models to determine whether distilling models from different architetures would affect the predictive performance.
- To show inference speed and memory usage improvements gained from distillation.


this project implemented a robust training pipeline that provides a user interface to change initial variables controlling the training procedure, model architectures, and hyperparameters. The pipeline will automatically logg all training metrics like loss function and accuracy score to a ML experiment tracking software, WandB, which allows us to easily compare and analyse the results of each model accordingly afterwards.

The interface enables quick experimentations to carry out desired training, whether it be distillation or normal fine-tuning, with ease. The script will then dynamically and automatically execute the training process, making it a seamless experience for training multiple different vision models.

## Results


## To-do List
1. Create a baseline pipeline for training and evaluating models.
2. Save the models and results on weights and biases storage.
3. Create a pipeline for knowledge distillation.
4. create a notebook to show inferece on test sets. Load saved models and make inference on test sets.

1. you can fill up cli arguments using yaml file or python!
2. models must met the image preprocessing requirements
3. best practices for L.LightningModule __init__ method is to create the base model there rather than create it outside and pass it in. This is because the model is a part of the module and should be created in the module.

## Environment Setup
- Python version       : 3.10.11
- IPython version      : 7.34.0

- torch       : 2.0.0+cu118
- lightning   : 2.0.1.post0
- wandb       : 0.15.0
- torchvision : 0.15.1+cu118
- torchmetrics: 0.11.4
- jsonargparse: 4.21.0