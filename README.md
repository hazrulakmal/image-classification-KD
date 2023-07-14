# Knowledge Distillation for Image Classification
Using knowledge distillation to improve lightweight vision model performance for image classification.

## Introduction
Deploying large deep models with billions of model parameters to devices with limited resources, such as mobile phones and embedded devices, is challenging due to high computational complexity and large storage requirements. Therefore, we aim to explore knowledge distillation techniques as a means of mitigating this problem by producing lightweight models while maximising their predictive performance for vision classification tasks.

## Objective 
- To demonstrate that the quality of student model predictive performance is influenced by the quality of the corresponding teacher model.
- Perform cross-architecture distillation between the teacher and student models to determine whether distilling models from different architetures would affect the predictive performance.
- To show inference speed and memory usage improvements gained from distillation.

This project implements a robust training pipeline that provides a user interface for changing initial variables that control the training procedure, model architectures, and hyperparameters. The pipeline automatically logs all training metrics, such as loss function and accuracy score, to a machine learning experiment tracking software called WandB. This allows us to easily compare and analyze the results of each model afterwards.

The interface enables quick experimentation with different desired training methods, such as distillation or normal fine-tuning. The script then dynamically and automatically executes the training process, making it a seamless experience for training multiple different vision models.

## Results


## Installation Guide



## Codebase Navigation
1. Data Preprocessing Steps
    - src -> data.py
2. Training and Distillation
    - src -> training.py
    - **LightningTraining** class is for finetuning vision models onto the dataset.
    - **DistilledTraining** class is for distilling a teacher model into a student model
    - Distil loss can be found in the training_step method of **DistilledTraining** class
3. Command Line Interface (CLI)
    - run.py
4. Project dependency 
    - requirements.txt

## Environment Setup
- Python version       : 3.10.11
- IPython version      : 7.34.0

- torch       : 2.0.0+cu118
- lightning   : 2.0.1.post0
- wandb       : 0.15.0
- torchvision : 0.15.1+cu118
- torchmetrics: 0.11.4
- jsonargparse: 4.21.0