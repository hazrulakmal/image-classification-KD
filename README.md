# Knowledge Distillation for Image Classification
Using knowledge distillation to improve lightweight vision model performance for image classification.

## Introduction
Deploying large deep models with billions of model parameters to devices with limited resources, such as mobile phones and embedded devices, is challenging due to high computational complexity and large storage requirements. Therefore, we aim to explore knowledge distillation techniques as a means of mitigating this problem by producing lightweight models while maximising their predictive performance for vision classification tasks.

## Objective 
- To demonstrate that the quality of student model **predictive performance** is influenced by the quality of the corresponding teacher model.
- Perform **cross-architecture distillation** between the teacher and student models to determine whether distilling models from different architetures would affect the predictive performance.
- To show **inference speed** and **memory usage** improvements gained from distillation.
- To build **high velocity**, **versioning** and **robust validation** training pipeline for distillation and fine-tuning.

## Training Pipeline
![Training Pipeline Overview](https://github.com/hazrulakmal/image-classification-KD/blob/main/asset/training_pipelines.png?raw=true)

This project implements a robust training pipeline that provides a user interface for changing initial variables that control the training procedure, model architectures, and hyperparameters. The pipeline automatically logs all training metrics, such as loss function and accuracy score, to a machine learning experiment tracking software called WandB. This allows us to easily compare and analyze the results of each model afterwards.

The interface enables quick experimentation with different desired training methods, such as distillation or normal fine-tuning. The script then dynamically and automatically executes the training process, making it a seamless experience for training multiple different vision models.

this project use Free Cloud GPU resources from Google Colab/Kaggle Notebook to train multiple models in parallel to significantly speed up the time required to train multiple vision models.

## Results
Results are summarized in public W&B report [here](https://api.wandb.ai/links/st311-project/c9zfjjli)

A quick summary of the results are as follows:
- Same-architecture Distillation improves student model performance by ____ on average.
- Cross-architecture distillation improves student model performance by ____ on average.
- for same-architecture distillation, the bigger the models, the better the distillation (knowledge transfer) performance but the same cannot be said for cross-architecture distillation.


### Same-architecture distillation
<p align="center">
  <img src="https://github.com/hazrulakmal/image-classification-KD/blob/main/asset/same_achitecture.png?raw=true"/>
</p>

### Cross-architecture distillation
<p align="center">
  <img src="https://github.com/hazrulakmal/image-classification-KD/blob/main/asset/cross_architecture.png?raw=true"/>
</p>


### All distillation
<p align="center">
  <img src="https://github.com/hazrulakmal/image-classification-KD/blob/main/asset/all_distillation.png?raw=true" width="630" height="400"/>
</p>

### Future Improvements
WIP


## Installation Guide
1. Google Colab/Kaggle Notebook **(Recommended)**
    - Open and Copy the notebook in Google Colab/Kaggle Notebook
    - The notebook
2. Local Machine
    - Clone the repository
    - Install the dependencies by running `pip install -r requirements.txt`
    - Run the cli.py script

## Codebase Navigation
1. Data Preprocessing Steps
    - src -> data.py
2. Training and Distillation
    - src -> training.py
    - **LightningTraining** class is for finetuning vision models on dataset.
    - **DistilledTraining** class is for distilling a teacher model into a student model.
    - Distil Loss can be found in the *training_step* method of **DistilledTraining** class
3. Command Line Interface Code (CLI)
    - run.py
4. Notebook
    - notebook -> training+interface.ipynb   
5. Project dependency 
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