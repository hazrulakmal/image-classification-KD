import torch.nn.functional as F
import lightning as L
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

class LightningTraining(L.LightningModule):
    def __init__(
        self, 
        model=None, 
        model_name:str = "resnet18",
        num_classes:int = 37,
        dropout_rates:float=0.5,
        learning_rate:float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        nesterov: bool = False,
        T_max : int = 10,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['teacher_model', "model"])
        self.model_name = model_name

        if model is None:
            self.model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="DEFAULT")
            in_features = self.model.fc.in_features
            self.model.fc = self._add_layers(in_features, int(in_features/2), self.hparams.num_classes, self.hparams.dropout_rates)
        else:
            self.model = model
            
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = torch.nn.CrossEntropyLoss()(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss, on_epoch=False, on_step=True)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)
        return [optimizer], [sch]
    
    def _add_layers(self, in_features, hidden_units, num_classes, dropout_rates):
        new_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rates),
            torch.nn.Linear(hidden_units, num_classes),
        )
        return new_layers   

class DistilledTraining(L.LightningModule):
    def __init__(
        self, 
        model=None,
        model_name:str = "resnet18",
        teacher_model=None,
        teacher_model_name:str = "resnet50",
        artifact_path:str = "resnet50-vanilla:latest", 
        alpha:float=0.5, 
        temperature:float=2.0,
        num_classes:int = 37,
        dropout_rates:float=0.5,
        learning_rate:float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        nesterov: bool = True,
        T_max : int = 10,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['teacher_model',"model"])
        self.model_name = model_name

        if model is None:
            self.model = self._load_model(model_name)
        else:
            self.model = model

        if teacher_model is None:
            self.teacher_model = self._fetch_model_artifact(artifact_path)
        elif teacher_model == "inference_mode":
            self.teacher_model = self._load_model(teacher_model_name) 
        else:
            self.teacher_model = teacher_model

        self.teacher_model.eval()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
    
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = torch.nn.CrossEntropyLoss()(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels
    
    def training_step(self, batch, batch_idx):
        features, true_labels = batch
        # Extract cross-entropy loss and logits from student
        logits_stu = self(features)
        loss_ce = torch.nn.CrossEntropyLoss()(logits_stu, true_labels)
        
        # Extract logits from teacher
        with torch.no_grad():
            logits_tea = self.teacher_model(features)
        
        # Soften probabilities and compute distillation loss
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.hparams.temperature ** 2 * loss_fct(
            F.log_softmax(logits_stu / self.hparams.temperature, dim=-1),
            F.softmax(logits_tea / self.hparams.temperature, dim=-1)
        )
        
        # Return weighted student loss
        loss = self.hparams.alpha * loss_ce + (1. - self.hparams.alpha) * loss_kd
        
        predicted_labels = torch.argmax(logits_stu, dim=1)
        self.log("train_loss", loss,  on_epoch=False, on_step=True, sync_dist=True)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
        self.parameters(), 
        lr=self.hparams.learning_rate,
        momentum=self.hparams.momentum,
        weight_decay=self.hparams.weight_decay,
        nesterov=self.hparams.nesterov
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)
        return [optimizer], [sch]
    
    def _add_layers(self, in_features, hidden_units, num_classes, dropout_rates):
        new_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rates),
            torch.nn.Linear(hidden_units, num_classes),
        )
        return new_layers
    
    def _fetch_model_artifact(self, artifact_path:str):
        artifact = WandbLogger.download_artifact(artifact_path)
        teacher_model = LightningTraining.load_from_checkpoint(
           artifact+"/model.ckpt", model_name=self.hparams.teacher_model_name
        )
        return teacher_model
    
    def _load_model(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="DEFAULT")
        in_features = model.fc.in_features
        model.fc = self._add_layers(in_features, int(in_features/2), self.hparams.num_classes, self.hparams.dropout_rates)
        return model
    