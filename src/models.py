import torch.nn.functional as F
import lightning as L
import torch
import torchmetrics
import torchvision

class PetModel(torch.nn.Module):
    def __init__(self, model_name:str, 
                 weights:str = None,
                 in_features:int = 512, 
                 num_classes:int =37):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="DEFAULT")
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 37)

    def forward(self, x):
        return self.model(x)
    
class LightningModel(L.LightningModule):
    def __init__(
        self, 
        model=None, 
        model_name:str = "resnet18",
        num_classes:int = 37,
        learning_rate:float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        nesterov: bool = False,
        T_max : int = 5,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['teacher_model', "model", "model_name"])
        self.model_name = model_name
        self.learning_rate = learning_rate

        if model is None:
            self.model = PetModel(self.model_name)
            
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=False, on_step=True)

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
    
class DistilledTrainingModule(LightningModel):
    def __init__(self, *args, teacher_model, alpha:float=0.5, temperature:float=2.0, **kwargs):
        super(DistilledTrainingModule, self).__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature

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
        loss_kd = self.temperature ** 2 * loss_fct(
            F.log_softmax(logits_stu / self.temperature, dim=-1),
            F.softmax(logits_tea / self.temperature, dim=-1)
        )
        
        # Return weighted student loss
        loss = self.alpha * loss_ce + (1. - self.alpha) * loss_kd
        
        predicted_labels = torch.argmax(logits_stu, dim=1)
        self.log("train_loss", loss,  sync_dist=True)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss