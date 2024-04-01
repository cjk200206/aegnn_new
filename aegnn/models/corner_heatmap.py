"""Partly copied from rpg-asynet paper: https://github.com/uzh-rpg/rpg_asynet"""
from collections import OrderedDict
import torch
import torch_geometric
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as pl_metrics
from torch_geometric.nn.models import GraphUNet

from torch.nn.functional import softmax
from typing import Any, Dict, Tuple
from .networks import by_name as model_by_name
from .networks.graph_epnt import EventPointNet

from .utils.d2s import labels2Dto3D


class CornerHeatMapModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                dim: int = 3, learning_rate: float = 5e-3, **model_kwargs):
        super(CornerHeatMapModel, self).__init__()
        #设置损失权重
        self.in_channels = 9
        self.out_channels = 16
        loss_weight = torch.ones(self.out_channels+1)
        # loss_weight[self.out_channels] = 16
        loss_weight.squeeze()

        self.optimizer_kwargs = dict(lr=learning_rate)
        # self.criterion = torch.nn.BCELoss() 
        self.criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
        self.num_outputs = num_classes
        self.dim = dim
        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        self.model = EventPointNet(input_shape=model_input_shape,in_channels=self.in_channels,out_channels=self.out_channels)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.pos = data.pos[:, :self.dim]
        data.edge_attr = data.edge_attr[:, :self.dim]
        return self.model.forward(data)

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)
        # #尝试模仿superpoint
        # label,label_indice = labels2Dto3D(batch.y,self.out_channels)
        label = batch.y

        loss = self.criterion(outputs.cuda(), target=label.cuda())
        accuracy = pl_metrics.accuracy(preds=y_prediction, target=label)
        recall = pl_metrics.recall(preds=y_prediction,target=label)
        self.logger.log_metrics({"Train/Loss": loss, "Train/Accuracy": accuracy,"Train/Recall": recall}, step=self.trainer.global_step)
        return loss















    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)
        #尝试模仿superpoint
        label,label_indice = labels2Dto3D(batch.y,self.out_channels)

        self.log("Val/Loss", self.criterion(outputs, target=label))
        # self.log("Val/Loss", self.criterion(predictions, target=label_indice))
        self.log("Val/Accuracy", pl_metrics.accuracy(preds=y_prediction, target=label_indice))
        self.log("Val/Recall",pl_metrics.recall(preds=y_prediction,target=label_indice)) #加入召回率评价

        if batch_idx == 9:
            print("\npred:",y_prediction[-20:], "\ngt:",label_indice[-20:]) #加入效果查看

        return predictions
    
    def predict_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-2, **self.optimizer_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRPolicy())
        return [optimizer], [lr_scheduler]


class LRPolicy(object):
    def __call__(self, epoch: int):
        if epoch < 20:
            return 5e-3
        else:
            return 5e-4

