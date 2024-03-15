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
# from .networks.graph_unet import Graph_UNet


class CornerModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                dim: int = 3, learning_rate: float = 5e-3, **model_kwargs):
        super(CornerModel, self).__init__()
        self.optimizer_kwargs = dict(lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([10.0,1.0])) #调整网络的权重
        self.num_outputs = num_classes
        self.dim = dim

        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        # self.model = model_by_name(network)(dataset, model_input_shape, num_outputs=num_classes, **model_kwargs)
        self.model = GraphUNet(in_channels=1,hidden_channels=64,out_channels=2,depth=4)


    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.pos = data.pos[:, :self.dim]
        data.edge_attr = data.edge_attr[:, :self.dim]
        return self.model.forward(data.x,data.edge_index,data.batch) #为了迎合UNET

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        y_prediction = torch.softmax(outputs, dim=-1)
        loss = self.criterion(outputs, target=batch.y)

        accuracy = pl_metrics.accuracy(preds=y_prediction, target=batch.y)
        recall = pl_metrics.recall(preds=y_prediction,target=batch.y, num_classes=2, average='none')
        self.logger.log_metrics({"Train/Loss": loss, "Train/Accuracy": accuracy,"Train/Recall": recall}, step=self.trainer.global_step)
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)

        self.log("Val/Loss", self.criterion(outputs, target=batch.y))
        self.log("Val/Accuracy", pl_metrics.accuracy(preds=y_prediction, target=batch.y))
        self.log("Val/Recall",pl_metrics.recall(preds=y_prediction,target=batch.y, num_classes=2, average='none')) #加入召回率评价

        if batch_idx == 9:
             print("\npred:",y_prediction[-10:], "gt",batch.y[-10:]) #加入效果查看

        k = min(3, self.num_outputs - 1)
        self.log(f"Val/Accuracy_Top{k}", pl_metrics.accuracy(preds=predictions, target=batch.y, top_k=k))
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

