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
from .utils.loss_functions import FocalLoss
from .utils.data_evaluation import data_evaluation


class CornerSuperpointModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                dim: int = 3, learning_rate: float = 5e-3, **model_kwargs):
        super(CornerSuperpointModel, self).__init__()
        #设置损失权重
        self.in_channels = 9
        self.out_channels = 16
        loss_weight = torch.ones(self.out_channels+1)
        # loss_weight[self.out_channels] = 16
        loss_weight.squeeze()

        self.optimizer_kwargs = dict(lr=learning_rate)
        self.criterion = torch.nn.BCELoss(reduction="none")
        # self.criterion = FocalLoss(gamma=2, alpha=torch.tensor([0.5]*(self.out_channels+1)))
        # self.num_outputs = num_classes
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
        batch_copy = batch.batch
        label = batch.y
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)
        
        ##计算掩膜，解决GNNbatch问题
        max_batch = batch_copy.max().item() + 1  # Calculate the number of batches,计算最多有多少个batch
        labels = []
        label_indices = []
        for i in range(max_batch):
            mask = (batch_copy == i)  # Create a mask for nodes in the current batch，计算一个掩膜
            label_batch = label[mask]  # 将掩膜用于labels
            label_batch,label_batch_indice = labels2Dto3D(label_batch,self.out_channels) # 计算新的label
            labels.append(label_batch)
            label_indices.append(label_batch_indice)            
        labels = torch.cat(labels, dim=0) 
        label_indices = torch.cat(label_indices, dim=0)

        ##评价指标
        # loss = self.criterion(outputs.cuda(), target=label_indices)
        loss = self.criterion(predictions, target=labels).sum()/labels.shape[0]
        accuracy = pl_metrics.accuracy(preds=y_prediction, target=label_indices)
        # valid_idxs = torch.where(label_indices != 16)
        # if len(valid_idxs[0]) != 0:
        #     accuracy_valid = pl_metrics.accuracy(preds=y_prediction[valid_idxs], target=label_indices[valid_idxs])
        #     loss_valid = self.criterion(outputs[valid_idxs].cuda(), target=label_indices[valid_idxs].cuda())
        # else:
        #     accuracy_valid = 0
        #     loss_valid = 0
        recall = pl_metrics.recall(preds=y_prediction,target=label_indices)
        # self.logger.log_metrics({"Train/Loss": loss, "Train/Loss_Valid": loss_valid,\
        #                          "Train/Accuracy": accuracy, "Train/Accuracy_Valid": accuracy_valid, "Train/Recall": recall}, step=self.trainer.global_step)
        # return loss+loss_valid
        
        self.logger.log_metrics({"Train/Loss": loss, "Train/Accuracy": accuracy, "Train/Recall": recall}, step=self.trainer.global_step)
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        #计算两类样本的相似度
        not_corner_idx = torch.where(batch.y!=1)
        is_corner_idx = torch.where(batch.y==1)
        not_corner = batch.x[not_corner_idx].cpu()
        is_corner = batch.x[is_corner_idx].cpu()
        # not_corner = tuple(not_corner[:])
        # is_corner = tuple(is_corner[:])
        # dis1,dis2,dis1_2 = data_evaluation(not_corner,is_corner)
        # print("\ndis1:",dis1,"\ndis2:",dis2,"\ndis1_2:",dis1_2)

        batch_copy = batch.batch
        label = batch.y
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)

        ##计算掩膜，解决GNNbatch问题
        max_batch = batch_copy.max().item() + 1  # Calculate the number of batches,计算最多有多少个batch
        labels = []
        label_indices = []
        for i in range(max_batch):
            mask = (batch_copy == i)  # Create a mask for nodes in the current batch，计算一个掩膜
            label_batch = label[mask]  # 将掩膜用于labels
            label_batch,label_batch_indice = labels2Dto3D(label_batch,self.out_channels) # 计算新的label
            labels.append(label_batch)
            label_indices.append(label_batch_indice)            
        labels = torch.cat(labels, dim=0) 
        label_indices = torch.cat(label_indices, dim=0)

        ##评价指标
        # self.log("Val/Loss", self.criterion(outputs, target=label_indices))
        self.log("Val/Loss", self.criterion(predictions, target=labels).sum()/labels.shape[0])
        self.log("Val/Accuracy", pl_metrics.accuracy(preds=y_prediction, target=label_indices))
        self.log("Val/Recall",pl_metrics.recall(preds=y_prediction,target=label_indices)) #加入召回率评价
        # valid_idxs = torch.where(label_indices != 16)
        # if batch_idx == 9:
        #     print("\npred:",y_prediction[valid_idxs], "\ngt:",label_indices[valid_idxs]) #加入效果查看

        # if len(valid_idxs[0]) != 0:
        #     self.log("Val/Accuracy_Valid", pl_metrics.accuracy(preds=y_prediction[valid_idxs], target=label_indices[valid_idxs]))
        #     self.log("Val/Loss_Valid", self.criterion(outputs[valid_idxs].cuda(), target=label_indices[valid_idxs].cuda()))
        # else:
        #     self.log("Val/Accuracy_Valid",0)
        #     self.log("Val/Loss_Valid",0)
        

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

