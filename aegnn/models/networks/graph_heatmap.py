"""
用来实现图上的UNet,然后用作类似superpoint的结构
"""

import torch
import torch_geometric

from torch.nn import ReLU
import torch.nn.functional as F
from torch_sparse import spspmm

from torch_geometric.nn import SplineConv, TopKPooling, BatchNorm,GCNConv,GATConv
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   sort_edge_index)
from torch_geometric.utils.repeat import repeat


class HeatMapNet(torch.nn.Module): 

    def __init__(self, input_shape: torch.Tensor,in_channels=9, out_channels=6,
                 pool_ratios=0.5, act=F.elu):
        super().__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"

        channels = [16,16,32,32,64,64,128,128] #encoder的通道数


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_ratios = pool_ratios
        self.act = act

        #Encoder-GCNConv
        self.conv1 = GCNConv(self.in_channels, channels[0], improved = True)
        self.norm1 = BatchNorm(in_channels=channels[0])
        self.conv2 = GCNConv(channels[0], channels[1], improved = True)
        self.norm2 = BatchNorm(in_channels=channels[1])
        self.pool1 = TopKPooling(channels[1], self.pool_ratios)

        self.conv3 = GCNConv(channels[1], channels[2], improved = True)
        self.norm3 = BatchNorm(in_channels=channels[2])
        self.conv4 = GCNConv(channels[2], channels[3], improved = True)
        self.norm4 = BatchNorm(in_channels=channels[3])
        self.pool2 = TopKPooling(channels[3], self.pool_ratios)

        self.conv5 = GCNConv(channels[3], channels[4], improved = True)
        self.norm5 = BatchNorm(in_channels=channels[4])
        self.conv6 = GCNConv(channels[4], channels[5], improved = True)
        self.norm6 = BatchNorm(in_channels=channels[5])
        self.pool3 = TopKPooling(channels[5], self.pool_ratios)

        self.conv7 = GCNConv(channels[5], channels[6], improved = True)
        self.norm7 = BatchNorm(in_channels=channels[6])
        self.conv8 = GCNConv(channels[6], channels[7], improved = True)
        self.norm8 = BatchNorm(in_channels=channels[7])
        #输出节点是原来的0.5^3，即1/8
        
        #Decoder-Heatmap
        self.convPa = GCNConv(channels[1], self.out_channels, improved = True)
        self.convPb = GCNConv(channels[3], self.out_channels, improved = True)
        self.convPc = GCNConv(channels[5], self.out_channels, improved = True)
        self.convPd = GCNConv(channels[7], self.out_channels, improved = True)
        self.normPHeatmap = BatchNorm(in_channels=self.out_channels) 

    #for GCNConv
    def forward(self,data: torch_geometric.data.Batch):
        # data.edge_weight = data.x.new_ones(data.edge_index.size(1))

        #Encoder
        data.x = self.act(self.conv1(data.x, data.edge_index))
        data.x = self.norm1(data.x)
        data.x = self.act(self.conv2(data.x, data.edge_index))
        data.x = self.norm2(data.x)
        HeatMapA = data.x #保存原始大小的HeatMap,edge_index,edge_weight
        edge_indexA = data.edge_index 
        # edge_weightA = data.edge_weight
        data.x, data.edge_index, data.edge_weight, data.batch, permB, _= self.pool1(data.x,data.edge_index) #perm是留下来节点的idx
        
        data.x = self.act(self.conv3(data.x, data.edge_index))
        data.x = self.norm3(data.x)
        data.x = self.act(self.conv4(data.x, data.edge_index))
        data.x = self.norm4(data.x)
        HeatMapB = data.x #保存0.5大小的HeatMap,edge_index,edge_weight
        edge_indexB = data.edge_index 
        # edge_weightB = data.edge_weight
        data.x, data.edge_index, data.edge_weight, data.batch, permC, _= self.pool2(data.x,data.edge_index)

        data.x = self.act(self.conv5(data.x, data.edge_index))
        data.x = self.norm5(data.x)
        data.x = self.act(self.conv6(data.x, data.edge_index))
        data.x = self.norm6(data.x)
        HeatMapC = data.x #保存0.25大小的HeatMap,edge_index,edge_weight
        edge_indexC = data.edge_index 
        # edge_weightC = data.edge_weight
        data.x, data.edge_index, data.edge_weight, data.batch, permD, _= self.pool3(data.x,data.edge_index)

        data.x = self.act(self.conv7(data.x, data.edge_index))
        data.x = self.norm7(data.x)
        data.x = self.act(self.conv8(data.x, data.edge_index))
        data.x = self.norm8(data.x)
        HeatMapD = data.x #保存0.125大小的HeatMap,edge_index,edge_weight
        edge_indexD = data.edge_index 
        # edge_weightD = data.edge_weight
        
        #Decoder
        HeatMapA = self.normPHeatmap(self.act(self.convPa(HeatMapA, edge_indexA))) #输出4个size的10通道heatmap
        HeatMapB = self.normPHeatmap(self.act(self.convPb(HeatMapB, edge_indexB)))
        HeatMapC = self.normPHeatmap(self.act(self.convPc(HeatMapC, edge_indexC)))
        HeatMapD = self.normPHeatmap(self.act(self.convPd(HeatMapD, edge_indexD)))
        
        upD = torch.zeros_like(HeatMapC) #将不同size的heatmap合到一起
        upD[permD] = HeatMapD
        HeatMapCD = upD+HeatMapC

        upC = torch.zeros_like(HeatMapB)
        upC[permC] = HeatMapCD
        HeatMapBCD = upC+HeatMapB

        upB = torch.zeros_like(HeatMapA)
        upB[permB] = HeatMapBCD
        HeatMapABCD = upB+HeatMapA

        x = self.normPHeatmap(HeatMapABCD)
        
        return x





        
