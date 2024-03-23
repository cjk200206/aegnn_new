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


class EventPointNet(torch.nn.Module): 

    def __init__(self, input_shape: torch.Tensor,in_channels=1, out_channels=16,
                 pool_ratios=0.5, act=F.elu):
        super().__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])
        kernel_size = 8
        # kernel_size = [8,8,8,8]

        channels = [8,8,16,16,32,32,64,64] #encoder的通道数
        # channels = [64,64,64,64,128,128,128,128,256] #encoder的通道数

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_ratios = pool_ratios
        self.act = act

        # #Encoder-SplineConv
        # self.conv1 = SplineConv(self.in_channels, channels[0], dim = dim, kernel_size = kernel_size)
        # self.norm1 = BatchNorm(in_channels=channels[0])
        # self.conv2 = SplineConv(channels[0], channels[1], dim = dim, kernel_size = kernel_size)
        # self.norm2 = BatchNorm(in_channels=channels[1])
        # self.pool1 = TopKPooling(channels[1], self.pool_ratios)

        # self.conv3 = SplineConv(channels[1], channels[2], dim = dim, kernel_size = kernel_size)
        # self.norm3 = BatchNorm(in_channels=channels[2])
        # self.conv4 = SplineConv(channels[2], channels[3], dim = dim, kernel_size = kernel_size)
        # self.norm4 = BatchNorm(in_channels=channels[3])
        # self.pool2 = TopKPooling(channels[3], self.pool_ratios)

        # self.conv5 = SplineConv(channels[3], channels[4], dim = dim, kernel_size = kernel_size)
        # self.norm5 = BatchNorm(in_channels=channels[4])
        # self.conv6 = SplineConv(channels[4], channels[5], dim = dim, kernel_size = kernel_size)
        # self.norm6 = BatchNorm(in_channels=channels[5])
        # self.pool3 = TopKPooling(channels[5], self.pool_ratios)

        # self.conv7 = SplineConv(channels[5], channels[6], dim = dim, kernel_size = kernel_size)
        # self.norm7 = BatchNorm(in_channels=channels[6])
        # self.conv8 = SplineConv(channels[6], channels[7], dim = dim, kernel_size = kernel_size)
        # self.norm8 = BatchNorm(in_channels=channels[7])
        # self.pool4 = TopKPooling(channels[7], self.pool_ratios)
        # #输出节点是原来的0.5^4，即1/16
        
        # #Detector
        # self.convPa = SplineConv(channels[7], self.out_channels+1,dim = dim, kernel_size = kernel_size)
        # self.normPa = BatchNorm(in_channels=self.out_channels+1) #多加一个dustbin,16+1

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
        self.pool4 = TopKPooling(channels[7], self.pool_ratios)
        #输出节点是原来的0.5^4，即1/16
        
        #Detector
        self.convPa = GCNConv(channels[7], self.out_channels+1, improved = True)
        self.normPa = BatchNorm(in_channels=self.out_channels+1) #多加一个dustbin,16+1
        

    # def forward(self,data: torch_geometric.data.Batch):
    #     #Encoder
    #     data.x = self.act(self.conv1(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm1(data.x)
    #     data.x = self.act(self.conv2(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm2(data.x)
    #     data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool1(data.x,data.edge_index,data.edge_attr)
        
    #     data.x = self.act(self.conv3(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm3(data.x)
    #     data.x = self.act(self.conv4(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm4(data.x)
    #     data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool2(data.x,data.edge_index,data.edge_attr)

    #     data.x = self.act(self.conv5(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm5(data.x)
    #     data.x = self.act(self.conv6(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm6(data.x)
    #     data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool3(data.x,data.edge_index,data.edge_attr)

    #     data.x = self.act(self.conv7(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm7(data.x)
    #     data.x = self.act(self.conv8(data.x, data.edge_index, data.edge_attr))
    #     data.x = self.norm8(data.x)
    #     data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool4(data.x,data.edge_index,data.edge_attr)

    #     #Detector
    #     data.x = self.act(self.convPa(data.x, data.edge_index, data.edge_attr))
    #     x = self.normPa(data.x)
        
    #     return x

    #for GCNConv
    def forward(self,data: torch_geometric.data.Batch):
        #Encoder
        data.x = self.act(self.conv1(data.x, data.edge_index))
        data.x = self.norm1(data.x)
        data.x = self.act(self.conv2(data.x, data.edge_index))
        data.x = self.norm2(data.x)
        data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool1(data.x,data.edge_index)
        
        data.x = self.act(self.conv3(data.x, data.edge_index))
        data.x = self.norm3(data.x)
        data.x = self.act(self.conv4(data.x, data.edge_index))
        data.x = self.norm4(data.x)
        data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool2(data.x,data.edge_index)

        data.x = self.act(self.conv5(data.x, data.edge_index))
        data.x = self.norm5(data.x)
        data.x = self.act(self.conv6(data.x, data.edge_index))
        data.x = self.norm6(data.x)
        data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool3(data.x,data.edge_index)

        data.x = self.act(self.conv7(data.x, data.edge_index))
        data.x = self.norm7(data.x)
        data.x = self.act(self.conv8(data.x, data.edge_index))
        data.x = self.norm8(data.x)
        data.x, data.edge_index, data.edge_attr, data.batch, _, _= self.pool4(data.x,data.edge_index)

        #Detector
        data.x = self.act(self.convPa(data.x, data.edge_index))
        x = self.normPa(data.x)
        
        return x





        
