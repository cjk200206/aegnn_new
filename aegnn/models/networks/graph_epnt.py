"""
用来实现图上的UNet,然后用作类似superpoint的结构
"""

import torch
import torch_geometric

import torch.nn.functional as F
from torch_sparse import spspmm

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   sort_edge_index)
from torch_geometric.utils.repeat import repeat


class EventPointNet(torch.nn.Module): 

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = pool_ratios
        self.act = act
        self.sum_res = sum_res

        channels = [4,8,16,32,64] #encoder的通道数

        self.encoder = torch.nn.ModuleList(

            GCNConv(in_channels, channels[0], improved=True),
            TopKPooling(channels[0], self.pool_ratios),
            GCNConv(channels[0], channels[1], improved=True),
            TopKPooling(channels[1], self.pool_ratios),
            GCNConv(channels[1], channels[2], improved=True),
            TopKPooling(channels[2], self.pool_ratios),
            GCNConv(channels[2], channels[3], improved=True),
            TopKPooling(channels[3], self.pool_ratios),
            GCNConv(channels[3], channels[4], improved=True), 
        )
        
        in_channels = channels if sum_res else 2 * channels


        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            layer.reset_parameters()


    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')



        
