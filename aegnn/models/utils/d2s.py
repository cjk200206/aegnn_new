"""
将标签转到16+1通道,变成符合输出的格式,并且加入dustbin,仿照superpoint
"""

import torch
import torch.nn as nn
import math
import numpy as np
from torch_geometric.data import Data 


# class DepthToSpace(nn.Module):
#     def __init__(self, block_size):
#         super(DepthToSpace, self).__init__()
#         self.block_size = block_size
#         self.block_size_sq = block_size*block_size

#     def forward(self, input):
#         output = input.permute(0, 2, 3, 1)
#         (batch_size, d_height, d_width, d_depth) = output.size()
#         s_depth = int(d_depth / self.block_size_sq)
#         s_width = int(d_width * self.block_size)
#         s_height = int(d_height * self.block_size)
#         t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
#         spl = t_1.split(self.block_size, 3)
#         stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
#         output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
#         output = output.permute(0, 3, 1, 2)
#         return output

class SeqToDepth(nn.Module):
    def __init__(self, seq_size):
        super(SeqToDepth, self).__init__()
        self.seq_size = seq_size


    def forward(self, input):
        length = math.ceil(len(input)/self.seq_size)
        output = np.resize(input.cpu(),(length,self.seq_size))
        # output = np.pad(output,(length,self.seq_size),mode="constant",constant_values=0)
        return output

def labels2Dto3D(labels,seq_size,add_dustbin=True,maxlabel = True):
    seq2depth = SeqToDepth(seq_size)
    labels = seq2depth(labels)
    
    if add_dustbin:
        dustbin = torch.from_numpy(labels).sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin<1.] = 0
        labels = torch.cat((torch.from_numpy(labels), dustbin.view(len(dustbin),1)), dim=1)
        ## norm
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
        if maxlabel: #保证只有一个标签是1
            labels_indices = torch.argmax(labels,dim=1)
            for row in range(len(labels)):
                labels[row,:] = 0
                labels[row,labels_indices[row]] = 1
    return labels.cuda(),labels_indices.cuda()

# def create_heatmap(batch: Data.batch,corner_num = 3):
#         batch.batch()
        
#         corner_idx = torch.where(batch.y==1)[0]
#         first_idx = corner_idx[:corner_num]
#         last_idx = corner_idx[-corner_num:]
#         new_idx = torch.cat([first_idx,last_idx]).unsqueeze(1)
#         heatmap = torch.zeros_like(batch.y).unsqueeze(0).expand(corner_num*2,-1) #创建多通道的heatmap
#         heatmap = heatmap.clone().scatter_(1,new_idx,torch.ones_like(new_idx)) #将指定位置标记，做成多通道热图
    
#     return heatmap