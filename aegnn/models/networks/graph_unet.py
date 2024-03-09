import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX


class Graph_UNet(GraphUNet):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                bias: bool = False, root_weight: bool = False):
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        # if dataset == "syn":
        #     kernel_size = 2
        #     n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
        #     pooling_outputs = 32
        # else:
        #     raise NotImplementedError(f"No model parameters for dataset {dataset}")

        in_channels = 1
        hidden_channels = 16 
        out_channels = num_outputs
        depth = 3
        super(Graph_UNet, self).__init__(in_channels,hidden_channels,out_channels,depth)
        
