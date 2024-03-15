from .graph_res import GraphRes
from .graph_wen import GraphWen
from .graph_res_new import GraphResNew
from .graph_epnt import EventPointNet

################################################################################################
# Access functions #############################################################################
################################################################################################
import torch


def by_name(name: str) -> torch.nn.Module.__class__:
    if name == "graph_res":
        return GraphRes
    elif name == "graph_wen":
        return GraphWen
    elif name == "graph_res_new":
        return GraphResNew
    elif name == "graph_epnt":
        return EventPointNet
    else:
        raise NotImplementedError(f"Network {name} is not implemented!")
