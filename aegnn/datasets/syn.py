import glob
import numpy as np
import os
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, List, Optional, Union

from .utils.normalization import normalize_time
from .ncaltech101 import NCaltech101


class Syn(NCaltech101):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8, pin_memory: bool = False,
                 transform: Optional[Callable[[Data], Data]] = None):
        super(Syn, self).__init__(batch_size, shuffle, num_workers, pin_memory=pin_memory, transform=transform)
        self.dims = (346, 260)  # overwrite image shape,改到davis346格式
        pre_processing_params = {"r": 3.0, "d_max": 32, "n_samples": 10000, "sampling": True}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
        events = np.loadtxt(raw_file)
        corner_label = events[-1][-1]
        return "corner" if corner_label == "1" else "not" #获取每个文本最后一个事件判断是否为角点

    @staticmethod
    def load(raw_file: str) -> Data:
        events = torch.from_numpy(np.loadtxt(raw_file)).float().cuda()
        x, pos = events[:, [-2]], events[:, :3]
        pos[:,:2]=pos[:,:2].float()
        pos[:,2]=pos[:,2]*1e-9 #把时间转换成秒的科学计数
        return Data(x=x, pos=pos)

    def pre_transform(self, data: Data) -> Data:
        params = self.hparams.preprocessing

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = normalize_time(data.pos[:, 2])

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])

        # Radius graph generation.
        data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
        return data

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    """
    syn_corner数据集通过syn2e建立,具体格式如下：
    /datasets
        /train
            /syn_polygon
                /augmented_events
                    /0
                        /0000000000.txt
                        /0000000001.txt
                        /others
                    /1
                    /2
                    /others
                /event_corners
                /events
                /others
            /syn_mutiple_polygons
            /others
        /val
    """
    
    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode,"*","augmented_events","*","*.txt")) #查找未处理过的源文件，mode代表train、val等

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode,"*","augmented_events","*","*"))

    @property
    def classes(self) -> List[str]:
        return ["not","corner"]
