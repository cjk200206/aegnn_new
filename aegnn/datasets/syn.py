import glob
import numpy as np
import os
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, Dict, List, Optional, Union

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
    def read_label(raw_file: str,end_idx) -> Optional[Union[str, List[str]]]:
        events = np.loadtxt(raw_file)
        corner_label = events[end_idx][-1]
        return "corner" if corner_label == 1 else "not" #获取每个文本最后一个事件判断是否为角点

    @staticmethod
    def load(raw_file: str) -> Data:
        events = np.loadtxt(raw_file)
        events,start_idx,end_idx = Syn.event_cropping(events,len(events)) #裁剪过后的事件
        events = torch.from_numpy(events).float().cuda()
        x, pos = events[:, [-2]], events[:, :3]
        pos[:,:2]=pos[:,:2].float()
        pos[:,2]=pos[:,2]*1e-9 #把时间转换成秒的科学计数
        return Data(x=x, pos=pos),start_idx,end_idx
    
    #修改预处理的流程，适应随机裁剪
    @staticmethod
    def processing(rf: str, pf: str, load_func: Callable[[str], Data],
                   class_dict: Dict[str, int], read_label: Callable[[str], str],
                   read_annotations: Callable[[str], np.ndarray], pre_transform: Callable = None):
        rf_wo_ext, _ = os.path.splitext(rf)

        # Load data from raw file. If the according loaders are available, add annotation, label and class id.
        device = "cpu"  # torch.device(torch.cuda.current_device())
        data_obj,start_idx,end_idx = load_func(rf) #修改load,包括输出裁剪的前后
        data_obj = data_obj.to(device)
        data_obj.file_id = os.path.basename(rf)
        if (label := read_label(rf,end_idx)) is not None: #加入end_idx,即判断裁剪后的片段
            data_obj.label = label if isinstance(label, list) else [label]
            data_obj.y = torch.tensor([class_dict[label] for label in data_obj.label])


        # Apply the pre-transform on the graph, to afterwards store it as .pt-file.
        assert data_obj.pos.size(1) == 3, "pos must consist of (x, y, t)"
        if pre_transform is not None:
            data_obj = pre_transform(data_obj)

        # Save the data object as .pt-torch-file. For the sake of a uniform processed
        # directory format make all output paths flat.
        os.makedirs(os.path.dirname(pf), exist_ok=True)
        torch.save(data_obj.to("cpu"), pf)
    
    def event_cropping(events,length,percent = 0.1): #随机裁剪一个事件的片段，作为输入
        start_idx = np.random.randint(0,length-percent*length-1)
        end_idx = int(start_idx+percent*length)
        cropped_events = events[start_idx:end_idx+1] #截取到包含随后一个事件

        return cropped_events,start_idx,end_idx

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
        return ["corner","not"]
