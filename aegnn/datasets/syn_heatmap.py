import glob
import numpy as np
import os
import torch
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, Dict, List, Optional, Union

from .utils.normalization import normalize_time
from .ncaltech101 import NCaltech101


class Syn_Heatmap(NCaltech101):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8, pin_memory: bool = False,
                 transform: Optional[Callable[[Data], Data]] = None):
        super(Syn_Heatmap, self).__init__(batch_size, shuffle, num_workers, pin_memory=pin_memory, transform=transform)
        self.dims = (346, 260)  # overwrite image shape,改到davis346格式
        # pre_processing_params = {"r": 3.0, "d_max": 32, "n_samples": 10000, "sampling": True}
        pre_processing_params = {"r": 3.0, "d_max": 9, "n_samples": 10000, "sampling": False}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def read_label(raw_file: str,start_idx,end_idx) -> Optional[Union[str, List[str]]]:
        events = np.loadtxt(raw_file)
        corner_label = events[start_idx:end_idx+1][:,-1]
        labels_new = []
        for label in corner_label: #将角点和周围的关联点都当成角点，扩充数据
            label = "corner" if label == 1 else "not" #将角点和周围的关联点都当成角点，扩充数据
            labels_new.append(label)
        return labels_new #获取事件段的标签

    @staticmethod
    def load(raw_file: str) -> Data:
        events = np.loadtxt(raw_file)
        events,start_idx,end_idx = Syn_Heatmap.event_cropping(events,len(events)) #裁剪过后的事件
        events = torch.from_numpy(events).float().cuda()
        # x, pos = events[:, [-2]], events[:, :3]
        x, pos = events[:, :3], events[:, :3] #尝试调整x的内容，加入x,y,t
        pos[:,:2]=pos[:,:2].float()
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
        if (label := read_label(rf,start_idx,end_idx)) is not None: #加入start_idx,end_idx,即判断裁剪后的片段
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
    
    def event_cropping(events,length,percent = 0.5): #随机裁剪一个事件的片段，作为输入
        start_idx = np.random.randint(0,length-percent*length-1)
        end_idx = int(start_idx+percent*length)
        cropped_events = events[start_idx:end_idx+1] #截取到包含随后一个事件

        return cropped_events,start_idx,end_idx
    
    #将时空事件流建立边关系后，转到3*3的角点空间表示中
    def create_corner_feature(self,data: Data) -> Data.x: 
        corner_template = torch.zeros(data.x.shape[0],3,3) #定义角点模板
        # corner_center = self.pos #定义时空上的角点中心
        edge_start = data.edge_index[0,:]
        edge_end = data.edge_index[1,:] #标记边的开头结尾
        related_pos = data.pos[edge_start]-data.pos[edge_end] #时空xyt做差,起点（边连接的末端）-终点（边连接的中心）
        combined_pos = torch.concat([edge_end.unsqueeze(dim=1),related_pos],dim=1) #node_end_idx,x,y,t

        combined_pos[:,1]=torch.where(combined_pos[:,1]<-1,torch.tensor(-1.0),combined_pos[:,1]) #将所有点的delta_x限制在-1,1之间
        combined_pos[:,1]=torch.where(combined_pos[:,1]>1,torch.tensor(1.0),combined_pos[:,1])
        combined_pos[:,2]=torch.where(combined_pos[:,2]<-1,torch.tensor(-1.0),combined_pos[:,2]) #将所有点的delta_y限制在-1,1之间
        combined_pos[:,2]=torch.where(combined_pos[:,2]>1,torch.tensor(1.0),combined_pos[:,2])

        prev_related_pos = combined_pos[torch.where(combined_pos[:,3]<=0)][:,:3] #t<=0,代表连接的边早于中心,筛选出先发生的事件边
        prev_related_pos[:,1] += 1 #坐标差值等价转换到角点模板索引
        prev_related_pos[:,2] += 1
        prev_related_pos = prev_related_pos.to(torch.long)

        corner_template[prev_related_pos[:,0],prev_related_pos[:,1],prev_related_pos[:,2]] += 1 #将边连接的事件按时空关系，填入角点模板
        corner_feature = corner_template.view(-1,9) #将角点模板转换到一维
        return corner_feature
    
    #选取开头和结尾各corner_num个角点
    def create_heatmap(self,data: Data,corner_num = 3) -> Data.y:
        corner_idx = torch.where(data.y==1)[0]
        first_idx = corner_idx[:corner_num]
        last_idx = corner_idx[-corner_num:]
        new_idx = torch.cat([first_idx,last_idx]).unsqueeze(1)
        heatmap = torch.zeros_like(data.y).unsqueeze(0).expand(corner_num*2,-1) #创建多通道的heatmap
        heatmap = heatmap.clone().scatter_(1,new_idx,torch.ones_like(new_idx)) #将指定位置标记，做成多通道热图
        
        return heatmap.T


    #生成关联点
    def create_related_points(self,data: Data) -> Data.y:
        corner_idx = torch.where(data.y==1)[0]
        related_points_indices_raw = torch.where(torch.isin(data.edge_index[1],corner_idx))[0] #找寻角点所在的边的索引
        related_points_idx_raw = data.edge_index[0,related_points_indices_raw] #找寻角点所连接的点的索引
        related_points_idx = related_points_idx_raw[torch.where(data.y[related_points_idx_raw]==0)[0]] #排除角点连接的角点
        data.y[related_points_idx[:]]=1 #赋值新的标记
        return data.y
    

    def pre_transform(self, data: Data) -> Data:
        params = self.hparams.preprocessing

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = normalize_time(data.pos[:, 2])

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        # data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"]) #不要下采样

        # Radius graph generation.
        data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])

        # 生成表示角点的corner_feature表示
        data.x = self.create_corner_feature(data)

        # 保留一部分的角点，生成heatmap
        data.y = self.create_heatmap(data)
        
        # # 生成关联点
        # data.y = self.create_related_points(data)

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

