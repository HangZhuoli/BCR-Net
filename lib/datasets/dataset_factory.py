from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path))
sys.path.append(os.path.join(current_path, '..'))
print(sys.path)

from lib.datasets.sample.ctdet_to_GrayGraph import CTDetDataset
from lib.datasets.dataset.visdrone2019DET import VisDrone2019DET
from lib.datasets.dataset.uavdt import UAVDT
from lib.datasets.dataset.crater import CraterDataset
#编写数据集的加载方案，同时实现并行的处理框架

dataset_factory = {
    'visdrone':VisDrone2019DET,
    'uavdt':UAVDT,
    
    # 新增加陨石坑处理数据集代码，用于处理陨石坑单一目标检测算法的识别
    'crater':CraterDataset
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], CTDetDataset):
        pass
    return Dataset
    
