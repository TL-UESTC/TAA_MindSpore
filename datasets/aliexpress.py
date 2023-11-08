# import numpy as np
# import pandas as pd
# import torch

# class AliExpressDataset(torch.utils.data.Dataset):
#     """
#     AliExpress Dataset
#     This is a dataset gathered from real-world traffic logs of the search system in AliExpress
#     Reference:
#         https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
#         Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
#     """

#     def __init__(self, dataset_path):
#         data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
#         self.categorical_data = data[:, :16].astype(np.int)
#         self.numerical_data = data[:, 16: -2].astype(np.float32)
#         self.labels = data[:, -2:].astype(np.float32)
#         self.numerical_num = self.numerical_data.shape[1]
#         self.field_dims = np.max(self.categorical_data, axis=0) + 1

#     def __len__(self):
#         return self.labels.shape[0]

#     def __getitem__(self, index):
#         return self.categorical_data[index], self.numerical_data[index], self.labels[index]

# def _convert_to_mindrecord(data_home,features,labels,weight_np=None, training=True):

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.common.dtype as mstype
import numpy as np
import pandas as pd

class AliExpressDataset():
    def __init__(self, dataset_path):
        self._index = 0
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
        self.categorical_data = data[:, :16].astype(np.int)
        self.numerical_data = data[:, 16: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

#         self.dataset = ds.NumpySlicesDataset(
#             [self.categorical_data, self.numerical_data, self.labels],
#             column_names=['categorical_data', 'numerical_data', 'labels'],
#             shuffle=False)
        
#         self.parent = None  # 添加parent属性
#         self.children = None
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]
