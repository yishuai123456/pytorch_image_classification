import os
from typing import Tuple, Union

import pathlib
import scipy.io
import numpy as np
import torch
import torchvision
import yacs.config

from torch.utils.data import Dataset

from pytorch_image_classification import create_transform


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


class MatDataset(Dataset):
    def __init__(self,mat_files,datasetaName,transform=None):
        self.datasetName=datasetaName
        self.transform=transform
        self.mat_files=mat_files

        self.receiver_depth_number=10
        self.receiver_depth=[20,60,100,140,180,220,260,300,340,380]

        # if self.datasetName=='Environment_Bty':
        #     for mat_file in mat_files:
        #         mat = scipy.io.loadmat(mat_file)
        #         if np.any(np.isnan(mat['tempBty'])): ##去掉nan值
        #             continue
        #
        #         data=mat['tempBty']
        #         label = np.squeeze(mat['label'])
        #
        #         self.data.append(data)
        #         self.labels.append(label)
        #
        # else:
        #     for mat_file in mat_files:
        #         mat = scipy.io.loadmat(mat_file)
        #         BTY = np.nan_to_num(mat['bty'])
        #         SSP = np.nan_to_num(mat['ssp'])
        #         data = np.zeros([80, 80, 116])
        #         data[:, :, 0] = BTY[0:52, 0:52]
        #         data[:, :, 1:116] = SSP
        #
        #         origin_label = mat['label']
        #         if (abs(origin_label[0]) > abs(origin_label[3])):
        #             if (origin_label[0] > 0):
        #                 label = 0
        #             else:
        #                 label = 2
        #         else:
        #             if (origin_label[3] > 0):
        #                 label = 1
        #             else:
        #                 label = 3
        #
        #         if (abs(origin_label[0]) < 0.1 and abs(origin_label[3]) < 0.1):
        #             label = 4
        #
        #         self.data.append(data)
        #         self.labels.append(label)



    def __len__(self):
        return len(self.mat_files)*self.receiver_depth_number

    def __getitem__(self, idx):
        idx=idx//self.receiver_depth_number
        depth_index=idx%self.receiver_depth_number

        ##非环境参数
        receiver_range=0
        frequency=0
        receiver_depth=(self.receiver_depth[depth_index]-np.mean(self.receiver_depth))/np.var(self.receiver_depth)
        env_info=np.array([receiver_depth,receiver_range,frequency])  #分别对应探测器深度，目标大致范围以及声源频率

        ##环境参数
        mat = scipy.io.loadmat(self.mat_files[idx])
        BTY = np.nan_to_num(mat['bty'])
        SSP = np.nan_to_num(mat['ssp'])
        data = np.zeros([80, 80, 251])
        data[:, :, 0] = BTY
        data[:, :, 1:251] = self.process_ssp(SSP)


        ##标签
        original_label = mat['label']
        label = original_label[depth_index][0]

        if self.transform:
            data=self.transform(data)

        #sample=torch.tensor(sample,dtype=torch.float32)
        label=torch.tensor(label,dtype=torch.long)
        env_info=torch.tensor(env_info,dtype=torch.float32)
        return data,env_info,label

    def process_ssp(self, ssp, return_ssp_depth_size=250):
        process_ssp = np.zeros([ssp.shape[0], ssp.shape[1], return_ssp_depth_size])

        # 用最后一个非 NaN 值填充 NaN 值
        nan_mask = np.isnan(ssp)
        non_nan_indices = np.where(~nan_mask, np.arange(nan_mask.shape[2])[None, None, :], 0)
        np.maximum.accumulate(non_nan_indices, axis=2, out=non_nan_indices)
        filled_ssp = ssp[np.arange(ssp.shape[0])[:, None, None],
        np.arange(ssp.shape[1])[None, :, None],
        non_nan_indices]

        # 处理填充后的数据，按照目标深度尺寸进行拷贝
        for i in range(ssp.shape[0]):
            for j in range(ssp.shape[1]):
                end_index = min(ssp.shape[2], return_ssp_depth_size)
                process_ssp[i, j, :end_index] = filled_ssp[i, j, :end_index]
                if end_index < return_ssp_depth_size:
                    process_ssp[i, j, end_index:] = filled_ssp[i, j, end_index - 1]

        return process_ssp


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST',
            'KMNIST',
    ]:
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    elif config.dataset.name == 'ImageNet':
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_dir / 'train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                       transform=val_transform)
        return train_dataset, val_dataset


    else:
        data_dir=config.dataset.dataset_dir
        mat_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')]

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(mat_files) * val_ratio)
        train_num = len(mat_files) - val_num
        lengths = [train_num, val_num]

        train_mat_files=np.random.choice(mat_files,size=train_num,replace=False)
        val_mat_files=np.setdiff1d(mat_files,train_mat_files)

        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = MatDataset(train_mat_files, config.dataset.name,train_transform)
        val_dataset = MatDataset(val_mat_files, config.dataset.name,val_transform)

        return train_dataset, val_dataset


