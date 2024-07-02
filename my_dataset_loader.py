import os.path

import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
data_dir="D:/project/environmentFactor/environmentFactorDataset/DataLayer"
# mat=scipy.io.loadmat('D:/project/environmentFactor/environmentFactorDataset/dataset_btyAndSSP/unlabeled/1.mat')
# print(mat['btyAndSSP'][:,:,50])

class MatDataset(Dataset):
    def __init__(self,mat_files,transform=None):
        self.data=[]
        self.labels=[]
        self.transform=transform

        for mat_file in mat_files:
            mat = scipy.io.loadmat(mat_file)
            BTY = np.nan_to_num(mat['BTY'])
            SSP = np.nan_to_num(mat['SSP'])
            data=np.zeros([52,52,2])
            data[:,:,0]=BTY[0:52,0:52]
            data[:,:,1]=SSP[0:52,0:52]

            origin_label=mat['Label4']
            if(abs(origin_label[0])>abs(origin_label[3])):
                if (origin_label[0]>0):
                    label=0
                else:
                    label=2
            else:
                if(origin_label[3]>0):
                    label=1
                else:
                    label=3

            if(abs(origin_label[0])<0.1 and abs(origin_label[3])<0.1):
                label=4

            #label=label.flatten()
            self.data.append(data)
            self.labels.append(label)
        self.data=np.array(self.data)
        self.labels = np.array(self.labels)
        print(self.labels.shape)
        # self.data=np.concatenate(self.data,axis=0)
        # self.labels=np.concatenate(self.labels,axis=0)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        label=self.labels[idx]

        if self.transform:
            sample=self.transform(sample)

        sample=torch.tensor(sample,dtype=torch.float32)
        label=torch.tensor(label,dtype=torch.int32)

        return sample,label

########accuracy test
# a=np.random.randint(0,4)
# from pytorch_image_classification.utils.metrics import accuracy
# output=torch.randint(0,6,size=[8,10])
# target=torch.ones(8,dtype=torch.long)
# print(accuracy(output,target,[1,5]))


########mydataset test
mat_files=[os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.mat')]
dataset=MatDataset(mat_files)
dataloader=DataLoader(dataset,batch_size=8,shuffle=True)

print(dataset.data.shape)
print(dataset.labels.shape)
for batch_data,batch_labels in dataloader:
    print(batch_data.shape)
    print(batch_labels.shape)
    break
