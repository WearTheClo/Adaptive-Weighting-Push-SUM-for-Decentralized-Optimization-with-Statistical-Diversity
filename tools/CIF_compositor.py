import numpy as np
import torch
import torch.utils.data.sampler as sampler

from torchvision import datasets,transforms

from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

data_root="../data"
Transform=transforms.ToTensor()

idx=[]

for i in range(10):
    idx.append([])

train_dataset=datasets.CIFAR10(data_root,train=True,download=False,transform=Transform)
train_sampler=sampler.SequentialSampler(train_dataset)
train_loader=data.DataLoader(train_dataset,sampler=train_sampler)

for b,(X_train,y_train) in enumerate(train_loader):
    lable=int(y_train)
    idx[lable].append(b)

tem=[]
for i in range(10):
    for j in idx[i]:
        tem.append(j)
tem_np=np.array(tem)
print(len(tem_np))
np.save("CIF_TRI_IDX",tem_np)

idx=[]

for i in range(10):
    idx.append([])

test_dataset=datasets.CIFAR10(data_root,train=False,download=False,transform=Transform)
test_sampler=sampler.SequentialSampler(test_dataset)
test_loader=data.DataLoader(test_dataset,sampler=test_sampler)

for b,(X_test,y_test) in enumerate(test_loader):
    lable=int(y_test)
    idx[lable].append(b)

tem=[]
for i in range(10):
    for j in idx[i]:
        tem.append(j)
tem_np=np.array(tem)
print(len(tem_np))
np.save("CIF_TES_IDX",tem_np)
print("Done!")