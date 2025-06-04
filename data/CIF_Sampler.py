# CIF_TRI_IDX.npy等文件以及将样本索引按照label排序
# 故sampler只需截取这些索引，便可以完成数据划分
# shuffle也是对这些索引的打乱

import math
import random
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

import numpy as np

T_co = TypeVar('T_co', covariant=True)

class CIF_NonIID_Sampler(Sampler[T_co]):
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 iid: bool = False, train: bool = True, path: str ="../tools/",
                 cifar: int = 10) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.iid = iid
        if not self.iid:
            self.train = train
            if self.train:
                if cifar == 10:
                    self.path = path + "CIF_TRI_IDX.npy"
                else:
                    self.path = path + "CIF_100_TRI_IDX.npy"
            else:
                if cifar == 10:
                    self.path = path + "CIF_TES_IDX.npy"
                else:
                    self.path = path + "CIF_100_TES_IDX.npy"

        if self.iid:
            self.num_clusters = 1
        elif self.train: #non-iid and train set has 2 clusters
            self.num_clusters = 2
        else: #non-iid test set has 1 cluster
            self.num_clusters = 1
        assert self.num_replicas % self.num_clusters == 0
        self.mem_clusters = self.num_replicas / self.num_clusters
        self.num_samples = math.ceil(len(self.dataset) / self.num_clusters)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        if self.iid:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
        else:
            indices = list(np.load(self.path,allow_pickle=True))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        #divide
        mem = int(self.rank // self.mem_clusters)
        idx = int(self.rank % self.mem_clusters)
        indices = indices[(mem + self.num_clusters * idx) * self.num_samples:(mem + self.num_clusters * idx + 1) * self.num_samples]
        if not self.iid and self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
