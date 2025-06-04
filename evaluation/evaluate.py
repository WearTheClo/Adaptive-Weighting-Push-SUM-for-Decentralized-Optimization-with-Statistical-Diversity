import torch
from torch import distributed

import os
import time
import numpy as np

class evaluate():
    def __init__(self,args):
        self.loss_item=[]
        self.c_r=[]
        self.epoch=[]
        self.L2Dis=[]
        self.args=args

    def append(self,loss_item,c_r,epoch,L2Dis):
        self.loss_item.append(loss_item)
        self.c_r.append(c_r)
        self.epoch.append(epoch)
        self.L2Dis.append(L2Dis)

    def all_reduce(self):
        value=torch.Tensor([self.loss_item[-1]/self.args.world_size,self.c_r[-1]/self.args.world_size,self.L2Dis[-1]/self.args.world_size]).cuda()
        distributed.all_reduce(value)
        value=value.cpu().float()
        self.loss_item[-1],self.c_r[-1],self.L2Dis[-1]=value[0],value[1],value[2]

    def read_save(self):
        if self.args.rank!=0:
            return

        #path_wor=self.args.evaluation+'wor.txt'
        if self.args.optimizer == 'SGP' or self.args.optimizer == 'SADDOPT':
            path_wor=self.args.evaluation+'lr_'+str(self.args.lr)+'_.txt'
        elif self.args.optimizer == 'MSGP':
            path_wor=self.args.evaluation+'lr_'+str(self.args.lr)+'_mr_'+str(self.args.mr)+'_.txt'
        elif self.args.optimizer == 'SGAP':
            path_wor=self.args.evaluation+'lr_'+str(self.args.lr)+'_v_'+str(self.args.v)+'_k_'+str(self.args.k)+'_.txt'
        elif self.args.optimizer == 'MSGAP':
            path_wor=self.args.evaluation+'lr_'+str(self.args.lr)+'_mr_'+str(self.args.mr)+'_v_'+str(self.args.v)+'_k_'+str(self.args.k)+'_.txt'

        with open(path_wor,'a') as f:
            f.write('epoch:%d Loss:%10.7f Accuracy:%10.7f%% L2Dis:%10.7f\n'%(self.epoch[-1],self.loss_item[-1],self.c_r[-1],self.L2Dis[-1]))
            f.close()

    def time_save(self):
        if self.args.rank!=0:
            return
        end=time.asctime()
        end_time=(int(end[11:13]),int(end[14:16]),int(end[17:19]))
        time_cost=0
        time_cost+=(end_time[0]-self.args.begin_time[0])*3600
        time_cost+=(end_time[1]-self.args.begin_time[1])*60
        time_cost+=end_time[2]-self.args.begin_time[2]
        if time_cost<0:
            time_cost+=86400
        time_cost=np.array(time_cost)

        print("Time cost:%d"%time_cost)
        np.save(self.args.evaluation+'Time.npy',time_cost)

    def np_save(self):
        if self.args.rank!=0:
            return

        if self.args.optimizer == 'SGP' or self.args.optimizer == 'SADDOPT':
            path_np=self.args.evaluation+'lr_'+str(self.args.lr)
        elif self.args.optimizer == 'MSGP':
            path_np=self.args.evaluation+'lr_'+str(self.args.lr)+'_mr_'+str(self.args.mr)
        elif self.args.optimizer == 'SGAP':
            path_np=self.args.evaluation+'lr_'+str(self.args.lr)+'_v_'+str(self.args.v)+'_k_'+str(self.args.k)
        elif self.args.optimizer == 'MSGAP':
            path_np=self.args.evaluation+'lr_'+str(self.args.lr)+'_mr_'+str(self.args.mr)+'_v_'+str(self.args.v)+'_k_'+str(self.args.k)

        e=np.array(self.epoch)
        l=np.array(self.loss_item)
        c=np.array(self.c_r)
        d=np.array(self.L2Dis)

        np.save(path_np+'Epoch.npy',e)
        np.save(path_np+'Loss.npy',l)
        np.save(path_np+'CR.npy',c)
        np.save(path_np+'Distance.npy',d)
