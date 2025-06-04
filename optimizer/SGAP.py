import random
import torch

from torch import distributed
from torch.optim.optimizer import Optimizer,required


class SGAP(Optimizer):
    def __init__(self,params,model,lr=required,momentum=False,mr=0.0,args=None):
        if lr is not required and lr<0.0:
            raise ValueError("Invalid learning rate")
        if mr<0.0:
            raise ValueError("Invalid momentum rate")

        if args.type=='ForceRandom':
            self.FR=True
        else:
            self.FR=False

        defaults=dict(lr=lr,mr=mr)
        super(SGAP,self).__init__(params,defaults)
        self.model=model
        self.momentum=momentum
        self.B=args.B
        self.auto = args.k # k 
        self.v = args.v # v

        self.iter_cnt=0
        self.epoch=-1
        self.rank=args.rank
        self.node_num=args.world_size

        self.topo=args.topo
        self.weight=[0.0]*self.node_num
        self.WM=torch.Tensor([[0.0]*self.node_num]*self.node_num).cuda()
        self.aux_var=torch.Tensor([1.0]).cuda()
        self.aux_vec=torch.Tensor([0.0]*self.node_num).cuda()
        self.rece_indx=[0]*self.node_num

        if self.rank>self.node_num:
            raise ValueError("Rank more than world size")

        if self.momentum:
            print("Momentum")
        
        self.communication_size=0
        print('Adaptive weighting Push-SUM')


    def reset_communication_size(self):
        self.communication_size=0


    def add_communication_size(self,each_send_size):
        tem = 0
        for i in range(self.node_num):
            if i==self.rank:
                continue
            if self.WM[i][self.rank]!=0.0:
                tem = tem + 1
        self.communication_size += each_send_size * tem


    def get_acc_communication_size(self):
        return self.communication_size


    def __setstate__(self,state):
        super(TVAWSGD,self).__setstate__(state)


    def compute_weight(self,other_rank):
        norm_sum=torch.Tensor([0.0]).cuda()

        if self.rece_indx[other_rank] == 0:
            return float(((1-self.v)/(1+self.v))*(-torch.exp(-norm_sum.mul(self.auto))+1.0+self.v))
        # 这个部分很难消除，因为论文原文要求保留过往模型，所以必须要给邻居节点开辟模型
        for group in self.param_groups:
            for p in group['params']:
                param_state=self.state[p]
                tem=param_state['past_param_buff'+str(self.rank)].add(param_state['past_param_buff'+str(other_rank)],alpha=-1.0)
                tem.mul_(tem)
                norm_sum.add_(tem.sum())
        return float(((1-self.v)/(1+self.v))*(-torch.exp(-norm_sum.mul(self.auto))+1.0+self.v))#Moreau weighting


    def auto_weight(self):
        #开始计算权重，Line4
        self.weight=[0.0]*self.node_num

        if self.FR and self.iter_cnt % self.B != 0:
            for i in range(self.node_num):
                if self.topo[i][self.rank]==0:
                    continue
                if random.random()<self.topo[i][self.rank]:
                    self.weight[i]=1.0
        else:
            if self.FR:
                idx = 0
            else:
                idx = self.iter_cnt % self.B
            for i in range(self.node_num):
                if self.topo[idx*self.node_num+i][self.rank]==0:
                    continue
                self.weight[i]=1.0

        for i in range(self.node_num):
            if i==self.rank or self.weight[i]==0.0:
                continue
            self.weight[i]=self.compute_weight(other_rank=i)
        
        out_degree=0.0
        for i in range(self.node_num):
            if self.weight[i]==0.0:
                continue
            out_degree+=1.0

        for i in range(self.node_num):
            if i==self.rank or self.weight[i]==0.0:
                continue
            self.weight[i]/=out_degree
        self.weight[self.rank]=2.0-sum(self.weight)


    def _update_PSSGD_params(self):
        # 计算权重后的通信，Line4
        for i in range(self.node_num):
            for j in range(self.node_num):
                if j==self.rank:
                    self.WM[i][j]=self.weight[i]
                else:
                    self.WM[i][j]=0.0
        distributed.barrier()
        distributed.all_reduce(self.WM)
        # Line4结束

        # 通信a，Line3
        for i in range(self.node_num):
            if i==self.rank:
                self.aux_vec[i]=float(self.aux_var)
            else:
                self.aux_vec[i]=0.0
        distributed.barrier()
        distributed.all_reduce(self.aux_vec)
        tem=torch.Tensor([0.0]).cuda()
        for i in range(self.node_num):
            tem.add_(self.aux_vec[i].mul(self.WM[self.rank][i]))
        self.aux_var=torch.clone(tem)
        #通信a结束，Line3

        # set rece_indx
        for i in range(self.node_num):
            if self.WM[self.rank][i]!=0.0:
                self.rece_indx[i]=1

        # gradient descent
        # Line2
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state=self.state[p]
                d_p=p.grad.data
                if 'param_buff' not in param_state:
                    param_state['param_buff']=torch.clone(p).detach()
                if 'momentum_buff' not in param_state and self.momentum:
                    param_state['momentum_buff']=torch.zeros_like(p.data)

                if self.momentum:
                    param_state['momentum_buff'].mul_(group['mr'])
                    param_state['momentum_buff'].add_(d_p)
                    param_state['param_buff'].add_(param_state['momentum_buff'],alpha=-group['lr'])
                else:
                    param_state['param_buff'].add_(d_p,alpha=-group['lr'])

        # consensus
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    # 通信x，Line3
                    param_state = self.state[p]
                    tem_param = torch.clone(param_state['param_buff']).detach().cuda()
                    tem_zero = torch.zeros_like(param_state['param_buff']).detach().cuda()
                    tem_param = torch.unsqueeze(tem_param, dim = 0)
                    tem_zero = torch.unsqueeze(tem_zero, dim = 0)
                    for i in range(self.node_num):
                        if i < self.rank:
                            tem_param = torch.cat((tem_zero, tem_param),dim = 0)
                        elif i > self.rank:
                            tem_param = torch.cat((tem_param, tem_zero),dim = 0)
                        else:
                            continue
                    distributed.barrier()
                    distributed.all_reduce(tem_param)
                    # tem_param 存储了当前层的全网参数，应该立刻共识
                    # 共识步x，Line6
                    tem_state = torch.zeros_like(p.data).cuda()
                    for i in range(self.node_num):
                        tem_state.add_(tem_param[i].mul(self.WM[self.rank][i]))

                    # 修正x，Line7
                    param_state['param_buff']=torch.clone(tem_state)
                    p.data=param_state['param_buff'].div(self.aux_var)

                    # set buffer
                    # buffer修改，Line5
                    for i in range(self.node_num):
                        if self.WM[self.rank][i]!=0.0:
                            param_state['past_param_buff'+str(i)]=torch.clone(tem_param[i])


    def step(self,closure=None,epoch=0,div_epoch=False):
        loss=None
        if closure is not None:
            loss=closure
        self.auto_weight()
        self._update_PSSGD_params()
        self.iter_cnt+=1
        return loss
