import copy
import torch
from torch.optim.optimizer import Optimizer,required

class L2DisCounter(Optimizer):
    def __init__(self,params,other_params):
        defaults=dict()
        super(L2DisCounter,self).__init__(params,defaults)
        self.p_g=copy.deepcopy(self.param_groups)
        super(L2DisCounter,self).__init__(other_params,defaults)
        self.o_p_g=copy.deepcopy(self.param_groups)

    def Counter(self):
        norm_sum=torch.Tensor([0.0]).cuda()
        for (group,other_group) in zip(self.p_g,self.o_p_g):
            for (p,o_p) in zip(group['params'],other_group['params']):
                tem=p.data.add(o_p.data,alpha=-1.0)
                tem.mul_(tem)
                norm_sum.add_(tem.sum())
        return float(norm_sum)