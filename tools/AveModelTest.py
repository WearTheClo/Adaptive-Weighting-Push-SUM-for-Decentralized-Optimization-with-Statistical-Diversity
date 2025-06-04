#Get the global model by all reduce
import torch
from torch import distributed
from torch.optim.optimizer import Optimizer,required


class AVEModelTest(Optimizer):
    def __init__(self,params,model,args=None):
        defaults=dict()
        super(AVEModelTest,self).__init__(params,defaults)
        self.model=model
        
        self.rank=args.rank
        self.node_num=args.world_size

        if self.rank>self.node_num:
            raise ValueError("Rank more than world size")


    def __setstate__(self,state):
        super(AVEModelTest,self).__setstate__(state)


    def ave_params(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    param_state=self.state[p]
                    # 这里直接all_reduce模型参数param_state['param_buff']即可，因为这里使用的是绝对平均
                    # 注意param_state['param_buff']存在性检查，以及最后的p.data = torch.clone(param_state['param_buff'])
                    if 'param_buff' not in param_state:
                        param_state['param_buff'] = torch.clone(p).detach()
                    
                    distributed.barrier()
                    distributed.all_reduce(param_state['param_buff'])
                    param_state['param_buff'].div_(self.node_num)
                    p.data = torch.clone(param_state['param_buff'])


    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure
        self.ave_params()
        return loss
