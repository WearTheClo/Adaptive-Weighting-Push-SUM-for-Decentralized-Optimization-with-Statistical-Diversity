import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import distributed
from torch.utils import data

import torchvision.models as models
from torchvision import datasets,transforms

import os
import copy
import argparse
import time
import random
import numpy as np

import topology.TopoLoador as TL
import evaluation.evaluate as eva
import data.CIF_Sampler as CIF_Sampler
import tools.AveModelTest as AMT
import tools.L2DisCounter as LDC

import optimizer.SGP as SGP
import optimizer.SGAP as SGAP
import optimizer.SADDOPT as SADDOPT

def work_process(rank,args):
    args.rank=rank
    ngpus_per_node = torch.cuda.device_count()
    gpu_id = args.rank % ngpus_per_node

    #torch.cuda.set_device(gpu_id) 
    #device=torch.device('cuda:'+str(gpu_id))
    torch.cuda.set_device(3) 
    device=torch.device('cuda:'+str(3))
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset=datasets.CIFAR10(root=args.data,train=True,download=False,transform=transform)
        test_dataset=datasets.CIFAR10(root=args.data,train=False,download=False,transform=transform)
        train_sampler=CIF_Sampler.CIF_NonIID_Sampler(train_dataset,num_replicas=args.world_size,rank=args.rank,train=True,path="./tools/",shuffle=True,iid=False,cifar=10)
        test_sampler=CIF_Sampler.CIF_NonIID_Sampler(test_dataset,num_replicas=args.world_size,rank=args.rank,train=False,path="./tools/",shuffle=True,iid=False,cifar=10)
    else:
        train_dataset=datasets.CIFAR100(root=args.data,train=True,download=False,transform=transform)
        test_dataset=datasets.CIFAR100(root=args.data,train=False,download=False,transform=transform)
        train_sampler=CIF_Sampler.CIF_NonIID_Sampler(train_dataset,num_replicas=args.world_size,rank=args.rank,train=True,path="./tools/",shuffle=True,iid=False,cifar=100)
        test_sampler=CIF_Sampler.CIF_NonIID_Sampler(test_dataset,num_replicas=args.world_size,rank=args.rank,train=False,path="./tools/",shuffle=True,iid=False,cifar=100)
    
    train_size=len(train_sampler)
    test_size=len(test_sampler)

    train_loader=data.DataLoader(train_dataset,args.batch_size,sampler=train_sampler)
    test_loader=data.DataLoader(test_dataset,args.batch_size,sampler=test_sampler)
    
    if args.model == 'ResNet18':
        model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        fc_features=model.fc.in_features
        if args.dataset == 'cifar10':
            model.fc=nn.Linear(fc_features,10)
        else:
            model.fc=nn.Linear(fc_features,100)
        model.to(device)
    else:
        model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        fc_features=model.fc.in_features
        if args.dataset == 'cifar10':
            model.fc=nn.Linear(fc_features,10)
        else:
            model.fc=nn.Linear(fc_features,100)
        model.to(device)

    loss_fun=nn.CrossEntropyLoss()

    lr = args.lr
    if args.optimizer == 'SGP':
        optimizer = SGP.SGP(model.parameters(),model=model,lr=lr,args=args)
    elif args.optimizer == 'MSGP':
        optimizer = SGP.SGP(model.parameters(),model=model,lr=lr,momentum=True,mr=args.mr,args=args)
    elif args.optimizer == 'SGAP':
        optimizer = SGAP.SGAP(model.parameters(),model=model,lr=lr,args=args)
    elif args.optimizer == 'MSGAP':
        optimizer = SGAP.SGAP(model.parameters(),model=model,lr=lr,momentum=True,mr=args.mr,args=args)
    elif args.optimizer == 'SADDOPT':
        optimizer = SADDOPT.SADDOPT(model.parameters(),model=model,lr=lr,args=args)

    e=eva.evaluate(args)
    
    for i in range(args.epochs):
        print("epoch number: {}".format(i+1))
        train_acc=0
        train_loss=0
        test_acc=0
        test_loss=0

        if (i+1)%10==0:
            lr/=10
            if args.optimizer == 'SGP':
                optimizer = SGP.SGP(model.parameters(),model=model,lr=lr,args=args)
            elif args.optimizer == 'MSGP':
                optimizer = SGP.SGP(model.parameters(),model=model,lr=lr,momentum=True,mr=args.mr,args=args)
            elif args.optimizer == 'SGAP':
                optimizer = SGAP.SGAP(model.parameters(),model=model,lr=lr,args=args)
            elif args.optimizer == 'MSGAP':
                optimizer = SGAP.SGAP(model.parameters(),model=model,lr=lr,momentum=True,mr=args.mr,args=args)
            elif args.optimizer == 'SADDOPT':
                optimizer = SADDOPT.SADDOPT(model.parameters(),model=model,lr=lr,args=args)
            print("step size decay to {}".format(lr))
        
        ave_model=copy.deepcopy(model)
        ave_optimizer=AMT.AVEModelTest(params=ave_model.parameters(),model=ave_model,args=args)
        ave_optimizer.ave_params()

        for b,(x_test,y_test) in enumerate(test_loader):
            x_test=x_test.to(device)
            y_test=y_test.to(device)
            y_val=ave_model(x_test)
            predicted=torch.max(y_val.data,1)[1]
            loss = loss_fun(y_val,y_test)

            test_loss += loss.item()
            test_acc += (predicted == y_test).sum()

        #L2 distance counter
        ave_loc_dis=LDC.L2DisCounter(model.parameters(),ave_model.parameters()).Counter()
        if rank==0:
            print('Distance between AVE and LOC: %10.7f'%(ave_loc_dis))

        e.append(loss_item=test_loss/test_size, c_r=100*test_acc/test_size, epoch=i, L2Dis=ave_loc_dis)
        e.all_reduce()
        e.read_save()
        print('Rank:%d Test accuracy%10.7f%%'%(args.rank,100*test_acc/test_size))

        for b,(x_train,y_train) in enumerate(train_loader):
            b+=1
            x_train=x_train.to(device)
            y_train=y_train.to(device)
            y_pred=model(x_train)
            predicted=torch.max(y_pred.data,1)[1]
            loss=loss_fun(y_pred,y_train)
            
            train_loss += loss.item()
            train_acc += (predicted == y_train).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b%25==0:
                train_acc_sum=train_acc/(b*args.batch_size)
                distributed.all_reduce(train_acc_sum)
                train_acc_sum.div_(args.world_size)
                train_loss_sum=train_loss/(b*args.batch_size)
                if rank==0:
                    print(f'epoch:{i:2} batch:{b:2} Train loss:{train_loss_sum:10.8f} Train accuracy:{100*train_acc_sum:10.8f}')

    #e.time_save()
    e.np_save()


def main():
    parser = argparse.ArgumentParser(description='Adaptive Weighting Push-SUM Protocol with Moreau Weighhting')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--mr', default=0.8,type=float, help='momentum rate')
    parser.add_argument('--B', default=1,type=int, help='strongly connected period')#别忘了调整
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--evaluation', type=str, default='./evaluation')

    parser.add_argument('--model', type=str, default='ResNet18')
    #parser.add_argument('--model', type=str, default='ResNet50')

    parser.add_argument('--dataset', type=str, default='cifar10')
    #parser.add_argument('--dataset', type=str, default='cifar100')

    #parser.add_argument('-opti','--optimizer', type=str, default='SGP')
    #parser.add_argument('-opti','--optimizer', type=str, default='MSGP')
    parser.add_argument('-opti','--optimizer', type=str, default='SGAP')
    #parser.add_argument('-opti','--optimizer', type=str, default='MSGAP')
    #parser.add_argument('-opti','--optimizer', type=str, default='SADDOPT')

    parser.add_argument('--k', default=10.0, type=float, help='AWPS hyper-parameter for rate')
    parser.add_argument('--v', default=0.1, type=float, help='AWPS hyper-parameter for lower bound')
    
    parser.add_argument('--type',type=str,default='Static')
    #parser.add_argument('--type',type=str,default='Exponential')
    #parser.add_argument('--type',type=str,default='ForceRandom')

    parser.add_argument('--topology',type=str,default='./topology/Full6.txt')
    #parser.add_argument('--topology',type=str,default='./topology/DC6.txt')
    #parser.add_argument('--topology',type=str,default='./topology/Exp6.txt')
    #parser.add_argument('--topology',type=str,default='./topology/Ran6.txt')
    
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:',
        help='URL specifying how to initialize the package.')
    parser.add_argument('--port', default=4562, type=int, help='port')

    parser.add_argument('-s', '--world-size', type=int, default=6, help='Network Scale')
    parser.add_argument('--gpu-num', default=1, type=int, help='gpu num')
    parser.add_argument('--batch-size', type=int, default=600, help='network full batch size')
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')

    args = parser.parse_args()
    
    args.init_method += str(args.port)
    args.batch_size = args.batch_size//args.world_size

    args.topo = TL.TopoLoador(args)

    if args.model == 'ResNet18':
        if args.dataset == 'cifar10':
            args.evaluation += '/R18_C10/'
        else:
            args.evaluation += '/R18_C100/'
    else:
        if args.dataset == 'cifar10':
            args.evaluation += '/R50_C10/'
        else:
            args.evaluation += '/R50_C100/'
    args.evaluation += args.optimizer + '/'
    args.evaluation += args.topology[11:-4] + '/'

    begin = time.asctime()
    args.begin_time = (int(begin[11:13]),int(begin[14:16]),int(begin[17:19]))
    print(args)
    mp.spawn(fn=work_process,args=(args,),nprocs=args.world_size)

if __name__ == "__main__":
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    main()