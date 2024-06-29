import argparse
import os
import random
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch    
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.inceptionv2 import *
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad
from fl_cifar import FacilityLocationCIFAR
from lazyGreedy import lazy_greedy_heap, algo1, k_medoids
from utils import *
from webvisiondataset import *

model_names = 'inception' ##????

parser = argparse.ArgumentParser(description="Webvision Training") #???
parser.add_argument('--dataset', default='webvision', help='dataset setting')
parser.add_argument('-a','--arch', metavar='ARCH', default='inception')
parser.add_argument('--exp-str',default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j','--workers', default=4, type=int, metavar='N',
                    help='number of data loading worker (deafault: 4)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--stop-epoch', type=int, default=90, metavar='N',
                    help='stop using crust after epoch')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restart)')
parser.add_argument('-b','--batch-size', default=32, type=int,metavar='N',
                    help='mini-batch-size')
parser.add_argument('--lr','--learning-rate',default=0.1,type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum',default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd','--weight-decay',default=5e-4, type=float,
                    metavar='W',help='weight-decay (default: 5e-4)',dest='weight_decay')
parser.add_argument('-p','--print-freq',default=10, type=int,
                    metavar='N',help='print frequency (deafault: 10)')
parser.add_argument('--resume', default='', type=str,metavar='PATH',
                    help='Path to latest checkpoint (deafault: none)')
parser.add_argument('-e','--evaluate',dest='evaluate',action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initialzing training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root-log', type=str, default='log')
parser.add_argument('--root-model', type=str, default='checkpoint')
parser.add_argument('--use_crust', action='store_true',
                    help="Whether to use clusters in dataset.")

parser.add_argument('--label-type', type=str, default='noisy',
                    help='noisy/pred')

parser.add_argument('--r',default=2.0, type=float,
                    help='Distance threshsold (i.e. radius) in caculating clusters.')
parser.add_argument('--fl-ratio', type=float,default=0.5,####???
                    help='Ratio for number of facilities.')
parser.add_argument('--crust-start',type=int,default=5)

parser.add_argument('--rand-number',type=int, default=0,
                    help='Ratio for number of facilities.') ###?????
parser.add_argument('--algo',type=str,default='lazy_greedy')
parser.add_argument('--crust_stop',type=int,default=90)

best_acc1 = 0

def main():
    args = parser.parse_args()
    if args.use_crust:
        args.store_name = '_'.join([args.dataset, args.arch, str(args.fl_ratio),str(args.r),args.exp_str])
    else:
        args.store_name = '_'.join([args.dataset, args.arch, args.exp_str])

    prepare_folder(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely'
                      'disable data parallelism.')
        
    ngpus_per_node = torch.cuda.device_count() # Cái này hình như không dùng
    main_worker(args.gpu,ngpus_per_node,args)##

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu #????

    if args.gpu is not None:
        print("Use GPU {} for training".format(args.gpu))

    #create model
    print('=> creating model "{}"'.format(args.arch))
    model = InceptionResNetV2(num_classes=50)

    if args.gpu is not None: ##???
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] ## Dòng 41???
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> load checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        
        else:
            print("=> No checkpoint at '{}'".format(args.resume))

    cudnn.benchmark = True#!!!

    # Data loading code
    transforms_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 

    transforms_val = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 

    if args.dataset =='webvision':
        train_dataset = webvision_dataset(
            root= './data',
            transform=transforms_train,
            mode='train',
            num_class=50
        )
        val_dataset = webvision_dataset(
            root= './data',
            transform=transforms_val,
            mode='test',
            num_class=50
        )
    # if args.dataset == 'cifar10':
    #     train_dataset = MISLABELCIFAR10(root ='./data',mislabel_type = args.mislabel_type, mislabel_ratio = args.mislabel_ratio, transform = transforms_train, download = True)
    #     val_dataset = datasets.CIFAR10(root ='./data', train = False, download = True, transform = transforms_val)

    # elif args.dataset =='cifar100':
    #     train_dataset = MISLABELCIFAR100(root ='./data', mislabel_type = args.mislabel_type, mislabel_ratio = args.mislabel_ratio,transform = transforms_train, download = True)
    #     val_dataset = datasets.CIFAR100(root = './data', train = False, download = True, transform = transforms_val)
    
    criterion = nn.CrossEntropyLoss(reduction = 'none').cuda(args.gpu)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers, pin_memory = True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers, pin_memory = True
    )

    trainval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = False,
        num_workers = args.workers, pin_memory = True
    )

    if args.evaluate:
        validate(val_loader,model, criterion,0, args)
        return
    
    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, ##!!!!
                    milestones = [30,60], last_epoch = args.start_epoch - 1)
    
    #init log for training #!!!
    log_training = open(os.path.join(args.root_log, args.store_name,'log.csv'),'w')
    with open(os.path.join(args.root_log, args.store_name,'args.txt'),'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir = os.path.join(args.root_log, args.store_name)) #!!!
    weights = [1] * len(train_dataset) #?? Dong 206
    weights = torch.FloatTensor(weights)#??


    for epoch in range(args.start_epoch, args.epochs):
        if epoch < args.crust_stop:
          train_dataset.switch_data()
          grads_all, all_preds, all_targets = estimate_grads(trainval_loader, model, criterion, args, epoch, log_training)
          if args.label_type == "pred":  
            labels = all_preds
          elif args.label_type == "noisy":
            labels = all_targets
          unique_preds = np.unique(labels)
          
          if args.use_crust and epoch > args.crust_start:
              #FL_part
              print("finding coreset")
              #per class clustering
              ssets = []
              #weights = []
              for c in unique_preds:
                  sample_ids = np.where((labels == c) == True)[0] #!!!
                  grads = grads_all[sample_ids]
                  dists = pairwise_distances(grads)
                  #weight = np.sum(dists < args.r, axis = 1)
                  B = int(args.fl_ratio * len(grads))
                  if args.algo == "lazy_greedy":
                      V = range(len(grads)) 
                      F = FacilityLocationCIFAR(V, D = dists)
                      sset, vals = lazy_greedy_heap(F,V,B)
                  else: 
                      sset = algo1(B,dists)
                  if len(list(sset))>0:
                      #weights.extend(weight[sset].tolist())
                      sset = sample_ids[np.array(sset)]
                      ssets += list(sset)

              #weights = torch.FloatTensor(weights)
              train_dataset.adjust_base_indx_temp(ssets)
              print('change train loader')

        #train for one epoch
        if args.use_crust and epoch > args.crust_start and epoch < args.stop_epoch:#???
            train(train_loader,model, criterion,weights,optimizer,epoch,args,log_training,tf_writer,fetch = True)#!!!
        else:
            train(train_loader,model, criterion,weights,optimizer,epoch,args,log_training,tf_writer,fetch=False)

        #evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_training,tf_writer)

        #remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1,best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)

        save_checkpoint(args,{
            'epoch': epoch + 1,
            'arch' : args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()

        },is_best)
        lr_schedular.step()#!!!
        print('best_acc1: {:.4f}'.format(best_acc1.item()))
def train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch = False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1',':6.2f')
    top5 = AverageMeter('Acc@5',':6.2f')

    progress = ProgressMetter(len(train_loader),batch_time,data_time,losses, top1,top5
                              ,prefix='Epoch: [{}]'.format(epoch))
    
    #switch to train mode
    model.train()
    end = time.time()
    
    for i,batch in enumerate(train_loader):
        input, target, index = batch 
        c_weights = np.ones(len([index]))
        
        # Convert numpy array to PyTorch tensor
        c_weights = torch.FloatTensor(c_weights)
        
        c_weights = c_weights / c_weights.sum()


        if args.gpu is not None:
            c_weights = c_weights.to(args.gpu, non_blocking = True) # So sanh voi dong 285, Khi nao dung cuda?

        #measure data loading time
        data_time.update(time.time() - end)

        input = input.type(torch.FloatTensor)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking = True)
            target = target.cuda(args.gpu, non_blocking = True)

        # compute output
        output, feats = model(input) #############################
        loss = criterion(output, target)
        loss = (loss * c_weights).sum()

        # measure accuracy and record loss #!!!
        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0],input.size(0))
        top5.update(acc5[0],input.size(0))

        #compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    tf_writer.add_scalar('loss/train',losses.avg,epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr',optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log_training=None, tf_writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMetter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.type(torch.FloatTensor)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, feats = model(input)
            loss = criterion(output, target)
            loss = loss.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
            log_training.write('epoch %d val acc: %f\n'%(epoch, top1.avg))
            print('epoch %d val acc: %f\n'%(epoch, top1.avg))

    return top1.avg

def estimate_grads(trainval_loader, model, criterion, args, epoch, log_training):
    # switch to train mode
    model.train()
    all_grads = []
    all_targets = []
    all_preds = []
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_on_noisy = AverageMeter('Acc@1', ':6.2f')

    for i, (input, target, idx) in enumerate(trainval_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        all_targets.append(target)
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        output, feat = model(input)
        _, pred = torch.max(output, 1)

        loss = criterion(output, target).mean()
        acc1_on_noisy, acc5_on_noisy = accuracy(output, target, topk=(1, 5))
        top1_on_noisy.update(acc1_on_noisy[0], input.size(0))
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())
    all_grads = np.vstack(all_grads)
    all_targets = np.hstack(all_targets)

    all_preds = np.hstack(all_preds)

    # In ra số phần tử khác nhau trong all_preds
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    count_dict = dict(zip(unique_preds, counts))
    #print("Number label of each class in predict:")
    #print(count_dict)

    log_training.write('epoch %d train acc on noisy: %f\n'%(epoch, top1_on_noisy.avg))
    print('epoch %d train acc on noisy: %f\n'%(epoch, top1_on_noisy.avg))
    return all_grads, all_preds, all_targets
    if args.label_type == "pred":    
        return all_grads, all_preds
    elif args.label_type == "noisy":
        return all_grads, all_targets
    else:
        return all_grads, all_targets_real
if __name__ == '__main__':
    main()
