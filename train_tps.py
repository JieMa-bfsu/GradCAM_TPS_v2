import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tps_grid_gen import TPSGridGen
from torch.autograd import Variable
from grid_sample import grid_sample
import itertools

import numpy as np
from torchsummary import summary
from tps_grid_gen import TPSGridGen
from torch.autograd import Variable
from grid_sample import grid_sample
import itertools
from torchcam.cams import GradCAM
import os, json
from easydict import EasyDict as edict
import argparse

# data_dir = '/sdb2/MJ/dataset/geo_151+129/'
# batch_size = 32
# train_sampler = None
# workers = 4
# arch = 'vgg'
# lr = 0.01
# momentum = 0.09
# weight_decay = 1e-4
# epoch_num = 100
# save_frequence = 10
# selected_layer = 'features.26'


parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='', help='option file location')
args = parser.parse_args()
file_dir = args.opt

with open(file_dir) as f:
        config = json.load(f)
config = edict(config)
print(config)

data_dir = config.data_dir
batch_size = config.batch_size
train_sampler = None
workers = config.workers
arch = config.arch
lr = config.lr
momentum = config.momentum
weight_decay = config.weight_decay
epoch_num = config.epoch_num
save_frequence = config.save_frequence
selected_layer = config.selected_layer
if_deform = config.if_deform

traindir = os.path.join(data_dir, 'train')
valdir = os.path.join(data_dir, 'validation')




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Classification_att(nn.Module):
    def __init__(self):
        super(Classification_att, self).__init__()
        self.features = models.vgg19(pretrained=True).features
        self.classifi = nn.Sequential(
            nn.Linear(32768, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifi(x)
        return x

def trans(image,imsize, rand_seed):
    # print(image.shape)
    # image = image.permute(2,1,0)

    image = image.squeeze(0)
    # print(image.shape)
    target_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / 4),
        torch.arange(-1.0, 1.00001, 2.0 / 4),
    )))
    source_control_points = target_control_points+rand_seed

    # print('initialize tps')
    tps = TPSGridGen(imsize, imsize, target_control_points)
    if imsize<256:
        # print(1111111111)
        image = image.permute(1,0,2,3)
        
    batchsize = image.shape[0]
    source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
    grid = source_coordinate.view(1, imsize, imsize, 2).cuda()
    grid = grid.repeat(batchsize,1,1,1)
    target_image = grid_sample(image.cuda(), grid)
    return target_image


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, criterion_tps, cam_extractor,optimizer, epoch, if_deform):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    print_freq = 5
    end = time.time()

    target_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / 4),
        torch.arange(-1.0, 1.00001, 2.0 / 4),
    )))
    rand_seed = torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1)


    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        model = model.cuda()

        # compute output
        output = model(input)
        loss_cla = criterion(output, target)
        if if_deform:
            activation_map = cam_extractor(1, output)
            cam = activation_map.clone()
            cam = cam.unsqueeze(0)
            cam = cam.unsqueeze(1)
            cam_tran = trans(cam, cam.shape[3], rand_seed)
            # cam_tran = cam_tran.squeeze(0)
            cam_tran = cam_tran.squeeze(1)
            im = input.clone()
            tran = trans(im, im.shape[3], rand_seed)
            tran_output = model(tran)
            tran_cam = cam_extractor(1, tran_output)
            loss_tps = criterion_tps(cam_tran, tran_cam)
            loss = loss_cla + loss_tps
        else:
            loss = loss_cla

        acc1= accuracy(output, target)
        
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    # Writing to log file
    try:
        with open('train_results.txt', 'w') as file:
            file.write('Epoch: [{0}]\t'
                       'Time {batch_time.avg:.3f}\t'
                       'Data {data_time.avg:.3f}\t'
                       'Loss {loss.avg:.4f}\t'
                       'Acc@1 {top1.avg:.3f}\t'
                       'Acc@5 {top5.avg:.3f}'.format(
                           epoch, batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1, top5=top5))
    except Exception as err:
        print(err)


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(
            train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()
criterion_tps = nn.MSELoss(reduce=True, size_average=True).cuda()
model = Classification_att()
cam_extractor = GradCAM(model, target_layer= selected_layer)
optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)


time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
if if_deform:
    save_dir = os.path.join('./checkpoint/',data_dir.split('/')[-2],'deform', time_now)
else:
    save_dir = os.path.join('./checkpoint/',data_dir.split('/')[-2],'normal', time_now)
mkdir(save_dir)

for epoch in range(1, epoch_num):

        adjust_learning_rate(optimizer, epoch, lr)
        # train for one epoch
        train(train_loader, model, criterion, criterion_tps, cam_extractor, optimizer, epoch, if_deform)

        if (epoch+1) % save_frequence == 0:
            model_path = save_dir + '/checkpoint-'+str(epoch) + '.pth.tar'
            torch.save(model.state_dict(), model_path)

        # evaluate on validation set

