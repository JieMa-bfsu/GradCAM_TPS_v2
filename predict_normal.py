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
from torchcam.cams import GradCAM
import cv2
import numpy as np
from torchsummary import summary
from tps_grid_gen import TPSGridGen
from torch.autograd import Variable
from grid_sample import grid_sample
import itertools



checkpoint_path = 'checkpoint/geo_151+129-checkpoint-99.pth.tar'
checkpoint = torch.load(checkpoint_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dir = '/sdb2/MJ/dataset/geo_151+129/validation/'
save_dir = os.path.join('geo_result', checkpoint_path.split('/')[-1])



def trans(image,imsize, rand_seed):
    # print(image.shape)
    # image = image.permute(2,1,0)
    target_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / 4),
        torch.arange(-1.0, 1.00001, 2.0 / 4),
    )))
    source_control_points = target_control_points+rand_seed

    # print('initialize tps')
    tps = TPSGridGen(imsize, imsize, target_control_points)

    source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
    grid = source_coordinate.view(1, imsize, imsize, 2).cuda()
    target_image = grid_sample(image.cuda(), grid)
    return target_image


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
        x = self.classifi(x)
        return x

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


def test(test_data, dataset, model, cam_extractor, criterion, save_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    target_control_points = torch.Tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / 4),
        torch.arange(-1.0, 1.00001, 2.0 / 4),
    )))
    rand_seed = torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1)

    end = time.time()
    for i, (input, target) in enumerate(test_data):
        im_name = dataset.imgs[i][0].split('/')[-1]
        input = input.cuda()
        target = target.cuda()
        model = model.cuda()
        # compute output
        output = model(input)
        # print(input.imgs)
        ##
        activation_map = cam_extractor(1, output)
        ac = (activation_map.clone().cpu().numpy()*255.).astype(np.uint8)
        ac[ac<0.2]=0
        ac = cv2.resize(ac.squeeze(), (512, 512), interpolation = cv2.INTER_CUBIC)
        
        cv2.imwrite(os.path.join(save_dir, im_name[0:-3]+'png'),ac)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1= accuracy(output, target)
        
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(nn.functional.softmax(output), target)

        if i % 5 == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(test_data), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False


mkdir(save_dir)
model = Classification_att().to(device)
model.load_state_dict(checkpoint)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

dataset = datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ]))
print(dataset.imgs[0][0])
val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

# summary(model, (3, 224, 224))

for n in model.named_modules():
    print (n)

criterion = nn.CrossEntropyLoss().cuda()
cam_extractor = GradCAM(model, target_layer= 'features.26')
test(val_loader, dataset, model, cam_extractor, criterion, save_dir)
