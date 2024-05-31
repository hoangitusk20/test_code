import torch
import torch.nn as nn 

import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_int(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1, downsample = None):
        super(BasicBlock,self).__init__()
        ####### 
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_planes,planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes,planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        ###
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet_s(nn.Module):
    def __init__(self, block, num_blocks, image_chanels = 3, num_classes = 10):
        super(ResNet_s,self).__init__()
        ######
        self.in_planes = 64

        self.conv1 = nn.Conv2d(image_chanels, self.in_planes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride= 1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride= 2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride= 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.apply(_weights_int)

    def make_layer(self, block, planes, num_blocks, stride = 1):
        downsample = None
        layers = []
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.in_planes, planes, stride, downsample,))
        self.in_planes = planes * block.expansion
        for _ in range(num_blocks - 1):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feats = x.view(x.size(0), -1)
        x = self.linear(feats)
        return x, feats
        ##
        return x

def resnet20(image_chanels = 3, num_classes = 10):
    return ResNet_s(BasicBlock,[3,3,3], image_chanels, num_classes)

def resnet32(image_chanels = 3, num_classes = 10):
    return ResNet_s(BasicBlock,[5,5,5], image_chanels, num_classes)

def resnet44(image_chanels = 3, num_classes = 10):
    return ResNet_s(BasicBlock,[7,7,7], image_chanels, num_classes)

def resnet56(image_chanels = 3, num_classes = 10):
    return ResNet_s(BasicBlock,[9,9,9], image_chanels, num_classes)

def resnet110(image_chanels = 3, num_classes = 10):
    return ResNet_s(BasicBlock,[18,18,18], image_chanels, num_classes)

def resnet1202(image_chanels = 3, num_classes = 10):
    return ResNet_s(BasicBlock,[200,200,200], image_chanels, num_classes)

def test():
    pass



