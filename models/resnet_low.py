'''ResNet in PyTorch.

ported from https://github.com/kuangliu/pytorch-cifar

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNet18LP', 'ResNet50LP']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, quant, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.quantA = quant()
        self.quantB = quant()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x = self.quantA(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.quantB(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.downsample(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.quant = quant()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x = quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = quant(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = quant(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, quant, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10, image_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        #block = BasicBlock
        #num_blocks = [2, 2, 2, 2]
        self.imagenet = (image_size>=224)

        if image_size >= 224:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
        elif image_size == 32:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], quant, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], quant, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], quant, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], quant, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) if self.imagenet else nn.AvgPool2d(4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.quant = quant()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, quant, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, quant, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # not quantising first conv layer?
        x = self.quant(x) #x if self.imagenet else self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x) if self.imagenet else x

        # blocks are quantised
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.quant(x) #x if self.imagenet else self.quant(x)
        x = self.avgpool(x) #F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out



class ResNet18LP:
    base = ResNet
    args = list()
    kwargs = {'block':BasicBlock, 'num_blocks':[2,2,2,2]}


class ResNet50LP:
    base = ResNet
    args = list()
    kwargs = {'block':Bottleneck, 'num_blocks':[3,4,6,3]}