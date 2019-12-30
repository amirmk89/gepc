"""
Resnet implementation courtesy of Yerlan Idelbayev.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn as nn

__all__ = ['ResNet', 'resnet20']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rm_linear=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.rm_linear = rm_linear
        self.outdim = 4 * self.in_planes if rm_linear else num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        if self.rm_linear:
            return out
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def pt_resnet(backbone='resnet', unfreeze_amount=0, load_pt=True, resize_final=None, rm_linear=True):
    if backbone is None:
        return nn.Identity()
    resnet_model = resnet20(num_classes=10, rm_linear=rm_linear)
    if load_pt:
        model_dict = torch.load('models/fe/resnet20.th')
        kw = 'net' if 'net' in model_dict.keys() else 'state_dict'
        if list(model_dict[kw].keys())[0].startswith('module.'):
            model_dict[kw] = {k[len('module.'):]: v for k, v in model_dict[kw].items()}
        resnet_model.load_state_dict(model_dict[kw])

    elif resize_final and (not rm_linear):
        resize_final_layer(resnet_model, resize_final)
    freeze_patch_fe(resnet_model, unfreeze_all=(not load_pt), unfreeze_amount=unfreeze_amount)
    return resnet_model.cuda()


def resize_final_layer(model, size):
    final_in_dim = model.linear.in_features
    model.linear = nn.Linear(final_in_dim, size)
    model.outdim = size


def freeze_patch_fe(model, unfreeze_all=False, unfreeze_amount=0):
    for param in model.parameters():
        param.requires_grad = unfreeze_all
    if unfreeze_all:
        return

    # Selective unfreezing
    if unfreeze_amount > 0:  # Unfreeze specified amount
        modules_to_unfreeze = list(model.children())[-unfreeze_amount:]
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
