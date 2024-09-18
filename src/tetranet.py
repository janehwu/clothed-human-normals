######################################################################
# Copyright 2021. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src import utils
from src.layers import partial
from src.layers.conv_layers import Hourglass
from src.vgg import Vgg19

class Bottleneck2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self.__class__, self).__init__()
        self.out_channels = out_channels
        skip = [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        self.skip = nn.Sequential(*skip)

        layers = [
            nn.Conv2d(in_channels, int(out_channels/2), kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(int(out_channels/2)),
            nn.Conv2d(int(out_channels/2), int(out_channels/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(int(out_channels/2)),
            nn.Conv2d(int(out_channels/2), out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] == self.out_channels:
            return x + self.layers(x)
        return self.skip(x) + self.layers(x)


class LastPartNet(nn.Module):
    def __init__(self, in_channels, adjLists_forPCN):
        super(self.__class__, self).__init__()
        self.adjLists_forPCN = adjLists_forPCN

        conv_block = [
            nn.Conv2d(in_channels, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2048*2*2, max(max(adjLists_forPCN[0]))+1),
            nn.ReLU(),
        ]
        self.conv_block = nn.Sequential(*conv_block)

        pc = self._iteratePCN()
        self.pc = nn.Sequential(*pc)

    def _iteratePCN(self):
        layers = []
        for i in range(len(self.adjLists_forPCN)):
            activation = "relu"
            if i == len(self.adjLists_forPCN)-1:
                activation = None
            layers.append(partial.PartialConnection(self.adjLists_forPCN[i], i, activation=activation))
        return layers

    def forward(self, x):
        features = self.conv_block(x)
        out = self.pc(features)
        return out, features

   
class TetraNet(nn.Module):
    def __init__(self, path_adjlists):
        super(self.__class__, self).__init__()
        # VGG features
        self.vgg = Vgg19(requires_grad=False)
        self.vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        # Load adjLists for PCN
        self.path_adjlists = path_adjlists
        self.adjLists_forPCN = utils.load_adjLists(path_adjlists + "/adjlist_[0-9]to[0-9].csv")
        if self.adjLists_forPCN == []:
            print("No adjlists_forPCN are loaded")
            return

        # Construct network
        conv_block = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ]
        self.conv_block = nn.Sequential(*conv_block)

        bottleneck_block1 = [
            Bottleneck2d(64, 128),
            nn.MaxPool2d(kernel_size=2),
        ]
        bottleneck_block2 = [
            Bottleneck2d(128, 128),
            Bottleneck2d(128, 128),
            Bottleneck2d(128, 256),
        ]
        self.bottleneck_block1 = nn.Sequential(*bottleneck_block1)
        self.bottleneck_block2 = nn.Sequential(*bottleneck_block2)
        '''
        nstack = 1
        hgl = [Hourglass(4, 256) for i in range(nstack)]
        self.hgl = nn.Sequential(*hgl)

        hgl_block = [
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        ]
        self.hgl_block = nn.Sequential(*hgl_block)
        '''
        out_block = [
            nn.Conv2d(in_channels=256, out_channels=8, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=256+128, kernel_size=1, padding=0),
        ]
        self.out_block = nn.Sequential(*out_block)
        self.cat1 = nn.Conv2d(in_channels=256+128, out_channels=256+128, kernel_size=1, padding=0)
        self.tsdf = LastPartNet(256+128, self.adjLists_forPCN)

    def forward(self, x, return_features=False):
        x_vgg = self.vgg_normalize(x)
        x_vgg = self.vgg(x_vgg)
        out = self.conv_block(x)
        pool_out = self.bottleneck_block1(out)
        out = self.bottleneck_block2(pool_out)
        out1 = self.out_block(x_vgg)
        cat1_out = torch.cat([x_vgg, pool_out], dim=1)
        cat1_out = self.cat1(cat1_out)
        int1 = out1 + cat1_out
        tsdf, features = self.tsdf(int1)
        # Convert from cm to m.
        tsdf = tsdf/10.
        if return_features:
            return tsdf, features
        return tsdf

