from torchvision import models
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import grad
import torchvision.transforms as transforms
import numpy as np

# Get pytorch version
TORCH_VERSION=int(torch.__version__.split('.')[0])
print('Pytorch version:', TORCH_VERSION)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        if TORCH_VERSION > 1:
            vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        else:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.features = torch.nn.Sequential(*[vgg_pretrained_features[i] for i in range(18)])
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.features(X)
        return out

