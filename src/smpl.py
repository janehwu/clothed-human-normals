######################################################################
# Copyright 2022. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
from src import utils

POSE_FEATURES = 72
CAM_FEATURES = 4

class SMPLPoseNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        conv_block = [
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        ]
        self.conv_block = nn.Sequential(*conv_block)

        self.pose_delta = nn.Linear(512*8*8, POSE_FEATURES)
        self.cam_delta = nn.Linear(512*8*8, CAM_FEATURES)
        torch.nn.init.constant_(self.pose_delta.weight, 0.0)
        torch.nn.init.constant_(self.cam_delta.weight, 0.0)

    def forward(self, x, init_pose=None, init_cam=None):
        out = self.conv_block(x)
        if init_pose is None or init_cam is None:
            pred_pose = self.pose_delta(out)
            pred_cam = self.cam_delta(out)
        else:
            pred_pose = init_pose + 0.05*torch.tanh(self.pose_delta(out))
            pred_cam = init_cam + 0.05*torch.tanh(self.cam_delta(out))
        return pred_pose, pred_cam
 
