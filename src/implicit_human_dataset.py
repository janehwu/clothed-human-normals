######################################################################
# Copyright 2021-2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import glob
import math
import os
import re
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('../'))
import _pickle as pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

#from generate_normal_map_renderpeople import rasterize_from_blender
from src import utils
from src.camera_transforms import RotateAxisAngle
from src.renderer import Renderer

from obj_io import *

class ImplicitHumanDataset(Dataset):
    '''
    Dataloader for image->TSDF and image->normal map examples.

    Args:
        datasetroot (str): Path to root directory containing all data
        imgdirlist (list): List of image directories
        tsdfdirlist (list): List of TSDF directories
        device (str): PyTorch device name
        resolution (int): Resolution of output normal map (square)
        renderer (Renderer): If not None, Renderer object for generating normal maps
    '''
    def __init__(self,
                 name,
                 datasetroot,
                 imgdirlist,
                 tsdfdirlist,
                 device,
                 resolution,
                 normal_img=False,
                 renderer=None,
                 use_pkl=False):
        self.name = name
        self.datasetroot = datasetroot
        self.imgdirlist = imgdirlist
        self.tsdfdirlist = tsdfdirlist

        self.device = device
        self.resolution = resolution
        self.normal_img = normal_img
        self.renderer = renderer
        self.use_pkl = use_pkl
        self.is_label_TSDF = True

        # Parse .txt
        self.imgdirPaths = []
        with open(self.imgdirlist, "r") as f:
            lines = f.read().split()
            for imgdirname in lines:
                imgdirpath = self.datasetroot + "/" + imgdirname
                print('IMG DIR:', imgdirpath)
                assert(os.path.exists(imgdirpath))
                self.imgdirPaths += [imgdirpath]
        print("Reading images from:", self.imgdirPaths)

        if not os.path.exists(self.tsdfdirlist):
            self.is_label_TSDF = False
            print("No TSDF labels. Skipping...")
        else:
            self.tsdfdirPaths = []
            with open(self.tsdfdirlist, "r") as f:
                lines = f.read().split()
                for tsdfdirname in lines:
                    tsdfdirpath = self.datasetroot + "/" + tsdfdirname
                    assert(os.path.exists(tsdfdirpath))
                    self.tsdfdirPaths += [tsdfdirpath]
            print("Reading TSDF values from:", self.tsdfdirPaths)

        # Count number of all dataset
        self.countList = []
        self.nameList_color = []
        self.nameList_label = []
        self.example_ids = []
        self.tracking_results = []

        if self.is_label_TSDF:
            labelPaths = self.tsdfdirPaths

        # Save filenames and tracking results by example scan
        for i in range(len(self.imgdirPaths)):
            imgdirpath = self.imgdirPaths[i]
            example_id = re.split('//|/', imgdirpath)[-2]
            print('Example id:', example_id)
            self.example_ids.append(example_id)
            searchpath_color_png = imgdirpath + "/Camera*.jpg"

            if self.is_label_TSDF:
                labeldirpath = labelPaths[i]
                searchpath_label = labeldirpath + "/*.bin"

            names_color = sorted(glob.glob(searchpath_color_png))
            names_color.sort()

            #if len(names_color) == len(names_label):
            self.countList += [len(names_color)]
            self.nameList_color.append(names_color)

        print("Num of available dataset: {0:d} (from {1:d} dir(s))".format(sum(self.countList), len(self.countList)))

        self.db, num_examples = self.load_db()

        self.Ids_all = np.arange(num_examples)
        self.num_samples = len(self.Ids_all)

    def __len__(self):
        return len(self.shuffle_idx) #self.num_samples

    def __getitem__(self, idx):
        sample = {}
        sample_id = self.Ids_all[idx]
        sample['normals'] = self.db['normals'][idx]
        sample['idx'] = sample_id
        sample['images'] = self.db['images'][idx]
        if self.is_label_TSDF:
            sample['tsdfs'] = self.db['tsdfs'][idx]
        sample['masks'] = self.db['masks'][idx]
        if self.use_pkl:
            sample['scales'] = self.db['scales'][idx]
            sample['translations'] = self.db['translations'][idx]
        sample['poses'] = self.db['poses'][idx]
        sample['camera_K'] = self.db['camera_K'][idx]
        sample['camera_R'] = self.db['camera_R'][idx]
        sample['camera_T'] = self.db['camera_T'][idx]
        return sample

    def preprocess_camera_params(self, root_dir, idx):
        # Assumes data from blender
        K = np.load(os.path.join(root_dir + '/cameras/Camera%s_K.npy' % idx))
        R_euler = np.load(os.path.join(root_dir + '/cameras/Camera%s_R.npy' % idx))
        T = np.load(os.path.join(root_dir + '/cameras/Camera%s_T.npy' % idx))

        # Transforms between blender and pytorch3d world coordinates
        blender_to_world = np.eye(3)
        blender_to_world[0,:] = [1, 0, 0]
        blender_to_world[1,:] = [0, 0, -1]
        blender_to_world[2,:] = [0, 1, 0]

        # Convert Euler angles from Blender to Pytorch3D
        X_r = RotateAxisAngle(-90, axis="X")
        X2_r = RotateAxisAngle(-R_euler[0], axis="X")
        Y_r = RotateAxisAngle(R_euler[1] +0, axis="Y")
        Z_r = RotateAxisAngle(R_euler[2] +0, axis="Z")
        R = X_r.compose(Z_r.compose(Y_r.compose(X2_r))).get_matrix()[0,:3,:3].numpy().transpose()

        # Apply transformation to translation
        T = np.matmul(blender_to_world, T)
        T[0] = T[0] * -1
        T = np.matmul(R, T)

        # Transpose rotation because Pytorch3D uses column-major order
        R = R.transpose()

        K = torch.unsqueeze(torch.tensor(K), 0).type(torch.float32).to(self.device)
        R = torch.unsqueeze(torch.tensor(R), 0).type(torch.float32).to(self.device)
        T = torch.unsqueeze(torch.tensor(T), 0).type(torch.float32).to(self.device)
        return K, R, T

    def load_db(self):
        db = {}
        Imgs = []
        TSDF = []
        Normals = []
        Masks = []
        VIBE_Inputs = []
        VIBE_Bboxes = []
        Poses = []
        Scales = []
        Translations = []
        Ks = []
        Rs = []
        Ts = []
        Vertices = []
        for i in range(len(self.countList)):
            # For each scan image
            Scan_Imgs = []
            for idx in range(len(self.nameList_color[i])):
                root_path = self.nameList_color[i][idx].split('image_data')[0]
                # print('Reading capture image:', self.nameList_color[i][idx])
                img = cv2.imread(self.nameList_color[i][idx], -1)[..., :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if img.shape[0] != 256:
                    img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
                img = img.astype(np.float32) / 255. # Image transformed to [0,1]
                Scan_Imgs += [img]
                if self.is_label_TSDF:
                    try:
                        TSDF += [utils.loadTSDF_bin(self.nameList_label[i][idx])]
                    except:
                        print("Got an error while reading {}".format(self.nameList_label[i][idx]))
                        sys.exit()

                if self.normal_img:
                    frame_idx = self.nameList_color[i][idx].split("/")[-1].split(".")[0][6:10]
                    normals_filename = os.path.join(root_path, 'normals_data', 'Camera_normals_%s.png' % frame_idx)
                    print('Normals:', normals_filename)
                    img = cv2.imread(normals_filename)
                    if img.shape[0] != self.resolution:
                        img = cv2.resize(img, (self.resolution, self.resolution), cv2.INTER_AREA)
                    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.
                    # Transform from [0,1] to [-1,1]
                    img = img*2-1

                    mask = np.linalg.norm(img, axis=-1) > 0.5
                    mask = np.expand_dims(mask,-1)
                    Normals.append(img)
                    Masks.append(mask)
                   
                    # Load and preprocess camera params, shared between examples!
                    K, R, T = self.preprocess_camera_params(root_path, frame_idx)
                    Ks.append(K)
                    Rs.append(R)
                    Ts.append(T)
                    # Repeat the same pose
                    if self.use_pkl:
                        pkl_filename = os.path.join(root_path, 'smpl/smpl_param.pkl')
                        with open(pkl_filename, mode="rb") as f:
                            param = pickle.load(f)
                        pose = param['body_pose'].ravel()
                        Poses.append(pose)
                        Scales.append(param['body_scale'])
                        translation = param['global_body_translation']
                        Translations.append(translation)
                    else:
                        pose_filename = os.path.join(root_path, 'pose.csv')
                        if os.path.isfile(pose_filename):
                            pose = np.genfromtxt(pose_filename, delimiter=',').astype(np.float32)
                        else:
                            pose = np.zeros(72, dtype=np.float32)
                        Poses.append(pose)

            Scan_Imgs = np.array(Scan_Imgs, dtype=np.float32)
            Imgs.append(Scan_Imgs)

        Imgs = np.concatenate(Imgs, axis=0)
        Imgs = Imgs.transpose((0, 3, 1, 2)) # Transpose for pytorch
        Imgs = torch.from_numpy(Imgs)
        TSDF = torch.from_numpy(np.array(TSDF, dtype=np.float32))
        Masks = torch.from_numpy(np.asarray(Masks))
        Poses = torch.from_numpy(np.asarray(Poses)).type(torch.float32)
        Scales = torch.from_numpy(np.asarray(Scales)).type(torch.float32)
        Translations = torch.from_numpy(np.asarray(Translations)).type(torch.float32)
        Ks = torch.cat(Ks)
        Rs = torch.cat(Rs)
        Ts = torch.cat(Ts)
        Normals = np.asarray(Normals, dtype=np.float32)
        Normals = torch.from_numpy(Normals)

        if self.is_label_TSDF:
            assert(np.amin(TSDF) >= 0 and np.amax(TSDF) <= 1)

        if self.normal_img and self.is_label_TSDF:
            Normals = []
            Masks = []
            self.renderer.eval()
            torch.set_grad_enabled(False)
            for i in range(Imgs.shape[0]):
                    NormalF, MaskF = self.renderer(TSDF[i]*2-1, Pose[i], Camera[i], self.resolution)
                    Normals.append(NormalF)
                    Masks.append(MaskF)
            Normals = torch.stack(Normals)
            Masks = torch.stack(Masks)

        device = self.device

        # TODO: Shuffling is currently disabled for debugging.
        print('Num examples:', len(Imgs))
        shuffle_idx = torch.arange(len(Imgs))
        self.shuffle_idx = shuffle_idx
        db['images'] = Imgs[shuffle_idx].to(device)
        db['masks'] = Masks[shuffle_idx].to(device) 
        if self.use_pkl:
            db['scales'] = Scales[shuffle_idx].to(device)
            db['translations'] = Translations[shuffle_idx].to(device)
        db['poses'] = Poses[shuffle_idx].to(device)
        db['camera_K'] = Ks[shuffle_idx].to(device)
        db['camera_R'] = Rs[shuffle_idx].to(device)
        db['camera_T'] = Ts[shuffle_idx].to(device)
        db['normals'] = Normals[shuffle_idx].to(device)
        print('All images:', db['images'].shape)
        print('All normals:', db['normals'].shape)
        print('All Ks:', db['camera_K'].shape)
        assert len(db['images']) == len(db['normals'])
        return db, len(Normals)

