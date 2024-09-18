######################################################################
# Copyright 2021-2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse
import os
import sys
import time
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# Get pytorch version
TORCH_VERSION=int(torch.__version__.split('.')[0])
print('Pytorch version:', TORCH_VERSION)

# Deterministic training
torch.manual_seed(42)
np.random.seed(42)

from src import tetranet
from src import utils
from src.implicit_human_dataset import ImplicitHumanDataset

from train_utils import *

# Inference functions
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import trimesh


def main():
    # torch.autograd.set_detect_anomaly(True)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Regress TSDF from single RGB image')

    parser.add_argument('--rundir', default='./rundir', type=str, help='Where logs and weights will be saved')
    parser.add_argument('--trial', default='test', required=True, type=str, help='Trial name')
    parser.add_argument('--datasetroot', default='./D_march', required=True, type=str, help='Path to the directory contains dataset')
    parser.add_argument(
        '--train_imgdirs',
        default="datasetpaths_train/imgdirs_train.txt",
        type=str,
        help='Relative path list to the imgs for training (from datasetroot)')
    parser.add_argument(
        '--train_tsdfdirs',
        default="datasetpaths_train/TSDFdirs_train.txt",
        type=str,
        help='Relative path list to the TSDFs for training (from datasetroot)')
    parser.add_argument(
        '--val_imgdirs',
        default="datasetpaths_val/imgdirs_val.txt",
        type=str,
        help='Relative path list to the imgs for validation (from datasetroot)')
    parser.add_argument(
        '--val_tsdfdirs',
        default="datasetpaths_val/TSDFdirs_val.txt",
        type=str,
        help='Relative path list to the TSDFs for validation (from datasetroot)')
    parser.add_argument('--cp', default="", type=str, help="Pretrained model weights")
    parser.add_argument('--adjlists', default="./adjLists", type=str, help="Adjlists to construct PCN")
    parser.add_argument('--mode', default=0, type=int, help="0:training, 1:prediction")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)
    parser.add_argument('--begin_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--save_obj', default=10, type=int)
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--normal_img', default=True, action='store_true')
    parser.add_argument('--plot_regularizers', default=False, action='store_true')
    parser.add_argument('--use_pkl', default=False, action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    ############ config ############
    # Mode: Train=0, Predict=1
    mode = args.mode
    lr = args.lr
    begin_epoch = args.begin_epoch
    end_epoch = args.end_epoch
    batch_size = args.batch_size
    val_ratio = args.val_ratio
    save_interval = args.save_interval
    save_obj = args.save_obj
    plot_regularizers = args.plot_regularizers
    use_pkl = args.use_pkl
    rundir = os.path.join(args.rundir, args.trial)
    path_adjlists = args.adjlists
    datasetroot = args.datasetroot

    imgdirs = {}
    tsdfdirs = {}
    imgdirs['train'] = args.train_imgdirs
    imgdirs['val'] = args.val_imgdirs
    tsdfdirs['train'] = os.path.join(datasetroot, args.train_tsdfdirs)
    tsdfdirs['val'] = os.path.join(datasetroot, args.val_tsdfdirs)

    num_workers = args.num_workers
    assert(num_workers > 0)
    resolution = args.resolution
    estimate_normal_img = args.normal_img
    verbose = args.verbose
    ################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)
    torch.cuda.init()

    # Initialze model
    model = tetranet.TetraNet(path_adjlists)
    model.to(device)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print('adding params to update', name)
            params_to_update.append(param)

    # Initialize optimizer
    optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=1e-4)
    #optimizer.add_param_group({'params': vibe_params_to_update, 'lr': 1e-5})

    renderer = None
    if estimate_normal_img:
        renderer = Renderer(device, batch_size=batch_size, img_dim=resolution)
        sdf_reg = SdfRegularizers(device)

    is_label_TSDF = not estimate_normal_img
    if mode == 0:
        print("Training mode")
        # Tensorboard setup
        writer = SummaryWriter(rundir)

        # Load weights
        if args.cp != "":
            print('Load checkpoint',args.cp)
            cp=torch.load(args.cp,map_location="cpu")
            print('trained for', cp['epoch'], 'epochs', 'val cost',cp['val_cost'])
            model.load_state_dict(cp['state_dict'])
            optimizer.load_state_dict(cp['optimizer'])
            '''
            vibe_model.load_state_dict(cp['vibe_state_dict'], strict=False)
            for param in optimizer.param_groups[0]['params']:
                param.requires_grad = False
            vibe_model.eval()
            '''
        # Add vibe opt group here.
        #optimizer.add_param_group({'params': vibe_params_to_update, 'lr': 1e-5})
        print("Model parameters:")
        summary(model, input_size=(3, 256, 256))

        print('Creating scheduler')
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                         patience=10, verbose=True, threshold=1e-04,
                                         threshold_mode='rel', cooldown=0,
                                         min_lr=1e-6, eps=1e-08)
        print('Old LR:', optimizer.param_groups[0]['lr'])
        optimizer.param_groups[0]['lr'] = args.lr
        print('New LR:', optimizer.param_groups[0]['lr'])

        criterion = nn.L1Loss()

        #  Dataloaders
        dataset_names = ['train', 'val']
        datasets = {}
        dataloaders = {}
        for name in dataset_names:
            dataset = ImplicitHumanDataset(name, datasetroot, imgdirs[name],
                                           tsdfdirs[name],
                                           device, resolution,
                                           normal_img=estimate_normal_img,
                                           renderer=renderer,
                                           use_pkl=use_pkl)
            datasets[name] = dataset
            shuffle = False #True
            if name == 'val':
                shuffle = False
            dataloaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if not os.path.exists(rundir):
            os.makedirs(rundir)
        print('rundir', rundir)

        # Training settings
        start_time = timer()

        train_cost_history = []
        val_cost_history = []

        #best_val_model = None
        best_val_cost = float('Inf')
        best_val_epoch = -1
        
        #best_train_model = None
        best_train_cost = float('Inf')
        best_train_epoch = -1

        save_dir = os.path.join(rundir, 'saved_models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_iter_log_filename = os.path.join(rundir, 'train_iter_log')
        train_epoch_log_filename = os.path.join(rundir, 'train_epoch_log')
        val_epoch_log_filename = os.path.join(rundir, 'val_epoch_log')

        train_iter_log = open(train_iter_log_filename, 'w')
        train_epoch_log = open(train_epoch_log_filename, 'w')
        val_epoch_log = open(val_epoch_log_filename, 'w')

        save_every_epoch = save_interval
        regularizer = 'eikonal'
        use_sdf_losses = True
        use_boundary_losses = True
        for epoch in range(begin_epoch, end_epoch):
            torch.cuda.empty_cache()
            print('-----------------------------------------------------')
            print('Epoch {}/{}'.format(epoch+1, end_epoch-begin_epoch))
            print('-----------------------------------------------------')
            running_losses = {
                'total': 0.0,
                'chamfer': 0.0,
                'normal': 0.0,
                'boundary': 0.0,
                'inflate': 0.0,
                'curvature': 0.0,
                'eikonal': 0.0,
            }
            torch.set_grad_enabled(True)
            start = 0
            end = batch_size
            data_count = 0
            for idx, batch in enumerate(dataloaders['train']):
                batch_start = timer()
                if TORCH_VERSION > 1:
                    optimizer.zero_grad(set_to_none=False)
                else:
                    optimizer.zero_grad()
                inputs = batch['images']
                outputs = model(inputs)
                if verbose:
                    print('Model inference:', timer() - batch_start)
                #outputs = all_outputs[[0]]
                #multiview_loss = criterion(torch.mean(all_outputs[1:], axis=0), outputs[0])

                loss = 0 #0.01*multiview_loss
                chamfer_loss = 0
                normal_loss = 0
                boundary_loss = 0
                inflate_loss = 0
                curvature_loss = 0
                eikonal_loss = 0
                if is_label_TSDF:
                    label = batch['tsdfs']
                    loss += criterion(outputs, label)
                if estimate_normal_img:
                    gt_normalf = batch['normals']
                    gt_masks = batch['masks']
                    if not use_pkl:
                        pose = batch['poses']
                    else:
                        pose = {
                            'pose': batch['poses'],
                            'scale': batch['scales'],
                            'translation': batch['translations']
                        }
                    cam = (batch['camera_K'], batch['camera_R'], batch['camera_T'])
                    losses, gt_normalf, pred_normalf = compute_normal_loss(pose, cam,
                                                                           renderer, sdf_reg,
                                                                           criterion,
                                                                           batch, outputs, device,
                                                                           epoch=epoch, step=idx,
                                                                           sdf_losses=use_sdf_losses,
                                                                           boundary_losses=use_boundary_losses,
                                                                           rundir=rundir,
                                                                           save_obj=((epoch+1)%save_obj==0),
                                                                           plot_regularizers=plot_regularizers,
                                                                           verbose=verbose)
                    loss += losses['normal']
                    normal_loss += losses['normal']
                    # 3D loss (TODO: remove later?)
                    #loss += losses['chamfer']
                    #chamfer_loss += losses['chamfer']
                    if use_boundary_losses:
                        loss += losses['boundary']
                        loss += losses['inflate']
                        boundary_loss += losses['boundary']
                        inflate_loss += losses['inflate']
                    if use_sdf_losses:
                        loss += losses['curvature']
                        loss += losses['eikonal']
                        curvature_loss += losses['curvature']
                        eikonal_loss += losses['eikonal']
                    if idx < 10 and epoch % 10 == 0:
                        writer.add_images('Images/train_image_%d' % idx, (batch['images']*255).type(torch.uint8), epoch)
                        pred_image = ((pred_normalf*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                        writer.add_images('Images/train_pd_%d' % idx, pred_image, epoch)
                        gt_image = ((gt_normalf*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                        writer.add_images('Images/train_gt_%d' % idx, gt_image, epoch)
                        gt_mask = batch['masks'].permute(0, 3, 1, 2)
                        writer.add_images('Images/train_gt_mask_%d' % idx, gt_mask, epoch)
                        if use_boundary_losses:
                            bd_image = (losses['boundary_mask']*255).permute(0, 3, 1, 2).type(torch.uint8)
                            writer.add_images('Images/train_shrink_%d' % idx, bd_image, epoch)
                            inflate_image = ((losses['inflate_normals']*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                            writer.add_images('Images/train_inflate_%d' % idx, inflate_image, epoch)
                            bd_image = (losses['inflate_mask']*255).permute(0, 3, 1, 2).type(torch.uint8)
                            writer.add_images('Images/train_inflate_boundary_%d' % idx, bd_image, epoch)

                        del pred_image, gt_image
                    del losses, gt_normalf, pred_normalf

                if verbose:
                    back_start = timer()
                loss.backward()
                if verbose:
                    print('Backprop:', timer() - back_start)
                # Clip gradient to avoid nan
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                loss_item = loss.item()
                running_losses['total'] += loss_item * inputs.shape[0]
                running_losses['normal'] += normal_loss.item() * inputs.shape[0]
                #running_losses['chamfer'] += chamfer_loss.item() * inputs.shape[0]
                if use_boundary_losses:
                    running_losses['boundary'] += boundary_loss.item() * inputs.shape[0]
                    running_losses['inflate'] += inflate_loss.item() * inputs.shape[0]
                if use_sdf_losses:
                    running_losses['curvature'] += curvature_loss.item() * inputs.shape[0]
                    running_losses['eikonal'] += eikonal_loss.item() * inputs.shape[0]
                data_count += inputs.shape[0]
                start += inputs.shape[0]
                end += inputs.shape[0]
                print('Batch time:', timer() - batch_start)

            cost = running_losses['total']/data_count
            writer.add_scalar('Loss/train', cost, epoch)
            writer.add_scalar('Loss/normal', running_losses['normal']/data_count, epoch)
            #writer.add_scalar('Loss/chamfer', running_losses['chamfer']/data_count, epoch)
            if use_boundary_losses:
                writer.add_scalar('Loss/boundary', running_losses['boundary']/data_count, epoch)
                writer.add_scalar('Loss/inflate', running_losses['inflate']/data_count, epoch)
            if use_sdf_losses:
                writer.add_scalar('Loss/curvature', running_losses['curvature']/data_count, epoch)
                writer.add_scalar('Loss/eikonal', running_losses['eikonal']/data_count, epoch)
            del running_losses
            print('Train loss: {:.4e}'.format(cost))
            train_epoch_log.write('%d %e\n' %(epoch, cost))
            train_epoch_log.flush()
            os.fsync(train_epoch_log)
            train_cost_history.append(cost)

            if epoch == 0 or (epoch+1) % 100 == 0:
                ###### VALIDATION ######
                torch.set_grad_enabled(False)
                start = 0
                end = batch_size
                running_val_loss = 0.0
                data_count = 0
                for idx, batch in enumerate(dataloaders['val']):
                    torch.cuda.empty_cache()
                    batch_start = timer()
                    inputs = batch['images']
                    outputs = model(inputs)

                    loss = 0
                    if is_label_TSDF:
                        label = batch['tsdfs']
                        loss += criterion(outputs, label)
                    if estimate_normal_img:
                        gt_normalf = batch['normals']
                        gt_masks = batch['masks']
                        if not use_pkl:
                            pose = batch['poses']
                        else:
                            pose = {
                                'pose': batch['poses'],
                                'scale': batch['scales'],
                                'translation': batch['translations']
                            }
                        cam = (batch['camera_K'], batch['camera_R'], batch['camera_T'])
                        losses, gt_normalf, pred_normalf = compute_normal_loss(pose, cam,
                                                                               renderer, sdf_reg,
                                                                               criterion,
                                                                               batch, outputs, device)
                        loss += losses['normal']

                        if idx < 3:
                            writer.add_images('Images/val_image_%d' % idx, (batch['images']*255).type(torch.uint8), epoch)
                            pred_image = ((pred_normalf*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                            pred_image = ((pred_normalf*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                            writer.add_images('Images/val_pd_%d' % idx, pred_image[:3], epoch)
                            gt_image = ((gt_normalf*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                            writer.add_images('Images/val_gt_%d' % idx, gt_image[:3], epoch)
                            del pred_image, gt_image
                        del outputs, losses, gt_normalf, pred_normalf

                    running_val_loss += loss.detach().item() * inputs.shape[0]
                    data_count += inputs.shape[0]
                    start += batch_size
                    end += batch_size
                    #print('Val batch time:', timer() - batch_start)

                val_cost = running_val_loss/data_count
                del running_val_loss
                #lr_scheduler.step(val_cost)
                #print('lr:', optimizer.param_groups[0]['lr'])
                print('Validation loss: {:.4e}'.format(val_cost))
                val_epoch_log.write('%d %e\n' %(epoch, val_cost))
                val_epoch_log.flush()
                os.fsync(val_epoch_log)
                val_cost_history.append(val_cost)

                writer.add_scalar('Loss/val', val_cost, epoch)
                '''
                if val_cost < best_val_cost and epoch % save_every_epoch == 0:
                    print('validation cost is lower than best before, saving model...')
                    best_val_cost = val_cost
                    save_path = os.path.join(save_dir, 'val_model_best.pth.tar')
                    save_model(model, optimizer, lr, val_cost, epoch, save_path)
                    print('saved model state', save_path)
                '''
            # save checkpoint
            if (epoch+1) % save_every_epoch == 0:
                save_path = os.path.join(save_dir, 'checkpoint-%d.pth.tar' %(epoch+1))
                save_model(model, optimizer, lr, 0, epoch, save_path)
                print('saved model state', save_path)

        elapsed = timer() - start_time
        print('Training completed in', time.strftime('%H:%M:%S', time.gmtime(elapsed)))
        #print('Best validation cost: {:.2e}'.format(best_val_cost))

        train_iter_log.close()
        train_epoch_log.close()
        val_epoch_log.close()

    elif mode == 1:
        print("Prediction mode")
        import _pickle as pickle

        # Load weights
        print('Load checkpoint',args.cp)
        cp=torch.load(args.cp,map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
        print('trained for', cp['epoch'], 'epochs', 'val cost',cp['val_cost'])
        model.load_state_dict(cp['state_dict'])
        optimizer.load_state_dict(cp['optimizer'])

        testdir = os.path.join(rundir, 'test')
        if not os.path.exists(testdir):
            os.makedirs(testdir)
        print('rundir test', testdir)

        criterion = nn.MSELoss()
        torch.set_grad_enabled(False)

        #  Dataloaders
        dataset_names = ['train']
        datasets = {}
        dataloaders = {}
        for name in dataset_names:
            dataset = ImplicitHumanDataset(name, datasetroot, imgdirs[name],
                                           tsdfdirs[name],
                                           device, resolution,
                                           normal_img=estimate_normal_img,
                                           renderer=renderer,
                                           use_pkl=use_pkl)
            datasets[name] = dataset
            dataloaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        start = 0
        end = batch_size
        running_val_loss = 0.0
        data_count = 0
        for idx, batch in enumerate(dataloaders['train']):
            torch.cuda.empty_cache()
            batch_start = timer()
            inputs = batch['images']
            outputs = model(inputs)
            '''
            frame_number = data_idx[idx] #min(100, 10*(data_count+1)+1)
            print('Frame:', frame_number)
            frame_root =  '/data/jwu/RenderPeople/rp_nathalie_4d_001_dancing_sequence/%04d/' % frame_number
            gt_fname = os.path.join(frame_root, 'mesh/gt_sdf.npy')
            outputs = np.expand_dims(np.load(gt_fname), 0)
            outputs = torch.tensor(outputs).to(device)
            print('GT:', outputs.shape)
            '''
            loss = 0
            if is_label_TSDF:
                label = batch['tsdfs']
                loss += criterion(outputs, label)
            if estimate_normal_img:
                gt_normalf = batch['normals']
                gt_masks = batch['masks']
                if not use_pkl:
                    pose = batch['poses']
                else:
                    pose = {
                        'pose': batch['poses'],
                        'scale': batch['scales'],
                        'translation': batch['translations']
                    }
                cam = (batch['camera_K'], batch['camera_R'], batch['camera_T'])
                losses, gt_normalf, pred_normalf, pred_depth = compute_normal_loss(pose, cam,
                                                                                   renderer, sdf_reg,
                                                                                   criterion,
                                                                                   batch, outputs, device,
                                                                                   epoch=0, step=idx,
                                                                                   rundir=testdir,
                                                                                   save_obj=True,
                                                                                   depth=True)
                loss += losses['normal']

                pred_mask = torch.linalg.norm(pred_normalf, axis=-1) > 0.5
                pred_mask = torch.unsqueeze(pred_mask, -1)

                # Compute correct normal error
                #print('Normal maps:', torch.unique(pred_normalf), torch.unique(gt_normalf))
                normal_errors = torch.clip(torch.sum(torch.multiply(pred_normalf, gt_normalf), dim=-1), -1, 1)
                normal_errors = torch.unsqueeze(normal_errors, -1)[0]
                normal_errors = .5*(1-normal_errors)

                # Two masks for intersecting and non-intersecting silhouettes
                mask = (gt_masks * pred_mask).type(torch.float32)[0]
                background_mask = ((1-gt_masks.type(torch.float32))*(1-pred_mask.type(torch.float32))).type(torch.float32)[0]

                # Nonoverlapping regions should have max error (1)
                normal_errors = mask*normal_errors + (1-mask)*torch.ones(normal_errors.shape, dtype=torch.float32).to(device)
                normal_errors = normal_errors*(1-background_mask)
                plot_errors(normal_errors[...,0], 1-background_mask, os.path.join(testdir, 'normals_error_%d.png' % idx))

                print('NORMAL ERROR:', torch.mean(torch.square(normal_errors)))
                '''
                # Ground truth depth map
                root_dir = frame_root
                obj_path = os.path.join(root_dir, 'mesh/mesh.obj')
                obj = trimesh.load(obj_path, process=False)
                gt_verts = (torch.from_numpy(obj.vertices).type(torch.float32).to(device), )
                gt_faces = (torch.from_numpy(obj.faces).to(device), )
                train_normal_img, train_mask, raster_vert_indices, raster_vert_weights = renderer.compute_normal_map(cam, (gt_verts[0],),
                                                                          gt_verts[0].unsqueeze(0), gt_faces[0].unsqueeze(0), output_weights=True)

                gt_depth = renderer.compute_depth_map(cam, gt_verts[0], raster_vert_indices, raster_vert_weights)
                plot_errors(gt_depth, gt_masks[0], os.path.join(testdir, 'gt_depth_%d.png' % idx))

                # Compute depth maps
                plot_errors(pred_depth, pred_mask[0], os.path.join(testdir, 'pred_depth_%d.png' % idx))

                # Compute depth map error
                depth_errors = pred_depth - gt_depth
                depth_errors = torch.unsqueeze(depth_errors, -1)
                depth_errors = mask*depth_errors + (1-mask)*(0.2)*torch.ones(depth_errors.shape, dtype=torch.float32).to(device)
                depth_errors = depth_errors*(1-background_mask)
                plot_errors(torch.clip(torch.abs(depth_errors[...,0]), 0, 0.2), 1-background_mask, os.path.join(testdir, 'depth_errors_%d.png' % idx))
                print('DEPTH ERROR:', torch.mean(torch.square(depth_errors)))
                '''
                # Save image
                pred_mask = pred_mask[0].detach().cpu().numpy()

                pred_image = ((pred_normalf*0.5+0.5)*255)[0]
                pred_image = pred_image.detach().cpu().numpy().astype(np.uint8)
                print('Normal MSE:', losses['normal'])
                pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
                pred_image = pred_image*pred_mask + (1-pred_mask)*np.ones(pred_image.shape, dtype=np.uint8)*255
                cv2.imwrite(os.path.join(testdir, 'pred_%d.png' % data_count), pred_image)

                # Save gt image
                pred_mask = gt_masks[0].detach().cpu().numpy()

                pred_image = ((gt_normalf*0.5+0.5)*255)[0]
                pred_image = pred_image.detach().cpu().numpy().astype(np.uint8)
                pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
                pred_image = pred_image*pred_mask + (1-pred_mask)*np.ones(pred_image.shape, dtype=np.uint8)*255
                cv2.imwrite(os.path.join(testdir, 'gt_%d.png' % data_count), pred_image)

            running_val_loss += loss.detach().item() * inputs.shape[0]
            data_count += inputs.shape[0]
            start += batch_size
            end += batch_size
            print('Val batch time:', timer() - batch_start)

        val_cost = running_val_loss/data_count
        print('Val cost:', val_cost)

def plot_errors(errors, mask, filename):
    print('Errors:', errors.shape)
    minima = torch.min(errors)
    maxima = torch.max(errors)
    print('Extrema:', minima, maxima)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    colors = (cm.jet(norm(errors.detach().cpu().numpy()))[...,:3]*255).astype(np.uint8)
    mask = mask.detach().cpu().numpy().astype(np.uint8)
    colors = mask * colors + (1-mask)*np.ones(colors.shape, dtype=np.uint8)*255
    cv2.imwrite(filename, cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))


if __name__=='__main__':
    main()
