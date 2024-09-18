######################################################################
# Copyright 2021-2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
import time
import os
from os.path import join, splitext, basename, exists, isdir, isfile
import argparse
import cv2
import numpy as np
import glob
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from timeit import default_timer as timer 

from src import tetranet
from src import utils
from src.dataloader import DataLoader
from src.renderer_single import Renderer
from src.sdf_regularizers import NormGradPhiFunction, SdfRegularizers, SmearedHeavisideFunction
from src.marching_tetrahedra_cuda import TSDF2MeshFunction


def save_model(model, optimizer, lr, val_cost, epoch, save_path):
    state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': lr,
            'val_cost': val_cost,
            'optimizer' : optimizer.state_dict(),
            }
    torch.save(state, save_path)


def norm_squared(x):
    return torch.sum(torch.square(x), dim=-1)


# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook
def main():
    parser = argparse.ArgumentParser(description='Regress TSDF from single RGB image')

    parser.add_argument(
        '--rundir',
        default='./rundir',
        type=str,
        help='Where logs and weights will be saved')
    parser.add_argument(
        '--trial',
        default='tetratsdf',
        type=str,
        help='Trial name')
    parser.add_argument(
        '--datasetroot',
        default='./D_march',
        type=str,
        help='Path to the directory contains dataset')

    parser.add_argument(
        '--imgdirs',
        default="datasetpaths_train/imgdirs_train.txt",
        type=str,
        help='Relative path list to the imgs for training (from datasetroot)')
    parser.add_argument(
        '--tsdfdirs',
        default="datasetpaths_train/TSDFdirs_train.txt",
        type=str,
        help='Relative path list to the TSDFs for training (from datasetroot)')
    parser.add_argument(
        '--normaldirs',
        default="datasetpaths_train/normaldirs_train.txt",
        type=str,
        help='Relative path list to the normal maps for training (from datasetroot)')
    parser.add_argument(
        '--cp',
        default="",
        type=str,
        help="Pretrained model weights")
    parser.add_argument(
        '--adjlists',
        default="./adjLists",
        type=str,
        help="Adjlists to construct PCN")
    parser.add_argument(
        '--mode',
        default=0,
        type=int,
        help="0:training, 1:prediction")
    parser.add_argument(
        '--lr',
        default=5e-04,
        type=float)
    parser.add_argument(
        '--begin_epoch',
        default=0,
        type=int)
    parser.add_argument(
        '--end_epoch',
        default=200,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int)
    parser.add_argument(
        '--val_ratio',
        default=0.1,
        type=float)
    parser.add_argument(
        '--save_interval',
        default=100,
        type=int)
    parser.add_argument('--normal_img', action='store_true')
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
    rundir = join(args.rundir, args.trial)
    path_adjlists = args.adjlists
    datasetroot = args.datasetroot
    imgdirs = os.path.join(datasetroot, args.imgdirs)
    tsdfdirs = os.path.join(datasetroot, args.tsdfdirs)
    normaldirs = os.path.join(datasetroot, args.normaldirs)
    estimate_normal_img = args.normal_img
    ################################

    # Set seed
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)

    model = tetranet.TetraNet(path_adjlists)
    model = model.to(device)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print('adding params to update', name)
            params_to_update.append(param)

    optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=1e-6)
    print('Creating scheduler')
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-06, threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08)

    # Initialize custom functions
    renderer = Renderer(device, img_dim=512)
    sdf_reg = SdfRegularizers(device)

    is_label_TSDF = True
    if not os.path.exists(tsdfdirs):
        is_label_TSDF=False
        print("No TSDF labels. Only use normal map loss...")

    if mode == 0:
        print("Training mode")
        print("Model parameters:")
        summary(model, input_size=(3, 256, 256))

        # Tensorboard setup
        writer = SummaryWriter(rundir)

         # Load weights
        if args.cp != "":
            print('Load checkpoint',args.cp)
            cp=torch.load(args.cp,map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
            print('trained for', cp['epoch'], 'epochs', 'val cost',cp['val_cost'])
            model.load_state_dict(cp['state_dict'])
            optimizer.load_state_dict(cp['optimizer'])
            optimizer.param_groups[0]['lr'] = lr

        criterion = nn.L1Loss()

        # Training and validation data loading
        dl = DataLoader(datasetroot, imgdirs, tsdfdirs, normaldirs, batch_size, val_ratio, None, rundir, device, renderer)
        train_x, train_y = dl.load_batches("train")
        n_train = train_x.shape[0]
        steps_per_epoch = dl.steps_per_epoch_train

        val_x, val_y = dl.load_batches("val")
        n_val = val_x.shape[0]
        #assert(len(dl.Ids_train)==n_train)
        #assert(len(dl.Ids_val)==n_val)

        # Create output path
        if not os.path.exists(rundir):
            os.makedirs(rundir)
        print('rundir', rundir)

        np.save(os.path.join(rundir, "training_ids.npy"), dl.Ids_train)
        np.save(os.path.join(rundir, "validation_ids.npy"), dl.Ids_val)

        # Training settings
        start_time = time.time()

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
        opt_curvature = False
        for epoch in range(begin_epoch, end_epoch):
            print('-----------------------------------------------------')
            print('Epoch {}/{}'.format(epoch+1, end_epoch-begin_epoch))
            print('-----------------------------------------------------')
            torch.cuda.empty_cache()
            running_loss = 0.0
            model.train()
            torch.set_grad_enabled(True)
            start = 0
            end = batch_size
            data_count = 0

            # Shuffle training data
            ''' 
            shuffle_idx = np.arange(n_train)
            np.random.shuffle(shuffle_idx)
            train_x = train_x[shuffle_idx]
            if is_label_TSDF:
                train_y = train_y[shuffle_idx]
            '''
            if estimate_normal_img:
                #train_pose = train_pose[shuffle_idx]
                #train_cam = train_cam[shuffle_idx]
                train_normal_imgs = train_normal_imgs[shuffle_idx]
                train_masks = train_masks[shuffle_idx]
            
            for itr in range(steps_per_epoch):
                #start_time = timer()
                # Data term
                inputs = train_x[start:min(end, n_train),:,:,:]
                label = train_y[start:min(end, n_train),:]
                outputs = model(inputs)
                data_loss = criterion(outputs, label)

                # Rescale output to be [-1, 1]
                tsdf_pred = outputs/10.
                tsdf_gt = label/10.
                #print('Data runtime:', timer()-start_time)

                # Regularization
                if len(inputs) >1:
                    sdf_idx = np.random.randint(0, len(inputs)-1)
                else:
                    sdf_idx = 0

                # Apply heaviside!
                heaviside_tsdf = SmearedHeavisideFunction.apply(tsdf_pred[sdf_idx], sdf_reg)
                tet_linear_coeffs, interface_tets = NormGradPhiFunction.apply(torch.unsqueeze(heaviside_tsdf,0), sdf_reg)
                #norm_grad = torch.linalg.norm(tet_linear_coeffs, axis=-1)
                norm_grad = norm_squared(tet_linear_coeffs)
                num_tets = len(norm_grad)
                curvature_scale = 100
                curvature_loss = curvature_scale * torch.sum(torch.square(sdf_reg.tet_volumes_cuda) * norm_grad) / float(num_tets)
                
               
                #start_time = timer()
                sdf_scale = 1e6
                sdf_tet_linear_coeffs, interface_tets = NormGradPhiFunction.apply(tsdf_pred[[sdf_idx]], sdf_reg)
                sdf_norm_grad = norm_squared(sdf_tet_linear_coeffs)
                num_tets = sdf_norm_grad.shape[-1]
                sdf_loss = sdf_scale * .5 *torch.sum(sdf_reg.tet_volumes_cuda * torch.square(sdf_norm_grad - 1)) / float(num_tets)
                #print('SDF runtime:', timer()-start_time)

                writer.add_scalar('Loss/data', data_loss, epoch)
                writer.add_scalar('Loss/curvature', curvature_loss, epoch)
                writer.add_scalar('Loss/sdf', sdf_loss, epoch)
                del sdf_tet_linear_coeffs, sdf_norm_grad

                loss = data_loss + sdf_loss
                ''' 
                if opt_curvature:
                    loss = curvature_loss
                else:
                    loss = sdf_loss
                '''
                start_time = timer()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                #print('Backprop runtime:', timer()-start_time)
                loss_item = loss.item()
                running_loss += loss_item * inputs.shape[0]
                data_count += inputs.shape[0]
                start += batch_size
                end += batch_size

                # Periodically save normal maps
                num_visualize = min(3, len(tsdf_pred))
                if (epoch%10==0) and (itr==0): 
                    if (epoch % 100) == 0:
                        plot_vals = heaviside_tsdf.detach().cpu().numpy()
                        plot_vals = [plot_vals[i] for i in range(len(heaviside_tsdf)) if plot_vals[i] > 0 and plot_vals[i]<1]
                        n, bins, patches = plt.hist(plot_vals, 100)
                        plt.ylabel("Count")
                        plt.xlabel("Smeared Heaviside values")
                        plt.savefig(os.path.join(rundir, 'heaviside_%i_%i.png' % (itr, epoch)))
                        plt.close()

                        for k in range(num_visualize): 
                            sdf_tet_linear_coeffs, interface_tets = NormGradPhiFunction.apply(tsdf_pred[[k]], sdf_reg)
                            sdf_norm_grad = norm_squared(sdf_tet_linear_coeffs[0])
                            interface = interface_tets[0].detach().cpu().numpy()
                            norm_grad_np = sdf_norm_grad.detach().cpu().numpy()
                            plot_vals = [norm_grad_np[i] for i in range(num_tets) if (interface[i])]
                            #print('Values:', np.unique(plot_vals))
                            n, bins, patches = plt.hist(plot_vals, 100)
                            plt.ylabel("Count")
                            #plt.xlabel("Phi on the tetrahedral mesh vertices")
                            plt.xlabel("Norm grad phi in each tetrahedra")
                            #plt.show()
                            plt.savefig(os.path.join(rundir, 'norm_grad_phi_%i_%i.png' % (k, epoch)))
                            plt.close()

                    #print('Verts:', verts_t.shape)
                    device = 'cuda'
                    camera = torch.tensor(np.array([0.8927, -0.8927, -0.0121, 0.1976])).to(device)
                    camera = torch.unsqueeze(camera, 0)

                    def write_normal_map(tsdf):
                        vert_tet_weights, vert_tet_indices, faces_t = TSDF2MeshFunction.apply(tsdf, renderer.mt)
                        tet_verts = renderer.mt.nodes[vert_tet_indices.long()]
                        tet_weights = renderer.mt.weights[vert_tet_indices.long()]
                        verts_t = torch.matmul(vert_tet_weights.unsqueeze(-2), tet_verts)[..., 0, :]

                        img,_ = renderer.compute_normal_map(camera, [verts_t],
                                                          torch.unsqueeze(verts_t, 0).type(torch.float32),
                                                          torch.unsqueeze(faces_t,0), azimuth=0.)
                        return ((img*0.5+0.5)*255).permute(0, 3, 1, 2).type(torch.uint8)
                    if epoch > 10 and (itr==0):
                        pred_images = torch.cat([write_normal_map(tsdf_pred[i]) for i in range(num_visualize)], 0)
                        writer.add_images('Images/train_images_%d' % itr, pred_images, epoch)
                        gt_images = torch.cat([write_normal_map(tsdf_gt[i]) for i in range(num_visualize)], 0)
                        writer.add_images('Images/gt_images_%d' % itr, gt_images, epoch)
                        writer.add_images('Images/input_images_%d' % itr, (inputs[:num_visualize]*255).type(torch.uint8), epoch)

                # Switch regularizers
                if itr == steps_per_epoch -1:
                    if epoch % 100 == 0:
                        if opt_curvature:
                            opt_curvature = False
                            print('Switching to SDF')
                        elif epoch % 300 == 0:
                            opt_curvature = True
                            print('Switching to curvature')
                
                print('Batch time:', timer() - start_time)

            assert(data_count == n_train)
            cost = running_loss/n_train
            print('Train loss: {:.4e}'.format(cost))
            train_epoch_log.write('%d %e\n' %(epoch, cost))
            train_epoch_log.flush()
            os.fsync(train_epoch_log)
            train_cost_history.append(cost)
            '''
            #TODO(jwu): Re-add this
            # Validation
            model.eval()
            torch.set_grad_enabled(False)
            start = 0
            end = batch_size
            running_val_loss = 0.0
            data_count = 0
            for itr in range(dl.steps_per_epoch_val):
                inputs = torch.Tensor(val_x[start:min(end, n_val), ...]).to(device)
                label = torch.Tensor(val_y[start:min(end, n_val), ...]).to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, label)
                running_val_loss += val_loss.item() * inputs.shape[0]
                data_count += inputs.shape[0]
                start += batch_size
                end += batch_size
            assert(data_count == n_val)
            val_cost = running_val_loss/n_val
            lr_scheduler.step(val_cost)
            print('lr:', optimizer.param_groups[0]['lr'])
            print('Validation loss: {:.4e}'.format(val_cost))
            val_epoch_log.write('%d %e\n' %(epoch, val_cost))
            val_epoch_log.flush()
            os.fsync(val_epoch_log)
            val_cost_history.append(val_cost)
            '''
            writer.add_scalar('Loss/train', cost, epoch)
            #writer.add_scalar('Loss/val', val_cost, epoch)

            # deep copy the model
            '''
            #TODO(jwu): Re-add this
            if val_cost < best_val_cost:
                print('validation cost is lower than best before, saving model...')
                best_val_cost = val_cost
                #best_val_model = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir, 'val_model_best.pth.tar')
                save_model(model, optimizer, lr, val_cost, epoch, save_path)
                print('saved model state', save_path)
            '''
            '''
            if cost < best_train_cost:
                print('training cost is lower than best before, saving model...')
                best_train_cost = cost
                #best_train_model = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir, 'train_model_best.pth.tar')
                save_model(model, optimizer, lr, cost, epoch, save_path)
                print('saved model state', save_path)
            '''
            # save checkpoint
            if (epoch+1) % save_every_epoch == 0:
                save_path = os.path.join(save_dir, 'checkpoint-%d.pth.tar' %(epoch+1))
                save_model(model, optimizer, lr, 0, epoch, save_path)
                print('saved model state', save_path)

        elapsed = time.time() - start_time
        print('Training completed in', time.strftime('%H:%M:%S', time.gmtime(elapsed)))
        #print('Best validation cost: {:.2e}'.format(best_val_cost))

        train_iter_log.close()
        train_epoch_log.close()
        val_epoch_log.close()

    elif mode == 1:
        print("Prediction mode")

        # Load weights
        print('Load checkpoint',args.cp)
        cp=torch.load(args.cp,map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
        print('trained for', cp['epoch'], 'epochs', 'val cost',cp['val_cost'])
        model.load_state_dict(cp['state_dict'])
        optimizer.load_state_dict(cp['optimizer'])

        criterion = nn.L1Loss()

        model.eval()
        torch.set_grad_enabled(False)

        # Training and validation data loading
        dl = DataLoader(datasetroot, imgdirs, tsdfdirs, batch_size, val_ratio, None, rundir, device, pifu, renderer, inference=True)
        if estimate_normal_img:
            val_x, val_y, val_pose, val_cam, val_normal_imgs, val_masks = dl.load_batches("train", estimate_normal_img)
        else:
            val_x, val_y = dl.load_batches("val", estimate_normal_img)
        n_val = val_x.shape[0]
        print("validation data:", val_x.shape, val_y.shape)

        assert(len(dl.Ids_train)==val_x.shape[0])

        running_val_loss = 0
        running_val_normal_loss = 0
        for i in range(20):
            inputs = torch.Tensor(np.expand_dims(val_x[i,:,:,:], 0)).to(device)
            label = torch.Tensor(np.expand_dims(val_y[i, ...], 0)).to(device)

            inference = model(inputs)
            val_loss = criterion(inference, label)
            running_val_loss += val_loss.item()
            del label
            if estimate_normal_img:
                pose = val_pose[i, ...]
                cam = val_cam[i, ...]
                tsdf = inference[0]*2-1 
                gt_normalf = val_normal_imgs[i,:]
                gt_mask = val_masks[i,:]
                normalf, mask = renderer(tsdf, pose, cam)
                normal_loss = criterion(normalf*mask*gt_mask, gt_normalf*gt_mask)
                running_val_normal_loss += normal_loss.item()
                        
            #print("inference:", dl.Ids_val[i], inference.shape, inference[0,:10])
            #utils.saveTSDF_bin(inference[0].detach().cpu().numpy(), os.path.join(rundir, 'test_val', 'test_%d.bin' % dl.Ids_val[i]))

        val_cost = running_val_loss/n_val
        val_normal_cost = running_val_normal_loss/n_val
        print('TSDF loss:', val_cost)
        print('Normal loss:', val_normal_cost)


if __name__=='__main__':
    main()
