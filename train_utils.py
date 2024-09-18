######################################################################
# Copyright 2021-2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from timeit import default_timer as timer 

#from chamferdist import ChamferDistance
import matplotlib.pyplot as plt
import numpy as np
import torch
#from lib.utils.demo_utils import convert_crop_cam_to_orig_img

from src.sdf_regularizers import NormGradPhiFunction, SdfRegularizers, SmearedHeavisideFunction
from src.renderer import Renderer


def save_model(model, optimizer, lr, val_cost, epoch, save_path):
    '''Save TetraTSDF and VIBE models, as well as other training parameters.'''
    state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': lr,
            'val_cost': val_cost,
            'optimizer' : optimizer.state_dict(),
            }
    torch.save(state, save_path)


def get_vibe_inference(vibe_model, batch):
    vibe_inputs = batch['vibe_inputs'].unsqueeze(1)
    B, seqlen = vibe_inputs.shape[:2]
    vibe_output = vibe_model(vibe_inputs)[-1]
    pose = vibe_output['theta'][:,:,3:75].reshape(B*seqlen, -1).detach()
    pred_cam = vibe_output['theta'][:, :, :3].reshape(B*seqlen, -1)
    cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=batch['vibe_bboxes'],
        img_width=256,
        img_height=256
    ).detach()
    return pose, cam

def norm_squared(x):
    '''Compute norm squared of a tensor along the last dimension.'''
    return torch.sum(torch.square(x), dim=-1)


def compute_sdf_losses(sdf, sdf_reg, epoch, rundir, device, plot_regularizers, verbose):
    '''Computes the Eikonal and motion by mean curvature regularization losses.

    Args:
        sdf: [B, N] containing SDF values on the tetrahedral mesh vertices.
        sdf_reg: SdfRegularizers object.
        epoch: Epoch number.
        rundir: Experiment root directory.
        verbose: Whether to print runtimes.

    Returns:
        A dictionary storing the Eikonal and motion by mean curvature losses.
    '''
    if verbose:
        start_time = timer()

    # If more than one batch element, choose a random index.
    if len(sdf) >1:
        sdf_idx = np.random.randint(0, len(sdf)-1)
    else:
        sdf_idx = 0

    # (0) Mask out outlier tets
    outlier_tet_idx = [169369, 540151, 1302954, 1474160]
    tet_mask = np.ones(1598018).astype(np.int64)
    tet_mask[outlier_tet_idx] = 0
    tet_mask = torch.tensor(tet_mask).to(device)

    # (1) Eikonal loss.
    sdf_scale = 1e5
    sdf_tet_linear_coeffs, interface_tets = NormGradPhiFunction.apply(sdf[[sdf_idx]], sdf_reg)
    sdf_norm_grad_sqrd = norm_squared(sdf_tet_linear_coeffs[0])
    num_interface_tets = torch.sum(interface_tets)
    sdf_loss = sdf_scale * .5*torch.sum(sdf_reg.tet_volumes_cuda * torch.square(sdf_norm_grad_sqrd - 1) * tet_mask) / float(num_interface_tets)

    # (2) Motion by mean curvature loss.
    curvature_scale = 5e3
    heaviside_tsdf = SmearedHeavisideFunction.apply(sdf[sdf_idx], sdf_reg)
    tet_linear_coeffs, interface_tets = NormGradPhiFunction.apply(torch.unsqueeze(heaviside_tsdf,0), sdf_reg)
    norm_grad = torch.linalg.norm(tet_linear_coeffs[0], axis=-1)
    norm_grad_squared = norm_squared(tet_linear_coeffs[0])
    num_tets = norm_grad.shape[-1]
    assert(interface_tets.shape[1] == num_tets)
    # Filter out tets wit large norm_grad_phi
    sdf_norm_grad = torch.linalg.norm(sdf_tet_linear_coeffs[0], axis=-1)
    valid_tets = torch.abs(sdf_norm_grad - 1.) < 1.
    valid_tets = valid_tets.type(torch.uint8) * (norm_grad_squared > 1e-12).type(torch.uint8)
    assert num_tets == len(valid_tets)
    curvature_loss = curvature_scale * torch.sum(sdf_reg.tet_volumes_cuda * norm_grad * valid_tets * tet_mask) / float(torch.sum(valid_tets))

    # Plot various values for debugging.
    if plot_regularizers and epoch % 100 == 0:
        # (1) SDF values.
        plot_vals = sdf[sdf_idx].detach().cpu().numpy()
        n, bins, patches = plt.hist(plot_vals, 100)
        plt.ylabel("Count")
        plt.xlabel("SDF values")
        plt.savefig(os.path.join(rundir, 'sdf_%i.png' % epoch))
        plt.close()

        # (2) Smeared heaviside values.
        plot_vals = heaviside_tsdf.detach().cpu().numpy()
        plot_vals = [plot_vals[i] for i in range(len(heaviside_tsdf)) if plot_vals[i] > 0 and plot_vals[i]<1]
        n, bins, patches = plt.hist(plot_vals, 100)
        plt.ylabel("Count")
        plt.xlabel("Smeared Heaviside values")
        plt.savefig(os.path.join(rundir, 'heaviside_%i.png' % epoch))
        plt.close()

        # (3) Norm grad SDF values near interface.
        interface = interface_tets[0].detach().cpu().numpy()
        norm_grad_np = torch.linalg.norm(sdf_tet_linear_coeffs[0], axis=-1).detach().cpu().numpy()
        plot_vals = [norm_grad_np[i] for i in range(num_tets) if (interface[i] and i not in outlier_tet_idx)]
        n, bins, patches = plt.hist(plot_vals, 100)
        plt.ylabel("Count")
        plt.xlabel("Norm grad phi in each tetrahedra")
        plt.savefig(os.path.join(rundir, 'norm_grad_phi_%i.png' % epoch))
        plt.close()

        # (4) Norm grad heaviside values.
        #norm_grad_np = (norm_grad[0]).detach().cpu().numpy()
        norm_grad_np = torch.linalg.norm(tet_linear_coeffs[0], axis=-1)
        valid_tet_idx = np.argwhere(interface == 1)
        plot_vals = (sdf_reg.tet_volumes_cuda * norm_grad_np).detach().cpu().numpy()[valid_tet_idx]
        n, bins, patches = plt.hist(plot_vals, 100)
        plt.ylabel("Count")
        plt.xlabel("Volume weighted norm grad Heaviside phi in each tetrahedra")
        plt.savefig(os.path.join(rundir, 'norm_grad_heaviside_phi_%i.png' % epoch))
        plt.close()

    outputs = {}
    outputs['curvature'] = curvature_loss
    outputs['eikonal'] = sdf_loss
    interface_norm_grad = torch.masked_select(sdf_norm_grad*tet_mask, mask=interface_tets[0].type(torch.BoolTensor).to(device))
    outputs['norm_grad_values'] = interface_norm_grad
    # outputs['outliers'] = outlier_tet_idx
    if verbose:
        print('SDF Regularizers:', timer() - start_time)
    return outputs


def compute_normal_loss(pose, camera_params, renderer, sdf_reg, criterion, batch, sdf,
                        device, epoch=-1, step =-1, sdf_losses=False, boundary_losses=False, rundir=None, save_obj=False,
                        depth=False, plot_regularizers=False, multiview=False, verbose=False):
    '''Computes the normal map loss and regularizer losses.

    Args:
        pose: [B, 72] SMPl pose parameters.
        camera_params: [B, 4] if using VIBE, or tuple storing K [B, 4, 4], R [B, 3, 3], and T[B, 3].
        renderer: Renderer object.
        sdf_reg: SdfRegularizers object.
        criterion: Loss criterion (for normal map loss only).
        batch: Dictionary of dataset batch.
        sdf: [B, N] containing SDF values on the tetrahedral mesh vertices.
        device: Device where data is stored.
        epoch: Epoch number.
        rundir: Experiment root directory.
        save_obj: Whether to save generated triangle mesh.
        verbose: Whether to print runtimes.
    '''
    losses = {}

    # Only compute regularization losses during training (could be documented better).
    if epoch >= 0 and sdf_losses:
        sdf_outputs = compute_sdf_losses(sdf, sdf_reg, epoch, rundir, device, plot_regularizers, verbose)
        losses['curvature'] = sdf_outputs['curvature']
        losses['eikonal'] = sdf_outputs['eikonal']
        losses['norm_grad_values'] = sdf_outputs['norm_grad_values']

    # Multiview normal loss.
    if multiview:
        multiview_camera_params = (batch['camera_K_multiview'], batch['camera_R_multiview'], batch['camera_T_multiview'])
        pred_normalf_mv, _ = renderer(sdf, pose, multiview_camera_params, epoch, step, rundir, save_obj, verbose)


    # Normal map loss.
    pred_normalf, pred_masks, verts, vert_tet_indices, raster_vert_indices, raster_vert_weights = renderer(sdf, pose, camera_params, epoch,
                                                                                                           step, rundir, save_obj, verbose, return_mesh=True)
    losses['normal'] = criterion(pred_normalf, batch['normals'])
    if multiview:
        losses['normal_mv'] = criterion(pred_normalf_mv, batch['normals_multiview'])
        losses['normal_pd_mv'] = pred_normalf_mv

    if boundary_losses:
        # Boundary loss: shrink.
        boundary_mask, boundary_image = compute_boundary_loss(pred_normalf, batch['masks'][[0]], sdf, vert_tet_indices, raster_vert_indices, renderer)
        eps = 5e-3
        boundary_heaviside = boundary_mask*(eps-sdf[0])
        losses['boundary'] = 0.1*norm_squared(boundary_heaviside)
        losses['boundary_mask'] = boundary_image

        # Boundary loss: expand.
        # Inflate sdf values iteratively...
        num_inflations = 1
        new_sdf = None
        for i in range(num_inflations):
            if i == 0:
                new_sdf = renderer.mt.inflate_sdf(sdf)
            else:
                new_sdf = renderer.mt.inflate_sdf(new_sdf)
        inflate_normalf, inflate_mask, _, inflate_vert_tet_indices, inflate_raster_vert_indices, _ = renderer(new_sdf, pose, camera_params, epoch, step, rundir, save_obj=False, return_mesh=True)
        losses['inflate_normals'] = inflate_normalf
        inflate_boundary_mask, inflate_boundary_image = compute_boundary_loss(batch['normals'][[0]], pred_masks.detach(), sdf, inflate_vert_tet_indices, inflate_raster_vert_indices, renderer, inflate=True)
        eps = 5e-3
        inflate_heaviside = inflate_boundary_mask*(eps+sdf[0])
        losses['inflate'] = 0.01*norm_squared(inflate_heaviside)
    
        # Color boundary image based on overlapping inflated triangles
        inflate_boundary_image = inflate_boundary_image.repeat(1, 1, 1, 3)
        inflate_image = inflate_boundary_image.long()
        inflate_image[..., 1:] -= (inflate_mask * inflate_boundary_image).long()[..., 1:]
        losses['inflate_mask'] = inflate_image.long()

    if depth:
        pred_depth = renderer.compute_depth_map(camera_params, verts[0], raster_vert_indices, raster_vert_weights)
        return losses, batch['normals'], pred_normalf, pred_depth
    return losses, batch['normals'], pred_normalf


#TODO: Handle batch training!!
def compute_boundary_loss(pred_normals, gt_masks, sdf, vert_tet_indices, raster_vert_indices, renderer, inflate=False):
    # Figure out which predicted pixels have no overlap with the ground truth
    pixels_mask = pred_normals * (1 - gt_masks.int())
    pixels_mask = torch.linalg.norm(pixels_mask, axis=-1) > 0.5
    pixels_to_shrink = torch.nonzero(pixels_mask, as_tuple=True)

    vertices_to_shrink = raster_vert_indices[pixels_to_shrink]
    unique_vert_idx = torch.unique(vertices_to_shrink).long()
    tet_idx = vert_tet_indices[unique_vert_idx]
    unique_tet_idx = torch.unique(tet_idx).long()

    # Create a SDF mask that selects tet vertices to be updated
    sdf_mask = torch.zeros(renderer.mt.nodes.shape[0]).to(renderer.device)
    for idx in unique_tet_idx:
        if inflate: #Only care about positive phi values
            if sdf[0, idx] > 0:
                sdf_mask[idx] = 1
        else: #Only care about negative phi values
            if sdf[0, idx] < 0:
                sdf_mask[idx] = 1
    return sdf_mask, pixels_mask.unsqueeze(-1)


def determine_regularizer(norm_grad_values):
    mean_val = torch.mean(norm_grad_values)
    mean_check = abs(mean_val - 1.) < 0.1

    outlier_val = torch.max(norm_grad_values)
    outlier_check = outlier_val < 4.

    quantile_val = torch.quantile(norm_grad_values, 0.95, 0, keepdim=False, interpolation='nearest')
    quantile_check = quantile_val < 2.

    regularizer = None
    if mean_check and quantile_check and outlier_check: #and other criteria
        regularizer = 'curvature'
    else:
        regularizer = 'eikonal'
    print('Regularizer:', regularizer, mean_val, quantile_val, outlier_val)
    return regularizer



