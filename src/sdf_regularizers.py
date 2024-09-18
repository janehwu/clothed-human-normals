######################################################################
# Copyright 2022. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
from timeit import default_timer as timer 
import os

import torch
import torch.autograd as autograd
#import pycuda.autoinit
import pycuda
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

import sys
sys.path.append('/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Scripts/learning/src')
sys.path.append('/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Scripts/learning')
from sdf_regularizer_cuda_kernels import mod
import utils as utils

BLOCK_SIZE = 128

ComputeEikonalGrads_gpu = mod.get_function('ComputeEikonalGrads')
ComputeLinearCoeffs_gpu = mod.get_function('ComputeLinearCoeffs')
ComputeSmearedHeaviside_gpu = mod.get_function('ComputeSmearedHeaviside')

drv.init()
#TODO: fix this for multiple devices!
pycuda_ctx = drv.Device(0).retain_primary_context()

SCALE = 0.015*3

def smeared_heaviside(x, eps=1.5*SCALE):
    assert eps > 0
    if x < -eps:
        return 0
    elif x > eps:
        return 1
    else:
        return 0.5 + x/(2.*eps) + 1/(2.*np.pi)*np.sin((np.pi*x)/eps)

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()

class SmearedHeavisideFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, sr):
        sdf = input.detach()
        output, grads = sr.smeared_heaviside(sdf)

        ctx.save_for_backward(grads)
        return output.to(sr.device)

    @staticmethod
    def backward(ctx, grad_heaviside):
        #start = timer()
        device = 'cuda'
        grad_input = grad_heaviside.clone()
        heaviside_grads = ctx.saved_tensors[0]
        return grad_heaviside * heaviside_grads, None


class NormGradPhiFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, sr):
        sdf = input.detach()
        batch_tet_coeffs = []
        batch_interface_tets = []
        for i in range(len(sdf)):
            tet_coeffs, interface_tets = sr.norm_grad_phi(sdf[i])
            batch_tet_coeffs.append(tet_coeffs)
            batch_interface_tets.append(interface_tets)
        batch_tet_coeffs = torch.stack(batch_tet_coeffs)
        batch_interface_tets = torch.stack(batch_interface_tets)

        ctx.save_for_backward(sr.grad_weights_wrt_sdf)
        #print('Unique coeffs:', np.unique(tet_coeffs.cpu().numpy()))
        return batch_tet_coeffs.to(sr.device), batch_interface_tets.to(sr.device)

    @staticmethod
    def backward(ctx, grad_linear_coeffs, grad_interface_tets):
        #start = timer()
        device = 'cuda'
        grad_input = grad_linear_coeffs.clone()

        #Input should be gradient wrt output
        grad_weights_wrt_sdf = ctx.saved_tensors[0]

        #grad_weights_flat = torch.unsqueeze(torch.flatten(grad_input), -1)
        grad_weights_flat = torch.reshape(grad_input, (len(grad_input), -1))
        grad_out = torch.sparse.mm(grad_weights_wrt_sdf.t(), grad_weights_flat.t()).t()
        #print('Backprop MT:', timer() - start)

        return grad_out, None


class SdfRegularizers():

    def __init__(self, device, grid=False):
        super(self.__class__, self).__init__()
        print('Initializing SdfRegularizers')
        start = timer()

        self.device = device
        # Load stored data
        root_dir = os.path.dirname(os.path.realpath(__file__))
        if grid:
            root_dir += 'grid_sphere'

        tetrahedra_idx = np.load(os.path.join(root_dir, 'delta_matrix_indices.npy'))
        delta_inv_matrix = np.load(os.path.join(root_dir, 'delta_inverse_matrix.npy')).astype(np.float32)
        tetrahedra_volumes = np.load(os.path.join(root_dir, 'tetrahedra_volumes.npy')).astype(np.float32)

        self.tetrahedra_idx = tetrahedra_idx
        self.tetrahedra_volumes = tetrahedra_volumes
        print('Loaded data:', tetrahedra_idx.shape, delta_inv_matrix.shape, tetrahedra_volumes)

        # Flatten and transfer data to GPU
        self.nb_tets = len(tetrahedra_idx)
        self.tet_idx_cuda = torch.from_numpy(tetrahedra_idx.flatten().astype(np.int32)).to(self.device).detach()
        self.delta_inv_matrix_cuda = torch.from_numpy(delta_inv_matrix.flatten().astype(np.float32)).to(self.device).detach()
        self.tet_volumes_cuda = torch.from_numpy(tetrahedra_volumes).to(self.device).detach()

        # Output arrays, only initialize arrays once
        tet_coeffs_np = np.zeros((self.nb_tets, 3), dtype=np.float32)
        self.tet_coeffs_cuda = torch.from_numpy(tet_coeffs_np.flatten().astype(np.float32)).to(self.device).detach()

        coeff_gradients_np = np.zeros((self.nb_tets, 3*4), dtype=np.float32) # dcoeffs / dphi
        self.coeff_gradients_cuda = torch.from_numpy(coeff_gradients_np.flatten().astype(np.float32)).to(self.device).detach()

        self.interface_tets_cuda = torch.from_numpy(np.zeros(self.nb_tets, dtype=np.int32)).to(self.device).detach()

        # Smeared heaviside outputs
        if grid:
            self.n_verts = 81530 #355487
        else:
            self.n_verts = 285213

        # Precompute norm grad phi gradients (constant in each tetrahedra)
        self.grad_weights_wrt_sdf = self.norm_grad_phi_gradients()
        print('Gradients for coefficients:', self.grad_weights_wrt_sdf.shape)

        del tetrahedra_idx, delta_inv_matrix, tet_coeffs_np, coeff_gradients_np
        print('Init time:', timer()-start)

    def norm_grad_phi_gradients(self):
        thread_size = self.nb_tets
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1
        #print('thread_size: {}'.format(thread_size))
        #print('blocks_per_grid: {}'.format(blocks_per_grid))
        
        params = np.array([self.nb_tets]).astype(np.float32)

        ComputeEikonalGrads_gpu(Holder(self.delta_inv_matrix_cuda),
                                Holder(self.coeff_gradients_cuda),
                                drv.In(params),
                                block=blockdim, grid=blocks_per_grid)
        coeff_grads = self.coeff_gradients_cuda.reshape((self.nb_tets, 3, 4))  # Reshape for backprop.
        coeff_tet_indices = self.tet_idx_cuda

        # Vectorized Jacobian computation
        num_tet_connections = 4
        idx = torch.unsqueeze(torch.arange(self.nb_tets*3).to(self.device).reshape((-1,3)), -1)
        idx = torch.tile(idx, (1, 1, num_tet_connections)).reshape((-1,1)) # Now has shape (n_verts*12, 1)

        tet_v = torch.reshape(coeff_tet_indices, (-1, 4))
        tet_v = torch.unsqueeze(torch.repeat_interleave(tet_v, 3, dim=0).flatten(), -1) # Now has shape (n_verts*12, 1)
        idx = torch.cat([idx, tet_v], -1)

        vals = coeff_grads.view(-1,)
        output = torch.sparse_coo_tensor(idx.t(), vals, (self.nb_tets*3, self.n_verts), dtype=torch.float32)
        return output

    def norm_grad_phi(self, sdf):
        thread_size = self.nb_tets
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1
        #print('thread_size: {}'.format(thread_size))
        #print('blocks_per_grid: {}'.format(blocks_per_grid))
        
        params = np.array([self.nb_tets]).astype(np.float32)
        sdf = sdf.flatten()

        ComputeLinearCoeffs_gpu(Holder(sdf),
                                Holder(self.tet_idx_cuda),
                                Holder(self.delta_inv_matrix_cuda),
                                Holder(self.tet_coeffs_cuda),
                                Holder(self.interface_tets_cuda),
                                drv.In(params),
                                block=blockdim, grid=blocks_per_grid)

        # Reshape all outputs
        tet_coeffs_out = self.tet_coeffs_cuda.reshape((self.nb_tets, 3))
        return tet_coeffs_out, self.interface_tets_cuda

    def smeared_heaviside(self, sdf):
        thread_size = len(sdf)
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1
        #print('thread_size: {}'.format(thread_size))
        #print('blocks_per_grid: {}'.format(blocks_per_grid))

        # Initialize output arrays
        heaviside_output = torch.zeros(thread_size, dtype=torch.float32).to(self.device).detach()
        heaviside_grads = torch.zeros(thread_size, dtype=torch.float32).to(self.device).detach()

        params = np.array([len(sdf), SCALE]).astype(np.float32)
        sdf = sdf.flatten()

        ComputeSmearedHeaviside_gpu(Holder(sdf),
                                    Holder(heaviside_output),
                                    Holder(heaviside_grads),
                                    drv.In(params),
                                    block=blockdim, grid=blocks_per_grid)
        return heaviside_output, heaviside_grads

if __name__ == '__main__':
    print('Testing sdf regularizer')

    # Load TSDF values
    tsdf_filename = '/data/jwu/D_march/TSDF/mesh_D_march_0000.bin'
    tsdf = utils.loadTSDF_bin(tsdf_filename)
    tsdf = tsdf*2-1
    print('TSDF:', tsdf.shape)
    print(np.unique(tsdf))

    tsdf_gpu = torch.tensor(tsdf, dtype=torch.float32, device='cuda')

    sr = SdfRegularizers(device='cuda:0')

    for i in range(20):
        start = timer()
        tet_coeffs, interface_tets = sr.norm_grad_phi(tsdf_gpu)
        print('SDF Reg:', timer()-start)
        print('Outputs:', tet_coeffs.shape)
        break
