######################################################################
# Copyright 2022. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import ctypes

from scipy import sparse
from timeit import default_timer as timer 
import torch
import torch.autograd as autograd
#import pycuda.autoinit
import pycuda
import pycuda.driver as drv

import sys
filedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(filedir)
sys.path.append(filedir + "/../")
from mesh_normals import verts_normals_list
from mt_cuda_kernels import mod
import outershell as osh
import utils as utils
from obj_io import *

# Load backprop kernel first
def get_cuda_compute_mt_gradients():
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cuda')
    dll = ctypes.CDLL(os.path.join(root_dir, 'mt_backprop_kernel.so'), mode=ctypes.RTLD_GLOBAL)
    func = dll.Compute_Vertex_Gradients
    func.argtypes = [
                     ctypes.c_void_p,
                     ctypes.c_void_p,
                     ctypes.c_void_p,
                     ctypes.c_void_p,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                    ]
    print('Loaded mt backprop kernel')
    return func
__cuda_compute_mt_gradients = get_cuda_compute_mt_gradients()


BLOCK_SIZE = 128

ComputeVertices_gpu = mod.get_function("ComputeVertices")
GenerateFaces_gpu = mod.get_function("GenerateFaces")

#drv.init()
pycuda_ctx = drv.Device(0).retain_primary_context()

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

class TSDF2MeshFunction(autograd.Function):
    #TODO: Deprecated?
    @staticmethod
    def compute_vertex_normals(verts, faces):
        norm = verts_normals_list([verts], [faces])[0]
        return norm

    @staticmethod
    def forward(ctx, input, mt, input_grad=None):
        sdf = input.detach()
        vert_tet_weights, vert_tet_indices, faces, weight_gradients = mt.marching_tets(sdf)
        
        ctx.save_for_backward(sdf, vert_tet_indices, weight_gradients)
        ctx.mark_non_differentiable(vert_tet_indices, faces)
        return vert_tet_weights.to(mt.device), vert_tet_indices.to(mt.device), faces.to(mt.device)

    @staticmethod
    def backward(ctx, grad_weights, grad_indices, grad_faces):
        start = timer()
        grad_input = grad_weights.clone()

        #Input should be gradient wrt output
        input, vert_tet_indices, weight_gradients = ctx.saved_tensors
        '''
        grad_out = compute_mt_gradients(grad_input.view(-1),
                                        vert_tet_indices.view(-1),
                                        weight_gradients.view(-1),
                                        input.shape[0])
        '''
        device = 'cuda:0'
        n_verts = vert_tet_indices.shape[0]
        # Vectorized Jacobian computation
        num_tet_connections = 2
        idx = torch.unsqueeze(torch.arange(n_verts*2).to(device).reshape((-1,2)), -1)
        idx = torch.tile(idx, (1, num_tet_connections, 1)).reshape((-1,1)) # Now has shape (n_verts*4, 1)
        tet_v = torch.reshape(vert_tet_indices, (-1,))
        tet_v = torch.unsqueeze(torch.repeat_interleave(tet_v, 2), -1) # Now has shape (n_verts*4, 1)
        idx = torch.cat([idx, tet_v], -1)

        vals = weight_gradients.view(-1,)
        output = torch.sparse_coo_tensor(idx.t(), vals, (n_verts*2, input.shape[0]), dtype=torch.float32)

        grad_weights_flat = torch.unsqueeze(torch.flatten(grad_input), -1)
        grad_out = torch.squeeze(torch.sparse.mm(output.t(), grad_weights_flat))

        print('Backprop MT:', timer() - start)
        return grad_out, None, None


#TODO: When calling, pass the actual device?
def compute_mt_gradients(grad_loss_wrt_weight, vert_tet_indices, grad_weight_wrt_phi, num_phi, device='cuda:0'):

    # Derivative of loss w.r.t. screen space vertices.
    assert num_phi == 285213
    grad_loss_wrt_phi = torch.zeros(num_phi, device=device).type(torch.float32).flatten().detach()

    num_verts = vert_tet_indices.shape[0] // 2
    start = timer()
    __cuda_compute_mt_gradients(ctypes.c_void_p(grad_loss_wrt_weight.data_ptr()),
                             ctypes.c_void_p(vert_tet_indices.data_ptr()),
                             ctypes.c_void_p(grad_weight_wrt_phi.data_ptr()),
                             ctypes.c_void_p(grad_loss_wrt_phi.data_ptr()),
                             num_verts,
                             grad_loss_wrt_weight.shape[0],
                             vert_tet_indices.shape[0],
                             grad_weight_wrt_phi.shape[0],
                             num_phi)
                             #block=blockdim, grid=blocks_per_grid)

    return grad_loss_wrt_phi


class MarchingTetrahedra():
    #TODO: remove default device
    def __init__(self, grid=False, device='cuda:0'):
        super(self.__class__, self).__init__()
        print('Initializing MarchingTetrahedra')
        self.device = device
        # Initialize tet mesh
        filename = "star_shell.ply"
        if grid:
            filename = "grid_sphere.ply"

        #nodes, tet_faces, tets, edges_sp = osh.loadOuterShell(filename)
        root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tet_mesh_data')
        nodes = np.load(os.path.join(root_dir, 'nodes.npy'))
        tet_faces = np.load(os.path.join(root_dir, 'tet_faces.npy'))
        tets = np.load(os.path.join(root_dir, 'tets.npy'))
        edges_sp = sparse.load_npz(os.path.join(root_dir, 'edges_sp.npz'))
        '''
        np.save('nodes.npy', nodes)
        np.save('tet_faces.npy', tet_faces)
        np.save('tets.npy', tets)
        sparse.save_npz('src/edges_sp.npz', edges_sp) 
        '''
        self.nodes = torch.from_numpy(nodes.astype(np.float32)).to(self.device)
        self.tets = torch.from_numpy(tets.astype(np.int64)).to(self.device)
        # Initialize tet skinning weights
        if grid:
            tet_skinning_weights = np.zeros((len(self.nodes), 24))
            self.weights = torch.zeros((len(self.nodes), 24)).float().to(self.device)
        else:
            tet_skinning_weights = np.load("coarseweights.npy")
            self.weights = torch.from_numpy(tet_skinning_weights.astype(np.float32)).to(self.device)

        self.nb_verts = nodes.shape[0] 
        self.nb_tets = tets.shape[0]
        self.nb_edges = edges_sp.count_nonzero()
        
        self.edges_sp_nz = edges_sp.nonzero()
        
        # Only initialize arrays once
        vert_tet_weights_np = np.zeros((self.nb_edges, 2), dtype=np.float32)
        vert_tet_indices_np = np.zeros((self.nb_edges, 2), dtype=np.int32) # 2 edge vertices
        weight_gradients_np = np.zeros((self.nb_edges, 4), dtype=np.float32) # dweights / dphi
        faces_np = np.zeros((2*self.nb_tets, 3), dtype=np.int32)
        
        nodes = nodes.flatten().astype(np.float32)
        tet_skin_w = tet_skinning_weights.flatten().astype(np.float32)
        edges_sp_nz_a = self.edges_sp_nz[0].flatten().astype(np.int32)
        edges_sp_nz_b = self.edges_sp_nz[1].flatten().astype(np.int32)
        edges_sp_row_ptr = np.array(edges_sp.indptr).flatten().astype(np.int32)
        edges_sp_columns = np.array(edges_sp.indices).flatten().astype(np.int32)
        tetra = tets.flatten().astype(np.int32)
        
        # Flatten and transfer output data to GPU
        self.vert_tet_weights_cuda = torch.from_numpy(vert_tet_weights_np.flatten().astype(np.float32)).to(self.device).detach()
        self.vert_tet_indices_cuda = torch.from_numpy(vert_tet_indices_np.flatten().astype(np.int32)).to(self.device).detach()
        self.weight_gradients_cuda = torch.from_numpy(weight_gradients_np.flatten().astype(np.float32)).to(self.device).detach()
        self.faces_cuda = torch.from_numpy(faces_np.flatten().astype(np.int32)).to(self.device).detach()
        del vert_tet_weights_np
        del vert_tet_indices_np
        del weight_gradients_np
        del faces_np
        
        # Transfer input data to GPU
        # 1. ComputeVertices
        self.sdf_gpu = torch.zeros(self.nb_verts, dtype=torch.float32).to(self.device).detach()
        self.nodes_gpu = torch.from_numpy(nodes).to(self.device).detach()
        self.tet_skin_w_gpu = torch.from_numpy(tet_skin_w).to(self.device).detach()
        self.edges_sp_nz_a_gpu = torch.from_numpy(edges_sp_nz_a).to(self.device).detach()
        self.edges_sp_nz_b_gpu = torch.from_numpy(edges_sp_nz_b).to(self.device).detach()
        
        # 2. GenerateFaces
        self.edges_sp_row_ptr_gpu = torch.from_numpy(edges_sp_row_ptr).to(self.device).detach()
        self.edges_sp_columns_gpu = torch.from_numpy(edges_sp_columns).to(self.device).detach()
        self.tetra_gpu = torch.from_numpy(tetra).to(self.device).detach()
        
        del nodes, tet_skin_w, edges_sp_nz_a, edges_sp_nz_b, edges_sp_row_ptr, edges_sp_columns, tetra
        print('Done initializing...')

    def marching_tets(self, sdf):
        # 1. Compute vertices    
        #start = timer()
        self.sdf_gpu[:] = sdf[:]

        thread_size = self.nb_edges
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1
        #print("thread_size: {}".format(thread_size))
        #print("blocks_per_grid: {}".format(blocks_per_grid))
        
        params = np.array([self.edges_sp_nz[0].shape[0], 0.0]).astype(np.float32)

        ComputeVertices_gpu(Holder(self.vert_tet_weights_cuda),
                            Holder(self.vert_tet_indices_cuda),
                            Holder(self.weight_gradients_cuda),
                            Holder(self.sdf_gpu),
                            Holder(self.nodes_gpu),
                            Holder(self.tet_skin_w_gpu),
                            Holder(self.edges_sp_nz_a_gpu),
                            Holder(self.edges_sp_nz_b_gpu),
                            drv.In(params),
                            block=blockdim, grid=blocks_per_grid)
        #print("ComputeVertices GPU:", timer()-start)

        # 2. Create the triangular faces
        #start = timer()
        thread_size = self.nb_tets
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1
        #print("thread_size: {}".format(thread_size))
        #print("blocks_per_grid: {}".format(blocks_per_grid))

        params = np.array([self.nb_tets, 0.0]).astype(np.float32)

        GenerateFaces_gpu(Holder(self.faces_cuda),
                          Holder(self.sdf_gpu),
                          Holder(self.edges_sp_row_ptr_gpu),
                          Holder(self.edges_sp_columns_gpu),
                          Holder(self.tetra_gpu),
                          drv.In(params),
                          block=blockdim, grid=blocks_per_grid)
        #print("GenerateFaces GPU:", timer()-start)

        # Reshape all outputs
        vert_tet_weights_out = self.vert_tet_weights_cuda.reshape((self.nb_edges, 2))
        vert_tet_indices_out = self.vert_tet_indices_cuda.reshape((self.nb_edges, 2))
        weight_gradients_out = self.weight_gradients_cuda.reshape((self.nb_edges, 4))
        faces_out = self.faces_cuda.reshape((2*self.nb_tets, 3))

        # Remove 0 entries
        nnz = torch.nonzero(torch.sum(vert_tet_weights_out,1))[:,0]
        #print(nnz.shape)
        nnz_vert_tet_weights = vert_tet_weights_out[nnz,:]
        nnz_vert_tet_indices = vert_tet_indices_out[nnz,:]
        nnz_weight_gradients = weight_gradients_out[nnz,:]
        #print(nnz_verts.shape)

        indices_v = torch.zeros((vert_tet_weights_out.shape[0]), dtype=torch.int32, device=self.device)
        indices_nnz_v = torch.arange(nnz_vert_tet_weights.shape[0], dtype=torch.int32, device=self.device)
        indices_v[nnz] = indices_nnz_v
        #print(indices_v)
        
        nnz_f = torch.nonzero(torch.sum(faces_out,1))[:,0]
        #print(nnz_f)
        nnz_faces = faces_out[nnz_f,:]
        nnz_faces[:,:] = indices_v[nnz_faces[:,:].long()]
        #print(nnz_faces.shape)

        return nnz_vert_tet_weights, nnz_vert_tet_indices, nnz_faces, nnz_weight_gradients


if __name__ == "__main__":
    print('Testing marching tetrahedra')
    '''
    # Initialize tet mesh
    nodes, tet_faces, tets, edges_sp = osh.loadOuterShell("../star_shell.ply")

    # Initialize tet skinning weights
    tet_skinning_weights = np.load("../coarseweights.npy")
    '''
    # Load TSDF values
    tsdf_filename = "/data/jwu/D_march/TSDF/mesh_D_march_0000.bin"
    tsdf = utils.loadTSDF_bin(tsdf_filename)
    print('TSDF:', tsdf.shape)
    print(np.unique(tsdf))

    tsdf_gpu = torch.tensor(tsdf, dtype=torch.float32, device='cuda')

    mt = MarchingTetrahedra()
    batch_size = 24
    all_runtimes = []
    for i in range(20):
        start = timer()
        for j in range(batch_size):
            vert_tet_weights, vert_tet_indices, faces, weight_gradients = mt.marching_tets(tsdf_gpu)
        tet_verts = mt.nodes[vert_tet_indices.long()]
        verts = torch.matmul(vert_tet_weights.unsqueeze(-2), tet_verts)[..., 0, :]
        runtime = timer()-start
        all_runtimes.append(runtime)
        print('Runtime:', runtime)
    avg_runtime = np.mean(all_runtimes)
    print('Avg runtime:', avg_runtime)
