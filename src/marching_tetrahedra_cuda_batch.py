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
from mt_cuda_kernels_batch import mod
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
InflateSDF_gpu = mod.get_function("InflateSDF")

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
    @staticmethod
    def compute_vertex_normals(verts, faces):
        norm = verts_normals_list([verts], [faces])[0]
        return norm

    @staticmethod
    def forward(ctx, input, mt, isocontour=0.0):
        sdf = input.detach()
        vert_tet_weights, vert_tet_indices, faces, weight_gradients, vert_counts = mt.marching_tets(sdf, isocontour)
        
        ctx.save_for_backward(sdf, vert_tet_indices, weight_gradients, vert_counts)
        ctx.mark_non_differentiable(vert_tet_indices, vert_counts)

        return vert_tet_weights, vert_tet_indices, faces, vert_counts

    @staticmethod
    def backward(ctx, grad_weights, grad_indices, grad_faces, grad_vert_counts):
        #start = timer()
        grad_input = grad_weights.clone()

        #Input should be gradient wrt output
        input, vert_tet_indices, weight_gradients, vert_counts = ctx.saved_tensors
        '''
        all_grad_out = []
        prev_vert = 0
        for i in range(len(input)):
            count = vert_counts[i]
            grad_input_i = grad_weights[prev_vert:prev_vert+count].view(-1)
            vert_tet_indices_i = vert_tet_indices[prev_vert:prev_vert+count].view(-1)
            weight_grads_i = weight_gradients[prev_vert:prev_vert+count].view(-1)

            grad_out = compute_mt_gradients(grad_input_i,
                                            vert_tet_indices_i,
                                            weight_grads_i,
                                            input.shape[1])
            all_grad_out.append(grad_out)
            prev_vert = vert_counts[i]
        all_grad_out = torch.stack(all_grad_out)
        '''
        # Assumes batch size of 1
        assert len(input) == 1
        grad_out = compute_mt_gradients(grad_input.view(-1),
                                        vert_tet_indices.view(-1),
                                        weight_gradients.view(-1),
                                        input.shape[1])
        #print('Backprop MT:', timer()-start)
        return grad_out.unsqueeze(0), None, None


#TODO: When calling, pass the actual device?
def compute_mt_gradients(grad_loss_wrt_weight, vert_tet_indices, grad_weight_wrt_phi, num_phi, device='cuda:0'):
    # Derivative of loss w.r.t. screen space vertices.
    assert num_phi == 285213
    grad_loss_wrt_phi = torch.zeros(num_phi, device=device).type(torch.float32).flatten().detach()

    num_verts = vert_tet_indices.shape[0] // 2
    #start = timer()
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
    #print('Backprop MT kernel:', timer()-start)
    return grad_loss_wrt_phi

class MarchingTetrahedra():
    #TODO: remove default device
    def __init__(self, batch_size, grid=False, device='cuda:0'):
        super(self.__class__, self).__init__()
        print('Initializing MarchingTetrahedra')
        self.device = device
        self.batch_size = batch_size
        # Initialize tet mesh
        filename = "star_shell.ply"
        if grid:
            filename = "grid_sphere.ply"
        #nodes, tet_faces, tets, edges_sp = osh.loadOuterShell(filename)
        root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tet_mesh_data')
        nodes = np.load(os.path.join(root_dir, 'nodes.npy'))

        # Normalize tet mesh vertices
        min_xyz = np.min(nodes, axis=0, keepdims=True)
        max_xyz = np.max(nodes, axis=0, keepdims=True)
        translation = -(min_xyz + max_xyz) * 0.5
        #nodes = nodes + translation
        self.translation = torch.tensor(translation, dtype=torch.float32, device=device)

        scale_inv = np.max(max_xyz-min_xyz)
        scale = 1.0 / scale_inv * 0.9
        #nodes *= scale
        self.scale = torch.tensor(scale, dtype=torch.float32, device=device)

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

        nodes = nodes.flatten().astype(np.float32)
        tet_skin_w = tet_skinning_weights.flatten().astype(np.float32)
        edges_sp_nz_a = self.edges_sp_nz[0].flatten().astype(np.int32)
        edges_sp_nz_b = self.edges_sp_nz[1].flatten().astype(np.int32)
        edges_sp_row_ptr = np.array(edges_sp.indptr).flatten().astype(np.int32)
        edges_sp_columns = np.array(edges_sp.indices).flatten().astype(np.int32)
        tetra = tets.flatten().astype(np.int32)

        # Transfer input data to GPU
        # 1. ComputeVertices
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

    def marching_tets(self, sdf, isocontour):
        vert_tet_weights_cuda = torch.zeros(self.batch_size*self.nb_edges*2, dtype=torch.float32).to(self.device).detach()
        vert_tet_indices_cuda = torch.zeros(self.batch_size*self.nb_edges*2, dtype=torch.int32).to(self.device).detach()
        weight_gradients_cuda = torch.zeros(self.batch_size*self.nb_edges*4, dtype=torch.float32).to(self.device).detach()
        faces_cuda = torch.zeros(self.batch_size*2*self.nb_tets*3, dtype=torch.int32).to(self.device).detach()

        # 1. Compute vertices
        #start = timer()
        batch_size = sdf.shape[0]
        thread_size = self.nb_edges
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), batch_size, 1
        #print("thread_size: {}".format(thread_size))
        #print("blocks_per_grid: {}".format(blocks_per_grid))
        
        params = np.array([self.nb_edges, self.nb_verts, isocontour]).astype(np.float32)

        ComputeVertices_gpu(Holder(vert_tet_weights_cuda),
                            Holder(vert_tet_indices_cuda),
                            Holder(weight_gradients_cuda),
                            Holder(sdf.flatten()),
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
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), batch_size, 1
        #print("thread_size: {}".format(thread_size))
        #print("blocks_per_grid: {}".format(blocks_per_grid))

        params = np.array([self.nb_tets, self.nb_verts, isocontour]).astype(np.float32)

        GenerateFaces_gpu(Holder(faces_cuda),
                          Holder(sdf.flatten()),
                          Holder(self.edges_sp_row_ptr_gpu),
                          Holder(self.edges_sp_columns_gpu),
                          Holder(self.tetra_gpu),
                          drv.In(params),
                          block=blockdim, grid=blocks_per_grid)
        #print("GenerateFaces GPU:", timer()-start)

        # Reshape all outputs
        vert_tet_weights_out = vert_tet_weights_cuda.reshape((self.batch_size, self.nb_edges, 2))[:batch_size]
        vert_tet_indices_out = vert_tet_indices_cuda.reshape((self.batch_size, self.nb_edges, 2))[:batch_size]
        weight_gradients_out = weight_gradients_cuda.reshape((self.batch_size, self.nb_edges, 4))[:batch_size]
        faces_out = faces_cuda.reshape((self.batch_size, 2*self.nb_tets, 3))
        #print('CUDA runtime:', timer()-start)

        start = timer()
        # Remove 0 entries
        weights_sum = torch.sum(vert_tet_weights_out,-1)
        # Returns a tuple of length D, where D is the number of dimensions (not batch size!)
        nnz = torch.nonzero(weights_sum, as_tuple=True)
        # Need to unpack to tuple...
        #print(nnz.shape)
        nnz_vert_tet_weights = vert_tet_weights_out[nnz]
        nnz_vert_tet_indices = vert_tet_indices_out[nnz]
        nnz_weight_gradients = weight_gradients_out[nnz]
        #print(nnz_verts.shape)

        indices_v = torch.zeros(vert_tet_weights_out.shape[:2], dtype=torch.int32, device=self.device)
        nnz_counts = torch.stack([torch.sum((nnz[0] - i)==0) for i in range(batch_size)])
        indices_nnz_v = torch.cat([torch.arange(count, dtype=torch.int32, device=self.device) for count in nnz_counts])
        indices_v[nnz] = indices_nnz_v
        indices_v = indices_v.flatten()

        faces_sum = torch.sum(faces_out,-1)
        nnz_f = torch.nonzero(faces_sum, as_tuple=True)
        # This is an array of shape (N,3)
        nnz_faces = faces_out[nnz_f]
        # Need to unpack to tuple...
        nnz_faces[:,:] = indices_v[nnz_faces[:,:].long()]
        #print('Remove nonzero runtime:', timer()-start)

        start = timer()
        nnz_f_counts = [torch.sum((nnz_f[0] - i)==0) for i in range(self.batch_size)]

        #nnz_vert_tet_weights_stacked = []
        #nnz_vert_tet_indices_stacked = []
        #nnz_weight_gradients_stacked = []
        nnz_faces_stacked = []

        prev_vert = 0
        prev_face = 0
        for i in range(batch_size):
            #nnz_vert_tet_weights_stacked.append(nnz_vert_tet_weights[prev_vert:nnz_counts[i]])
            #nnz_vert_tet_indices_stacked.append(nnz_vert_tet_indices[prev_vert:nnz_counts[i]])
            #nnz_weight_gradients_stacked.append(nnz_weight_gradients[prev_vert:nnz_counts[i]])
            nnz_faces_stacked.append(nnz_faces[prev_face:prev_face+nnz_f_counts[i]])
            prev_vert = nnz_counts[i]
            prev_face = nnz_f_counts[i]
        #print('Stacking runtime:', timer()-start)
        return nnz_vert_tet_weights, nnz_vert_tet_indices, nnz_faces_stacked, nnz_weight_gradients, nnz_counts

    #TODO: This assumes a batch size of 1!
    def inflate_sdf(self, sdf):
        # 1. Compute vertices
        #start = timer()
        assert len(sdf.shape) == 2
        inflated_sdf = torch.zeros(self.nb_verts, dtype=torch.float32).to(self.device).detach()
        inflated_sdf[:] = sdf.flatten()

        batch_size = sdf.shape[0]
        thread_size = self.nb_edges
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), batch_size, 1
        #print("thread_size: {}".format(thread_size))
        #print("blocks_per_grid: {}".format(blocks_per_grid))

        params = np.array([self.nb_edges, self.nb_verts]).astype(np.float32)

        InflateSDF_gpu(Holder(sdf.flatten()),
                       Holder(inflated_sdf),
                       Holder(self.edges_sp_nz_a_gpu),
                       Holder(self.edges_sp_nz_b_gpu),
                       drv.In(params),
                       block=blockdim, grid=blocks_per_grid)
        #print("ComputeVertices GPU:", timer()-start)
        return inflated_sdf.unsqueeze(0).detach()


if __name__ == "__main__":
    print('Testing marching tetrahedra')
    # Load TSDF values
    batch_size = 1
    batch_tsdf = []
    for i in range(batch_size):
        tsdf_filename = "/data/jwu/D_march/TSDF/mesh_D_march_{:04d}.bin".format(i)
        tsdf = utils.loadTSDF_bin(tsdf_filename)
        batch_tsdf.append(tsdf)
        #print('TSDF:', tsdf.shape)
        #print(np.unique(tsdf))
    batch_tsdf = torch.tensor(batch_tsdf, dtype=torch.float32, device='cuda')

    mt = MarchingTetrahedra(batch_size)
    #batch_tsdf = torch.unsqueeze(tsdf_gpu, 0).repeat(batch_size, 1)
    print('Batch tsdf:', batch_tsdf.shape)
    avg_runtime = []
    for i in range(20):
        start = timer()
        vert_tet_weights, vert_tet_indices, faces, vert_counts = TSDF2MeshFunction.apply(batch_tsdf, mt)
        tet_verts = mt.nodes[vert_tet_indices.long()]
        verts = torch.matmul(vert_tet_weights.unsqueeze(-2), tet_verts)[..., 0, :]
        runtime = timer()-start
        avg_runtime.append(runtime)
        print('Runtime:', runtime)
    print('Avg runtime:', np.mean(avg_runtime))
    assert len(verts) == 81182
    assert len(faces[0]) == 162364
