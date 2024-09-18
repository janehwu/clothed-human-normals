#####################################################################
# Copyright 2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import ctypes

import torch
import torch.autograd as autograd
#import pycuda.autoinit
import pycuda.driver as drv

from raster_cuda_kernels import mod
from timeit import default_timer as timer 

# Load backprop kernel first
def get_cuda_compute_gradients():
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cuda')
    dll = ctypes.CDLL(os.path.join(root_dir, 'raster_backprop_kernel.so'), mode=ctypes.RTLD_GLOBAL)
    func = dll.Rasterize_Triangle_Mesh_Gradients
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
    print('Loaded raster backprop kernel')
    return func
__cuda_compute_gradients = get_cuda_compute_gradients()


RasterizeMesh_gpu = mod.get_function("Rasterize_Triangle_Mesh")
RasterizeMeshGrad_gpu = mod.get_function("Rasterize_Triangle_Mesh_Gradients")
PreRasterizeMeshGrad_gpu = mod.get_function("Pre_Rasterize_Triangle_Mesh_Gradients")
BinTriangles_gpu = mod.get_function("Bin_Triangles")
BLOCK_SIZE = 1 #128

drv.init()
pycuda_ctx = drv.Device(0).retain_primary_context()


def read_txt(fname):
    data = []
    with open(fname, 'r') as f:
        first_line = True
        for l in f.readlines():
            if first_line:
                size = int(l.split()[-1])
                first_line = False
            else:
                vals = l.split()
                vals = [float(v) for v in vals]
                data.append(vals)
    return np.asarray(data)

def divUp(x,y):
    if x % y == 0:
        return x/y
    else:
        return (x+y-1)/y

class Holder(drv.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()

class Rasterizer():
    def __init__(self, img_dim, device):
        self.img_dim = img_dim
        self.device = device
        self.max_triangles_per_bin = 20

    def rasterize_mesh(self, verts_screen, faces):
        # Define output tensors
        img_dim = self.img_dim
        dists = torch.from_numpy(np.zeros(img_dim*img_dim).astype(np.float32)).to(self.device).detach()
        alphas = torch.from_numpy(np.zeros(img_dim*img_dim*3).astype(np.float32)).to(self.device).detach()
        vert_indices = torch.from_numpy(np.zeros(img_dim*img_dim*3).astype(np.int32)).to(self.device).detach()
        bins_idx = torch.zeros(self.img_dim*self.img_dim, device=self.device).type(torch.int32).detach()
        bins2tri = torch.zeros(self.img_dim*self.img_dim*self.max_triangles_per_bin,
                               device=self.device).type(torch.int32).detach()-1
        bins2z = torch.zeros(self.img_dim*self.img_dim*self.max_triangles_per_bin,
                             device=self.device).type(torch.int32).detach()

        if verts_screen[0,2] < 0:
            bins2z[:] = -999
        else:
            bins2z[:] = -1

        num_faces = len(faces)

        # Parallelize over faces
        thread_size = num_faces
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([self.max_triangles_per_bin, self.img_dim, self.img_dim]).astype(np.int32)
        '''
        print('Raster start')
        print("torch.cuda.total_memory: %fGB"%(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        '''
        #start = timer()
        BinTriangles_gpu(Holder(verts_screen.flatten()),
                          Holder(faces.flatten()),
                          Holder(bins2tri),
                          Holder(bins2z),
                          Holder(bins_idx),
                          drv.In(params),
                          block=blockdim, grid=blocks_per_grid)
        #print('1. BINNING', timer()-start)
        #print('Bins:', np.unique(self.bins_idx.detach().cpu().numpy()))

        # Parallelize over pixels
        thread_size = self.img_dim*self.img_dim
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([self.max_triangles_per_bin, self.img_dim]).astype(np.int32)

        #start = timer()
        RasterizeMesh_gpu(Holder(verts_screen.flatten()),
                          Holder(faces.flatten()),
                          Holder(bins2tri),
                          Holder(bins2z),
                          Holder(bins_idx),
                          Holder(dists),
                          Holder(alphas),
                          Holder(vert_indices),
                          drv.In(params),
                          block=blockdim, grid=blocks_per_grid)
        #print('2. RASTER', timer()-start)

        alphas = alphas.reshape((self.img_dim, self.img_dim, 3)).permute((1,0,2))
        vert_indices = vert_indices.reshape((self.img_dim, self.img_dim, 3)).permute((1,0,2))

        del dists, bins_idx, bins2tri, bins2z
        return alphas, vert_indices

    def precompute_gradients(self, verts_screen, pix_vert_indices):
        # We want to compute d(alpha)/d(verts) -> each alpha depends on 2 vertices
        grad_alphas = torch.zeros(self.img_dim*self.img_dim*3*6, device='cuda').type(torch.float32).detach()
        grad_alphas_idx = torch.zeros(self.img_dim*self.img_dim*3*6, device='cuda').type(torch.int32).detach()

        # Parallelize over pixels
        img_dim = pix_vert_indices.shape[0]
        thread_size = img_dim*img_dim
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([img_dim]).astype(np.int32)

        # Derivative of loss w.r.t. screen space vertices.
        grad_verts_screen = torch.zeros(verts_screen.shape, device='cuda').type(torch.float32).flatten().detach()

        start = timer()
        # grad_alphas and pix_vert_indices need to be permuted back to (H, W, 3)
        PreRasterizeMeshGrad_gpu(Holder(verts_screen.flatten()),
                                 Holder(pix_vert_indices.permute((1,0,2)).flatten()),
                                 Holder(grad_alphas),
                                 Holder(grad_alphas_idx),
                                 drv.In(params),
                                 block=blockdim, grid=blocks_per_grid)
        #print('1. BACKPROP', timer()-start)
        return grad_alphas, grad_alphas_idx

    # TODO: Remove if unused
    @staticmethod
    def compute_gradients(verts_screen, pix_vert_indices, grad_alphas, device='cuda:0'):
        global pycuda_ctx
        pycuda_ctx.push()

        # Parallelize over pixels
        img_dim = pix_vert_indices.shape[0]
        thread_size = img_dim*img_dim
        blockdim = BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([img_dim]).astype(np.int32)
        
        # Derivative of loss w.r.t. screen space vertices.
        grad_verts_screen = torch.zeros(verts_screen.shape, device=device).type(torch.float32).flatten().detach()

        start = timer()
        # grad_alphas and pix_vert_indices need to be permuted back to (H, W, 3)
        RasterizeMeshGrad_gpu(Holder(verts_screen.flatten()),
                          Holder(pix_vert_indices.permute((1,0,2)).flatten()),
                          Holder(grad_alphas.permute((1,0,2)).flatten()),
                          Holder(grad_verts_screen),
                          drv.In(params),
                          block=blockdim, grid=blocks_per_grid)
        print('1. BACKPROP', timer()-start)

        pycuda_ctx.pop()  # very important
        return grad_verts_screen.reshape(verts_screen.shape)


def compute_gradients(verts_screen, pix_vert_indices, grad_alphas, device='cuda:0'):
    # Derivative of loss w.r.t. screen space vertices.
    grad_verts_screen = torch.zeros(verts_screen.shape, device=device).type(torch.float32).flatten().detach()

    img_dim = pix_vert_indices.shape[0]

    # grad_alphas and pix_vert_indices need to be permuted back to (H, W, 3)
    __cuda_compute_gradients(ctypes.c_void_p(verts_screen.flatten().data_ptr()),
                             ctypes.c_void_p(pix_vert_indices.permute((1,0,2)).flatten().data_ptr()),
                             ctypes.c_void_p(grad_alphas.permute((1,0,2)).flatten().data_ptr()),
                             ctypes.c_void_p(grad_verts_screen.data_ptr()),
                             img_dim,
                             grad_verts_screen.shape[0],
                             pix_vert_indices.flatten().shape[0],
                             pix_vert_indices.flatten().shape[0],
                             grad_verts_screen.shape[0])
                             #block=blockdim, grid=blocks_per_grid)
    return grad_verts_screen.reshape(verts_screen.shape)


class RasterizerFunction(autograd.Function):
    @staticmethod
    def forward(ctx, verts_screen, faces, rasterizer):
        alphas, pix_vert_indices = rasterizer.rasterize_mesh(verts_screen.detach(),
                                                             faces.detach())
        grad_alphas, grad_alphas_idx = rasterizer.precompute_gradients(verts_screen.detach(),
                                                                       pix_vert_indices)
        ctx.save_for_backward(verts_screen, grad_alphas, grad_alphas_idx, pix_vert_indices)
        ctx.mark_non_differentiable(pix_vert_indices)

        # Add extra dimension to alphas for matmul later.
        return alphas.to(rasterizer.device), pix_vert_indices.to(rasterizer.device)

    @staticmethod
    def backward(ctx, grad_alphas, grad_vert_indices):
        #start = timer()
        grad_input = grad_alphas.clone()

        # METHOD 1
        verts_screen, alpha_gradients, grad_alphas_idx, pix_vert_indices = ctx.saved_tensors

        #grad_verts_screen = Rasterizer.compute_gradients(verts_screen.detach(), pix_vert_indices.detach(),
        #                                                 grad_input.detach())
        grad_verts_screen = compute_gradients(verts_screen.detach(),
                                              pix_vert_indices.detach(),
                                              grad_input.detach())

        #print('METHOD 1:', grad_verts_screen_gt)
        '''
        # METHOD 2
        verts_screen, alpha_gradients, grad_alphas_idx = ctx.saved_tensors

        n_verts = verts_screen.shape[0]
        # Vectorized Jacobian computation
        # The Jacobian is (H x W x 3, N x 3)
        # Where there are 6 connections per alpha value
        img_dim = int(np.sqrt(len(grad_alphas_idx) // 18))
        num_tet_connections = 6
        idx = torch.unsqueeze(torch.arange(img_dim*img_dim*3).to(grad_alphas.device).reshape((-1,3)), -1)
        #print(idx)
        idx = torch.tile(idx, (1, 1, num_tet_connections)).reshape((-1,1)) # Now has shape (n_verts*18, 1)
        #print(idx)
        v = torch.unsqueeze(grad_alphas_idx, -1) # Now has shape (n_verts*18, 1)
        idx = torch.cat([idx, v], -1)

        vals = alpha_gradients
        output = torch.sparse_coo_tensor(idx.t(), vals, (img_dim*img_dim*3, n_verts*3), dtype=torch.float32)

        grad_weights_flat = torch.unsqueeze(grad_input.permute((1,0,2)).flatten(), -1)
        grad_verts_screen = torch.squeeze(torch.sparse.mm(output.t(), grad_weights_flat))
        #print('METHOD 2:', grad_verts_screen.view(-1,3))
        '''
        #print('Backprop raster:', timer() - start)
        return grad_verts_screen.view(-1, 3), None, None


def test_renderpeople():
    import matplotlib.pyplot as plt
    #idx = 0
    r = Rasterizer(img_dim=512)
    all_times = []
    for idx in range(1):
        verts_world= read_txt("/data/jwu/raster_data/verts_camera_%d.txt" % idx).astype(np.float32)
        verts_screen = read_txt("/data/jwu/raster_data/verts_screen_%d.txt" % idx).astype(np.float32)
        faces = read_txt("/data/jwu/raster_data/faces.txt").astype(np.int32)
        vert_norms = read_txt("/data/jwu/raster_data/verts_norms.txt").astype(np.float32)

        verts_screen = torch.from_numpy(verts_screen).to(device).detach()
        verts_world = torch.from_numpy(verts_world).to(device).detach()
        vert_norms = torch.from_numpy(vert_norms).to(device).detach()
        faces = torch.from_numpy(faces).to(device).detach()

        start = timer()
        bary, indices = RasterizerFunction.apply(verts_screen, faces, r)
        pixel_vert_norms = vert_norms[indices.detach().cpu().numpy()]
        raster_out = torch.matmul(bary.unsqueeze(-2), pixel_vert_norms)[..., 0, :]
        print('RASTER OUT:', raster_out.shape)

        raster_time = timer() - start
        print('Rasterization time:', raster_time)
        all_times.append(raster_time)
    print('Avg raster time:', np.mean(all_times))

    # Visualization stuff
    raster_out = raster_out / torch.unsqueeze(torch.norm(raster_out+1e-8, p=2, dim=-1), -1)

    normal_map = raster_out.cpu().numpy()
    print('Values:', np.unique(normal_map))
    valid_mask = normal_map != 0
    pixel_map = normal_map * 0.5 + 0.5
    pixel_map = np.clip(pixel_map, 0., 1.)
    pixel_map *= 255 * valid_mask
    print('Pixel map:', pixel_map.shape)
    pixel_map = pixel_map.astype(np.uint8)
    plt.imshow(pixel_map)
    plt.show()


class Context():
    def __init__(self):
        self.saved_tensors = []

def test_backprop():
    device = 'cuda:0'
    r = Rasterizer(img_dim=512, device=device)

    for idx in range(1):
        verts_world= read_txt("/data/jwu/raster_data/verts_camera_%d.txt" % idx).astype(np.float32)
        verts_screen = read_txt("/data/jwu/raster_data/verts_screen_%d.txt" % idx).astype(np.float32)
        faces = read_txt("/data/jwu/raster_data/faces.txt").astype(np.int32)
        vert_norms = read_txt("/data/jwu/raster_data/verts_norms.txt").astype(np.float32)

        verts_screen = torch.from_numpy(verts_screen).to(device).detach()
        verts_world = torch.from_numpy(verts_world).to(device).detach()
        vert_norms = torch.from_numpy(vert_norms).to(device).detach()
        faces = torch.from_numpy(faces).to(device).detach()

        start = timer()
        bary, indices = RasterizerFunction.apply(verts_screen, faces, r)
        pixel_vert_norms = vert_norms[indices.detach().cpu().numpy()]
        raster_out = torch.matmul(bary.unsqueeze(-2), pixel_vert_norms)[..., 0, :]

        raster_time = timer() - start
        print('Rasterization time:', raster_time)

    # Now test backprop
    print('Testing backprop...')
    ctx = Context()
   
    # verts_screen, vert_norms, alphas, pixel_to_face_vertex_idx, faces = ctx.saved_tensors
    #grad_alphas = torch.ones(bary.shape).cuda().type(torch.float32)
    #print('Alphas:', grad_alphas.shape)
    grad_alphas, grad_alphas_idx = r.precompute_gradients(verts_screen, indices)
    print(np.unique(grad_alphas.cpu().detach().numpy()))
    print(np.unique(grad_alphas_idx.cpu().detach().numpy()))

def test_inference():
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('../')
    from obj_io import read_obj

    device = 'cuda:0'
    r = Rasterizer(img_dim=256, device=device)

    root = "/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/build"

    #obj = read_obj(os.path.join(root, "skinned_mesh.obj"))
    #faces = obj.f.astype(np.int32)
    #vert_norms = obj.vn.astype(np.float32)

    faces = read_txt(os.path.join(root, "faces.txt")).astype(np.int32)
    vert_norms = read_txt(os.path.join(root, "verts_norms.txt")).astype(np.float32)

    #np.savetxt('faces.txt', faces, header='%d' % len(faces))
    #np.savetxt('verts_norms.txt', vert_norms, header='%d' % num_verts)

    verts_world= read_txt(os.path.join(root, "verts_camera.txt")).astype(np.float32)
    verts_screen = read_txt(os.path.join(root, "verts_screen.txt")).astype(np.float32)

    verts_screen = torch.from_numpy(verts_screen).to(device).detach()
    verts_world = torch.from_numpy(verts_world).to(device).detach()
    vert_norms = torch.from_numpy(vert_norms).to(device).detach()
    faces = torch.from_numpy(faces).to(device).detach()

    print('Inputs:', verts_world.shape, verts_screen.shape, vert_norms.shape, faces.shape)

    for i in range(50):
        start = timer()
        bary, indices = RasterizerFunction.apply(verts_screen, faces, r)
        pixel_vert_norms = vert_norms[indices.detach().cpu().numpy()]
        raster_out = torch.matmul(bary.unsqueeze(-2), pixel_vert_norms)[..., 0, :]

        raster_time = timer() - start
        print('Rasterization time:', raster_time)

    # Visualization stuff
    raster_out = raster_out / torch.unsqueeze(torch.norm(raster_out+1e-8, p=2, dim=-1), -1)

    normal_map = raster_out.cpu().numpy()
    print('Values:', np.unique(normal_map))
    valid_mask = normal_map != 0
    pixel_map = normal_map * 0.5 + 0.5
    pixel_map = np.clip(pixel_map, 0., 1.)
    pixel_map *= 255 * valid_mask
    print('Pixel map:', pixel_map.shape)
    pixel_map = pixel_map.astype(np.uint8)
    plt.imshow(pixel_map)
    plt.show()


if __name__=='__main__':
    #test_inference()
    #test_renderpeople()
    test_backprop()
