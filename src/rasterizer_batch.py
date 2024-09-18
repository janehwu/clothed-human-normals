#####################################################################
# Copyright 2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import ctypes

import torch
import torch.autograd as autograd
import pycuda.autoinit
import pycuda.driver as drv

from raster_cuda_kernels_batch import mod
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
    def __init__(self, img_dim, batch_size, device):
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.device = device

        #TODO: change this back on A100!
        self.max_triangles_per_bin = 6#20

        self.dists = torch.zeros(self.batch_size*img_dim*img_dim, dtype=torch.float32).to(self.device).detach()
        self.alphas = torch.zeros(self.batch_size*img_dim*img_dim*3, dtype=torch.float32).to(self.device).detach()
        self.vert_indices = torch.zeros(self.batch_size*img_dim*img_dim*3, dtype=torch.int32).to(self.device).detach()
        self.bins_idx = torch.zeros(self.batch_size*self.img_dim*self.img_dim, device=self.device).type(torch.int32).detach()
        self.bins2tri = torch.zeros(self.batch_size*self.img_dim*self.img_dim*self.max_triangles_per_bin,
                                    device=self.device).type(torch.int32).detach()-1
        self.bins2z = torch.zeros(self.batch_size*self.img_dim*self.img_dim*self.max_triangles_per_bin,
                                  device=self.device).type(torch.int32).detach()

        # TODO: remove if precompute_gradients() not called!
        # We want to compute d(alpha)/d(verts) -> each alpha depends on 2 vertices
        self.grad_alphas = torch.zeros(batch_size*img_dim*img_dim*3*6, dtype=torch.float32).cuda().detach()
        self.grad_alphas_idx = torch.zeros(batch_size*img_dim*img_dim*3*6, device='cuda').type(torch.int32).detach()

    def rasterize_mesh(self, verts_screen_packed, faces_packed, faces_padded, verts_packed_first_idx, faces_packed_first_idx):
        '''
        In order to batch rasterize triangle meshes with different # verts and faces, each batch is flattened.

        Args:
            verts_screen_packed: [N*3] where N is the sum of all batch vertices
            faces_packed: [F*3] where F is the sum of all batch faces
            verts_packed_first_idx: [B] indicating what index the first vertex is located in for each batch item.
            faces_packed_first_idx: [B] indicating what index the first face is located in for each batch item.
        '''
        # Reset output tensors
        
        self.dists[:] = 0
        self.alphas[:] = 0
        self.vert_indices[:] = 0
        self.bins_idx[:] = 0
        self.bins2tri[:] = -1

        # The last batch can be smaller than self.batch_size
        batch_size = len(verts_packed_first_idx)-1

        #TODO: Fix this? Why is this needed?
        for i in range(batch_size):
            batch_elements = self.img_dim*self.img_dim*self.max_triangles_per_bin
            if verts_screen_packed[verts_packed_first_idx[i]+2] < 0:
                self.bins2z[i*batch_elements:(i+1)*batch_elements] = -99999
            else:
                self.bins2z[i*batch_elements:(i+1)*batch_elements] = -1

        num_faces = faces_padded.shape[1]
        # Parallelize over faces
        thread_size = num_faces
        blockdim = batch_size*BLOCK_SIZE, 1, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([self.max_triangles_per_bin, self.img_dim, self.img_dim, num_faces]).astype(np.int32)
        '''
        print('Raster start')
        print("torch.cuda.total_memory: %fGB"%(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        '''
        #start = timer()
        BinTriangles_gpu(Holder(verts_screen_packed.clone()),
                         Holder(faces_packed.clone()),
                         Holder(verts_packed_first_idx),
                         Holder(faces_packed_first_idx),
                         Holder(self.bins2tri),
                         Holder(self.bins2z),
                         Holder(self.bins_idx),
                         drv.In(params),
                         block=blockdim, grid=blocks_per_grid)
        #print('1. BINNING', timer()-start)
        #print('Bins:', np.unique(self.bins_idx.detach().cpu().numpy()))

        # Parallelize over pixels
        thread_size = self.img_dim*self.img_dim
        blockdim = BLOCK_SIZE, batch_size, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([self.max_triangles_per_bin, self.img_dim]).astype(np.int32)

        #start = timer()
        RasterizeMesh_gpu(Holder(verts_screen_packed.clone()),
                          Holder(faces_packed.clone()),
                          Holder(verts_packed_first_idx),
                          Holder(faces_packed_first_idx),
                          Holder(self.bins2tri),
                          Holder(self.bins2z),
                          Holder(self.bins_idx),
                          Holder(self.dists),
                          Holder(self.alphas),
                          Holder(self.vert_indices),
                          drv.In(params),
                          block=blockdim, grid=blocks_per_grid)
        #print('2. RASTER', timer()-start)

        alphas = self.alphas.reshape((self.batch_size, self.img_dim, self.img_dim, 3)).permute((0,2,1,3))
        vert_indices = self.vert_indices.reshape((self.batch_size, self.img_dim, self.img_dim, 3)).permute((0,2,1,3))
        return alphas[:batch_size], vert_indices[:batch_size]

    def precompute_gradients(self, verts_screen_packed, verts_packed_first_idx, pix_vert_indices, device):
        self.grad_alphas[:] = 0
        self.grad_alphas_idx[:] = 0
        # Parallelize over pixels

        assert len(pix_vert_indices.shape) == 4
        batch_size = pix_vert_indices.shape[0]
        img_dim = pix_vert_indices.shape[1]
        thread_size = img_dim*img_dim
        blockdim = BLOCK_SIZE, batch_size, 1
        threads_per_block = BLOCK_SIZE
        blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1

        params = np.array([img_dim]).astype(np.int32)

        # Derivative of loss w.r.t. screen space vertices.
        grad_verts_screen = torch.zeros(verts_screen_packed.shape, device=device).type(torch.float32).detach()

        #start = timer()

        # pix_vert_indices need to be permuted back to (H, W, 3)
        PreRasterizeMeshGrad_gpu(Holder(verts_screen_packed),
                                 Holder(verts_packed_first_idx),
                                 Holder(pix_vert_indices.permute((0,2,1,3)).flatten()),
                                 Holder(self.grad_alphas),
                                 Holder(self.grad_alphas_idx),
                                 drv.In(params),
                                 block=blockdim, grid=blocks_per_grid)
        #print('1. BACKPROP', timer()-start)

        # Reshape by batch
        grad_alphas = self.grad_alphas.reshape((batch_size,-1))
        grad_alphas_idx = self.grad_alphas_idx.reshape((batch_size,-1))
        return grad_alphas, grad_alphas_idx


def compute_gradients(verts_screen, pix_vert_indices, grad_alphas, device='cuda:0'):
    '''
    # Parallelize over pixels
    img_dim = pix_vert_indices.shape[0]
    thread_size = img_dim*img_dim
    blockdim = BLOCK_SIZE, 1, 1
    threads_per_block = BLOCK_SIZE
    blocks_per_grid = int(divUp(thread_size, threads_per_block)), 1, 1
    '''

    # Derivative of loss w.r.t. screen space vertices.
    grad_verts_screen = torch.zeros(verts_screen.shape, device=device).type(torch.float32).flatten().detach()

    img_dim = pix_vert_indices.shape[0]
    '''
    params = torch.tensor(
                 np.array([img_dim,
                       grad_verts_screen.shape[0],
                       pix_vert_indices.flatten().shape[0],
                       pix_vert_indices.flatten().shape[0],
                       grad_verts_screen.shape[0]])).type(torch.int32).to(device)
    print('Params:', params)
    '''
    start = timer()
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

    #print('1. BACKPROP', timer()-start)
    return grad_verts_screen.reshape(verts_screen.shape)


class RasterizerFunction(autograd.Function):
    @staticmethod
    def forward(ctx, verts_screen_packed, faces_packed, faces_padded,
                verts_packed_first_idx, faces_packed_first_idx, rasterizer):
        alphas, pix_vert_indices = rasterizer.rasterize_mesh(verts_screen_packed,
                                                             faces_packed,
                                                             faces_padded,
                                                             verts_packed_first_idx,
                                                             faces_packed_first_idx)
        '''
        grad_alphas, grad_alphas_idx = rasterizer.precompute_gradients(verts_screen_packed,
                                                                       verts_packed_first_idx,
                                                                       pix_vert_indices,
                                                                       rasterizer.device)
        '''
        # Saving for backward pass:
        # verts_screen_packed: flattened vertices
        # verts_packed_first_idx: first vertex index for each mesh
        # grad_alphas: for each normal map
        # grad_alphas_idx: for each normal map
        ctx.save_for_backward(verts_screen_packed, verts_packed_first_idx,
                              pix_vert_indices)
        ctx.mark_non_differentiable(pix_vert_indices)

        # Add extra dimension to alphas for matmul later.
        return alphas.to(rasterizer.device), pix_vert_indices.to(rasterizer.device)

    @staticmethod
    def backward(ctx, grad_alphas, grad_vert_indices):
        grad_input = grad_alphas.clone()

        verts_screen_packed, verts_packed_first_idx, pix_vert_indices = ctx.saved_tensors

        grad_verts_screen = compute_gradients(verts_screen_packed.detach(), pix_vert_indices[0].detach(),
                                              grad_input[0].detach())
        return grad_verts_screen, None, None, None, None, None


def test_renderpeople():
    import matplotlib.pyplot as plt
    #idx = 0
    batch_size = 1
    r = Rasterizer(img_dim=512, batch_size=batch_size, device='cuda:0')
    all_times = []
    for idx in range(20):
        verts_world= read_txt("/data/jwu/raster_data/verts_camera_%d.txt" % idx).astype(np.float32)
        verts_screen = read_txt("/data/jwu/raster_data/verts_screen_%d.txt" % idx).astype(np.float32)
        faces = read_txt("/data/jwu/raster_data/faces.txt").astype(np.int32)
        vert_norms = read_txt("/data/jwu/raster_data/verts_norms.txt").astype(np.float32)

        verts_screen = torch.from_numpy(verts_screen).to(r.device).detach()
        verts_world = torch.from_numpy(verts_world).to(r.device).detach()
        vert_norms = torch.from_numpy(vert_norms).to(r.device).detach()
        faces = torch.from_numpy(faces).to(r.device).detach()

        verts_screen = verts_screen.unsqueeze(0).repeat(batch_size,1,1)
        faces = faces.unsqueeze(0).repeat(batch_size,1,1)

        #TODO: actually do padding!
        faces_padded = faces

        #TODO: Pack elsewhere?
        verts_screen_packed = verts_screen.flatten().detach()
        faces_packed = faces.flatten().detach()

        verts_packed_first_idx = []
        faces_packed_first_idx = []
        for i in range(len(verts_screen)+1):
            if i == 0:
                verts_packed_first_idx.append(0)
                faces_packed_first_idx.append(0)
            elif i == len(verts_screen):
                verts_packed_first_idx.append(len(verts_screen_packed)//3)
                faces_packed_first_idx.append(len(faces_packed)//3)
            else:  # Otherwise, add length of previous item
                verts_packed_first_idx.append(verts_packed_first_idx[-1] + len(verts_screen[i-1])*3)
                faces_packed_first_idx.append(faces_packed_first_idx[-1] + len(faces[i-1])*3)

        verts_packed_first_idx = torch.tensor(verts_packed_first_idx, dtype=torch.int32).to(r.device).detach()
        faces_packed_first_idx = torch.tensor(faces_packed_first_idx, dtype=torch.int32).to(r.device).detach()

        start = timer()
        bary, indices = RasterizerFunction.apply(verts_screen_packed, faces_packed, faces_padded,
                                                 verts_packed_first_idx, faces_packed_first_idx, r)
        raster_time = timer() - start
        print('Rasterization time:', raster_time)
        vert_norms_batch = vert_norms.unsqueeze(0).repeat(batch_size,1,1)
        indices_np = indices.detach().cpu().numpy()
      
        for i in range(batch_size):
            pixel_vert_norms = vert_norms[indices_np[i]]
            raster_out = torch.matmul(bary[i].unsqueeze(-2), pixel_vert_norms)[..., 0, :]
        #print('RASTER OUT:', raster_out.shape)

        all_times.append(raster_time)
    print('Total time:', np.sum(all_times))
    #print('Avg raster time:', np.mean(all_times))

    # Visualization stuff
    raster_out = raster_out / torch.unsqueeze(torch.norm(raster_out+1e-8, p=2, dim=-1), -1)
    '''
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
    '''
    return np.sum(all_times)


def test_renderpeople_batch():
    import matplotlib.pyplot as plt
    #idx = 0
    batch_size = 20
    r = Rasterizer(img_dim=512, batch_size=batch_size, device='cuda:0')

    all_verts = []
    all_faces = []
    all_vert_norms = []
    for idx in range(batch_size):
        verts_screen = read_txt("/data/jwu/raster_data/verts_screen_%d.txt" % idx).astype(np.float32)
        faces = read_txt("/data/jwu/raster_data/faces.txt").astype(np.int32)
        vert_norms = read_txt("/data/jwu/raster_data/verts_norms.txt").astype(np.float32)

        verts_screen = torch.from_numpy(verts_screen).to(r.device).detach()
        vert_norms = torch.from_numpy(vert_norms).to(r.device).detach()
        faces = torch.from_numpy(faces).to(r.device).detach()

        all_verts.append(verts_screen.flatten())
        all_faces.append(faces)
        all_vert_norms.append(vert_norms)

    start = timer()
    faces_padded = torch.nn.utils.rnn.pad_sequence(all_faces, batch_first=True)
    verts_screen_packed = torch.cat(all_verts, axis=0).detach()
    faces_packed = torch.cat([f.flatten() for f in all_faces], axis=0).detach()

    verts_packed_first_idx = []
    faces_packed_first_idx = []
    for i in range(len(all_verts)+1):
        if i == 0:
            verts_packed_first_idx.append(0)
            faces_packed_first_idx.append(0)
        elif i == len(verts_screen):
            verts_packed_first_idx.append(len(verts_screen_packed)//3)
            faces_packed_first_idx.append(len(faces_packed)//3)
        else:  # Otherwise, add length of previous item
            verts_packed_first_idx.append(verts_packed_first_idx[-1] + len(all_verts[i-1]))
            faces_packed_first_idx.append(faces_packed_first_idx[-1] + len(all_faces[i-1]))

    verts_packed_first_idx = torch.tensor(verts_packed_first_idx, dtype=torch.int32).to(r.device).detach()
    faces_packed_first_idx = torch.tensor(faces_packed_first_idx, dtype=torch.int32).to(r.device).detach()
    print('Preprocessing mesh:', timer()-start)

    start = timer()
    bary, indices = RasterizerFunction.apply(verts_screen_packed, faces_packed, faces_padded,
                                             verts_packed_first_idx, faces_packed_first_idx, r)
    raster_time = timer() - start
    print('Rasterization time:', raster_time)
    vert_norms_batch = vert_norms.unsqueeze(0).repeat(batch_size,1,1)
    indices_np = indices.detach().cpu().numpy()
  
    for i in range(batch_size):
        pixel_vert_norms = all_vert_norms[i][indices_np[i]]
        raster_out = torch.matmul(bary[i].unsqueeze(-2), pixel_vert_norms)[..., 0, :]
    #print('RASTER OUT:', raster_out.shape)
    # Visualization stuff
    raster_out = raster_out / torch.unsqueeze(torch.norm(raster_out+1e-8, p=2, dim=-1), -1)

    '''
    normal_map = raster_out.cpu().numpy()
    print('Values:', np.unique(normal_map))
    valid_mask = normal_map != 0
    pixel_map = normal_map * 0.5 + 0.5
    pixel_map = np.clip(pixel_map, 0., 1.)
    pixel_map *= 255 * valid_mask
    print('Pixel map:', pixel_map.shape)
    pixel_map = pixel_map.astype(np.uint8)
    #plt.imshow(pixel_map)
    #plt.show()
    '''
    return raster_time


class Context():
    def __init__(self):
        self.saved_tensors = []

def test_backprop():
    device = 'cuda:0'
    batch_size = 1
    r = Rasterizer(img_dim=512, batch_size=batch_size, device=device)

    for idx in range(1):
        #verts_world= read_txt("/data/jwu/raster_data/verts_camera_%d.txt" % idx).astype(np.float32)
        verts_screen = read_txt("/data/jwu/raster_data/verts_screen_%d.txt" % idx).astype(np.float32)
        faces = read_txt("/data/jwu/raster_data/faces.txt").astype(np.int32)
        vert_norms = read_txt("/data/jwu/raster_data/verts_norms.txt").astype(np.float32)

        verts_screen = torch.from_numpy(verts_screen).to(device).detach()
        #verts_world = torch.from_numpy(verts_world).to(device).detach()
        vert_norms = torch.from_numpy(vert_norms).to(device).detach()
        faces = torch.from_numpy(faces).to(device).detach()

        start = timer()
        '''
        bary, indices = RasterizerFunction.apply(verts_screen, faces, r)
        pixel_vert_norms = vert_norms[indices.detach().cpu().numpy()]
        raster_out = torch.matmul(bary.unsqueeze(-2), pixel_vert_norms)[..., 0, :]
        '''
        bary, indices = RasterizerFunction.apply(verts_screen.unsqueeze(0).repeat(batch_size,1,1),
                                                 faces.unsqueeze(0).repeat(batch_size,1,1), r)
        vert_norms_batch = vert_norms.unsqueeze(0).repeat(batch_size,1,1)
        indices_np = indices.detach().cpu().numpy()

        for i in range(batch_size):
            pixel_vert_norms = vert_norms[indices_np[i]]
            raster_out = torch.matmul(bary[i].unsqueeze(-2), pixel_vert_norms)[..., 0, :]

        raster_time = timer() - start
        print('Rasterization time:', raster_time)

    # Now test backprop
    print('Testing backprop...')
    ctx = Context()
   
    # verts_screen, vert_norms, alphas, pixel_to_face_vertex_idx, faces = ctx.saved_tensors
    #grad_alphas = torch.ones(bary.shape).cuda().type(torch.float32)
    #print('Alphas:', grad_alphas.shape)
    grad_alphas, grad_alphas_idx = r.precompute_gradients(verts_screen, indices, device)
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
    all_times = []
    for i in range(20):
        all_times.append(test_renderpeople_batch())
    print('Total time:', np.sum(all_times))
    print('Avg raster time:', np.mean(all_times))

    #test_backprop()
