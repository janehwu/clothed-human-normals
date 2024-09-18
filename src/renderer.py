######################################################################
# Copyright 2021-2023. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh
from timeit import default_timer as timer 

from src.marching_tetrahedra_cuda_batch import MarchingTetrahedra, TSDF2MeshFunction
from src.mesh_normals import verts_normals_list
from src.pytorch3d_cameras import FoVPerspectiveCameras
from src.rasterizer import Rasterizer, RasterizerFunction
import src.outershell as osh
from src.skinning import Skinning

from obj_io import *
#from pytorch3d.renderer.cameras import FoVPerspectiveCameras as Pytorch3D_FoVCameras

# optimizing rigid transformation
from src.so3 import so3_exp_map


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    #rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


class Renderer(nn.Module):

    def __init__(self, device, batch_size, img_dim):
        super(self.__class__, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.mt = MarchingTetrahedra(batch_size)
        self.skinning = Skinning(device)
        self.rasterizer = Rasterizer(img_dim, device)

    def forward(self, sdf, pose, camera_params, epoch, step,
                rundir=None, save_obj=False, verbose=False, isocontour=0.0, return_mesh=False):
        if verbose:
            start = timer()
        all_verts = []
        all_weights = []
        all_lengths = []
        all_poses = []
        vert_tet_weights, vert_tet_indices, all_faces, vert_counts = TSDF2MeshFunction.apply(sdf, self.mt, isocontour)

        prev_vert = 0
        for i in range(sdf.shape[0]):
            if type(pose) is dict:
                body_pose = pose['pose']
                body_scale = pose['scale']
                body_translation = pose['translation']
            else:
                body_pose = pose
                body_scale = self.mt.scale
                body_translation = self.mt.translation

            count = vert_counts[i]
            tet_verts = self.mt.nodes[vert_tet_indices[prev_vert:prev_vert+count].long()]
            tet_weights = self.mt.weights[vert_tet_indices[prev_vert:prev_vert+count].long()]
            verts_t = torch.matmul(vert_tet_weights[prev_vert:prev_vert+count].unsqueeze(-2), tet_verts)[..., 0, :]
            # Transform vertices before skinning!
            verts_t = verts_t * body_scale + body_translation
            weights_t = torch.matmul(vert_tet_weights[prev_vert:prev_vert+count].unsqueeze(-2), tet_weights)[..., 0, :]

            all_verts.append(verts_t)
            all_poses.append((body_pose, body_scale, body_translation))
            all_weights.append(weights_t)
            all_lengths.append(verts_t.shape[0])
            prev_vert = count

        all_verts = torch.nn.utils.rnn.pad_sequence(all_verts, batch_first=True)
        all_weights = torch.nn.utils.rnn.pad_sequence(all_weights, batch_first=True)
        if verbose:
            print('MT:', timer() - start)
            start = timer()
        # Pose can be either csv (use default translation/scaling) or pkl
        # TODO: Handle larger batch size?
        body_pose, body_scale, body_translation = all_poses[0]
        skinned_verts = self.skinning(all_verts, body_pose, all_weights, body_scale, body_translation)
        skinned_verts_unpad = [skinned_verts[i, :all_lengths[i]] for i in range(len(skinned_verts))]
        if verbose:
            print('Skinning:', timer() - start)
            start = timer()
        if save_obj and not verbose:
            obj = Obj(v=skinned_verts_unpad[0].detach().cpu().numpy(), f=all_faces[0].detach().cpu().numpy())
            write_obj(obj, os.path.join(rundir, 'skinned_body_%d_%d.obj' % (epoch, step)))
            #obj = Obj(v=all_verts[0].detach().cpu().numpy(), f=all_faces[0].detach().cpu().numpy())
            #write_obj(obj, os.path.join(rundir, 'unskinned_body_%d_%d.obj' % (epoch, step)))
        #vibe_to_world = batch_rodrigues(pose[:,:3])
        imgs, masks, raster_vert_indices, raster_vert_weights = self.compute_normal_map(camera_params, skinned_verts_unpad,
                                                                                        skinned_verts, all_faces, output_weights=True)
        if verbose:
            print('Normals:', timer() - start)
        if return_mesh:
            return imgs, masks, skinned_verts, vert_tet_indices, raster_vert_indices, raster_vert_weights
        return imgs, masks

    def fov_camera_from_parameters(self, camera, azimuth=0., elev=0.):
        device = self.device
        # Initialize a camera.
        R = torch.eye(3).unsqueeze(0).to(self.device)
        T = torch.tensor([[0., 0., 1.]]).to(self.device)
        batch_size = camera.shape[0]
        R = R.repeat(batch_size, 1, 1)
        T = T.repeat(batch_size, 1)

        # Weak perspective camera only changes calibration matrix
        K = torch.eye(4).type(torch.float32).to(device)
        K = K.repeat(batch_size, 1, 1)
        K[:, 0, 0] = -camera[:,0]
        K[:, 1, 1] = -camera[:,1]
        K[:, 0, 3] = camera[:,2] * camera[:,0]
        K[:, 1, 3] = -camera[:,3] * camera[:,1]
        K[:, 2, 2] = -1*torch.ones(batch_size).to(device)

        cameras = FoVPerspectiveCameras(device=device, K=K, R=R, T=T)
        return cameras

    def compute_normal_map(self, camera_params, verts, verts_padded, faces, azimuth=180., elev=0., output_weights=False):
        #start = timer()
        vert_norms =  verts_normals_list(verts, faces, self.device)

        if type(camera_params) is tuple:
            if len(camera_params) == 5:
                K, R, T, log_D_R, D_T = camera_params
                D_R = so3_exp_map(log_D_R)
                assert D_R.shape == (1, 3, 3)
                R = torch.matmul(R, D_R)
                assert R.shape == (1, 3, 3)
                T = (torch.matmul(T, D_R) + D_T)[0]
                assert T.shape == (1, 3)
            else:
                K, R, T = camera_params
            cameras = FoVPerspectiveCameras(device=self.device, K=K, R=R, T=T)
        else:
            cameras = self.fov_camera_from_parameters(camera_params, azimuth, elev)

        # Convert to camera space
        vert_norms = torch.nn.utils.rnn.pad_sequence(vert_norms, batch_first=True)
        vert_norms = cameras.get_world_to_view_transform().transform_normals(vert_norms)
        #vert_norms = [v / torch.unsqueeze(torch.norm(v, p=2, dim=-1)+1e-8, -1) for v in vert_norms]

        # Convert vertices
        verts_screen = cameras.transform_points_screen(verts_padded, image_size=(self.img_dim, self.img_dim)).type(torch.float32)

        #print('Pre-raster computatiion:', timer()-start)
        # Prepare rasterizer parameters
        all_normals = []
        all_indices = []
        if output_weights:
            all_bary = []
        for i in range(len(verts)):
            #start = timer()
            bary, indices = RasterizerFunction.apply(verts_screen[i], faces[i].type(torch.int32), self.rasterizer)
            pixel_vert_norms = vert_norms[i][indices.long()]
            normal_map = torch.matmul(bary.unsqueeze(-2), pixel_vert_norms)[..., 0, :]

            normal_map = normal_map / torch.unsqueeze(torch.norm(normal_map+1e-8, p=2, dim=-1), -1)
            #print('Raster time:', timer()-start)
            all_normals.append(normal_map)
            all_indices.append(indices)
            if output_weights:
                all_bary.append(bary)
        all_normals = torch.stack(all_normals)

        all_masks = torch.linalg.norm(all_normals, axis=-1) > 0.5
        all_masks = torch.unsqueeze(all_masks, -1)

        raster_vert_indices = torch.stack(all_indices)
        if output_weights:
            raster_vert_weights = torch.stack(all_bary)
            return all_normals, all_masks, raster_vert_indices, raster_vert_weights
        return all_normals, all_masks, raster_vert_indices

    def compute_depth_map(self, camera_params, verts, raster_vert_indices, raster_vert_weights):
        batch_size = self.batch_size
        if type(camera_params) is tuple:
            K, R, T = camera_params
            cameras = Pytorch3D_FoVCameras(device=self.device, K=K, R=R, T=T)
        else:
            cameras = self.fov_camera_from_parameters(camera_params, azimuth, elev)

        # Actual barycentric coordinates should sum to 1.
        raster_vert_weights = raster_vert_weights / (torch.sum(raster_vert_weights, dim=-1, keepdim=True)+1e-10)
        pixel_tris = verts[raster_vert_indices[0].long()]
        pixel_points = torch.matmul(raster_vert_weights[0].unsqueeze(-2), pixel_tris)[..., 0, :]

        depth = torch.linalg.norm(pixel_points - torch.unsqueeze(T,0), dim=-1)
        return depth


if __name__=="__main__":
    from src.camera_transforms import RotateAxisAngle
    from pytorch3d.renderer.cameras import FoVPerspectiveCameras as Pytorch3D_FoVCameras
    def preprocess_camera_params(root_dir, idx):
        # Data from blender
        K = np.load(os.path.join('/data/jwu/RenderPeople/cameras/Camera%s_K.npy' % idx))
        R_euler = np.load(os.path.join('/data/jwu/RenderPeople/cameras/Camera%s_R.npy' % idx))
        T = np.load(os.path.join('/data/jwu/RenderPeople/cameras/Camera%s_T.npy' % idx))

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

        K = torch.unsqueeze(torch.tensor(K), 0).type(torch.float32)
        R = torch.unsqueeze(torch.tensor(R), 0).type(torch.float32)
        T = torch.unsqueeze(torch.tensor(T), 0).type(torch.float32)
        return K, R, T

    #TODO: Remove this later, testing w/ SMPL
    device = 'cuda'
    renderer = Renderer(device, batch_size=1, img_dim=512)
    #camera = torch.from_numpy(np.array([0.7967, 0.7967, 0.0180, 0.2367])).to(device)
    #obj_path = '../gradient_tests/rp_output/output_mesh.obj'

    root_dir = '/data/jwu/RenderPeople/rp_aaron_posed_001'
    obj_path = os.path.join(root_dir, 'mesh/mesh.obj')


    for i in range(10):
        # Load and preprocess camera params
        K, R, T = preprocess_camera_params(root_dir, '%04d' % i)
        K = K.to(device)
        R = R.to(device)
        T = T.to(device)

        obj = trimesh.load(obj_path, process=False)
        gt_verts = (torch.from_numpy(obj.vertices).type(torch.float32).to(device), )
        gt_faces = (torch.from_numpy(obj.faces).to(device), )
        train_normal_img, train_mask, raster_vert_indices, raster_vert_weights = renderer.compute_normal_map((K, R, T), (gt_verts[0],),
                                                                  gt_verts[0].unsqueeze(0), gt_faces[0].unsqueeze(0), output_weights=True)

        #output_image = ((0.5*train_normal_img[0]+0.5)*255).detach().cpu().numpy()
        #cv2.imwrite('test_normal_map.png', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

        renderer.compute_depth_map(i, (K, R, T), gt_verts[0], raster_vert_indices, raster_vert_weights, train_normal_img, train_mask)

