######################################################################
# Copyright 2021. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import cv2
#import cPickle as pickle
import pickle
import src.outershell as osh
from obj_io import *
import torch
import torch.nn as nn
from timeit import default_timer as timer 


class Skinning(nn.Module):
    def __init__(self, device="cuda:0"):
        super(self.__class__, self).__init__()
        self.device = device

        # Load files
        project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        J = np.load(os.path.join(project_dir,'TshapeCoarseJoints.npy')) # Neutral joint coordinates
        J_shapedir = np.load(os.path.join(project_dir,'J_shapedir.npy')) # J_shapedir (can deform joints depend on 10 betas)
        self.kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
                                    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]) # Define joint relationships
        # Define star-pose
        starpose = np.zeros(24*3)
        starpose[3:6] = np.array([0,  0,  0.5])
        starpose[6:9] = np.array([0,  0,  -0.5])
        self.starpose = torch.tensor(starpose).type(torch.float32).to(device)
        self.J = torch.tensor(J).type(torch.float32).to(device)

    def forward(self, vertices, pose, weights, scale, translation):
        device = self.device

        # Generate rotation matrix
        batch_size = vertices.shape[0]
        J = self.J * scale + translation
        A2_all, A2_global = global_rigid_transformation_batch(pose, self.kintree_table, J, device)


        A1,_ = global_rigid_transformation_torch(self.starpose, self.kintree_table, J, device)
        A1_inv = torch.dstack([torch.inverse(A1[:,:,i]) for i in range(24)])
        A1_inv_all = A1_inv.repeat(batch_size, 1, 1, 1)

        weights_T = weights.permute((0,2,1))
        R1 = torch.einsum('ijkl,ilm->ijkm', A1_inv_all, weights_T)
        R2 = torch.einsum('ijkl,ilm->ijkm', A2_all, weights_T)

        # Deform GT mesh to star-pose
        vertices_T = vertices.permute((0,2,1))
        vertices_4dim = torch.cat((vertices_T, torch.ones((batch_size, 1, vertices.shape[1])).to(device)), dim=1)

        v_Tpose = R1[:,:,0] * vertices_4dim[:,0].unsqueeze(1).repeat(1,4,1)
        v_Tpose += R1[:,:,1] * vertices_4dim[:,1].unsqueeze(1).repeat(1,4,1)
        v_Tpose += R1[:,:,2] * vertices_4dim[:,2].unsqueeze(1).repeat(1,4,1)
        v_Tpose += R1[:,:,3] * vertices_4dim[:,3].unsqueeze(1).repeat(1,4,1)

        v_starpose = R2[:,:,0] * v_Tpose[:,0].unsqueeze(1).repeat(1,4,1)
        v_starpose += R2[:,:,1] * v_Tpose[:,1].unsqueeze(1).repeat(1,4,1)
        v_starpose += R2[:,:,2] * v_Tpose[:,2].unsqueeze(1).repeat(1,4,1)
        v_starpose += R2[:,:,3] * v_Tpose[:,3].unsqueeze(1).repeat(1,4,1)
        v_starpose = v_starpose.permute((0,2,1))[:,:,:3]

        return v_starpose


def rodrigues(r):
    '''
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    '''
    #r = r.to(self.device)
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
    m = torch.stack(
      (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
       -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
             + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R


def global_rigid_transformation(pose, kintree_table, J):

    results = {}

    pose = pose.reshape((-1,3))
    
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}

    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    
    rodrigues = lambda x : cv2.Rodrigues(x)[0]
    rodriguesJB = lambda x : cv2.Rodrigues(x)[1]
    with_zeros = lambda x : np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))
    pack = lambda x : np.hstack([np.zeros((4, 3)), x.reshape((4,1))])
    

    results[0] = with_zeros(np.hstack((rodrigues(pose[0,:]), J[0,:].reshape((3,1)))))        
    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(np.hstack((rodrigues(pose[i,:]),(J[i] - J[parent[i]]).reshape(3,1)))))

    Rt_Global = np.dstack([results[i] for i in sorted(results.keys())])
    Rt_A = np.dstack([results[i] - (pack(results[i].dot(np.concatenate( ( (J[i,:]), (0,) ) )))) for i in range(len(results))])

    return Rt_A, Rt_Global


def global_rigid_transformation_torch(pose, kintree_table, J, device):

    results = {}

    #pose = torch.reshape(pose, (-1,3))
    
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}

    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    with_zeros = lambda x : torch.vstack((x, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(device)))
    pack = lambda x : torch.hstack([torch.zeros((4, 3)).to(device), torch.reshape(x, (4,1))])

    R_cube_big = rodrigues(pose.view(-1, 1, 3))

    results[0] = with_zeros(torch.hstack((R_cube_big[0], torch.reshape(J[0,:], (3,1)))))
    for i in range(1, kintree_table.shape[1]):
        parent_result = results[parent[i]]
        temp = with_zeros(torch.hstack((R_cube_big[i],(J[i] - J[parent[i]]).reshape(3,1))))
        results[i] = torch.matmul(parent_result, temp)

    Rt_Global = torch.dstack([results[i] for i in sorted(results.keys())])
    Rt_A = torch.dstack([results[i] - (pack(torch.matmul(results[i], torch.cat([J[i,:], torch.tensor([0]).to(device)])))) for i in range(len(results))])

    return Rt_A, Rt_Global


def global_rigid_transformation_batch(pose, kintree_table, J, device):
    batch_size = pose.shape[0]
    results = {}
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}
    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    pad = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(batch_size,1,1).to(device)
    with_zeros = lambda x : torch.cat((x, pad), dim=1) # (B,3,4) + (B,1,4) = (B,4,4)
    pack = lambda x : torch.cat([torch.zeros((batch_size, 4, 3)).to(device), x], dim=2) # (B,4,3) + (B,4,1) = (B,4,4)

    R_cube_big = rodrigues(torch.reshape(pose, (-1, 1, 3)))
    R_cube_big = torch.reshape(R_cube_big, (batch_size, 24, 3, 3))

    J_root = torch.reshape(J[0,:], (3,1)).repeat(batch_size, 1, 1)
    results[0] = with_zeros(torch.cat((R_cube_big[:,0], J_root), dim=2))

    for i in range(1, kintree_table.shape[1]):
        parent_result = results[parent[i]]
        J_parent = (J[i] - J[parent[i]]).reshape(3,1).repeat(batch_size, 1, 1)
        temp = with_zeros(torch.cat((R_cube_big[:,i], J_parent), dim=2))
        results[i] = torch.matmul(parent_result, temp)

    Rt_Global = torch.stack([results[i] for i in sorted(results.keys())], dim=3)
    Rt_A = torch.stack([results[i] - (pack(torch.matmul(results[i], torch.cat([J[i,:], torch.tensor([0]).to(device)]).repeat(batch_size, 1).unsqueeze(2)))) for i in range(len(results))], dim=3)
    return Rt_A, Rt_Global


def skin_mesh_from_T_pose(gtmesh_vertices, targetpose, weights, device='cpu'):
    '''
    Skins triangle mesh from T pose to targetpose

    Args:
        gtmesh_vertices: Batch triangle mesh vertices of shape (B, V, 3)
        targetpose: Batch SMPL pose parameters of shape (B, 72)
        weights: Batch skinning weights of shape (B, V, 24)
    Returns:
        Skinned vertices of shape (B, V, 3)
    '''
    #global J, J_shapedir, kintree_table, starpose, A1_inv
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    J = np.load(os.path.join(project_dir, 'TshapeCoarseJoints.npy'))
    J = torch.tensor(J).type(torch.float32).to(device)

    J_shapedir = np.load(os.path.join(project_dir, 'J_shapedir.npy'))
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
                                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]) # Define joint relationships
    # Define star-pose
    starpose = np.zeros(24*3)
    starpose = torch.tensor(starpose).type(torch.float32).to(device)

    A1,_ = global_rigid_transformation_torch(starpose, kintree_table, J, device)
    A1_inv = torch.dstack([torch.inverse(A1[:,:,i]) for i in range(24)])


    # Generate rotation matrix
    batch_size = gtmesh_vertices.shape[0]
    A2_all, A2_global = global_rigid_transformation_batch(targetpose, kintree_table, J, device)

    A1_inv_all = A1_inv.repeat(batch_size, 1, 1, 1)

    weights_T = weights.permute((0,2,1))
    R1 = torch.einsum('ijkl,ilm->ijkm', A1_inv_all, weights_T)
    R2 = torch.einsum('ijkl,ilm->ijkm', A2_all, weights_T)

    # Deform GT mesh to star-pose
    gtmesh_vertices_T = gtmesh_vertices.permute((0,2,1))
    gtmesh_vertices_4dim = torch.cat((gtmesh_vertices_T, torch.ones((batch_size, 1, gtmesh_vertices.shape[1])).to(device)), dim=1)

    v_Tpose = R1[:,:,0] * gtmesh_vertices_4dim[:,0].unsqueeze(1).repeat(1,4,1)
    v_Tpose += R1[:,:,1] * gtmesh_vertices_4dim[:,1].unsqueeze(1).repeat(1,4,1)
    v_Tpose += R1[:,:,2] * gtmesh_vertices_4dim[:,2].unsqueeze(1).repeat(1,4,1)
    v_Tpose += R1[:,:,3] * gtmesh_vertices_4dim[:,3].unsqueeze(1).repeat(1,4,1)

    v_starpose = R2[:,:,0] * v_Tpose[:,0].unsqueeze(1).repeat(1,4,1)
    v_starpose += R2[:,:,1] * v_Tpose[:,1].unsqueeze(1).repeat(1,4,1)
    v_starpose += R2[:,:,2] * v_Tpose[:,2].unsqueeze(1).repeat(1,4,1)
    v_starpose += R2[:,:,3] * v_Tpose[:,3].unsqueeze(1).repeat(1,4,1)
    v_starpose = v_starpose.permute((0,2,1))[:,:,:3]

    return v_starpose


def skin_mesh_from_star_pose(gtmesh_vertices, targetpose, weights, device='cpu'):
    ''' 
    Skins triangle mesh from star pose to targetpose

    Args:
        gtmesh_vertices: Batch triangle mesh vertices of shape (B, V, 3)
        targetpose: Batch SMPL pose parameters of shape (B, 72)
        weights: Batch skinning weights of shape (B, V, 24)
    Returns:
        Skinned vertices of shape (B, V, 3)
    '''
    #global J, J_shapedir, kintree_table, starpose, A1_inv
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    J = np.load(os.path.join(project_dir, 'TshapeCoarseJoints.npy'))
    J = torch.tensor(J).type(torch.float32).to(device)

    J_shapedir = np.load(os.path.join(project_dir, 'J_shapedir.npy'))
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
                                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]) # Define joint relationships
    # Define star-pose
    starpose = np.zeros(24*3)
    starpose[3:6] = np.array([0,  0,  0.5])
    starpose[6:9] = np.array([0,  0,  -0.5])
    starpose = torch.tensor(starpose).type(torch.float32).to(device)

    A1,_ = global_rigid_transformation_torch(starpose, kintree_table, J, device)
    A1_inv = torch.dstack([torch.inverse(A1[:,:,i]) for i in range(24)])


    # Generate rotation matrix
    batch_size = gtmesh_vertices.shape[0]
    A2_all, A2_global = global_rigid_transformation_batch(targetpose, kintree_table, J, device)

    A1_inv_all = A1_inv.repeat(batch_size, 1, 1, 1)

    weights_T = weights.permute((0,2,1))
    R1 = torch.einsum('ijkl,ilm->ijkm', A1_inv_all, weights_T)
    R2 = torch.einsum('ijkl,ilm->ijkm', A2_all, weights_T)

    # Deform GT mesh to star-pose
    gtmesh_vertices_T = gtmesh_vertices.permute((0,2,1))
    gtmesh_vertices_4dim = torch.cat((gtmesh_vertices_T, torch.ones((batch_size, 1, gtmesh_vertices.shape[1])).to(device)), dim=1)

    v_Tpose = R1[:,:,0] * gtmesh_vertices_4dim[:,0].unsqueeze(1).repeat(1,4,1)
    v_Tpose += R1[:,:,1] * gtmesh_vertices_4dim[:,1].unsqueeze(1).repeat(1,4,1)
    v_Tpose += R1[:,:,2] * gtmesh_vertices_4dim[:,2].unsqueeze(1).repeat(1,4,1)
    v_Tpose += R1[:,:,3] * gtmesh_vertices_4dim[:,3].unsqueeze(1).repeat(1,4,1)

    v_starpose = R2[:,:,0] * v_Tpose[:,0].unsqueeze(1).repeat(1,4,1)
    v_starpose += R2[:,:,1] * v_Tpose[:,1].unsqueeze(1).repeat(1,4,1)
    v_starpose += R2[:,:,2] * v_Tpose[:,2].unsqueeze(1).repeat(1,4,1)
    v_starpose += R2[:,:,3] * v_Tpose[:,3].unsqueeze(1).repeat(1,4,1)
    v_starpose = v_starpose.permute((0,2,1))[:,:,:3]

    return v_starpose

def skin_tet_mesh(pose, gtmesh_vertices, tet_skinning_weights):
    J = np.load(os.path.join(project_dir, 'TshapeCoarseJoints.npy'))
    J_shapedir = np.load(os.path.join(project_dir, 'J_shapedir.npy'))
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
                                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]) # Define joint relationships

    # Define star-pose
    starpose = np.zeros(24*3)
    starpose[3:6] = np.array([0,  0,  0.5])
    starpose[6:9] = np.array([0,  0,  -0.5])

    # Generate rotation matrix
    A1, A1_global = global_rigid_transformation(starpose, kintree_table, J)
    A1_inv = np.dstack([np.linalg.inv(A1[:,:,i]) for i in range(24)])
    A2, A2_global = global_rigid_transformation(pose, kintree_table, J)

    R1 = A1_inv.dot(tet_skinning_weights.T)
    R2 = A2.dot(tet_skinning_weights.T)

    # Deform GT mesh to star-pose
    gtmesh_vertices_4dim = np.vstack((gtmesh_vertices.T, np.ones((1, gtmesh_vertices.shape[0]))))
    v_Tpose = (R1[:,0] * gtmesh_vertices_4dim[0] + R1[:,1] * gtmesh_vertices_4dim[1] + R1[:,2] * gtmesh_vertices_4dim[2] + R1[:,3] * gtmesh_vertices_4dim[3])
    v_starpose = (R2[:,0] * v_Tpose[0] + R2[:,1] * v_Tpose[1] + R2[:,2] * v_Tpose[2] + R2[:,3] * v_Tpose[3]).T[:,:3]

    return v_starpose


if __name__ == '__main__':
    params_dir = '/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/TetraTSDF/D_march/smplparams_centered'
    parampath = os.path.join(params_dir,'param_D_march_101.pkl')
    
    # Load pose
    with open(parampath, mode='rb') as f:
        param = pickle.load(f, encoding='latin1')
    pose = param['pose']
    print('SMPL pose:', pose.shape)
    pose = pose.ravel()

    pose = torch.from_numpy(pose).type(torch.float32)
    # Initialize tet skinning weights
    weights = torch.from_numpy(np.load('coarseweights.npy')).type(torch.float32)

    # Initialize tet mesh
    verts,faces,_,_ = osh.loadOuterShell('star_shell.ply')
    verts = torch.from_numpy(verts).type(torch.float32)

    verts = verts.repeat(16,1,1)
    pose = pose.repeat(16,1)
    weights = weights.repeat(16,1,1)

    print('Skinning inputs:', verts.shape, pose.shape, weights.shape)
    print('Faces:', faces.shape)
    start = timer()
    new_verts = skin_mesh_from_star_pose(verts.to('cuda'), pose.to('cuda'), weights.to('cuda'), device='cuda')
    print('Skinning:', timer() - start)
    obj = Obj(new_verts[0].detach().cpu().numpy(), faces)
    write_obj(obj, 'test_tpose_mesh.obj')

