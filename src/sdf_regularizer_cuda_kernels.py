######################################################################
# Copyright 2022. Jane Wu.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

import os
import torch

phi = torch.cuda.FloatTensor(9)
tetrahedra = torch.cuda.IntTensor(9)
inv_delta_v = torch.cuda.FloatTensor(9)
coeff = torch.cuda.FloatTensor(9)
coeff_gradients = torch.cuda.FloatTensor(9)
output = torch.cuda.FloatTensor(9)
grad = torch.cuda.FloatTensor(9)
boundary_tet = torch.cuda.IntTensor(9)
params = torch.cuda.FloatTensor(9)

mod = SourceModule("""
// matmul
inline __host__ __device__ void matmul_3x3_3x1(float *a, float *b, int start_idx, float *result)
{
    // Assuming a is 3x3 and b is 3x1.
    for(int i=0;i<3;i++)
        result[i] = a[start_idx+3*i]*b[0] + a[start_idx+3*i+1]*b[1] + a[start_idx+3*i+2]*b[2];
}

extern "C"
__global__ void ComputeSmearedHeaviside(float *phi, float *output, float *grad, float *params){
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    int nb_verts = int(params[0]);
    float scale = params[1];

    // assuming x and y inputs are same length
    if (idx >= nb_verts)
        return;
    
    float eps = 1.5*scale;
    float pi = 3.14159265;

    if(phi[idx] < -eps){
        output[idx] = 0;
        grad[idx] = 0;
    }
    else if(phi[idx] > eps){
        output[idx] = 1;
        grad[idx] = 0;
    }
    else{
        output[idx] = 0.5 + phi[idx]/(2.*eps) + 1/(2.*pi)*sin((pi*phi[idx])/eps);
        grad[idx] = 1/(2.*eps)*(1 + cos((pi*phi[idx])/eps));
    }
}

extern "C"
__global__ void  ComputeEikonalGrads(float *inv_delta_v, float *coeff_gradients, float *params){
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    int nb_tets = int(params[0]);

    // assuming x and y inputs are same length
    if (idx >= nb_tets)
        return;

    int start_idx = 9*idx;
    // Gradients
    for(int i=0;i<12;i++){
        if(i%4==0){
            int row = i/4;
            coeff_gradients[12*idx+i] = inv_delta_v[start_idx+3*row] +
                                        inv_delta_v[start_idx+3*row+1] +
                                        inv_delta_v[start_idx+3*row+2];
        }
        else{
            int row = i/4;
            int col = i%4-1;
            coeff_gradients[12*idx+i] = -inv_delta_v[start_idx+3*row+col];
        }
    }
}

extern "C"
__global__ void  ComputeLinearCoeffs(float *phi, int *tetrahedra, float *inv_delta_v, float *coeff,
                                     int *boundary_tet, float *params){
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    int nb_tets = int(params[0]);

    // assuming x and y inputs are same length
    if (idx >= nb_tets)
        return;

    // Compute delta phi values.
    int source_idx = tetrahedra[4*idx];
    int edge_idx[3] = {
                       tetrahedra[4*idx+1], 
                       tetrahedra[4*idx+2],
                       tetrahedra[4*idx+3]
                      };
    float phi_edge[3];
    for(int i=0;i<3;i++){
        phi_edge[i] = phi[source_idx] - phi[edge_idx[i]];
        if(phi[source_idx]*phi[edge_idx[i]]<0)
            boundary_tet[idx] = 1;
    }

    // Compute matmul to calculate coefficients of linear approximation.
    float result[3];
    int start_idx = 9*idx;
    matmul_3x3_3x1(inv_delta_v, phi_edge, start_idx, result);

    // Store values
    coeff[3*idx] = result[0];
    coeff[3*idx+1] = result[1];
    coeff[3*idx+2] = result[2];
}
""", no_extern_c=True)
