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

vert_tet_weights = torch.cuda.FloatTensor(9)
weight_gradients = torch.cuda.FloatTensor(9)
vert_tet_indices = torch.cuda.IntTensor(9)
sdf = torch.cuda.FloatTensor(9)
nodes = torch.cuda.FloatTensor(9)
tet_skinning_weights = torch.cuda.FloatTensor(9)
edges_a = torch.cuda.IntTensor(9)
edges_b = torch.cuda.IntTensor(9)
params = torch.cuda.FloatTensor(9)

#grad_phi = torch.cuda.FloatTensor(9)
#grad_weights = torch.cuda.FloatTensor(9)

mod = SourceModule("""
#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8
#define THREAD_SIZE_Z 8

#define TRUNCATE 0.4f

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float norm(float3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

extern "C"
__global__ void  GenerateFaces(int *Faces, float *TSDF, int *Edges_row_ptr, int *Edges_columns, 
int *Tetra, float *params) {
    // Thread id in a 1D block
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int voxid = tx + ty * bw;

    unsigned int batch_idx = blockIdx.y;

    int nb_tets = int(params[0]);
    int nb_verts = int(params[1]);
    float m_iso = params[2];

    // assuming x and y inputs are same length
    if (voxid >= nb_tets-1)
        return;

    float a, b, c, d; //4 summits if the tetrahedra voxel

    //Value of the TSDF
    a = TSDF[nb_verts*batch_idx + Tetra[voxid * 4]];
    b = TSDF[nb_verts*batch_idx + Tetra[voxid * 4 + 1]];
    c = TSDF[nb_verts*batch_idx + Tetra[voxid * 4 + 2]];
    d = TSDF[nb_verts*batch_idx + Tetra[voxid * 4 + 3]];

    int count = 0;
    if (a >= m_iso)
        count += 1;
    if (b >= m_iso)
        count += 1;
    if (c >= m_iso)
        count += 1;
    if (d >= m_iso)
        count += 1;

    if (count == 0 || count == 4) //return;
    {
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+0] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+1] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+2] = 0;

        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+3] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+4] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+5] = 0;
    }

    //! Three vertices are inside the volume
    else if (count == 3) {
        int2 list[6] = { make_int2(0,1), make_int2(0,2), make_int2(0,3), 
        make_int2(1,2), make_int2(1,3), make_int2(2,3) };
        //! Make sure that fourth value lies outside
        if (d < m_iso)
        {
        }
        else if (c < m_iso)
        {
            list[0] = make_int2(0, 3);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(0, 2);
            list[3] = make_int2(1, 3);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(1, 2);
        }
        else if (b < m_iso)
        {
            list[0] = make_int2(0, 2);
            list[1] = make_int2(0, 3);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(1, 2);
            list[5] = make_int2(1, 3);
        }
        else
        {
            list[0] = make_int2(1, 3);
            list[1] = make_int2(1, 2);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(0, 3);
            list[5] = make_int2(0, 2);
        }
        //ad
        int sum1 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].x] : Tetra[voxid * 4 + list[2].y];
        int sum2 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].y] : Tetra[voxid * 4 + list[2].x];
        int idx_ad = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ad = k;
                break;
            }
        }

        //bd
        sum1 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].x] : Tetra[voxid * 4 + list[4].y];
        sum2 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].y] : Tetra[voxid * 4 + list[4].x];
        int idx_bd = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_bd = k;
                break;
            }
        }

        //cd
        sum1 = Tetra[voxid * 4 + list[5].x] < Tetra[voxid * 4 + list[5].y] ? Tetra[voxid * 4 + list[5].x] : Tetra[voxid * 4 + list[5].y];
        sum2 = Tetra[voxid * 4 + list[5].x] < Tetra[voxid * 4 + list[5].y] ? Tetra[voxid * 4 + list[5].y] : Tetra[voxid * 4 + list[5].x];
        int idx_cd = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_cd = k;
                break;
            }
        }

        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+0] = idx_ad;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+1] = idx_cd;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+2] = idx_bd;

        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+3] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+4] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+5] = 0;

    }

    //! Two vertices are inside the volume
    else if (count == 2) {
        //! Make sure that the last two points lie outside
        int2 list[6] = { make_int2(0,1), make_int2(0,2), make_int2(0,3), 
        make_int2(1,2), make_int2(1,3), make_int2(2,3) };
        if (a >= m_iso && b >= m_iso)
        {
        }
        else if (a >= m_iso && c >= m_iso)
        {
            list[0] = make_int2(0, 2);
            list[1] = make_int2(0, 3);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(1, 2);
            list[5] = make_int2(1, 3);
        }
        else if (a >= m_iso && d >= m_iso)
        {
            list[0] = make_int2(0, 3);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(0, 2);
            list[3] = make_int2(1, 3);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(1, 2);
        }
        else if (b >= m_iso && c >= m_iso)
        {
            list[0] = make_int2(1, 2);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(1, 3);
            list[3] = make_int2(0, 2);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(0, 3);
        }
        else if (b >= m_iso && d >= m_iso)
        {
            list[0] = make_int2(1, 3);
            list[1] = make_int2(1, 2);
            list[2] = make_int2(0, 1);
            list[3] = make_int2(2, 3);
            list[4] = make_int2(0, 3);
            list[5] = make_int2(0, 2);
        }
        else //c && d > m_iso
        {
            list[0] = make_int2(2, 3);
            list[1] = make_int2(0, 2);
            list[2] = make_int2(1, 2);
            list[3] = make_int2(0, 3);
            list[4] = make_int2(1, 3);
            list[5] = make_int2(0, 1);
        }

        //ac
        int sum1 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].x] : Tetra[voxid * 4 + list[1].y];
        int sum2 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].y] : Tetra[voxid * 4 + list[1].x];
        int idx_ac = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ac = k;
                break;
            }
        }

        //ad
        sum1 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].x] : Tetra[voxid * 4 + list[2].y];
        sum2 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].y] : Tetra[voxid * 4 + list[2].x];
        int idx_ad = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ad = k;
                break;
            }
        }

        //bc
        sum1 = Tetra[voxid * 4 + list[3].x] < Tetra[voxid * 4 + list[3].y] ? Tetra[voxid * 4 + list[3].x] : Tetra[voxid * 4 + list[3].y];
        sum2 = Tetra[voxid * 4 + list[3].x] < Tetra[voxid * 4 + list[3].y] ? Tetra[voxid * 4 + list[3].y] : Tetra[voxid * 4 + list[3].x];
        int idx_bc = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_bc = k;
                break;
            }
        }

        //bd
        sum1 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].x] : Tetra[voxid * 4 + list[4].y];
        sum2 = Tetra[voxid * 4 + list[4].x] < Tetra[voxid * 4 + list[4].y] ? Tetra[voxid * 4 + list[4].y] : Tetra[voxid * 4 + list[4].x];
        int idx_bd = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_bd = k;
                break;
            }
        }

        // storeTriangle(ac,bc,ad);
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+0] = idx_ac;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+1] = idx_bc;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+2] = idx_ad;

        //storeTriangle(bc,bd,ad);
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+3] = idx_bc;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+4] = idx_bd;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+5] = idx_ad;
    }
    //! One vertex is inside the volume
    else if (count == 1) {
        //! Make sure that the last three points lie outside
        int2 list[6] = { make_int2(0,1), make_int2(0,2), make_int2(0,3),
            make_int2(1,2), make_int2(1,3), make_int2(2,3) };
        if (a >= m_iso)
        {
        }
        else if (b >= m_iso)
        {
            list[0] = make_int2(1, 2);
            list[1] = make_int2(0, 1);
            list[2] = make_int2(1, 3);
            list[3] = make_int2(0, 2);
            list[4] = make_int2(2, 3);
            list[5] = make_int2(0, 3);
        }
        else if (c >= m_iso)
        {
            list[0] = make_int2(0, 2);
            list[1] = make_int2(1, 2);
            list[2] = make_int2(2, 3);
            list[3] = make_int2(0, 1);
            list[4] = make_int2(0, 3);
            list[5] = make_int2(1, 3);
        }
        else // d > m_iso
        {
            list[0] = make_int2(2, 3);
            list[1] = make_int2(1, 3);
            list[2] = make_int2(0, 3);
            list[3] = make_int2(1, 2);
            list[4] = make_int2(0, 2);
            list[5] = make_int2(0, 1);
        }

        //ab
        int sum1 = Tetra[voxid * 4 + list[0].x] < Tetra[voxid * 4 + list[0].y] ? Tetra[voxid * 4 + list[0].x] : Tetra[voxid * 4 + list[0].y];
        int sum2 = Tetra[voxid * 4 + list[0].x] < Tetra[voxid * 4 + list[0].y] ? Tetra[voxid * 4 + list[0].y] : Tetra[voxid * 4 + list[0].x];
        int idx_ab = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ab = k;
                break;
            }
        }

        //ac
        sum1 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].x] : Tetra[voxid * 4 + list[1].y];
        sum2 = Tetra[voxid * 4 + list[1].x] < Tetra[voxid * 4 + list[1].y] ? Tetra[voxid * 4 + list[1].y] : Tetra[voxid * 4 + list[1].x];
        int idx_ac = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ac = k;
                break;
            }
        }

        //ad
        sum1 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].x] : Tetra[voxid * 4 + list[2].y];
        sum2 = Tetra[voxid * 4 + list[2].x] < Tetra[voxid * 4 + list[2].y] ? Tetra[voxid * 4 + list[2].y] : Tetra[voxid * 4 + list[2].x];
        int idx_ad = 0;
        for (int k = Edges_row_ptr[sum1]; k < Edges_row_ptr[sum1 + 1]; k++) {
            if (Edges_columns[k] == sum2) {
                idx_ad = k;
                break;
            }
        }

        //storeTriangle(ab,ad,ac);
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+0] = idx_ab;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+1] = idx_ad;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+2] = idx_ac;

        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+3] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+4] = 0;
        Faces[(2*nb_tets*3)*batch_idx + 6 * (voxid)+5] = 0;
    }
}

extern "C"
__global__ void  ComputeVertices(float *vert_tet_weights, int *vert_tet_indices, float *weight_gradients, float *sdf, float *nodes, float *tet_skinning_weights, int *edges_a, int *edges_b, float *params) {
    //unsigned int tx = threadIdx.x + threadIdx.y * THREAD_SIZE_X + threadIdx.z * THREAD_SIZE_X * THREAD_SIZE_Y;
    //

    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    // Batch index
    unsigned int batch_idx = blockIdx.y;


    int nb_edges = int(params[0]);
    int nb_verts = int(params[1]);
    float level = params[2];

    // assuming x and y inputs are same length
    if (idx >= nb_edges)
        return;

    int sum_1 = edges_a[idx];
    int sum_2 = edges_b[idx];

    float sdf_1 = sdf[nb_verts*batch_idx+sum_1] - level;
    float sdf_2 = sdf[nb_verts*batch_idx+sum_2] - level;
    float eps = 1e-8;

    if (abs(sdf_1) < eps){
        if (signbit(sdf_1))
            sdf_1 = -1e-8;
        else
            sdf_1 = 1e-8;
    }
    if (abs(sdf_2) < eps){
        if (signbit(sdf_2))
            sdf_2 = -1e-8;
        else
            sdf_2 = 1e-8;
    }

    if (sdf_1 * sdf_2 < 0.0f) {
        float weight_1 = -sdf_2 / (sdf_1 - sdf_2 + eps);
        float weight_2 = sdf_1 / (sdf_1 - sdf_2 + eps);
        /*
        verts[3*idx] = weight_1 * nodes[3*sum_1] + weight_2 * nodes[3*sum_2];
        verts[3*idx+1] = weight_1 * nodes[3*sum_1+1] + weight_2 * nodes[3*sum_2+1];
        verts[3*idx+2] = weight_1 * nodes[3*sum_1+2] + weight_2 * nodes[3*sum_2+2];
        */
        vert_tet_weights[(nb_edges*2)*batch_idx+2*idx] = weight_1;
        vert_tet_weights[(nb_edges*2)*batch_idx+2*idx+1] = weight_2;
        vert_tet_indices[(nb_edges*2)*batch_idx+2*idx] = sum_1;
        vert_tet_indices[(nb_edges*2)*batch_idx+2*idx+1] = sum_2;

        float scalar = 1 / pow(sdf_1 - sdf_2 + eps, 2);
        weight_gradients[(nb_edges*4)*batch_idx+4*idx] = scalar*sdf_2;
        weight_gradients[(nb_edges*4)*batch_idx+4*idx+1] = -scalar*sdf_2;
        weight_gradients[(nb_edges*4)*batch_idx+4*idx+2] = -scalar*sdf_1;
        weight_gradients[(nb_edges*4)*batch_idx+4*idx+3] = scalar*sdf_1;
    }
}

extern "C"
__global__ void  InflateSDF(float *sdf, float *inflated_sdf, int *edges_a, int *edges_b, float *params) {
    //unsigned int tx = threadIdx.x + threadIdx.y * THREAD_SIZE_X + threadIdx.z * THREAD_SIZE_X * THREAD_SIZE_Y;
    //

    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    // Batch index
    unsigned int batch_idx = blockIdx.y;


    int nb_edges = int(params[0]);
    int nb_verts = int(params[1]);

    // assuming x and y inputs are same length
    if (idx >= nb_edges)
        return;

    int sum_1 = edges_a[idx];
    int sum_2 = edges_b[idx];

    float sdf_1 = sdf[nb_verts*batch_idx+sum_1];
    float sdf_2 = sdf[nb_verts*batch_idx+sum_2];

    float eps = 1e-4;
    if (sdf_1 * sdf_2 < 0.0f) {
        if (sdf_1 > 0.0f)
            atomicExch(&inflated_sdf[nb_verts*batch_idx+sum_1],-eps);
        else
            atomicExch(&inflated_sdf[nb_verts*batch_idx+sum_2],-eps);
    }
}
""", no_extern_c=True)
