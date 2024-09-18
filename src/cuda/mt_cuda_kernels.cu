#include <cuda.h>
#include <cuda_runtime_api.h>

#define TRUNCATE 0.4f
#define THREAD_SIZE_X 128
#define divUp(x,y) (x%y) ? ((x+y-1)/y) : (x/y)

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

__global__ void  Compute_Vertex_Gradients_Kernel(float *grad_loss_wrt_weight, int *vert_tet_indices,
                                                 float *grad_weight_wrt_phi, 
                                                 float *grad_loss_wrt_phi, int nb_verts) {
    //unsigned int tx = threadIdx.x + threadIdx.y * THREAD_SIZE_X + threadIdx.z * THREAD_SIZE_X * THREAD_SIZE_Y;
    //

    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    // parallelizing over vertices
    if (idx >= nb_verts)
        return;

    int sum_1 = vert_tet_indices[2*idx];
    int sum_2 = vert_tet_indices[2*idx+1];

    //dL/dw_j * dw_j/dphi_i
    atomicAdd(&(grad_loss_wrt_phi[sum_1]), grad_loss_wrt_weight[2*idx]*grad_weight_wrt_phi[4*idx] +
                                           grad_loss_wrt_weight[2*idx+1]*grad_weight_wrt_phi[4*idx+1]);
    atomicAdd(&(grad_loss_wrt_phi[sum_2]), grad_loss_wrt_weight[2*idx]*grad_weight_wrt_phi[4*idx+2] +
                                           grad_loss_wrt_weight[2*idx+1]*grad_weight_wrt_phi[4*idx+3]);
}

extern "C"{
void Compute_Vertex_Gradients(float *grad_loss_wrt_weight, int *vert_tet_indices,
                              float *grad_weight_wrt_phi, 
                              float *grad_loss_wrt_phi, int nb_verts,
                              int grad_loss_wrt_weight_size,
                              int vert_tet_indices_size,
                              int grad_weight_wrt_phi_size,
                              int grad_loss_wrt_phi_size)
{
    Compute_Vertex_Gradients_Kernel <<< ceil(nb_verts/128.), 128 >>> (grad_loss_wrt_weight, vert_tet_indices,
                                                                      grad_weight_wrt_phi, grad_loss_wrt_phi, nb_verts);
}

}

