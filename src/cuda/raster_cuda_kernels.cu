#include <cuda.h>
#include <cuda_runtime_api.h>

inline __device__ float Min3(float a, float b, float c)
{
    return min(min(a,b),c);
}

inline __device__ float Max3(float a, float b, float c)
{
    return max(max(a,b),c);
}

inline __device__ float Norm_3D(float x, float y, float z)
{
    return sqrt(pow(x,2) + pow(y,2) + pow(z,2));
}

__device__ float Triangle_Area_2D(float x0, float y0, float x1, float y1, float x2, float y2)
{
    float u[] = {x1-x0,y1-y0};
    float v[] = {x2-x0,y2-y0};
    float cross = u[0]*v[1] - v[0]*u[1];
    return cross / 2.;
}

__device__ bool Barycentric_Coordinates_2D(float *weights, float px, float py,
                                float x0, float y0, float x1, float y1,
                                float x2, float y2)
{
    float total_area = Triangle_Area_2D(x0, y0, x1, y1, x2, y2);
    if(total_area == 0) return false;

    weights[0] = Triangle_Area_2D(px, py, x1, y1, x2, y2) / total_area;
    if(weights[0] < 0) return false;

    weights[1] = Triangle_Area_2D(x0, y0, px, py, x2, y2) / total_area;
    if(weights[1] < 0 || weights[0]+weights[1] > 1) return false;

    weights[2] = Triangle_Area_2D(x0, y0, x1, y1, px, py) / total_area;
    if(weights[2] < 0) return false;
    return true;
}

// This is the gradient of alpha w.r.t. v1 or v2 (negate_sign).
__device__ void Grad_Alpha(float x1, float y1, float z1,
                           float x2, float y2, float z2, 
                           float px, float py, bool negate_sign, float* result)
{
    float sign = 1;
    if(negate_sign)
        sign = -1;

    //Make sure area is positive.
    float z_area = -1;
    if(negate_sign)
        z_area = Triangle_Area_2D(px, py, x2, y2, x1, y1);
    else
        z_area = Triangle_Area_2D(px, py, x1, y1, x2, y2);

    //assert(z_area>=0);
    //if(z_area<0)
    //    sign = sign * -1;
    //    z_area = z_area * -1;
    result[0]=z1*z2*(y2-py)*sign/2.;
    result[1]=z1*z2*(px-x2)*sign/2.;
    result[2]=z2*z_area;

}

__global__ void Rasterize_Triangle_Mesh_Gradients_Kernel(float *verts_screen, int *vert_indices,
                                                         float *grad_alphas, float *grad_verts_screen,
                                                         int height)
{
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;
  
    // Pixel 2D indices
    int x = idx / height;
    int y = idx % height;
    float px_center = x+0.5;
    float py_center = y+0.5;

    // Vertex indices of the face intersecting with this pixel
    int raster_idx = 3*height*x + 3*y;
    int i = vert_indices[raster_idx];
    int j = vert_indices[raster_idx+1];
    int k = vert_indices[raster_idx+2];

    // If all zeros, skip
    if(i==0 && j==0 && k==0) return;
    
    // Check counterclockwise ordering!!
    float triangle_area = Triangle_Area_2D(verts_screen[i*3], verts_screen[i*3+1],
                                           verts_screen[j*3], verts_screen[j*3+1],
                                           verts_screen[k*3], verts_screen[k*3+1]);
    // For each alpha, compute d alpha / d v_i
    int idx1, idx2;
    bool negate_sign;
    int other_verts[2] = {-1};
    float d_alpha_v[3]={0};
    for(int m=0;m<3;m++){
        // Figure out other two vertices...
        if(m==0){
            other_verts[0]=j;
            other_verts[1]=k;
        }
        else if(m==1){
            other_verts[0]=k;
            other_verts[1]=i;
        }
        else{
            other_verts[0]=i;
            other_verts[1]=j;
        }
        for (int n=0;n<2;n++){
            if(n==0){
                idx1=other_verts[0];
                idx2=other_verts[1];
                negate_sign=false;
            }
            else{
                idx1=other_verts[1];
                idx2=other_verts[0];
                negate_sign=true;
            }
            Grad_Alpha(verts_screen[idx1*3], verts_screen[idx1*3+1], verts_screen[idx1*3+2],
                       verts_screen[idx2*3], verts_screen[idx2*3+1], verts_screen[idx2*3+2],
                       px_center, py_center, negate_sign, d_alpha_v);

            // Add gradients to output
            atomicAdd(&(grad_verts_screen[idx1*3]), grad_alphas[raster_idx+m]*d_alpha_v[0]);
            atomicAdd(&(grad_verts_screen[idx1*3+1]), grad_alphas[raster_idx+m]*d_alpha_v[1]);
            atomicAdd(&(grad_verts_screen[idx1*3+2]), grad_alphas[raster_idx+m]*d_alpha_v[2]);
        }
    }
}
extern "C" {
void Rasterize_Triangle_Mesh_Gradients(float *verts_screen, int *vert_indices,
                                       float *grad_alphas, float *grad_verts_screen,
                                       int height,
                                       int verts_screen_size,
                                       int vert_indices_size,
                                       int grad_alphas_size,
                                       int grad_verts_screen_size){
    //int height = params[0];
    /*
    int verts_screen_size = params[1];
    int vert_indices_size = params[2];
    int grad_alphas_size = params[3];
    int grad_verts_screen_size = params[4];

    float *verts_screen_in, *grad_alphas_in, *grad_verts_screen_out;
    int *vert_indices_in;
    cudaMalloc((void **)&verts_screen_in, verts_screen_size * sizeof(float));
    cudaMalloc((void **)&vert_indices_in, vert_indices_size * sizeof(int));
    cudaMalloc((void **)&grad_alphas_in, grad_alphas_size * sizeof(float));
    cudaMalloc((void **)&grad_verts_screen_out, grad_verts_screen_size * sizeof(float));

    cudaMemcpy(verts_screen_in, verts_screen, verts_screen_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vert_indices_in, vert_indices, vert_indices_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_alphas_in, grad_alphas, grad_alphas_size * sizeof(float), cudaMemcpyHostToDevice);
    */
    Rasterize_Triangle_Mesh_Gradients_Kernel <<< ceil(height*height/1.), 1 >>> (verts_screen, vert_indices, grad_alphas, grad_verts_screen, height);

    //cudaMemcpy(grad_verts_screen, grad_verts_screen_out, grad_verts_screen_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free all allocated memory.
    /*cudaFree(verts_screen_in);
    cudaFree(vert_indices_in);
    cudaFree(grad_alphas_in);
    cudaFree(grad_verts_screen_out);*/
}

}
