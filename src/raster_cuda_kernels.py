#import pycuda.autoinit  # Enable if running rasterizer script
import pycuda.driver as drv
import numpy
import torch
from pycuda.compiler import SourceModule

drv.init()
pycuda_ctx = drv.Device(0).retain_primary_context()

verts_screen = torch.cuda.FloatTensor(9)
faces = torch.cuda.IntTensor(9)
dists = torch.cuda.FloatTensor(9)
alphas = torch.cuda.FloatTensor(9)
params = torch.cuda.FloatTensor(9)
bins2tri = torch.cuda.IntTensor(9)
bins2z = torch.cuda.IntTensor(9)
bins_idx = torch.cuda.IntTensor(9)
grad_alphas = torch.cuda.FloatTensor(9)
grad_alphas_idx = torch.cuda.IntTensor(9)
grad_verts_screen = torch.cuda.FloatTensor(9)

mod = SourceModule("""
#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8
#define THREAD_SIZE_Z 8

#define TRUNCATE 0.4f

extern "C" {

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


__global__ void Bin_Triangles(float *verts_screen, int *faces, int *bins2tri, int *bins2z, int *bins_idx, int *params)
{
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int f = tx + ty * bw;

    //NOTE: f is the triangle idx. So we are storing 1 bin index (flattened) for each triangle.

    //Parameters
    int max_triangles_per_bin = params[0];
    int width = params[1];
    int height = params[2];

    int i,j,k;
    int min_x, min_y;
    int max_x, max_y;
    i = faces[f*3];
    j = faces[f*3+1];
    k = faces[f*3+2];

    //Compute triangle bbox
    min_x = floor(Min3(verts_screen[i*3],verts_screen[j*3],verts_screen[k*3]));
    max_x = ceil(Max3(verts_screen[i*3],verts_screen[j*3],verts_screen[k*3]));
    min_y = floor(Min3(verts_screen[i*3+1],verts_screen[j*3+1],verts_screen[k*3+1]));
    max_y = ceil(Max3(verts_screen[i*3+1],verts_screen[j*3+1],verts_screen[k*3+1]));

    //Clip against screen bounds
    min_x = max(min_x, 0);
    min_y = max(min_y, 0);
    max_x = min(max_x, width-1);
    max_y = min(max_y, height-1);

    //For each pixel, add triangle index to list
    for(int x=min_x;x<=max_x;x++){
        for(int y=min_y;y<=max_y;y++){
            float px_center = x+0.5;
            float py_center = y+0.5;
            // Check if pixel center is inside triangle while...
            // computing screen space barycentric coordinates of pixel center
            float w_screen[3];
            bool is_inside = Barycentric_Coordinates_2D(w_screen, px_center, py_center,
                                 verts_screen[i*3], verts_screen[i*3+1],
                                 verts_screen[j*3], verts_screen[j*3+1],
                                 verts_screen[k*3], verts_screen[k*3+1]);
            if(!is_inside) continue;
            int position_idx = atomicAdd(&(bins_idx[height*x + y]), 1);
            if(position_idx>=max_triangles_per_bin) continue;

            int bin_array_idx = max_triangles_per_bin*height*x + max_triangles_per_bin*y + position_idx;
            bins2tri[bin_array_idx] = f;

            // Compute distance to triangle
            float dist = w_screen[0]*verts_screen[i*3+2] +
                         w_screen[1]*verts_screen[j*3+2] +
                         w_screen[2]*verts_screen[k*3+2];
            bins2z[bin_array_idx] = (int)(dist * 10000);
        }
    }
}

__global__ void Rasterize_Triangle_Mesh(float *verts_screen, int *faces,
                             int *bins2tri, int *bins2z, int *bins_idx,
                             float *dists, float *alphas, int *vert_indices, int *params)
{
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    //Parameters
    int max_triangles_per_bin = params[0];
    int height = params[1];
  
    // Pixel 2D indices
    int x = idx / height;
    int y = idx % height;

    int i, j, k; //vertex indices of face

    // Do nothing if no triangles
    int bins2tri_idx = max_triangles_per_bin*height*x + max_triangles_per_bin*y;
    if(bins2tri[bins2tri_idx]<0) return;

    // Determine which triangle is closest
    int max_idx = max(bins_idx[idx], max_triangles_per_bin);
    for(int f_idx=0;f_idx<max_idx;f_idx++){
        int zdist = bins2z[bins2tri_idx + f_idx];
        if(zdist> dists[idx] || dists[idx] == 0){

        dists[idx] = zdist;
        int f = bins2tri[bins2tri_idx + f_idx];
        i = faces[f*3];
        j = faces[f*3+1];
        k = faces[f*3+2];
        
        float px_center = x+0.5;
        float py_center = y+0.5;
        // Check if pixel center is inside triangle while...
        // computing screen space barycentric coordinates of pixel center
        float w_screen[3];
        bool is_inside = Barycentric_Coordinates_2D(w_screen, px_center, py_center,
                             verts_screen[i*3], verts_screen[i*3+1],
                             verts_screen[j*3], verts_screen[j*3+1],
                             verts_screen[k*3], verts_screen[k*3+1]);

        // Compute world barycentric coordinates of pixel center
        float z0 = verts_screen[i*3+2];
        float z1 = verts_screen[j*3+2];
        float z2 = verts_screen[k*3+2];
        float alpha0 = z1*z2*w_screen[0];
        float alpha1 = z0*z2*w_screen[1];
        float alpha2 = z0*z1*w_screen[2];

        int raster_idx = 3*height*x + 3*y;

        // Check counterclockwise ordering!!
        float triangle_area = Triangle_Area_2D(verts_screen[i*3], verts_screen[i*3+1],
                                               verts_screen[j*3], verts_screen[j*3+1],
                                               verts_screen[k*3], verts_screen[k*3+1]);
        if(triangle_area<0){
            int temp = j;
            j = k;
            k = temp;
            float alpha_temp = alpha1;
            alpha1 = alpha2;
            alpha2 = alpha_temp;
        }
        triangle_area = Triangle_Area_2D(verts_screen[i*3], verts_screen[i*3+1],
                                         verts_screen[j*3], verts_screen[j*3+1],
                                         verts_screen[k*3], verts_screen[k*3+1]);
        //assert(triangle_area>=0);

        // Save alphas for gradient
        alphas[raster_idx] = alpha0;
        alphas[raster_idx+1] = alpha1;
        alphas[raster_idx+2] = alpha2;

        // Save triangle vertex indices per pixel for gradient
        vert_indices[raster_idx] = i;
        vert_indices[raster_idx+1] = j;
        vert_indices[raster_idx+2] = k;
        }
    }
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

__global__ void Rasterize_Triangle_Mesh_Gradients(float *verts_screen, int *vert_indices,
                                                  float *grad_alphas, float *grad_verts_screen,
                                                  int *params)
{
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    //Parameters
    int height = params[0];
  
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
    //assert(triangle_area>0);

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

__global__ void Pre_Rasterize_Triangle_Mesh_Gradients(float *verts_screen, int *vert_indices,
                                                      float *grad_alphas, int *grad_alphas_idx,
                                                      int *params)
{
    unsigned int tx = threadIdx.x;
    // Block id in a 1D grid
    unsigned int ty = blockIdx.x;
    // Block width, i.e. number of threads per block
    unsigned int bw = blockDim.x;
    // Compute flattened index inside the array
    unsigned int idx = tx + ty * bw;

    //Parameters
    int height = params[0];

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
    //assert(triangle_area>0);

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

            // Add vert idx
            grad_alphas_idx[18*idx+6*m+3*n]=3*idx1;
            grad_alphas_idx[18*idx+6*m+3*n+1]=3*idx1+1;
            grad_alphas_idx[18*idx+6*m+3*n+2]=3*idx1+2;

            // Add gradients to output
            grad_alphas[18*idx+6*m+3*n]=d_alpha_v[0];
            grad_alphas[18*idx+6*m+3*n+1]=d_alpha_v[1];
            grad_alphas[18*idx+6*m+3*n+2]=d_alpha_v[2];
        }
    }
}


}
""", no_extern_c=True)


if __name__=="__main__":
    import numpy as np

    raster = mod.get_function("Rasterize_Triangle_Mesh")
