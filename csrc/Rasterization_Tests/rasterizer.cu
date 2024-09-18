#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

inline __host__ __device__ float Norm_3D(float x, float y, float z)
{
    return sqrt(pow(x,2) + pow(y,2) + pow(z,2));
}

__global__ float Triangle_Area_2D(float x0, float y0, float x1, float y1, float x2, float y2)
{
    float u[] = {x1-x0,y1-y0};
    float v[] = {x2-x0,y2-y0};
    float cross = u[0]*v[1] - v[0]*u[1];
    return cross / 2.;
}

__global__ bool Barycentric_Coordinates_2D(float *weights, float px, float py,
                                float x0, float y0, float x1, float y1,
                                float x2, float y2)
{
    float total_area = Triangle_Area_2D(x0, y0, x1, y1, x2, y2);
    weights[0] = Triangle_Area_2D(px, py, x1, y1, x2, y2) / total_area;
    if(weights[0] < 0) return false;

    weights[1] = Triangle_Area_2D(x0, y0, px, py, x2, y2) / total_area;
    if(weights[1] < 0 || weights[0]+weights[1] > 1) return false;

    weights[2] = 1 - weights[0] - weights[1]; //Triangle_Area_2D(x0, y0, x1, y1, px, py) / total_area;
    if(weights[2] < 0) return false;
    return true;
}

__global__ void Rasterize_Triangle_Mesh(float *verts_screen, float *verts_world, float *vert_norms,
                             float *faces, float *output, int num_vertices, int num_faces,
                             int width=512, int height=512)
{
    //Initialize pixels
    float dists[width][height] = {0};
    float raster_out[512][512][3] = {0};
    float intersections[512][512][3] = {0};

    //Assumes vertices are already in screen space
    int i, j, k; //vertex indices of face
    int min_x, min_y, max_x, max_y;
    for(int f=0;f<num_faces;f++){
        i = faces[f*3];
        j = faces[f*3+1];
        k = faces[f*3+2];

        //Compute triangle bbox
        min_x = floor(std::min({verts_screen[i*3],verts_screen[j*3],verts_screen[k*3]}));
        max_x = ceil(std::max({verts_screen[i*3],verts_screen[j*3],verts_screen[k*3]}));
        min_y = floor(std::min({verts_screen[i*3+1],verts_screen[j*3+1],verts_screen[k*3+1]}));
        max_y = ceil(std::max({verts_screen[i*3+1],verts_screen[j*3+1],verts_screen[k*3+1]}));

        //Clip against screen bounds
        min_x = std::max(min_x, 0);
        min_y = std::max(min_y, 0);
        max_x = std::min(max_x, width-1);
        max_y = std::min(max_y, height-1);

        //For each pixel, check if this is the nearest triangle
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

                // Compute distance to triangle
                float dist = w_screen[0]*verts_screen[i*3+2] +
                             w_screen[1]*verts_screen[j*3+2] +
                             w_screen[2]*verts_screen[k*3+2];

                // Update raster_out if curr_dist < dists[x][y]
                if(dist>dists[x][y] || dists[x][y] == 0){
                    dists[x][y] = dist;

                    // Compute world barycentric coordinates of pixel center
                    float z0 = verts_world[i*3+2];
                    float z1 = verts_world[j*3+2];
                    float z2 = verts_world[k*3+2];
                    float alpha0 = z1*z2*w_screen[0];
                    float alpha1 = z0*z2*w_screen[1];
                    float alpha2 = z0*z1*w_screen[2];
                    float denom = alpha0+alpha1+alpha2;
                    alpha0 = alpha0 / denom;
                    alpha1 = alpha1 / denom;
                    alpha2 = alpha2 / denom;

                    raster_out[x][y][0] = alpha0*vert_norms[i*3] + alpha1*vert_norms[j*3] + alpha2*vert_norms[k*3];
                    raster_out[x][y][1] = alpha0*vert_norms[i*3+1] + alpha1*vert_norms[j*3+1] + alpha2*vert_norms[k*3+1];
                    raster_out[x][y][2] = alpha0*vert_norms[i*3+2] + alpha1*vert_norms[j*3+2] + alpha2*vert_norms[k*3+2];
                    // Normalize
                    float norm = Norm_3D(raster_out[x][y][0], raster_out[x][y][1], raster_out[x][y][2]);
                    raster_out[x][y][0] = raster_out[x][y][0] / norm;
                    raster_out[x][y][1] = raster_out[x][y][1] / norm;
                    raster_out[x][y][2] = raster_out[x][y][2] / norm;

                    // Also store world space intersection
                    intersections[x][y][0] = alpha0*verts_world[i*3] + alpha1*verts_world[j*3] + alpha2*verts_world[k*3];
                    intersections[x][y][1] = alpha0*verts_world[i*3+1] + alpha1*verts_world[j*3+1] + alpha2*verts_world[k*3+1];
                    intersections[x][y][2] = alpha0*verts_world[i*3+2] + alpha1*verts_world[j*3+2] + alpha2*verts_world[k*3+2];

                }
        }}
    }
}

}
