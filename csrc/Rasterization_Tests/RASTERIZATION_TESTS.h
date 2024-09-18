//#####################################################################
// Copyright 2022, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class RASTERIZATION_TESTS
//#####################################################################
//#####################################################################
#ifndef __RASTERIZATION_TESTS__
#define __RASTERIZATION_TESTS__

#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>
#include <vector>

namespace PhysBAM{

template<class T, class RW>
class RASTERIZATION_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Read_Txt
//#####################################################################
int Read_Txt(std::string fname, float* &data){
    std::string line;
    std::ifstream file(fname);
    if(file.is_open()){
        getline(file,line);
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        int size = std::stoi(words[1]);
        std::cout << "number of elements: " << size << std::endl;
        data = new float[size*3];
        
        for (int i = 0; i < size; i++) { // Read vertices
            getline(file,line);
            std::istringstream iss(line);
            float a,b,c;
            if(!(iss >> a >> b >> c)) break;
            data[i*3] = a;
            data[i*3+1] = b;
            data[i*3+2] = c;
        }
        file.close();
        return size;
    }
    else {
        file.close();
        std::cout << "could not load file: " << fname << std::endl;
        return -1;
    }
}

//#####################################################################
// Function Rasterize_Example
//#####################################################################
void Rasterize_Example()
{
    std::string verts_world_fname = "/data/jwu/raster_data/verts_camera_179.txt";
    std::string verts_screen_fname = "/data/jwu/raster_data/verts_screen_179.txt";
    std::string faces_fname = "/data/jwu/raster_data/faces.txt";
    std::string verts_norms_fname = "/data/jwu/raster_data/verts_norms.txt";

    float *verts_world, *verts_screen, *vert_norms, *faces;
    int num_vertices = Read_Txt(verts_world_fname, verts_world);
    Read_Txt(verts_screen_fname, verts_screen);
    Read_Txt(verts_norms_fname, vert_norms);
    int num_faces = Read_Txt(faces_fname, faces);

    Rasterize_Triangle_Mesh(verts_screen, verts_world, vert_norms,
                            faces, num_vertices, num_faces);

}

//#####################################################################
// Function Backprop_Rasterize_Example
//#####################################################################
void Backprop_Rasterize_Example()
{
    std::string verts_screen_fname = "/data/jwu/raster_data/ellipse_data/verts_screen.txt";
    std::string pix_vert_indices_fname = "/data/jwu/raster_data/ellipse_data/pix_vert_indices.txt";
    std::string grad_alphas_fname = "/data/jwu/raster_data/ellipse_data/grad_alphas.txt";

    float *verts_screen, *pix_vert_indices, *grad_alphas;
    int num_vertices = Read_Txt(verts_screen_fname, verts_screen);

    Read_Txt(pix_vert_indices_fname, pix_vert_indices);
    int num_pixels = sqrt((float) Read_Txt(grad_alphas_fname, grad_alphas));

    Rasterize_Triangle_Mesh_Gradients(verts_screen, pix_vert_indices, grad_alphas,
                                      num_vertices, num_pixels);

}

//#####################################################################
// Function Bin_Triangles
//#####################################################################
void Bin_Triangles(float *verts_screen, float *faces, std::vector<uint32_t> *bins,
                   int num_faces, int max_faces_per_bin=512, int width=512, int height=512)
{
    //Compute bin size based on image dimensions
    int max_image_dim = max(width, height);
    int bin_size = int(pow(2, max(ceil(log2(max_image_dim)) - 4, 4.)));
    std::cout<<"Bin size: "<<bin_size<<std::endl;

    int num_bins_per_dim = ceil(max_image_dim/((float) bin_size));
    std::cout<<"Num bins: "<<num_bins_per_dim<<std::endl;

    //Initialize bins
    //std::vector<uint32_t> bins[num_bins_per_dim*num_bins_per_dim];

    int i,j,k;
    int min_x, min_y, max_x, max_y;
    int min_bx, min_by, max_bx, max_by;
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
        min_x = max(min_x, 0);
        min_y = max(min_y, 0);
        max_x = min(max_x, width-1);
        max_y = min(max_y, height-1);

        //Which bins?
        min_bx = min_x / bin_size;
        min_by = min_y / bin_size;
        max_bx = min(max_x / bin_size, num_bins_per_dim-1);
        max_by = min(max_y / bin_size, num_bins_per_dim-1);

        //For each bin, add triangle index to list
        for(int bx=min_bx;bx<=max_bx;bx++)
            for(int by=min_by;by<=max_by;by++)
                bins[bx*num_bins_per_dim + by].push_back(f);
    }
}

//#####################################################################
// Function Norm_3D
//#####################################################################
float Norm_3D(float x, float y, float z)
{
    return sqrt(pow(x,2) + pow(y,2) + pow(z,2));
}

//#####################################################################
// Function Triangle_Area
//#####################################################################
float Triangle_Area_2D(float x0, float y0, float x1, float y1, float x2, float y2)
{
    float u[] = {x1-x0,y1-y0};
    float v[] = {x2-x0,y2-y0};
    float cross = u[0]*v[1] - v[0]*u[1];
    return cross / 2.;
}

//#####################################################################
// Function Barycentric_Coordinates_2D
//#####################################################################
bool Barycentric_Coordinates_2D(float *weights, float px, float py,
                                float x0, float y0, float x1, float y1,
                                float x2, float y2)
{
    float total_area = Triangle_Area_2D(x0, y0, x1, y1, x2, y2);
    std::cout<<"Total area: "<<total_area<<std::endl;
    if(total_area == 0) return false;

    weights[0] = Triangle_Area_2D(px, py, x1, y1, x2, y2) / total_area;
    if(weights[0] < 0) return false;

    weights[1] = Triangle_Area_2D(x0, y0, px, py, x2, y2) / total_area;
    if(weights[1] < 0 || weights[0]+weights[1] > 1) return false;

    weights[2] = 1 - weights[0] - weights[1]; //Triangle_Area_2D(x0, y0, x1, y1, px, py) / total_area;
    if(weights[2] < 0) return false;
    return true;
}

//#####################################################################
// Function Rasterize_Triangle_Mesh
//#####################################################################
void Rasterize_Triangle_Mesh(float *verts_screen, float *verts_world, float *vert_norms,
                             float *faces, int num_vertices, int num_faces,
                             int width=512, int height=512)
{
    //Initialize pixels
    float dists[width][height] = {0};
    float raster_out[512][512][3] = {0};
    //float intersections[512][512][3] = {0};
    int intersecting_triangles[512][512][3] = {-1};

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
        min_x = max(min_x, 0);
        min_y = max(min_y, 0);
        max_x = min(max_x, width-1);
        max_y = min(max_y, height-1);

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

                    // Store vertices of intersecting face
                    intersecting_triangles[x][y][0] = i;
                    intersecting_triangles[x][y][1] = j;
                    intersecting_triangles[x][y][2] = k;
                    /*// Store alphas
                    alphas[x][y][0] = alpha0;
                    alphas[x][y][1] = alpha1;
                    alphas[x][y][2] = alpha2;*/

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

                    /*// Also store world space intersection
                    intersections[x][y][0] = alpha0*verts_world[i*3] + alpha1*verts_world[j*3] + alpha2*verts_world[k*3];
                    intersections[x][y][1] = alpha0*verts_world[i*3+1] + alpha1*verts_world[j*3+1] + alpha2*verts_world[k*3+1];
                    intersections[x][y][2] = alpha0*verts_world[i*3+2] + alpha1*verts_world[j*3+2] + alpha2*verts_world[k*3+2];*/

                }
        }}
    }

    // Save offsets
    auto array_to_flat_vec=[](float (*array)[512][3]){
        int width=512;
        int height=512;
        std::vector<T> vec(width*height*3);size_t cnt=0;
        for(int i=0;i<width;i++){
            for(int j=0;j<height;j++){
                for(int k=0;k<3;k++)vec[cnt++]=array[i][j][k];
            }
        }
        return vec;
    };
    
    auto output_vec=array_to_flat_vec(raster_out);
    cnpy::npy_save("raster_out_179.npy",&output_vec[0],{width,height,3},"w");
    //output_vec=array_to_flat_vec(intersections);
    //cnpy::npy_save("world_intersections_0.npy",&output_vec[0],{width,height,3},"w");

}

//#####################################################################
// Function Rasterize_Triangle_Mesh_Bins
//#####################################################################
void Rasterize_Triangle_Mesh_Bins(float *verts_screen, float *verts_world, float *vert_norms,
                                  float *faces, std::vector<uint32_t> *bins, int num_bins, int num_vertices, int num_faces,
                                  int width=512, int height=512)
{
    //Initialize pixels
    float dists[width][height] = {0};
    float raster_out[512][512][3] = {0};
    float intersections[512][512][3] = {0};

    //Assumes vertices are already in screen space
    int i, j, k; //vertex indices of face
    int min_x, min_y, max_x, max_y;
    uint32_t f;
    for(int b=0;b<num_bins;b++){
    for(int bf=0;bf<bins[b].size();bf++){
        f = bins[b][bf];
        i = faces[f*3];
        j = faces[f*3+1];
        k = faces[f*3+2];

        //Compute triangle bbox
        min_x = floor(std::min({verts_screen[i*3],verts_screen[j*3],verts_screen[k*3]}));
        max_x = ceil(std::max({verts_screen[i*3],verts_screen[j*3],verts_screen[k*3]}));
        min_y = floor(std::min({verts_screen[i*3+1],verts_screen[j*3+1],verts_screen[k*3+1]}));
        max_y = ceil(std::max({verts_screen[i*3+1],verts_screen[j*3+1],verts_screen[k*3+1]}));

        //Clip against screen bounds
        min_x = max(min_x, 0);
        min_y = max(min_y, 0);
        max_x = min(max_x, width-1);
        max_y = min(max_y, height-1);

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

    // Save offsets
    auto array_to_flat_vec=[](float (*array)[512][3]){
        int width=512;
        int height=512;
        std::vector<T> vec(width*height*3);size_t cnt=0;
        for(int i=0;i<width;i++){
            for(int j=0;j<height;j++){
                for(int k=0;k<3;k++)vec[cnt++]=array[i][j][k];
            }
        }
        return vec;
    };
    
    auto output_vec=array_to_flat_vec(raster_out);
    cnpy::npy_save("raster_out_179.npy",&output_vec[0],{width,height,3},"w");
    //output_vec=array_to_flat_vec(intersections);
    //cnpy::npy_save("world_intersections_0.npy",&output_vec[0],{width,height,3},"w");

}

//#####################################################################
// Function Grad_Alpha
//#####################################################################
// This is the gradient of alpha w.r.t. v1 or v2 (negate_sign).
void Grad_Alpha(float x1, float y1, float z1,
                float x2, float y2, float z2, 
                float px, float py, bool negate_sign, float* result)
{
    int sign = 1;
    if(negate_sign) sign = -1;

    result[0]=z1*z2*(y2-py)*sign/2.;
    result[1]=z1*z2*(px-x2)*sign/2.;
    result[2]=z2*Triangle_Area_2D(px, py, x1, y1, x2, y2)*sign;
}

//#####################################################################
// Function Rasterize_Triangle_Mesh_Gradients
//#####################################################################
void Rasterize_Triangle_Mesh_Gradients(float *verts_screen, float *vert_indices,
                                       float *grad_alphas,
                                       int num_verts, int height)
{
    // Initialize output
    float grad_verts_screen[256*256*3] = {0};
    std::cout<<"Image height: "<<height<<std::endl;
    for(int x=0;x<height;x++){
    for(int y=0;y<height;y++){
    // Pixel 2D indices
    //int x = idx / height;
    //int y = idx % height;

    // Vertex indices of the face intersecting with this pixel
    int raster_idx = 3*height*x + 3*y;
    int i = vert_indices[raster_idx];
    int j = vert_indices[raster_idx+1];
    int k = vert_indices[raster_idx+2];
    float triangle_area = Triangle_Area_2D(verts_screen[i*3], verts_screen[i*3+1],
                                           verts_screen[j*3], verts_screen[j*3+1],
                                           verts_screen[k*3], verts_screen[k*3+1]);

    // For each alpha, compute d alpha / d v_i
    int idx1, idx2;
    bool negate_sign;
    int other_verts[2] = {-1};
    float d_alpha_v[3]={0};
    for(int m=0;m<3;m++){
        if(grad_alphas[raster_idx+m] == 0) continue;

        if(triangle_area > 0)
        std::cout<<"Triangle 2D area: "<<triangle_area<<std::endl;
        // Figure out other two vertices...
        switch(m){
            case 0:
                other_verts[0]=j;
                other_verts[1]=k;
                break;
            case 1:
                other_verts[0]=k;
                other_verts[1]=i;
                break;
            case 2:
                other_verts[0]=i;
                other_verts[1]=j;
                break;
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
                       x, y, negate_sign, d_alpha_v);

            //std::cout<<"d_alpha_v: "<<d_alpha_v[0]<<", "<<d_alpha_v[1]<<", "<<d_alpha_v[2]<<std::endl;
            // Add gradients to output
            grad_verts_screen[idx1*3] += grad_alphas[raster_idx+m]*d_alpha_v[0];
            grad_verts_screen[idx1*3+1] += grad_alphas[raster_idx+m]*d_alpha_v[1];
            grad_verts_screen[idx1*3+2] += grad_alphas[raster_idx+m]*d_alpha_v[2];
        }

    }

    }}
}

//#####################################################################
};
}
#endif
