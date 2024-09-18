//#####################################################################
// Copyright 2021, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class SKIN_TESTS
//#####################################################################
//#####################################################################
#ifndef __SKIN_TESTS__
#define __SKIN_TESTS__

#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include "../External/cnpy.h"
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class SKIN_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Interpolate_Skinning_Weights_From_Scan_Mesh
//#####################################################################
void Interpolate_Skinning_Weights_From_Scan_Mesh(const std::string tri_path,const std::string weights_output_path)
{
    std::string scan_path="/data/jwu/D_march/meshes_starpose/star_mesh_D_march_101.tri.gz";
    std::string skinning_weights_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/weights_smpl_flat.npy";

    TRIANGULATED_SURFACE<T>& scan_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),scan_path,scan_surface);

    scan_surface.Update_Triangle_List();
    scan_surface.Initialize_Hierarchy();
    scan_surface.mesh.Initialize_Node_On_Boundary();
    scan_surface.mesh.Initialize_Neighbor_Nodes();
    scan_surface.mesh.Initialize_Incident_Elements();

    cnpy::NpyArray arr=cnpy::npy_load(skinning_weights_path);
    std::vector<float> loaded_data=arr.as_vec<float>();
    std::cout <<"total skinning weights = "<<arr.shape[0]<<std::endl;
    std::cout<<"second weight = "<<loaded_data[1]<<std::endl;

    int num_weights=arr.shape[0];
    ARRAY<T> scan_skinning_weights(num_weights);
    for(int i=0;i<num_weights;i++)
        scan_skinning_weights(i+1)=loaded_data[i];

    TRIANGULATED_SURFACE<T>& triangulated_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);
    triangulated_surface.Update_Triangle_List();
    std::cout<<"Output surface particles = "<<triangulated_surface.particles.array_collection->Size()<<std::endl;
    ARRAY<T> new_skinning_weights(triangulated_surface.particles.array_collection->Size()*24);
    for(int i=1;i<=triangulated_surface.particles.array_collection->Size();i++){
        int t;TV closest_point=scan_surface.Surface(triangulated_surface.particles.X(i),(T)1e-1,0,&t);
        TRIANGLE_3D<T> triangle=scan_surface.Get_Element(t);
        TV weights=triangle.Barycentric_Coordinates(closest_point);
        TV_INT parents=scan_surface.mesh.elements(t);
        for(int j=1;j<=24;j++){ //Interpolate each skinning weight
            new_skinning_weights((i-1)*24+j)=weights(1)*scan_skinning_weights((parents(1)-1)*24+j) +
                                             weights(2)*scan_skinning_weights((parents(2)-1)*24+j) +
                                             weights(3)*scan_skinning_weights((parents(3)-1)*24+j);
        }
    }
    // Save offsets
    auto array_to_flat_vec=[](const ARRAY<T>& array){
        std::vector<RW> vec(array.Size());size_t cnt=0;
        for(int i=1;i<=array.Size();i++){
            vec[cnt++]=array(i);
        }
        return vec;
    };
    
    auto weights_vec=array_to_flat_vec(new_skinning_weights);
    cnpy::npy_save(weights_output_path,&weights_vec[0],{new_skinning_weights.m},"w");

}

//#####################################################################
// Function Interpolate_Skinning_Weights_From_Tet_Mesh
//#####################################################################
void Interpolate_Skinning_Weights_From_Tet_Mesh(const std::string tri_path,const std::string weights_output_path)
{
    std::string tet_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/star_shell.tet.gz";
    std::string skinning_weights_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/coarseweights_flat.npy";

    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Initialize_Hierarchy();
    tetrahedralized_volume.Initialize_Triangulated_Surface();
    tetrahedralized_volume.triangulated_surface->Initialize_Hierarchy();

    TRIANGULATED_SURFACE<T>& boundary_surface=tetrahedralized_volume.Get_Boundary_Object();
    boundary_surface.Update_Triangle_List();

    cnpy::NpyArray arr=cnpy::npy_load(skinning_weights_path);
    std::vector<float> loaded_data=arr.as_vec<float>();
    std::cout <<"total skinning weights = "<<arr.shape[0]<<std::endl;
    std::cout<<"second weight = "<<loaded_data[1]<<std::endl;

    int num_weights=arr.shape[0];
    ARRAY<T> tet_skinning_weights(num_weights);
    for(int i=0;i<num_weights;i++)
        tet_skinning_weights(i+1)=loaded_data[i];

    TRIANGULATED_SURFACE<T>& triangulated_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);
    triangulated_surface.Update_Triangle_List();
    std::cout<<"Output surface particles = "<<triangulated_surface.particles.array_collection->Size()<<std::endl;
    ARRAY<T> new_skinning_weights(triangulated_surface.particles.array_collection->Size()*24);
    const GEOMETRY_PARTICLES<TV> &particles=triangulated_surface.particles;
    for(int i=1;i<=particles.array_collection->Size();i++){
        bool inside_success=tetrahedralized_volume.Inside(particles.X(i));
        if(inside_success){
            ARRAY<int> intersection_list;
            tetrahedralized_volume.hierarchy->Intersection_List(particles.X(i),intersection_list);
            for(int j=1;j<=intersection_list.m;j++){
                int t=intersection_list(j);
                TETRAHEDRON<T> tetrahedron=tetrahedralized_volume.Get_Element(t);;
                if(tetrahedron.Inside(particles.X(i))){
                    VECTOR<int,4> parents=tetrahedralized_volume.mesh.elements(t);
                    VECTOR<T,4> weights=tetrahedron.Barycentric_Coordinates(particles.X(i));
                    for(int j=1;j<=24;j++){ //Interpolate each skinning weight
                    new_skinning_weights((i-1)*24+j)=weights(1)*tet_skinning_weights((parents(1)-1)*24+j) +
                                                     weights(2)*tet_skinning_weights((parents(2)-1)*24+j) +
                                                     weights(3)*tet_skinning_weights((parents(3)-1)*24+j) +
                                                     weights(4)*tet_skinning_weights((parents(4)-1)*24+j);
                     }
                     break;
                }
            }
        }
        else{
            int t;TV closest_point=boundary_surface.Surface(triangulated_surface.particles.X(i),(T)1e-1,0,&t);
            TRIANGLE_3D<T> triangle=boundary_surface.Get_Element(t);
            TV weights=triangle.Barycentric_Coordinates(closest_point);
            TV_INT parents=boundary_surface.mesh.elements(t);
            for(int j=1;j<=24;j++){ //Interpolate each skinning weight
                new_skinning_weights((i-1)*24+j)=weights(1)*tet_skinning_weights((parents(1)-1)*24+j) +
                                                 weights(2)*tet_skinning_weights((parents(2)-1)*24+j) +
                                                 weights(3)*tet_skinning_weights((parents(3)-1)*24+j);
            }
        }
    }
    // Save offsets
    auto array_to_flat_vec=[](const ARRAY<T>& array){
        std::vector<RW> vec(array.Size());size_t cnt=0;
        for(int i=1;i<=array.Size();i++){
            vec[cnt++]=array(i);
        }
        return vec;
    };
    
    auto weights_vec=array_to_flat_vec(new_skinning_weights);
    cnpy::npy_save(weights_output_path,&weights_vec[0],{new_skinning_weights.m},"w");

}

//#####################################################################
// Function Closest_Skinning_Weights
//#####################################################################
void Closest_Skinning_Weights(const std::string tri_path,const std::string weights_output_path)
{
    std::string scan_path="/data/jwu/D_march/meshes_starpose/star_mesh_D_march_101.tri.gz";
    std::string skinning_weights_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/weights_smpl_flat.npy";

    TRIANGULATED_SURFACE<T>& scan_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),scan_path,scan_surface);

    scan_surface.Update_Triangle_List();
    scan_surface.Initialize_Hierarchy();
    scan_surface.mesh.Initialize_Node_On_Boundary();
    scan_surface.mesh.Initialize_Neighbor_Nodes();
    scan_surface.mesh.Initialize_Incident_Elements();

    cnpy::NpyArray arr=cnpy::npy_load(skinning_weights_path);
    std::vector<float> loaded_data=arr.as_vec<float>();
    std::cout <<"total skinning weights = "<<arr.shape[0]<<std::endl;
    std::cout<<"second weight = "<<loaded_data[1]<<std::endl;

    int num_weights=arr.shape[0];
    ARRAY<T> scan_skinning_weights(num_weights);
    for(int i=0;i<num_weights;i++)
        scan_skinning_weights(i+1)=loaded_data[i];

    TRIANGULATED_SURFACE<T>& triangulated_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);
    triangulated_surface.Update_Triangle_List();
    std::cout<<"Output surface particles = "<<triangulated_surface.particles.array_collection->Size()<<std::endl;
    ARRAY<T> new_skinning_weights(triangulated_surface.particles.array_collection->Size()*24);
    for(int i=1;i<=triangulated_surface.particles.array_collection->Size();i++){
        TV target_point=triangulated_surface.particles.X(i);
        // Find closest vertex in scan mesh
        T closest_distance=(T)0.;
        bool closest_distance_set=false;
        int closest_particle=0;
        for(int j=1;j<=scan_surface.particles.array_collection->Size();j++){
            T distance=(scan_surface.particles.X(j)-target_point).Magnitude();
            if(distance<closest_distance||!closest_distance_set){
                closest_distance_set=true;
                closest_distance=distance;
                closest_particle=j;}
        }
        for(int k=1;k<=24;k++)
            new_skinning_weights((i-1)*24+k)=scan_skinning_weights((closest_particle-1)*24+k);
    }
    // Save offsets
    auto array_to_flat_vec=[](const ARRAY<T>& array){
        std::vector<RW> vec(array.Size());size_t cnt=0;
        for(int i=1;i<=array.Size();i++){
            vec[cnt++]=array(i);
        }
        return vec;
    };
    
    auto weights_vec=array_to_flat_vec(new_skinning_weights);
    cnpy::npy_save(weights_output_path,&weights_vec[0],{new_skinning_weights.m},"w");

}
//#####################################################################
};
}
#endif
