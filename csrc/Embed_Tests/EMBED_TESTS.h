//#####################################################################
// Copyright 2020, Jane Wu, Zhenglin Geng.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class EMBED_TESTS
//#####################################################################
//#####################################################################
#ifndef __EMBED_TESTS__
#define __EMBED_TESTS__

#include <PhysBAM_Geometry/Spatial_Acceleration/TETRAHEDRON_HIERARCHY.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include "../External/cnpy.h"
//#include "Utils/DEBUG_VIZ_UTIL.h"
//#include "../opengl_3d_proto/Proto/PROTO_DEBUG_UTILS.h"
#include <iostream>
#include <fstream>

namespace PhysBAM{

template<class T, class RW>
class EMBED_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Compute_Deformations
//#####################################################################
void Compute_Deformations(const std::string gt_surface_path,const std::string embedded_surface_path,const std::string output_path)
{
    TRIANGULATED_SURFACE<T>* gt_surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(gt_surface_path,*gt_surface);
    gt_surface->Initialize_Segment_Lengths();
    
    TRIANGULATED_SURFACE<T>* emb_surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(embedded_surface_path,*emb_surface);
    emb_surface->Initialize_Segment_Lengths();

    int n_edges = emb_surface->segment_lengths->Size();
    ARRAY<T> old_segments = *gt_surface->segment_lengths;
    ARRAY<T> new_segments = *emb_surface->segment_lengths;
    ARRAY<T> deformations(n_edges);

    std::ofstream file;
    file.open(output_path);
    for(int i=1;i<=n_edges;i++){
        file<<new_segments(i)/old_segments(i);
        if(i<n_edges)file<<"\n";
    }
    file.close();
}

//#####################################################################
// Function Apply_Embedding
//#####################################################################
void Apply_Embedding(const std::string ref_tet_path,const std::string ref_surface_path,const std::string embedding_path,const std::string offset_path)
{
    std::cout << embedding_path << std::endl;
    TETRAHEDRALIZED_VOLUME<T>* tet=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(ref_tet_path,*tet);
    tet->Initialize_Hierarchy();
    tet->Initialize_Triangulated_Surface();
    tet->triangulated_surface->Initialize_Hierarchy();

    TRIANGULATED_SURFACE<T>* surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(ref_surface_path,*surface);

    ARRAY<ARRAY<int> > p;
    ARRAY<ARRAY<T> > w;
    FILE_UTILITIES::Read_From_File<RW>(embedding_path,p,w);

    // Compute vertex positions from embeddings
    int n_vertices=surface->particles.array_collection->Size();
    ARRAY<TV> vertices(n_vertices);
    ARRAY<TV> offsets(n_vertices);
    for(int i=1;i<=n_vertices;i++){
        TV vertex;
        vertex(1)=w(i)(1)*tet->particles.X(p(i)(1)).x+w(i)(2)*tet->particles.X(p(i)(2)).x+w(i)(3)*tet->particles.X(p(i)(3)).x+w(i)(4)*tet->particles.X(p(i)(4)).x;
        vertex(2)=w(i)(1)*tet->particles.X(p(i)(1)).y+w(i)(2)*tet->particles.X(p(i)(2)).y+w(i)(3)*tet->particles.X(p(i)(3)).y+w(i)(4)*tet->particles.X(p(i)(4)).y;
        vertex(3)=w(i)(1)*tet->particles.X(p(i)(1)).z+w(i)(2)*tet->particles.X(p(i)(2)).z+w(i)(3)*tet->particles.X(p(i)(3)).z+w(i)(4)*tet->particles.X(p(i)(4)).z;
        vertices(i)=vertex;
        TV offset;
        offset(1)=vertex(1)-surface->particles.X(i)(1);
        offset(2)=vertex(2)-surface->particles.X(i)(2);
        offset(3)=vertex(3)-surface->particles.X(i)(3);
        offsets(i)=offset;
    }

    // Save offsets
    auto array_to_flat_vec=[](const ARRAY<TV>& array){
        std::vector<T> vec(array.Size()*3);size_t cnt=0;
        for(int i=1;i<=array.Size();i++){
            for(int d=1;d<=3;d++)vec[cnt++]=array(i)(d);
        }
        return vec;
    };
    
    auto offsets_vec=array_to_flat_vec(offsets);
 
    //cnpy::npy_save(offset_path,&offsets_vec[0],{offsets.Size(),3},"w");

    // Generate cloth from embedding
    TRIANGULATED_SURFACE<T>* out_surface=TRIANGULATED_SURFACE<T>::Create();
    out_surface->mesh.elements=surface->mesh.elements;
    out_surface->particles.array_collection->Add_Elements(vertices.m);
    out_surface->particles.X=vertices;
    out_surface->Update_Number_Nodes();
    FILE_UTILITIES::Write_To_File<RW>(offset_path,*out_surface);
}

//#####################################################################
// Function Compute_Errors
//#####################################################################
void Compute_Errors(const std::string tet_path,const std::string surface_path,const std::string output_path)
{
    std::cout << tet_path << std::endl;
    std::cout << surface_path << std::endl;
    TETRAHEDRALIZED_VOLUME<T>* tet=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tet_path,*tet);
    tet->Initialize_Hierarchy();
    tet->Initialize_Triangulated_Surface();
    tet->triangulated_surface->Initialize_Hierarchy();

    TRIANGULATED_SURFACE<T>* surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(surface_path,*surface);
    Initialize_Embedding_Errors(*tet,surface->particles,output_path);
}

//#####################################################################
// Function Initialize_Embedding_Errors
//#####################################################################
void Initialize_Embedding_Errors(TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume,const GEOMETRY_PARTICLES<TV>& particles,const std::string output_path)
{
    TRIANGULATED_SURFACE<T>& triangulated_surface=tetrahedralized_volume.Get_Boundary_Object();
    triangulated_surface.Update_Triangle_List();
    std::ofstream file;
    file.open(output_path);

    for(int i=1;i<=particles.array_collection->Size();i++){
        if(!tetrahedralized_volume.Inside(particles.X(i))){
            file<<i-1;
            if(i<particles.array_collection->Size())file<<"\n";
        }
    }
    file.close();
}

//#####################################################################
// Function Compute_Embedding
//#####################################################################
void Compute_Embedding(const std::string tet_path,const std::string surface_path,const std::string embedding_path,const std::string output_path)
{
    std::cout << tet_path << std::endl;
    std::cout << surface_path << std::endl;
    TETRAHEDRALIZED_VOLUME<T>* tet=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tet_path,*tet);
    tet->Initialize_Hierarchy();
    tet->Initialize_Triangulated_Surface();
    tet->triangulated_surface->Initialize_Hierarchy();

    TRIANGULATED_SURFACE<T>* surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(surface_path,*surface);
    ARRAY<ARRAY<int> > p;
    ARRAY<ARRAY<T> > w;
    ARRAY<TV> embedding_offsets;
    Initialize_Embedding(*tet,*surface,p,w,embedding_offsets);

    std::cout<<"write to "<<embedding_path<<std::endl;
    FILE_UTILITIES::Write_To_File<RW>(embedding_path,p,w);

    // Compute vertex positions from embeddings
    int n_vertices=surface->particles.array_collection->Size();
    ARRAY<TV> vertices(n_vertices);
    for(int i=1;i<=n_vertices;i++){
        float sum = w(i)(1)+w(i)(2) + w(i)(3) + w(i)(4);
        assert(sum == 1);
        TV vertex;
        vertex(1)=w(i)(1)*tet->particles.X(p(i)(1)).x+w(i)(2)*tet->particles.X(p(i)(2)).x+w(i)(3)*tet->particles.X(p(i)(3)).x+w(i)(4)*tet->particles.X(p(i)(4)).x;
        vertex(2)=w(i)(1)*tet->particles.X(p(i)(1)).y+w(i)(2)*tet->particles.X(p(i)(2)).y+w(i)(3)*tet->particles.X(p(i)(3)).y+w(i)(4)*tet->particles.X(p(i)(4)).y;
        vertex(3)=w(i)(1)*tet->particles.X(p(i)(1)).z+w(i)(2)*tet->particles.X(p(i)(2)).z+w(i)(3)*tet->particles.X(p(i)(3)).z+w(i)(4)*tet->particles.X(p(i)(4)).z;
        vertices(i)=vertex;
    }

    // Generate cloth from embedding
    TRIANGULATED_SURFACE<T>* out_surface=TRIANGULATED_SURFACE<T>::Create();
    out_surface->mesh.elements=surface->mesh.elements;
    out_surface->particles.array_collection->Add_Elements(vertices.m);
    out_surface->particles.X=vertices;
    out_surface->Update_Number_Nodes();
    FILE_UTILITIES::Write_To_File<RW>(output_path,*out_surface);
}

//#####################################################################
// Function Compute_Embedding_Fix
//#####################################################################
void Compute_Embedding_Fix(const std::string surface_path,const std::string embedding_path,const std::string output_path)
{
    const std::string tpose_cloth_path = "/data/jwu/PhysBAM/Private_Projects/body_embedded_cloth/Tpose_embedded_shirt.tri.gz";
    const std::string tpose_kdsm_path = "/data/jwu/PhysBAM/Private_Projects/body_embedded_cloth/Data/Tpose_kdsm.tet.gz";

    std::cout << surface_path << std::endl;

    // Initialize T-pose data
    TETRAHEDRALIZED_VOLUME<T>* tet=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tpose_kdsm_path,*tet);
    tet->Initialize_Hierarchy();
    tet->Initialize_Triangulated_Surface();
    tet->triangulated_surface->Initialize_Hierarchy();

    TRIANGULATED_SURFACE<T>* tpose_surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tpose_cloth_path,*tpose_surface);

    TRIANGULATED_SURFACE<T>* surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(surface_path,*surface);

    ARRAY<ARRAY<int> > p;
    ARRAY<ARRAY<T> > w;
    FILE_UTILITIES::Read_From_File<RW>(embedding_path,p,w);

    // Compute max incident edge lengths
    T max_dist = tpose_surface->Maximum_Edge_Length();
    std::cout << "max edge length: " << max_dist << std::endl;

    // Compute vertex positions from embeddings
    int n_vertices=surface->particles.array_collection->Size();
    ARRAY<TV> vertices(n_vertices);
    ARRAY<int> bad_vertices;
    for(int i=1;i<=n_vertices;i++){
        TV vertex;
        vertex(1)=w(i)(1)*tet->particles.X(p(i)(1)).x+w(i)(2)*tet->particles.X(p(i)(2)).x+w(i)(3)*tet->particles.X(p(i)(3)).x+w(i)(4)*tet->particles.X(p(i)(4)).x;
        vertex(2)=w(i)(1)*tet->particles.X(p(i)(1)).y+w(i)(2)*tet->particles.X(p(i)(2)).y+w(i)(3)*tet->particles.X(p(i)(3)).y+w(i)(4)*tet->particles.X(p(i)(4)).y;
        vertex(3)=w(i)(1)*tet->particles.X(p(i)(1)).z+w(i)(2)*tet->particles.X(p(i)(2)).z+w(i)(3)*tet->particles.X(p(i)(3)).z+w(i)(4)*tet->particles.X(p(i)(4)).z;
        vertices(i)=vertex;
        T dist = sqrt(pow(vertex(1)-tpose_surface->particles.X(i)(1),2) + pow(vertex(2)-tpose_surface->particles.X(i)(2),2)
                    + pow(vertex(3)-tpose_surface->particles.X(i)(3),2));

        if(dist > 5*max_dist){
            std::cout << i << std::endl;
            bad_vertices.Append(i);
            vertices(i) = tpose_surface->particles.X(i);
        }
    }
    std::cout << output_path << std::endl;
    // Generate cloth from embedding
    TRIANGULATED_SURFACE<T>* out_surface=TRIANGULATED_SURFACE<T>::Create();
    out_surface->mesh.elements=surface->mesh.elements;
    out_surface->particles.array_collection->Add_Elements(vertices.m);
    out_surface->particles.X=vertices;
    out_surface->Update_Number_Nodes();
    FILE_UTILITIES::Write_To_File<RW>(output_path,*out_surface);

}

//#####################################################################
// Function Initialize_Embedding
//#####################################################################
void Initialize_Embedding(TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume,TRIANGULATED_SURFACE<T> &posed_cloth,ARRAY<ARRAY<int> >& embedding_parents,ARRAY<ARRAY<T> >& embedding_weights,ARRAY<TV> &embedding_offsets)
{
    const GEOMETRY_PARTICLES<TV> &particles=posed_cloth.particles;
    TRIANGULATED_SURFACE<T>& triangulated_surface=tetrahedralized_volume.Get_Boundary_Object();
    triangulated_surface.Update_Triangle_List();
    embedding_parents.Resize(particles.array_collection->Size(),false,false);
    embedding_weights.Resize(particles.array_collection->Size(),false,false);
    embedding_offsets.Resize(particles.array_collection->Size(),false,false);

    for(int i=1;i<=particles.array_collection->Size();i++){
        embedding_parents(i).Remove_All();embedding_weights(i).Remove_All();
        bool inside_success=tetrahedralized_volume.Inside(particles.X(i));
        if(inside_success){
            ARRAY<int> intersection_list;
            tetrahedralized_volume.hierarchy->Intersection_List(particles.X(i),intersection_list);
            for(int j=1;j<=intersection_list.m;j++){
                int t=intersection_list(j);
                TETRAHEDRON<T> tetrahedron=tetrahedralized_volume.Get_Element(t);;
                if(tetrahedron.Inside(particles.X(i))){
                    TETRAHEDRON<T> tetrahedron=tetrahedralized_volume.Get_Element(t);
                    embedding_parents(i)=tetrahedralized_volume.mesh.elements(t);
                    embedding_weights(i)=tetrahedron.Barycentric_Coordinates(particles.X(i));
                    embedding_offsets(i)=TV();
                    break;
                }
            }
        }
        if(!inside_success){
            std::cout<<"i:"<<i<<",not inside"<<std::endl;
            int t;TV closest_point=triangulated_surface.Surface(particles.X(i),(T)5e-4,0,&t);
            TRIANGLE_3D<T> triangle=triangulated_surface.Get_Element(t);
            TV weights=triangle.Barycentric_Coordinates(closest_point);
            TV_INT parents = triangulated_surface.mesh.elements(t);
            // Iterate over tets to find fourth parent
            for(int k=1;k<=tetrahedralized_volume.mesh.elements.m;k++){
                int a,b,c,d;tetrahedralized_volume.mesh.elements(k).Get(a,b,c,d);
                if(parents(1)==a || parents(1)==b || parents(1)==c || parents(1)==d){ 
                    if(parents(2)==a || parents(2)==b || parents(2)==c || parents(2)==d){
                        if(parents(3)==a || parents(3)==b || parents(3)==c || parents(3)==d){
                            embedding_parents(i)=tetrahedralized_volume.mesh.elements(k);
                            // Set weights
                            ARRAY<T> tet_weights(4);
                            for(int l=1;l<=4;l++){
                                if(parents(1)==embedding_parents(i)(l))
                                    tet_weights(l)=weights(1);
                                else if(parents(2)==embedding_parents(i)(l))
                                    tet_weights(l)=weights(2);
                                else if(parents(3)==embedding_parents(i)(l))
                                    tet_weights(l)=weights(3);
                                else
                                    tet_weights(l)=0.0;
                            }
                            embedding_weights(i)=tet_weights;
                            break;}}}}
            embedding_offsets(i)=particles.X(i)-closest_point;
        }
    }
}
//#####################################################################
};
}
#endif
