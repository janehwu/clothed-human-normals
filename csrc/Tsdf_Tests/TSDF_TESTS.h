//#####################################################################
// Copyright 2020, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class TSDF_TESTS
//#####################################################################
//#####################################################################
#ifndef __TSDF_TESTS__
#define __TSDF_TESTS__

#include <PhysBAM_Geometry/Implicit_Objects_Uniform/LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Spatial_Acceleration/TETRAHEDRON_HIERARCHY.h>
#include <PhysBAM_Geometry/Topology/TETRAHEDRON_MESH.h>
#include <PhysBAM_Geometry/Topology/POINT_SIMPLEX_MESH.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Implicit_Objects_Uniform/READ_WRITE_LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry_Level_Sets/LEVELSET_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Topology_Based_Geometry_Level_Sets/READ_WRITE_LEVELSET_TETRAHEDRALIZED_VOLUME.h>
//#include "Topology_Based_Geometry_Level_Sets/FAST_MARCHING_METHOD_TETRAHEDRALIZED_VOLUME.h"
#include "../External/cnpy.h"
//#include "Utils/DEBUG_VIZ_UTIL.h"
//#include "../opengl_3d_proto/Proto/PROTO_DEBUG_UTILS.h"
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class TSDF_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function TSDF_To_Level_Set
//#####################################################################
void TSDF_To_Level_Set(const std::string tet_path,const std::string tsdf_path)
{
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.mesh.Initialize_Neighbor_Nodes();
    tetrahedralized_volume.mesh.Initialize_Boundary_Nodes();
    tetrahedralized_volume.Initialize_Triangulated_Surface();
    tetrahedralized_volume.triangulated_surface->mesh.Initialize_Incident_Elements();
    tetrahedralized_volume.triangulated_surface->Update_Triangle_List();
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.Initialize_Hierarchy();
    std::cout<<"total vertices = "<<tetrahedralized_volume.particles.array_collection->Size()<<std::endl;
    std::cout<<"total tetrahedra = "<<tetrahedralized_volume.mesh.elements.m<<std::endl;

    cnpy::NpyArray arr = cnpy::npy_load(tsdf_path);
    std::vector<float> loaded_data = arr.as_vec<float>();
    std::cout <<"total tsdf values = "<<arr.shape[0]<<std::endl;

    int num_vertices=arr.shape[0];
    ARRAY<T> tsdf(num_vertices);
    ARRAY<int> seed_indices;
    for(int i=1;i<=num_vertices;i++){
        tsdf(i)=loaded_data[i-1];
        if((tsdf(i)>-1)&&(tsdf(i)<1))
            seed_indices.Append(i);
    }

    // Signed distance field
    T stopping_distance = 0;
    std::cout<<"Starting fast marching"<<std::endl;
    /*FAST_MARCHING_METHOD_TETRAHEDRALIZED_VOLUME<TV> fmm_mesh_object(tetrahedralized_volume,tsdf);
    std::cout<<"done1"<<std::endl;
    fmm_mesh_object.Fast_Marching_Method(&seed_indices);
    std::cout<<"done2"<<std::endl;*/
}

//#####################################################################
// Function Compute_Correspondences
//#####################################################################
void Compute_Correspondences(const std::string advected_scan_path, const std::string smpl_path, const std::string original_scan_path)
{
    TRIANGULATED_SURFACE<T>& advected_scan=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),advected_scan_path,advected_scan);
    std::cout<<"Scan mesh vertices: "<<advected_scan.particles.array_collection->Size()<<std::endl;

    TRIANGULATED_SURFACE<T>& original_scan=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),original_scan_path,original_scan);
    std::cout<<"Scan mesh vertices: "<<original_scan.particles.array_collection->Size()<<std::endl;

    TRIANGULATED_SURFACE<T>& smpl_template=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),smpl_path,smpl_template);
    std::cout<<"SMPL mesh vertices: "<<smpl_template.particles.array_collection->Size()<<std::endl;

    // Also load skinning weights for scanned mesh
    std::string skinning_weights_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/weights_smpl_subdiv_flatten.npy";
    cnpy::NpyArray arr = cnpy::npy_load(skinning_weights_path);
    std::vector<float> loaded_data = arr.as_vec<float>();
    std::cout <<"total skinning weights = "<<arr.shape[0]<<std::endl;

    ARRAY<ARRAY<T> >skinning(original_scan.particles.array_collection->Size());
    for(int i=1;i<=arr.shape[0];i++){
        skinning(((i-1)/24)+1).Append(loaded_data[i-1]);
    }
    std::cout<<"scan skinnning weights: "<<skinning(1)<<std::endl;

    advected_scan.Initialize_Hierarchy();
    advected_scan.Update_Triangle_List();
    ARRAY<ARRAY<int> > p(smpl_template.particles.array_collection->Size());
    ARRAY<ARRAY<T> > w(smpl_template.particles.array_collection->Size());  

    const GEOMETRY_PARTICLES<TV> &particles=smpl_template.particles;

    // For every SMPL vertex, embed in an advected scan triangle
    for(int i=1;i<=particles.array_collection->Size();i++){
        /*
        bool inside_success=advected_scan.Boundary(particles.X(i),(T)5e-4);
        if(inside_success){
            ARRAY<int> intersection_list;
            advected_scan.hierarchy->Intersection_List(particles.X(i),intersection_list);
            for(int j=1;j<=intersection_list.m;j++){
                int t=intersection_list(j);
                embedding_parents(i)=advected_scan.mesh.elements(t);
                embedding_weights(i)=advected_scan.Get_Element(t).Barycentric_Coordinates(particles.X(i));
                // Just take first intersection
                break;
            }
        }
        */
        if(true){
            //std::cout<<"i:"<<i<<",not on boundary"<<std::endl;
            int t;TV closest_point=advected_scan.Surface(particles.X(i),(T)5e-4,0,&t);
            p(i)=advected_scan.mesh.elements(t);
            w(i)=advected_scan.Get_Element(t).Barycentric_Coordinates(closest_point);
        }
    }

    // Compute vertex positions from embeddings
    int n_vertices=particles.array_collection->Size();
    ARRAY<TV> smpl_vertices(n_vertices);

    // Also compute new SMPL skinning weights
    std::vector<T> smpl_skinning_weights;

    for(int i=1;i<=n_vertices;i++){
        float sum = w(i)(1)+w(i)(2) + w(i)(3);
        assert(sum == 1);
        TV vertex;
        vertex(1)=w(i)(1)*original_scan.particles.X(p(i)(1)).x+w(i)(2)*original_scan.particles.X(p(i)(2)).x+w(i)(3)*original_scan.particles.X(p(i)(3)).x;
        vertex(2)=w(i)(1)*original_scan.particles.X(p(i)(1)).y+w(i)(2)*original_scan.particles.X(p(i)(2)).y+w(i)(3)*original_scan.particles.X(p(i)(3)).y;
        vertex(3)=w(i)(1)*original_scan.particles.X(p(i)(1)).z+w(i)(2)*original_scan.particles.X(p(i)(2)).z+w(i)(3)*original_scan.particles.X(p(i)(3)).z;
        smpl_vertices(i)=vertex;

        for(int s=1;s<=24;s++){
            smpl_skinning_weights.push_back(w(i)(1)*skinning(p(i)(1))(s)+w(i)(2)*skinning(p(i)(2))(s)+w(i)(3)*skinning(p(i)(3))(s));
        }
    }
    std::cout<<smpl_skinning_weights.size()/24<<std::endl;
    
    auto offsets_vec=smpl_skinning_weights;
    cnpy::npy_save("new_weights.npy",&offsets_vec[0],{smpl_skinning_weights.size(),1},"w");

    std::cout<<"Done"<<std::endl;
    // Generate cloth from embedding
    TRIANGULATED_SURFACE<T>* out_surface=TRIANGULATED_SURFACE<T>::Create();
    out_surface->mesh.elements=smpl_template.mesh.elements;
    out_surface->particles.array_collection->Add_Elements(smpl_vertices.m);
    out_surface->particles.X=smpl_vertices;
    out_surface->Update_Number_Nodes();
    FILE_UTILITIES::Write_To_File<RW>("out_projection.tri.gz",*out_surface);

}

//#####################################################################
// Function Project_SMPL_To_Phi
//#####################################################################
void Project_SMPL_To_Phi(const std::string smpl_path, const std::string phi_path)
{
    TRIANGULATED_SURFACE<T>& smpl_template=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),smpl_path,smpl_template);
    std::cout<<"SMPL mesh vertices: "<<smpl_template.particles.array_collection->Size()<<std::endl;

    LEVELSET_IMPLICIT_OBJECT<TV>& body_implicit_surface=*LEVELSET_IMPLICIT_OBJECT<TV>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),phi_path,body_implicit_surface);

    T step_size=1e-4;
    int num_iterations=2000;

    T last_avg_dist=0;
    for(int i=1;i<=num_iterations;i++){
        T avg_dist=0;
        for(int j=1;j<=smpl_template.particles.array_collection->Size();j++){
            T phi=body_implicit_surface(smpl_template.particles.X(j));
            TV grad_phi=body_implicit_surface.Normal(smpl_template.particles.X(j)).Normalized();
            TV new_X=smpl_template.particles.X(j)-sign(phi)*step_size*grad_phi;
            T new_phi=body_implicit_surface(new_X);
            //Check if crossing interface
            if(sign(phi)!=sign(new_phi)){
                T ratio=new_phi/(new_phi-phi);
                new_X=ratio*smpl_template.particles.X(j)+(1-ratio)*new_X;
            }

            smpl_template.particles.X(j)=new_X;
            avg_dist+=abs(phi);
        }
        avg_dist=avg_dist/smpl_template.particles.array_collection->Size();
        std::cout<<i<<" --- avg phi: "<<avg_dist<<std::endl;
        //if(i>1 && last_avg_dist < avg_dist) break;
        last_avg_dist=avg_dist;
    }
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"advected_surface.tri.gz",smpl_template);
}

//#####################################################################
// Function Sample_Phi_On_Tetrahedra
//#####################################################################
void Phi_To_TSDF(const std::string tet_path, const std::string phi_path, const std::string tsdf_output_path)
{
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.Initialize_Hierarchy();

    std::cout<<"Tet mesh vertices: "<<tetrahedralized_volume.particles.array_collection->Size()<<std::endl;
    std::cout<<"Tet mesh elements: "<<tetrahedralized_volume.mesh.elements.m<<std::endl;

    LEVELSET_IMPLICIT_OBJECT<TV>& body_implicit_surface=*LEVELSET_IMPLICIT_OBJECT<TV>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),phi_path,body_implicit_surface);

    int num_vertices=tetrahedralized_volume.particles.array_collection->Size();
    ARRAY<T> phi(num_vertices);
    for(int i=1;i<=num_vertices;i++)
        // TODO: remove min/max for full level set
        phi(i)=std::max((T)-1.0,std::min(body_implicit_surface(tetrahedralized_volume.particles.X(i)),(T)1.0));

    // Fix phi if negative on boundary
    TRIANGULATED_SURFACE<T>& tet_boundary=tetrahedralized_volume.Get_Boundary_Object();
    T avg_edge_length=tet_boundary.Average_Edge_Length();
    for(int t=1;t<=tet_boundary.mesh.elements.m;t++){
        int i,j,k;
        tet_boundary.mesh.elements(t).Get(i,j,k);
        if(phi(i)<0)
            phi(i)=((T)0.01)*avg_edge_length;
        if(phi(j)<0)
            phi(j)=((T)0.01)*avg_edge_length;
        if(phi(k)<0)
            phi(k)=((T)0.01)*avg_edge_length;
    }

    // Save offsets
    auto array_to_flat_vec=[](const ARRAY<T>& array){
        std::vector<RW> vec(array.Size());size_t cnt=0;
        for(int i=1;i<=array.Size();i++){
            vec[cnt++]=array(i);
        }
        return vec;
    };
    
    auto phi_vec=array_to_flat_vec(phi);
    cnpy::npy_save(tsdf_output_path,&phi_vec[0],{num_vertices},"w");

}

void Boundary_Mesh(const std::string tri_path)
{
    TRIANGULATED_SURFACE<T>& triangulated_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);
    std::cout<<"SMPL mesh vertices: "<<triangulated_surface.particles.array_collection->Size()<<std::endl;


    triangulated_surface.Update_Triangle_List();
    triangulated_surface.Initialize_Hierarchy();

    triangulated_surface.mesh.Initialize_Node_On_Boundary();
    triangulated_surface.mesh.Initialize_Neighbor_Nodes();
    triangulated_surface.mesh.Initialize_Incident_Elements();
    triangulated_surface.mesh.Initialize_Boundary_Mesh();
    triangulated_surface.mesh.Initialize_Edge_Triangles();
    std::cout<<"Num triangles: "<<triangulated_surface.mesh.elements.Size()<<std::endl;
    std::cout<<"Boundary elements: "<<(*triangulated_surface.mesh.boundary_mesh).elements.m<<std::endl;

    ARRAY<VECTOR<int,3> > triangles;
    for(int i=1;i<=(*triangulated_surface.mesh.boundary_mesh).elements.m;i++){
        int j=(*triangulated_surface.mesh.boundary_mesh).elements(i)(1);
        int k=(*triangulated_surface.mesh.boundary_mesh).elements(i)(2);
        // Find intersecting triangles (only 1)
        ARRAY<int> triangles_on_edge;
        int tris=triangulated_surface.mesh.Triangles_On_Edge(j,k,&triangles_on_edge);
        PHYSBAM_ASSERT(tris==1);
        for(int t=1;t<=tris;t++){
            int idx=(triangles_on_edge)(t);
            triangles.Append(triangulated_surface.mesh.elements(idx));
        }
    }
    TRIANGULATED_SURFACE<T>* boundary_surface=TRIANGULATED_SURFACE<T>::Create();
    boundary_surface->mesh.elements=triangles;
    boundary_surface->particles.array_collection->Add_Elements(triangulated_surface.particles.array_collection->Size());
    boundary_surface->particles.X=triangulated_surface.particles.X;
    boundary_surface->Update_Number_Nodes();
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"boundary_surface.tri.gz",*boundary_surface);
}

//#####################################################################
// Function Sample_Phi_On_Tetrahedra
//#####################################################################
void Sample_Phi_On_Tetrahedra(const std::string tet_path, const std::string phi_path)
{
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.Initialize_Hierarchy();

    std::cout<<"Tet mesh vertices: "<<tetrahedralized_volume.particles.array_collection->Size()<<std::endl;
    std::cout<<"Tet mesh elements: "<<tetrahedralized_volume.mesh.elements.m<<std::endl;

    // TODO: Remove
    // Check for inverted tets
    /*
    tetrahedralized_volume.mesh.Initialize_Segment_Mesh();
    tetrahedralized_volume.mesh.Initialize_Element_Edges();

    ARRAY<int> inverted_tets;
    tetrahedralized_volume.Inverted_Tetrahedrons(inverted_tets);
    std::cout<<"Num inverted tets: "<<inverted_tets.Size()<<std::endl;

    TETRAHEDRON_MESH* inverted_mesh=new TETRAHEDRON_MESH();
    for(int t=1;t<=inverted_tets.Size();t++){
        inverted_mesh->elements.Append(tetrahedralized_volume.mesh.elements(t));
    }
    tetrahedralized_volume.mesh.elements=inverted_mesh->elements;
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.mesh.Initialize_Segment_Mesh();
    tetrahedralized_volume.mesh.Initialize_Element_Edges();

    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"inverted_tets.tet.gz",tetrahedralized_volume);
    */

    LEVELSET_IMPLICIT_OBJECT<TV>& body_implicit_surface=*LEVELSET_IMPLICIT_OBJECT<TV>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),phi_path,body_implicit_surface);

    int num_vertices=tetrahedralized_volume.particles.array_collection->Size();
    ARRAY<T> phi(num_vertices);
    for(int i=1;i<=num_vertices;i++)
        phi(i)=body_implicit_surface(tetrahedralized_volume.particles.X(i));

    // Fix phi if negative on boundary
    TRIANGULATED_SURFACE<T>& tet_boundary=tetrahedralized_volume.Get_Boundary_Object();
    T avg_edge_length=tet_boundary.Average_Edge_Length();
    for(int t=1;t<=tet_boundary.mesh.elements.m;t++){
        int i,j,k;
        tet_boundary.mesh.elements(t).Get(i,j,k);
        if(phi(i)<0)
            phi(i)=((T)0.01)*avg_edge_length;
        if(phi(j)<0)
            phi(j)=((T)0.01)*avg_edge_length;
        if(phi(k)<0)
            phi(k)=((T)0.01)*avg_edge_length;
    }

    LEVELSET_TETRAHEDRALIZED_VOLUME<T>* tsdf_tet= new LEVELSET_TETRAHEDRALIZED_VOLUME<T>(tetrahedralized_volume,phi);
    //FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"test_levelset_tet.gz",*tsdf_tet);

    // Output tri mesh as well
    TRIANGULATED_SURFACE<T>& tsdf_surface=*TRIANGULATED_SURFACE<T>::Create();
    tsdf_tet->Calculate_Triangulated_Surface_From_Marching_Tetrahedra(tsdf_surface,(T)0);

    // TODO: Remove
    /////////////////////////////////////////////////////////////////////////////////////
    // Output boundary mesh
    tsdf_surface.Update_Triangle_List();
    tsdf_surface.Initialize_Hierarchy();

    tsdf_surface.mesh.Initialize_Node_On_Boundary();
    tsdf_surface.mesh.Initialize_Neighbor_Nodes();
    tsdf_surface.mesh.Initialize_Incident_Elements();
    tsdf_surface.mesh.Initialize_Boundary_Mesh();
    tsdf_surface.mesh.Initialize_Edge_Triangles();
    std::cout<<"Num triangles: "<<tsdf_surface.mesh.elements.Size()<<std::endl;
    std::cout<<"Boundary elements: "<<(*tsdf_surface.mesh.boundary_mesh).elements.m<<std::endl;
    /*
    ARRAY<VECTOR<int,3> > triangles;
    for(int i=1;i<=(*tsdf_surface.mesh.boundary_mesh).elements.m;i++){
        int j=(*tsdf_surface.mesh.boundary_mesh).elements(i)(1);
        int k=(*tsdf_surface.mesh.boundary_mesh).elements(i)(2);
        // Find intersecting triangles (only 1)
        ARRAY<int> triangles_on_edge;
        int tris=tsdf_surface.mesh.Triangles_On_Edge(j,k,&triangles_on_edge);
        PHYSBAM_ASSERT(tris==1);
        for(int t=1;t<=tris;t++){
            int idx=(triangles_on_edge)(t);
            triangles.Append(tsdf_surface.mesh.elements(idx));
        }
    }
    TRIANGULATED_SURFACE<T>* triangulated_surface=TRIANGULATED_SURFACE<T>::Create();
    triangulated_surface->mesh.elements=triangles;
    triangulated_surface->particles.array_collection->Add_Elements(tsdf_surface.particles.array_collection->Size());
    triangulated_surface->particles.X=tsdf_surface.particles.X;
    triangulated_surface->Update_Number_Nodes();
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"boundary_mesh.tri.gz",*triangulated_surface);
    */
    /////////////////////////////////////////////////////////////////////////////////////

    // TODO: Remove
    // Run tests on tri mesh...
    // 1) Compute unit normal of centroid of each triangle
    // 2) Compute norm grad phi at that centroid
    // 3) Dot product and delete if <= 0
    /*
    TRIANGLE_MESH* inverted_mesh=new TRIANGLE_MESH();
    for(int t=1;t<=tsdf_surface.mesh.elements.m;t++){
        int i,j,k;
        tsdf_surface.mesh.elements(t).Get(i,j,k);
        TV centroid=(T)one_third*(tsdf_surface.particles.X(i)+tsdf_surface.particles.X(j)+tsdf_surface.particles.X(k));
        TV normal=TRIANGLE_3D<T>::Normal(tsdf_surface.particles.X(i),tsdf_surface.particles.X(j),tsdf_surface.particles.X(k));
        if(TV::Dot_Product(normal,body_implicit_surface.Normal(centroid))<0){
            std::cout<<"Inverted triangle! "<<t<<std::endl;
            inverted_mesh->elements.Append(tsdf_surface.mesh.elements(t));
        }
    }
    std::cout<<"Number of poorly correlated triangles: "<<inverted_mesh->elements.m<<std::endl;
    //tsdf_surface.mesh.elements=inverted_mesh->elements;
    */

    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"level_set_surface.tri.gz",tsdf_surface);
}

//#####################################################################
// Function TSDF_To_Mesh
//#####################################################################
void TSDF_To_Mesh(const std::string tet_path,const std::string tsdf_path, const std::string tri_path)
{
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.Initialize_Triangulated_Surface();
    tetrahedralized_volume.triangulated_surface->Update_Triangle_List();
    //std::cout<<"total vertices = "<<tetrahedralized_volume.particles.array_collection->Size()<<std::endl;
    //std::cout<<"total tetrahedra = "<<tetrahedralized_volume.mesh.elements.m<<std::endl;

    cnpy::NpyArray arr = cnpy::npy_load(tsdf_path);
    std::vector<float> loaded_data = arr.as_vec<float>();
    //std::cout <<"total tsdf values = "<<arr.shape[0]<<std::endl;

    int num_vertices=arr.shape[0];
    ARRAY<T> tsdf(num_vertices);
    for(int i=0;i<num_vertices;i++){
        tsdf(i+1)=loaded_data[i];
    }

    LEVELSET_TETRAHEDRALIZED_VOLUME<T>* tsdf_tet= new LEVELSET_TETRAHEDRALIZED_VOLUME<T>(tetrahedralized_volume,tsdf);
    //FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"test_levelset_tet.gz",*tsdf_tet);

    // Output tri mesh as well
    TRIANGULATED_SURFACE<T>& tsdf_surface=*TRIANGULATED_SURFACE<T>::Create();
    tsdf_tet->Calculate_Triangulated_Surface_From_Marching_Tetrahedra(tsdf_surface,(T)0);
    std::cout<<"\nfile = "<<tsdf_path<<std::endl;
    std::cout<<"volume = "<<tsdf_surface.Volumetric_Volume()<<std::endl;
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),tri_path,tsdf_surface);
}
//#####################################################################
};
}
#endif
