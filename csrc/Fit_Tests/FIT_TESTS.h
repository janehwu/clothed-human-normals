//#####################################################################
// Copyright 2021, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class FIT_TESTS
//#####################################################################
//#####################################################################
#ifndef __FIT_TESTS__
#define __FIT_TESTS__

#include <PhysBAM_Geometry/Spatial_Acceleration/TETRAHEDRON_HIERARCHY.h>
#include <PhysBAM_Geometry/Topology/TETRAHEDRON_MESH.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>

//#include "../External/cnpy.h"
//#include "Utils/DEBUG_VIZ_UTIL.h"
//#include "../opengl_3d_proto/Proto/PROTO_DEBUG_UTILS.h"
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

#include "Fit.h"

namespace PhysBAM{

template<class T, class RW>
class FIT_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

void LoadLevelSet(std::string path, float *tsdf, int nb_nodes) {
    
    std::ifstream file (path, std::ios::binary);
    file.read((char*) tsdf, nb_nodes*sizeof(float));
    //file.write((char*)weights, 24*nb_nodes*sizeof(float));
    file.close();
    
}

//#####################################################################
// Function TSDF_To_Tri
//#####################################################################
void TSDF_To_Mesh(const std::string tet_path,const std::string tsdf_path, const std::string tri_path)
{
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tet_path,tetrahedralized_volume);
    //tetrahedralized_volume.mesh.Initialize_Neighbor_Nodes();
    //tetrahedralized_volume.mesh.Initialize_Boundary_Nodes();
    tetrahedralized_volume.Initialize_Triangulated_Surface();
    tetrahedralized_volume.triangulated_surface->Update_Vertex_Normals();
    //tetrahedralized_volume.triangulated_surface->mesh.Initialize_Incident_Elements();
    //tetrahedralized_volume.triangulated_surface->Update_Triangle_List();
    //tetrahedralized_volume.Update_Tetrahedron_List();
    //tetrahedralized_volume.Initialize_Hierarchy();

    int num_vertices=tetrahedralized_volume.particles.array_collection->Size();
    int num_tetrahedra=tetrahedralized_volume.mesh.elements.m;
    std::cout<<"total vertices = "<<num_vertices<<std::endl;
    std::cout<<"total tetrahedra = "<<num_tetrahedra<<std::endl;

    // Create tet data
    //std::vector<float> tet_nodes(num_vertices*3);
    //std::vector<int> tet_tetra(num_tetrahedra*4);
    float *tet_nodes=new float[num_vertices*3];
    int *tet_tetra=new int[num_tetrahedra*4];

    for(int i=1;i<=num_vertices;i++){
        tet_nodes[(i-1)*3]=tetrahedralized_volume.particles.X(i)(1);
        tet_nodes[(i-1)*3+1]=tetrahedralized_volume.particles.X(i)(2);
        tet_nodes[(i-1)*3+2]=tetrahedralized_volume.particles.X(i)(3);
    }
    for(int i=1;i<=num_tetrahedra;i++){
        tet_tetra[(i-1)*4]=tetrahedralized_volume.mesh.elements(i)(1);
        tet_tetra[(i-1)*4+1]=tetrahedralized_volume.mesh.elements(i)(2);
        tet_tetra[(i-1)*4+2]=tetrahedralized_volume.mesh.elements(i)(3);
        tet_tetra[(i-1)*4+3]=tetrahedralized_volume.mesh.elements(i)(4);
    }

    // Tet surface mesh
    TRIANGULATED_SURFACE<T>& triangulated_surface = tetrahedralized_volume.Get_Boundary_Object();
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"boundary_surface.tri.gz",triangulated_surface);

    int num_surface_vertices=triangulated_surface.particles.array_collection->Size();
    int num_surface_triangles=triangulated_surface.mesh.elements.m;
    std::cout<<"surface vertices = "<<num_surface_vertices<<std::endl;
    std::cout<<"surface triangles = "<<num_surface_triangles<<std::endl;
    //std::vector<float> surface_nodes(num_surface_vertices*3);
    //std::vector<float> surface_normals(num_surface_vertices*3);
    //std::vector<float> surface_triangles(num_surface_triangles*3);
    float *surface_nodes=new float[num_surface_vertices*3];
    float *surface_normals=new float[num_surface_vertices*3];
    int *surface_triangles=new int[num_surface_triangles*3];

    TV normal;
    for(int i=1;i<=num_surface_vertices;i++){
        surface_nodes[(i-1)*3]=triangulated_surface.particles.X(i)(1);
        surface_nodes[(i-1)*3+1]=triangulated_surface.particles.X(i)(2);
        surface_nodes[(i-1)*3+2]=triangulated_surface.particles.X(i)(3);
        normal=(*triangulated_surface.vertex_normals)(i);
        surface_normals[(i-1)*3]=normal(1);
        surface_normals[(i-1)*3+1]=normal(2);
        surface_normals[(i-1)*3+2]=normal(3);
    }

    for(int i=1;i<=num_surface_triangles;i++){
        surface_triangles[(i-1)*3]=triangulated_surface.mesh.elements(i)(1);
        surface_triangles[(i-1)*3+1]=triangulated_surface.mesh.elements(i)(2);
        surface_triangles[(i-1)*3+2]=triangulated_surface.mesh.elements(i)(3);
    }

    // Print data
    std::cout<<"Tet mesh verts: "<<tet_nodes[0]<<std::endl;
    std::cout<<"Tet mesh tetra: "<<tet_tetra[0]<<std::endl;
    std::cout<<"Tri mesh verts: "<<surface_nodes[0]<<std::endl;
    std::cout<<"Tri mesh tris:  "<<surface_triangles[0]<<std::endl;
    std::cout<<"Tri mesh norms: "<<surface_normals[0]<<std::endl;

    // Load level set
    int idx_file=0;
    float *sdf = new float[num_vertices];
    LoadLevelSet("/data/jwu/jane-standford/Fitting/TSDF/mesh_D_march_" + std::to_string(idx_file) + ".bin", sdf, num_vertices);
    std::cout << "The level set is generated: " << sdf[0]<<" "<<sdf[num_vertices-1]<< std::endl;

    fitLevelSet::FitToLevelSet(sdf, tet_nodes, tet_tetra, num_tetrahedra,
                               surface_nodes, surface_normals, surface_triangles,
                               num_surface_vertices, num_surface_triangles);

    // Save fitted mesh
    /*
    std::cout<<"Saving fitted mesh"<<std::endl;
    for(int i=1;i<=num_vertices;i++){
        triangulated_surface.particles.X(i)(1)=surface_nodes[(i-1)*3];
        triangulated_surface.particles.X(i)(2)=surface_nodes[(i-1)*3+1];
        triangulated_surface.particles.X(i)(3)=surface_nodes[(i-1)*3+2];
    }
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);
    */
}
//#####################################################################
};
}
#endif
