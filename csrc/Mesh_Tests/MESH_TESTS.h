//#####################################################################
// Copyright 2020, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class MESH_TESTS
//#####################################################################
//#####################################################################
#ifndef __MESH_TESTS__
#define __MESH_TESTS__

#include <PhysBAM_Geometry/Spatial_Acceleration/TETRAHEDRON_HIERARCHY.h>
#include <PhysBAM_Geometry/Topology/TETRAHEDRON_MESH.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>

#include "../External/cnpy.h"
//#include "Utils/DEBUG_VIZ_UTIL.h"
//#include "../opengl_3d_proto/Proto/PROTO_DEBUG_UTILS.h"
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class MESH_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Ply_To_Tet
// (From Diego)
//#####################################################################
void Ply_To_Tet(std::string path){
    int *result = new int[3];
    result[0] = -1; result[1] = -1; result[2] = -1;

    std::string line;
    std::ifstream plyfile (path, std::ios::binary);
    if (plyfile.is_open()) {
        
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        // get number of vertices
        std::istringstream iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        result[0] = std::stoi(words[2]);
        std::cout << "number of vertices: " << result[0] << std::endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        // get number of faces
        std::istringstream iss2(line);
        std::vector<std::string> words2((std::istream_iterator<std::string>(iss2)), std::istream_iterator<std::string>());
        result[1]  = std::stoi(words2[2]);
        std::cout << "number of faces: " << result[1]  << std::endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        std::istringstream iss3(line);
        std::vector<std::string> words3((std::istream_iterator<std::string>(iss3)), std::istream_iterator<std::string>());
        result[2] = std::stoi(words3[2]);
        std::cout << "number of voxels: " << result[2] << std::endl;

        getline (plyfile,line); // PLY
        getline (plyfile,line); // PLY
        std::cout << line << std::endl;

        ARRAY<TV> vertices;
        ARRAY<VECTOR<int,3> > triangles;
        ARRAY<VECTOR<int,4> > tetrahedron_list;

        for (int i = 0; i < result[0]; i++) { // Read vertices
            double x,y,z;
            plyfile.read((char *) &x, 8);
            plyfile.read((char *) &y, 8);
            plyfile.read((char *) &z, 8);
            vertices.Append(TV(x,y,z));
        }

        for (int i = 0; i < result[1]; i++) { // Read triangles
            unsigned char a;
            int b;
            plyfile.read((char *) &a, 1);

            if (int(a) != 3) {
                std::cout << "Error non triangle faces" << std::endl;
            }
            VECTOR<int,3> tri;
            plyfile.read((char *) &b, 4);
            tri(1) = b;
            plyfile.read((char *) &b, 4);
            tri(2) = b;
            plyfile.read((char *) &b, 4);
            tri(3) = b;
            triangles.Append(tri);
        }

        for (int i = 0; i < result[2]; i++) { // Read tetrahedra
            unsigned char a;
            int b;
            plyfile.read((char *) &a, 1);

            if (int(a) != 4) {
                std::cout << "Error non tetrahedral voxel" << std::endl;
            }
            VECTOR<int,4> tet;
            plyfile.read((char *) &b, 4);
            tet(1) = b+1;
            plyfile.read((char *) &b, 4);
            tet(2) = b+1;
            plyfile.read((char *) &b, 4);
            tet(3) = b+1;
            plyfile.read((char *) &b, 4);
            tet(4) = b+1;
            tetrahedron_list.Append(tet);
        }

        plyfile.close();

        std::cout << "Done processing" << std::endl;
        std::cout << vertices(1).x << "," << vertices(2).y << "," << vertices(3).z << std::endl;
        std::cout << triangles(1)(1) << "," << triangles(1)(2) << "," << triangles(1)(3) << std::endl;
        std::cout << tetrahedron_list(1)(1) << "," << tetrahedron_list(1)(2) << "," << tetrahedron_list(1)(3) << "," << tetrahedron_list(1)(4) << std::endl;

        //Construct tet mesh
        PARTICLES<TV>* particles = new PARTICLES<TV>();
        particles->array_collection->Add_Elements(vertices.m);
        particles->X=vertices;
        TETRAHEDRON_MESH* tetrahedron_mesh=new TETRAHEDRON_MESH(vertices.m, tetrahedron_list);
        std::cout << "done creating tetrahedron mesh" << std::endl;

        //Change particles
        std::string star_surface_path="network/pose_shell_network.tri.gz";
        TRIANGULATED_SURFACE<T>& star_surface=*TRIANGULATED_SURFACE<T>::Create();
        FILE_UTILITIES::Read_From_File<RW>(star_surface_path,star_surface);

        TETRAHEDRALIZED_VOLUME<T>* tetrahedralized_volume=new TETRAHEDRALIZED_VOLUME<T>(*tetrahedron_mesh, star_surface.particles);

        std::string output_filename = "network/pose_shell_network.tet.gz";
        FILE_UTILITIES::Write_To_File<RW>(output_filename,*tetrahedralized_volume);
    }
    else {
        plyfile.close();
        std::cout << "could not load file: " << path << std::endl;
    }
}

void Set_Vertices(const std::string tet_path, const std::string tri_path)
{
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
   
   //Change particles
   TRIANGULATED_SURFACE<T>& triangulated_surface=*TRIANGULATED_SURFACE<T>::Create();
   FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);

   tetrahedralized_volume.particles.X=triangulated_surface.particles.X;
   std::string output_filename = "pose_kdsm.tet.gz";
   FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),output_filename,tetrahedralized_volume);

}

//#####################################################################
// Function Tet_To_Ply
//#####################################################################
void Tet_To_Ply(const std::string tet_path)
{
    //Construct tet mesh
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.mesh.Initialize_Triangle_Mesh();

    int num_vertices = tetrahedralized_volume.particles.array_collection->Size();
    int num_faces = tetrahedralized_volume.mesh.triangle_mesh->elements.m;
    int num_tets = tetrahedralized_volume.mesh.elements.m;
    std::cout<<"total vertices = "<<num_vertices<<std::endl;
    std::cout<<"total triangles = "<<num_faces<<std::endl;
    std::cout<<"total tetrahedra = "<<num_tets<<std::endl;

    std::ofstream plyfile;
    plyfile.open("coarsehuman.ply");
    plyfile << "ply\nformat ascii 1.0\nelement vertex " << num_vertices << "\n";
    plyfile << "property float x\nproperty float y\nproperty float z\n";
    plyfile << "element face " << num_faces << "\n";
    plyfile << "property list uchar int vertex_indices\n";
    plyfile << "element voxel " << num_tets << "\n";
    plyfile << "property list uchar int vertex_indices\nend_header\n";

    // Write vertices
    for(int i=1;i<=num_vertices;i++){
        plyfile << tetrahedralized_volume.particles.X(i).x << " ";
        plyfile << tetrahedralized_volume.particles.X(i).y << " ";
        plyfile << tetrahedralized_volume.particles.X(i).z << "\n";
    }
    // Write faces
    for(int i=1;i<=num_faces;i++){
        TV_INT face = tetrahedralized_volume.mesh.triangle_mesh->elements(i);
        plyfile << "3 ";
        plyfile << face(1)-1 << " " << face(2)-1 << " " << face(3)-1 << "\n";
    }
    // Write tets
    for(int i=1;i<=num_tets;i++){
        VECTOR<int,4> tet = tetrahedralized_volume.mesh.elements(i);
        plyfile << "4 ";
        plyfile << tet(1)-1 << " " << tet(2)-1 << " " << tet(3)-1 << " " << tet(4)-1 << "\n";
    }
    plyfile.close();
}

//#####################################################################
// Function Tri_To_Tet
//#####################################################################
void Tri_To_Tet(const std::string tri_path,const std::string tet_path,const std::string output_tet_path)
{
    //Construct tet mesh
    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Update_Tetrahedron_List();
    tetrahedralized_volume.Initialize_Triangulated_Surface();
    tetrahedralized_volume.triangulated_surface->Update_Triangle_List();
    std::cout<<"total vertices = "<<tetrahedralized_volume.particles.array_collection->Size()<<std::endl;
    std::cout<<"total tetrahedra = "<<tetrahedralized_volume.mesh.elements.m<<std::endl;

    //Change particles
    TRIANGULATED_SURFACE<T>& pose_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tri_path,pose_surface);
    std::cout<<"tri vertices = "<<pose_surface.particles.array_collection->Size()<<std::endl;

    //Set  vertices
    tetrahedralized_volume.particles.X=pose_surface.particles.X;

    FILE_UTILITIES::Write_To_File<RW>(output_tet_path,tetrahedralized_volume);

}

//#####################################################################
// Function Compute_Embedding
//#####################################################################
void Compute_Embedding(const std::string tet_path,const std::string tsdf_path)
{
    //TETRAHEDRALIZED_VOLUME<T>* star_tet=TETRAHEDRALIZED_VOLUME<T>::Create();
    TRIANGULATED_SURFACE<T>* star_tet=TRIANGULATED_SURFACE<T>::Create();

    FILE_UTILITIES::Read_From_File<RW>(tet_path,*star_tet);
    star_tet->Initialize_Hierarchy();
    star_tet->Initialize_Segment_Lengths();
    star_tet->Update_Triangle_List();
    std::cout << "Shell: " << star_tet->particles.array_collection->Size() << std::endl;
    //star_tet->Initialize_Triangulated_Surface();
    //star_tet->triangulated_surface->Initialize_Hierarchy();

    cnpy::NpyArray arr = cnpy::npy_load(tsdf_path);
    std::vector<float> loaded_data = arr.as_vec<float>();
    std::cout << arr.shape[0]  <<std::endl;
    for(int i=1;i<=arr.shape[0];i++)
        std::cout << loaded_data[i] << std::endl;
}
//#####################################################################
};
}
#endif
