//#####################################################################
// Copyright 2022, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class SDF_REGULARIZATION_TESTS
//#####################################################################
//#####################################################################
#ifndef __SDF_REGULARIZATION_TESTS__
#define __SDF_REGULARIZATION_TESTS__

#include <PhysBAM_Geometry/Spatial_Acceleration/TETRAHEDRON_HIERARCHY.h>
#include <PhysBAM_Geometry/Topology/TETRAHEDRON_MESH.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include <PhysBAM_Tools/Matrices/MATRIX_3X3.h>
#include <PhysBAM_Geometry/Basic_Geometry/TETRAHEDRON.h>
#include <PhysBAM_Dynamics/Meshing/TETRAHEDRAL_MESHING.h>
#include <PhysBAM_Geometry/Grids_Uniform_Computations/LEVELSET_MAKER_UNIFORM.h>
#include <PhysBAM_Geometry/Implicit_Objects_Uniform/LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Read_Write/Implicit_Objects_Uniform/READ_WRITE_LEVELSET_IMPLICIT_OBJECT.h>


#include "../External/cnpy.h"
//#include "Utils/DEBUG_VIZ_UTIL.h"
//#include "../opengl_3d_proto/Proto/PROTO_DEBUG_UTILS.h"
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class SDF_REGULARIZATION_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Compute_Tetrahedra_Volumes
//#####################################################################
void Compute_Tetrahedra_Volumes(const std::string tet_path)
{
    TETRAHEDRALIZED_VOLUME<T>* tetrahedralized_volume=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tet_path,*tetrahedralized_volume);
    tetrahedralized_volume->Initialize_Hierarchy();
    tetrahedralized_volume->Update_Tetrahedron_List();
    tetrahedralized_volume->Compute_Nodal_And_Tetrahedron_Volumes();

    int num_tetrahedra=tetrahedralized_volume->mesh.elements.m;
    std::cout<<"Tet mesh vertices: "<<tetrahedralized_volume->particles.array_collection->Size()<<std::endl;
    std::cout<<"Tet mesh elements: "<<num_tetrahedra<<std::endl;

    //Open the output file
    std::ofstream volume_file;
    volume_file.open("tetrahedra_volumes.txt");
    for(int t=1;t<=num_tetrahedra;t++){
        T element_size = tetrahedralized_volume->Element_Size(t);
        volume_file<<element_size<<"\n";
    }
    volume_file.close();
    std::cout<<"Done"<<std::endl;
}

//#####################################################################
// Function Compute_Tetrahedral_Mesh_Boundary
//#####################################################################
void Compute_Tetrahedral_Mesh_Boundary(const std::string tet_path)
{
    TETRAHEDRALIZED_VOLUME<T>* tetrahedralized_volume=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tet_path,*tetrahedralized_volume);
    tetrahedralized_volume->Initialize_Hierarchy();
    tetrahedralized_volume->Update_Tetrahedron_List();
    tetrahedralized_volume->Compute_Nodal_And_Tetrahedron_Volumes();

    int num_tetrahedra=tetrahedralized_volume->mesh.elements.m;
    std::cout<<"Tet mesh vertices: "<<tetrahedralized_volume->particles.array_collection->Size()<<std::endl;
    std::cout<<"Tet mesh elements: "<<num_tetrahedra<<std::endl;

    // Fix phi if negative on boundary
    TRIANGULATED_SURFACE<T>& tet_boundary=tetrahedralized_volume->Get_Boundary_Object();
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"star_shell_boundary.tri.gz",tet_boundary);
}

//#####################################################################
// Function Compute_Tetrahedra_Inverse_Matrices
//#####################################################################
void Compute_Tetrahedra_Inverse_Matrices(const std::string tet_path)
{
    TETRAHEDRALIZED_VOLUME<T>* tetrahedralized_volume=TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File<RW>(tet_path,*tetrahedralized_volume);
    tetrahedralized_volume->Initialize_Hierarchy();
    tetrahedralized_volume->Update_Tetrahedron_List();

    int num_tetrahedra=tetrahedralized_volume->mesh.elements.m;
    std::cout<<"Tet mesh vertices: "<<tetrahedralized_volume->particles.array_collection->Size()<<std::endl;
    std::cout<<"Tet mesh elements: "<<num_tetrahedra<<std::endl;

    //Open the output files
    std::ofstream delta_matrix_file;
    std::ofstream delta_indices_file;
    delta_matrix_file.open("delta_inverse_matrix.txt");
    delta_indices_file.open("delta_matrix_indices.txt");

    int i,j,k,l;
    int source_idx;
    for(int t=1;t<=num_tetrahedra;t++){
        tetrahedralized_volume->mesh.elements(t).Get(i,j,k,l);
        // For each vertex position, use 3 connecting edges to compute matrix.
        // Take the solution whose inverse matrix has the largest determinant.
        MATRIX<T,3,3> final_inv_matrix;
        T largest_determinant = -9999999999.;
        ARRAY<int>final_edge_idx(3);
        int final_s=-1;
        for(int s=1;s<=4;s++){
            ARRAY<int> edge_idx(3);
            switch(s){
                case 1:
                    source_idx=i;
                    edge_idx(1)=j;
                    edge_idx(2)=k;
                    edge_idx(3)=l;
                    break;
                case 2:
                    source_idx=j;
                    edge_idx(1)=i;
                    edge_idx(2)=k;
                    edge_idx(3)=l;
                    break;
                case 3:
                    source_idx=k;
                    edge_idx(1)=i;
                    edge_idx(2)=j;
                    edge_idx(3)=l;
                    break;
                case 4:
                    source_idx=l;
                    edge_idx(1)=i;
                    edge_idx(2)=j;
                    edge_idx(3)=k;
                    break;
            }
            MATRIX<T,3,3> deltas;
            // For each edge, compute per-axis lengths and add to matrix.
            for(int e=1;e<=3;e++){
                TV edge_lengths = tetrahedralized_volume->particles.X(source_idx) - tetrahedralized_volume->particles.X(edge_idx(e));
                deltas(e,1) = edge_lengths(1); //dx
                deltas(e,2) = edge_lengths(2); //dy
                deltas(e,3) = edge_lengths(3); //dz
            }
            deltas.Invert();
            T determinant = deltas.Determinant();
            if(determinant>largest_determinant){
                largest_determinant=determinant;
                final_inv_matrix=deltas;
                final_s=source_idx;
                final_edge_idx=edge_idx;
            }
        }
        PHYSBAM_ASSERT(largest_determinant > 0);

        for(int m=1;m<=3;m++){
            for(int n=1;n<=3;n++){
                delta_matrix_file<<final_inv_matrix(m,n)<<" ";
            }
        }
        delta_matrix_file<<"\n";
        delta_indices_file<<final_s<<" ";
        for(int m=1;m<=3;m++){
            delta_indices_file<<final_edge_idx(m)<<" ";
        }
        delta_indices_file<<"\n";
    }
    delta_matrix_file.close();
    delta_indices_file.close();
    std::cout<<"Done"<<std::endl;
}


//#####################################################################
// Function Phi_To_Tet
//#####################################################################
void Phi_To_Tet(const std::string phi_filename) {
    T curvature_subdivision_threshold = 0.6;
    int max_subdivision_levels = 1;
    T bcc_lattice_cell_size = 0.05;
 
    LEVELSET_IMPLICIT_OBJECT<TV>* surface = LEVELSET_IMPLICIT_OBJECT<TV>::Create();
    FILE_UTILITIES::Read_From_File<RW>(phi_filename, *surface);

    GRID<TV> grid = surface->levelset.grid;
    STREAM_TYPE ST((RW()));
    TETRAHEDRAL_MESHING<T> tet(ST);

    ARRAY<T,VECTOR<int,3> > phi(grid.Domain_Indices());
    phi.Fill(FLT_MAX);
    T fmm_stopping_distance=0.;
    T phi_offset=0.;
    ARRAY<VECTOR<int,3> > initialized_indices;
    initialized_indices.Exact_Resize(0);initialized_indices.Preallocate(20);

    // Compute distances from grid boundary for each cell
    for(int i=1;i<=grid.counts.x;i++){
        for(int j=1;j<=grid.counts.y;j++){
            for(int k=1;k<=grid.counts.z;k++){
                // Boundary of the mesh
                if(i==1 || i==grid.counts.x || j==1 || j==grid.counts.y || k==1 || k==grid.counts.z) {
                    initialized_indices.Append(TV_INT(i,j,k));
                    phi(i,j,k)=0.;
                }
                // Otherwise, distance is smallest index
                else{
                   T dx=min(i,grid.counts.x-i);
                   T dy=min(j,grid.counts.y-j);
                   T dz=min(k,grid.counts.z-k);
                   phi(i,j,k)=-min(dx,min(dy,dz));
                }
            }
        }
    }
    /*
    for(int i=1;i<=grid.counts.x;i++) for(int j=1;j<=grid.counts.y;j++) for(int k=1;k<=grid.counts.z;k++) 
        phi(i,j,k)=clamp(phi(i,j,k),-10*grid.min_dX,10*grid.min_dX); // clamp away from FLT_MAX to avoid floating point exceptions
    GRID<TV> grid_copy=grid;LEVELSET_3D<GRID<TV> > levelset(grid_copy,phi);
    levelset.Fast_Marching_Method(0,fmm_stopping_distance,phi_offset?&initialized_indices:0);
    */
    LEVELSET_IMPLICIT_OBJECT<TV> phi_grid(grid, phi);
    FILE_UTILITIES::Write_To_File<RW>("grid_boundary.phi.gz", phi_grid);

    ///////////////////////////////

    // Generate a tet mesh from the grid that was used.
    tet.Initialize(&phi_grid);
    phi_grid.Compute_Normals();
    phi_grid.Update_Box();

    //T bcc_lattice_cell_size = (T)0;
    bool use_adaptive_refinement = true;
    // int max_subdivision_levels = max_subdivision_levels;
    // T curvature_subdivision_threshold = (T)curvature_subdivision_threshold;
    T interpolation_error_subdivision_threshold = (T).5;//.08;
    int number_of_initial_optimization_steps = 0;
    int number_of_final_optimization_steps = 0;

    std::cout<<"Creating initial mesh..."<<std::endl;
    tet.Set_Curvature_Subdivision_Threshold(curvature_subdivision_threshold);
    tet.Set_Interpolation_Error_Subdivision_Threshold(interpolation_error_subdivision_threshold);
    tet.Create_Initial_Mesh(bcc_lattice_cell_size, use_adaptive_refinement, max_subdivision_levels, true, false);
    // tet.Initialize_Optimization();
    // tet.Create_Final_Mesh_With_Optimization(number_of_initial_optimization_steps, number_of_final_optimization_steps);
    // tet.Snap_Nodes_To_Level_Set_Boundary(5);

    std::cout<<"Writing output..."<<std::endl;
    std::string tet_filename = "grid.tet.gz";
    FILE_UTILITIES::Write_To_File(ST, tet_filename, *tet.tetrahedralized_volume);
    //Tet_To_Obj<T>(tet.tetrahedralized_volume, FILE_UTILITIES::Get_Basename(tet_filename) + ".v.obj");
    delete surface;

}


//#####################################################################
};
}
#endif
