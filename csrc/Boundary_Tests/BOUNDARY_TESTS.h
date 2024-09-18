//#####################################################################
// Copyright 2021, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class BOUNDARY_TESTS
//#####################################################################
//#####################################################################
#ifndef __BOUNDARY_TESTS__
#define __BOUNDARY_TESTS__

#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>

#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class BOUNDARY_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Compute_Boundary_Mesh
//#####################################################################
void Compute_Boundary_Mesh(const std::string tri_path)
{
    TRIANGULATED_SURFACE<T>& triangulated_surface=*TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tri_path,triangulated_surface);

    triangulated_surface.Update_Triangle_List();
    triangulated_surface.Initialize_Hierarchy();

    triangulated_surface.mesh.Initialize_Node_On_Boundary();
    triangulated_surface.mesh.Initialize_Neighbor_Nodes();
    triangulated_surface.mesh.Initialize_Incident_Elements();
    triangulated_surface.mesh.Initialize_Boundary_Mesh();
    triangulated_surface.mesh.Initialize_Edge_Triangles();
    std::cout<<"Num triangles: "<<triangulated_surface.mesh.elements.Size()<<std::endl;
    std::cout<<"Boundary elements: "<<(*triangulated_surface.mesh.boundary_mesh).elements.m<<std::endl;
}
//#####################################################################
};
}
#endif
