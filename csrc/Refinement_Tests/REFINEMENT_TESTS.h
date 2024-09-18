//#####################################################################
// Copyright 2021, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class REFINEMENT_TESTS
//#####################################################################
//#####################################################################
#ifndef __REFINEMENT_TESTS__
#define __REFINEMENT_TESTS__

#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include <PhysBAM_Dynamics/Meshing/RED_GREEN_TETRAHEDRA.h>
#include "../External/cnpy.h"
#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class REFINEMENT_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:


//#####################################################################
// Function Refine_Skinned_Tet_Mesh
//#####################################################################
void Refine_Skinned_Tet_Mesh()
{
    //std::string tet_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/star_shell.tet.gz";
    std::string tet_path="/data/jwu/D_march/tet_meshes/shell_D_march_101.tet.gz";
    std::string skinning_weights_path="/data/jwu/PhysBAM/Private_Projects/cloth_inverse_render/Data/coarseweights_flat.npy";

    TETRAHEDRALIZED_VOLUME<T>& tetrahedralized_volume=*TETRAHEDRALIZED_VOLUME<T>::Create();
    FILE_UTILITIES::Read_From_File(STREAM_TYPE((RW)1.0),tet_path,tetrahedralized_volume);
    tetrahedralized_volume.Initialize_Hierarchy();
    tetrahedralized_volume.Initialize_Triangulated_Surface();
    tetrahedralized_volume.triangulated_surface->Initialize_Hierarchy();

    RED_GREEN_TETRAHEDRA<T> redgreen(tetrahedralized_volume);
    ARRAY<int> tets_to_refine;
    std::cout<<"Tets: "<<tetrahedralized_volume.mesh.elements.m<<std::endl;
    for(int t=1;t<=tetrahedralized_volume.mesh.elements.m;t++)
        tets_to_refine.Append(t);
    std::cout<<"Refining tets..."<<std::endl;
    redgreen.Refine_Simplex_List(tets_to_refine);
    std::cout<<"Refined tets: "<<tetrahedralized_volume.mesh.elements.m<<std::endl;
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),"refined_shell_D_march_101.tet.gz",tetrahedralized_volume);
}
//#####################################################################
};
}
#endif
