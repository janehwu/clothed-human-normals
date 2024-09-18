//#####################################################################
// Copyright 2020, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <PhysBAM_Tools/Log/LOG.h>
#include <PhysBAM_Tools/Parsing/PARSE_ARGS.h>
#include <PhysBAM_Tools/Read_Write/Grids_Uniform_Arrays/READ_WRITE_ARRAYS.h>
#include <PhysBAM_Tools/Read_Write/Arrays/READ_WRITE_ARRAY.h>
#include <PhysBAM_Tools/Read_Write/Utilities/FILE_UTILITIES.h>
#include <PhysBAM_Tools/Utilities/PROCESS_UTILITIES.h>
#include <PhysBAM_Geometry/Basic_Geometry/TETRAHEDRON.h>
#include <PhysBAM_Geometry/Implicit_Objects_Uniform/LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Implicit_Objects_Uniform/READ_WRITE_LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Topology/SEGMENT_MESH.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include <PhysBAM_Dynamics/Meshing/TETRAHEDRAL_MESHING.h>
#include <PhysBAM_Dynamics/Particles/PARTICLES_FORWARD.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/DEFORMABLES_DRIVER.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Forces/IMPLICIT_ZERO_LENGTH_SPRINGS.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include "Mesh_Tests/MESH_TESTS.h"
#include "Tsdf_Tests/TSDF_TESTS.h"
#include "Embed_Tests/EMBED_TESTS.h"
#include "Fit_Tests/FIT_TESTS.h"
#include "Skin_Tests/SKIN_TESTS.h"
#include "Reconstruction_Tests/RECONSTRUCTION_TESTS.h"
#include "Refinement_Tests/REFINEMENT_TESTS.h"
#include "Rasterization_Tests/RASTERIZATION_TESTS.h"
#include "Sdf_Regularization_Tests/SDF_REGULARIZATION_TESTS.h"
//#include "Skin_Tests/SKIN_WEIGHTS_TESTS.h"

using namespace PhysBAM;

using T = double;
using RW = float;
using TV = VECTOR<T,3>;

int main(int argc,char *argv[])
{
    LOG::Initialize_Logging(false,false,1<<30,true,1);

    PROCESS_UTILITIES::Set_Floating_Point_Exception_Handling(true);
    Initialize_Read_Write_General_Structures();
    PARSE_ARGS parse_args;

    if(PARSE_ARGS::Find_And_Remove("-tet",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tri>","<tet> input tri file");
        parse_args.Set_Extra_Arguments(2,"<tet>","<tet> input tet file");
        parse_args.Set_Extra_Arguments(3,"<tet>","<tet> output tet file");
     
        parse_args.Parse(argc, argv);
        std::string input_tri_filename = parse_args.Extra_Arg(1);
        std::string input_tet_filename = parse_args.Extra_Arg(2);
        std::string output_tet_filename = parse_args.Extra_Arg(3);

        MESH_TESTS<T, RW> mesh_test;
        mesh_test.Tri_To_Tet(input_tri_filename,input_tet_filename,output_tet_filename);

    }
    else if(PARSE_ARGS::Find_And_Remove("-mesh",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tet>","<tet> input tet file");
        parse_args.Parse(argc, argv);
        std::string input_tet_filename = parse_args.Extra_Arg(1);

        MESH_TESTS<T, RW> mesh_test;
        mesh_test.Tet_To_Ply(input_tet_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-tsdf",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tet>","<tet> input tet file");
        parse_args.Set_Extra_Arguments(2, "<input tsdf>", "<input tsdf> input tsdf file");
        parse_args.Set_Extra_Arguments(3, "<output tri>", "<output tri> output tri file");
        //parse_args.Set_Extra_Arguments(3, "<output tsdf>", "<output tsdf> output tsdf file");
        //parse_args.Set_Extra_Arguments(3, "<input scan>", "<input scan> input tsdf file");
        
        parse_args.Parse(argc, argv);
        std::string input_tet_filename = parse_args.Extra_Arg(1);
        std::string input_tsdf_filename = parse_args.Extra_Arg(2);
        std::string output_tsdf_filename = parse_args.Extra_Arg(3);
        //std::string input_scan = parse_args.Extra_Arg(3);

        TSDF_TESTS<T, RW> tsdf_test;
        tsdf_test.TSDF_To_Mesh(input_tet_filename, input_tsdf_filename, output_tsdf_filename);
        //tsdf_test.Compute_Correspondences(input_tet_filename, input_tsdf_filename, input_scan);
        //tsdf_test.Project_SMPL_To_Phi(input_tet_filename, input_tsdf_filename);
        //tsdf_test.Phi_To_TSDF(input_tet_filename, input_tsdf_filename, output_tsdf_filename);
        //tsdf_test.TSDF_To_Level_Set(input_tet_filename, input_tsdf_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-boundary",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tri>","<tri> input tri file");
        parse_args.Parse(argc, argv);
        std::string input_tri_filename = parse_args.Extra_Arg(1);

        TSDF_TESTS<T, RW> tsdf_test;
        tsdf_test.Boundary_Mesh(input_tri_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-recon",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tri>","<tri> input tri file");
        parse_args.Set_Extra_Arguments(2,"<tri>","<tri> output tri file");
        parse_args.Parse(argc, argv);
        std::string input_tri_filename = parse_args.Extra_Arg(1);
        std::string output_tri_filename = parse_args.Extra_Arg(2);

        RECONSTRUCTION_TESTS<T, RW> recon_test;
        recon_test.Flood_Fill_Mesh(input_tri_filename, output_tri_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-embed",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tet>","<tet> input tet file");
        parse_args.Set_Extra_Arguments(2, "<input tri>", "<input tri> input tri file");
        parse_args.Set_Extra_Arguments(3, "<embedding>", "<embedding> output embedding file");
        parse_args.Set_Extra_Arguments(4, "<output tri>", "<output tri> output cloth file");
        parse_args.Parse(argc, argv);

        std::string input_tet_filename = parse_args.Extra_Arg(1);
        std::string input_tri_filename = parse_args.Extra_Arg(2);
        std::string embedding_filename = parse_args.Extra_Arg(3);
        std::string output_tri_filename = parse_args.Extra_Arg(4);

        EMBED_TESTS<T, RW> cloth_embedding;
        cloth_embedding.Apply_Embedding(input_tet_filename, input_tri_filename, embedding_filename, output_tri_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-fit",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tet>","<tet> input tet file");
        parse_args.Set_Extra_Arguments(2, "<input tsdf>", "<input tsdf> input tsdf file");
        parse_args.Set_Extra_Arguments(3, "<output tri>", "<output tri> output tri file");
     
        parse_args.Parse(argc, argv);
        std::string input_tet_filename = parse_args.Extra_Arg(1);
        std::string input_tsdf_filename = parse_args.Extra_Arg(2);
        std::string output_tri_filename = parse_args.Extra_Arg(3);

        FIT_TESTS<T, RW> fit_test;
        fit_test.TSDF_To_Mesh(input_tet_filename, input_tsdf_filename, output_tri_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-skin",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tri>","<tri> input tri file");
        parse_args.Set_Extra_Arguments(2, "<output weights>", "<output weights> output skinning weights file");

        parse_args.Parse(argc, argv);
        std::string input_tri_filename = parse_args.Extra_Arg(1);
        std::string output_weights_filename = parse_args.Extra_Arg(2);

        SKIN_TESTS<T, RW> skin_test;
        skin_test.Interpolate_Skinning_Weights_From_Tet_Mesh(input_tri_filename, output_weights_filename);
    }
    else if(PARSE_ARGS::Find_And_Remove("-refine",argc,argv)){
        REFINEMENT_TESTS<T, RW> refinement_test;
        refinement_test.Refine_Skinned_Tet_Mesh();
    }
    else if(PARSE_ARGS::Find_And_Remove("-raster",argc,argv)){
        RASTERIZATION_TESTS<T, RW> rasterization_test;
        //rasterization_test.Rasterize_Example();
        rasterization_test.Backprop_Rasterize_Example();
    }
    else if(PARSE_ARGS::Find_And_Remove("-regularization",argc,argv)){
        parse_args.Set_Extra_Arguments(1,"<tet>","<tet> input tet file");
        parse_args.Parse(argc, argv);
        std::string input_tet_filename = parse_args.Extra_Arg(1);

        SDF_REGULARIZATION_TESTS<T, RW> sdf_regularization_test;
        //sdf_regularization_test.Phi_To_Tet(input_tet_filename);
        sdf_regularization_test.Compute_Tetrahedra_Inverse_Matrices(input_tet_filename);
        sdf_regularization_test.Compute_Tetrahedra_Volumes(input_tet_filename);
        //sdf_regularization_test.Compute_Tetrahedral_Mesh_Boundary(input_tet_filename);
    }
    /*
    else if(PARSE_ARGS::Find_And_Remove("-skin",argc,argv)){
        SKIN_WEIGHTS_TESTS<T> *example=new SKIN_WEIGHTS_TESTS<T>(stream_type);
        // example->want_mpi_world=true;
        example->Parse(argc,argv);
        // if(example->mpi_world->initialized){
        //     example->deformable_body_collection.Set_Mpi_Solids(new MPI_SOLIDS<TV>);
        //     example->output_directory+=STRING_UTILITIES::string_sprintf("/%d",(example->deformable_body_collection.mpi_solids->rank+1));
        //     FILE_UTILITIES::Create_Directory(example->output_directory);
        //     FILE_UTILITIES::Create_Directory(example->output_directory+"/common");
        //     LOG::Instance()->Copy_Log_To_File(example->output_directory+"/common/log.txt",example->restart);
        // }

        DEFORMABLES_DRIVER<TV> driver(*example);
        driver.Execute_Main_Program();

        delete example;
    }*/
    else{
    std::cout<<"No flags passed. Exiting."<<std::endl;
    }
} 
