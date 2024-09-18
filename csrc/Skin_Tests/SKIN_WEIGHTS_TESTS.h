//#####################################################################
// Copyright 2021, Zhenglin Geng.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class SKIN_WEIGHTS_TESTS
//#####################################################################
//#####################################################################
#ifndef __SKIN_WEIGHTS_TESTS__
#define __SKIN_WEIGHTS_TESTS__


#include <PhysBAM_Tools/Data_Structures/PAIR.h>
#include <PhysBAM_Tools/Grids_Uniform/GRID.h>
#include <PhysBAM_Tools/Grids_Uniform/UNIFORM_GRID_ITERATOR_NODE.h>
#include <PhysBAM_Tools/Grids_Uniform_Arrays/ARRAYS_ND.h>
#include <PhysBAM_Tools/Grids_Uniform_Arrays/GRID_ARRAYS_POLICY_UNIFORM.h>
#include <PhysBAM_Tools/Grids_Uniform_Interpolation/LINEAR_INTERPOLATION_UNIFORM.h>
#include <PhysBAM_Tools/Log/LOG.h>
#include <PhysBAM_Tools/Parsing/PARSE_ARGS.h>
#include <PhysBAM_Tools/Read_Write/Grids_Uniform/READ_WRITE_GRID.h>
#include <PhysBAM_Tools/Read_Write/Utilities/FILE_UTILITIES.h>
#include <PhysBAM_Tools/Utilities/PROCESS_UTILITIES.h>
#include <PhysBAM_Geometry/Basic_Geometry/ORIENTED_BOX.h>
#include <PhysBAM_Geometry/Geometry_Particles/REGISTER_GEOMETRY_READ_WRITE.h>
#include <PhysBAM_Geometry/Grids_Dyadic_Computations/LEVELSET_MAKER_DYADIC.h>
#include <PhysBAM_Geometry/Grids_Uniform_Computations/LEVELSET_MAKER_UNIFORM.h>
#include <PhysBAM_Geometry/Implicit_Objects_Dyadic/DYADIC_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Implicit_Objects_Uniform/LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Grids_Dyadic_Level_Sets/READ_WRITE_LEVELSET_OCTREE.h>
#include <PhysBAM_Geometry/Read_Write/Implicit_Objects_Uniform/READ_WRITE_LEVELSET_IMPLICIT_OBJECT.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TETRAHEDRALIZED_VOLUME.h>
#include <PhysBAM_Geometry/Solids_Geometry/DEFORMABLE_GEOMETRY_COLLECTION.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/DEFORMABLES_EXAMPLE.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Bindings/BINDING_LIST.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Bindings/SOFT_BINDINGS.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Bindings/LINEAR_BINDING.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Bindings/LINEAR_BINDING_DYNAMIC.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Parallel_Computation/MPI_SOLIDS.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/DEFORMABLES_DRIVER.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Standard_Tests/DEFORMABLES_STANDARD_TESTS.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include <PhysBAM_Dynamics/Poisson_Equations/LAPLACE_SIMPLICIAL_NODE.h>
#include <PhysBAM_Dynamics/Morphing/LAPLACE_MORPH_UNIFORM.h>
#include <PhysBAM_Dynamics/Meshing/TETRAHEDRAL_MESHING.h>

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <map>

//#include "proto/cpp/skin_weights.pb.h"
//#include "../../../opengl_3d_proto/Proto/PROTO_DEBUG_UTILS.h"

using namespace std;
namespace PhysBAM{

template<class T>
class SKIN_WEIGHTS_TESTS:public DEFORMABLES_EXAMPLE<VECTOR<T,3> >
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;
    typedef typename TV::SPIN T_SPIN;
    typedef typename GRID_ARRAYS_POLICY<GRID<TV> >::ARRAYS_SCALAR T_ARRAYS_SCALAR;
    typedef typename T_ARRAYS_SCALAR::template REBIND<TV>::TYPE T_ARRAYS_VECTOR;
    typedef typename T_ARRAYS_SCALAR::template REBIND<bool>::TYPE T_ARRAYS_BOOL;
public:
    typedef DEFORMABLES_EXAMPLE<TV> BASE;
    using BASE::deformables_parameters;using BASE::data_directory;using BASE::first_frame;using BASE::last_frame;using BASE::frame_rate;using BASE::output_directory;using BASE::stream_type;using BASE::deformable_body_collection;using BASE::rigid_geometry_collection;using BASE::deformables_evolution;using BASE::parse_args;using BASE::test_number;

    DEFORMABLES_STANDARD_TESTS<TV> tests;

    ARRAY<ARRAY<int> > attached_nodes;

    enum{
        COMPUTE_TET_SKIN_WEIGHTS=1,
        COMPUTE_GRID_SKIN_WEIGHTS=2
    };

    std::string input_kdsm_tet_path,input_body_tri_path,input_body_skin_weights_path,output_path;
    int n_bones;
    int max_iters;
    int cell_size;
    T relative_tol;

    bool use_proto_debug;

    SKIN_WEIGHTS_TESTS(const STREAM_TYPE stream_type):BASE(stream_type),tests(*this,deformable_body_collection),use_proto_debug(false)
    {}

    virtual ~SKIN_WEIGHTS_TESTS() 
    {
    }

    // Unused callbacks
    void Post_Initialization() PHYSBAM_OVERRIDE {}
    void Postprocess_Solids_Substep(const T time,const int substep) PHYSBAM_OVERRIDE {}
    void Apply_Constraints(const T dt,const T time) PHYSBAM_OVERRIDE {}
    void Add_External_Impulses_Before(ARRAY_VIEW<TV> V,const T time,const T dt) PHYSBAM_OVERRIDE {}
    void Add_External_Impulses(ARRAY_VIEW<TV> V,const T time,const T dt) PHYSBAM_OVERRIDE {}
    void Add_External_Impulse(ARRAY_VIEW<TV> V,const int node,const T time,const T dt) PHYSBAM_OVERRIDE {}
    void Limit_Solids_Dt(T& dt,const T time) PHYSBAM_OVERRIDE {}
    void Set_External_Velocities(ARRAY_VIEW<TV> V,const T velocity_time,const T current_position_time) PHYSBAM_OVERRIDE {}
    void Set_External_Velocities(ARRAY_VIEW<TWIST<TV> > twist,const T velocity_time,const T current_position_time) PHYSBAM_OVERRIDE {}
    void Set_External_Velocities(ARRAY_VIEW<TV> V,ARRAY_VIEW<T_SPIN> angular_velocity,const T velocity_time,const T current_position_time) PHYSBAM_OVERRIDE {}
    void Zero_Out_Enslaved_Velocity_Nodes(ARRAY_VIEW<TV> V,const T velocity_time,const T current_position_time) PHYSBAM_OVERRIDE {}
    void Align_Deformable_Bodies_With_Rigid_Bodies() PHYSBAM_OVERRIDE {}
    void Preprocess_Solids_Substep(const T time,const int substep) PHYSBAM_OVERRIDE {}
    void Update_Solids_Parameters(const T time) PHYSBAM_OVERRIDE {}
    void Preprocess_Substep(const T dt,const T time) PHYSBAM_OVERRIDE {}
    void Postprocess_Substep(const T dt,const T time) PHYSBAM_OVERRIDE {}
    void Self_Collisions_Begin_Callback(const T time,const int substep) PHYSBAM_OVERRIDE {}
    void Filter_Velocities(const T dt,const T time,const bool velocity_update) PHYSBAM_OVERRIDE {}
    bool Set_Kinematic_Velocities(TV& V,T_SPIN& angular_velocity,const T time,const int id) PHYSBAM_OVERRIDE {return false;}
    bool Set_Kinematic_Velocities(TWIST<TV>& twist,const T time,const int id) PHYSBAM_OVERRIDE {return false;}
    bool Set_Kinematic_Positions(FRAME<TV>& frame,const T time,const int id) PHYSBAM_OVERRIDE {return false;}
    void Add_External_Forces(ARRAY_VIEW<TV> F,const T time) PHYSBAM_OVERRIDE {}
    void Update_Time_Varying_Material_Properties(const T time) PHYSBAM_OVERRIDE
    {}
    void Set_Deformable_Particle_Is_Simulated(ARRAY<bool>& particle_is_simulated) PHYSBAM_OVERRIDE {}

//#####################################################################
// Function Register_Options
//#####################################################################
void Register_Options()
{
    BASE::Register_Options();
    parse_args->Add_String_Argument("-input_kdsm_tet", "kdsm_tpose.tet.gz",""); 
    parse_args->Add_String_Argument("-input_body_tri","body.tri.gz","");
    parse_args->Add_String_Argument("-input_body_skin_weights","","");
    parse_args->Add_String_Argument("-output", "diffuse_dirichlet_kdsm_skin_weights.txt");
    parse_args->Add_Integer_Argument("-n_bones",163,"number of bones");
    parse_args->Add_Integer_Argument("-max_iters",10000,"max number of cg iterations");
    parse_args->Add_Double_Argument("-relative_tol",1e-6,"");
    parse_args->Add_Integer_Argument("-cell_size",10,"");
}
//#####################################################################
// Function Parse_Options
//#####################################################################
void Parse_Options()
{
    BASE::Parse_Options();
    std::cout<<"Running Test Number "<<test_number<<std::endl;
    input_kdsm_tet_path = parse_args->Get_String_Value("-input_kdsm_tet");
    input_body_tri_path=parse_args->Get_String_Value("-input_body_tri");
    input_body_skin_weights_path = parse_args->Get_String_Value("-input_body_skin_weights");
    output_path = parse_args->Get_String_Value("-output");
    n_bones=parse_args->Get_Integer_Value("-n_bones");
    max_iters=parse_args->Get_Integer_Value("-max_iters");
    relative_tol=parse_args->Get_Double_Value("-relative_tol");
    cell_size=parse_args->Get_Integer_Value("-cell_size");
}
//#####################################################################
// Function Parse_Late_Options
//#####################################################################
void Parse_Late_Options() PHYSBAM_OVERRIDE {BASE::Parse_Late_Options();}
//#####################################################################
// Function Initialize_Bodies
//#####################################################################
void Initialize_Bodies() PHYSBAM_OVERRIDE
{
    PARTICLES<TV>& particles=deformable_body_collection.particles;
    BINDING_LIST<TV>& binding_list=deformable_body_collection.binding_list;
    SOFT_BINDINGS<TV>& soft_bindings=deformable_body_collection.soft_bindings;

    switch(test_number){
        case COMPUTE_TET_SKIN_WEIGHTS:{
            Compute_Tet_Skin_Weights();
            last_frame=1;
            break;
        }
        case COMPUTE_GRID_SKIN_WEIGHTS:{
            Compute_Grid_Skin_Weights();
            last_frame=1;
            break;
        }
        default:
            std::cout<<"unrecognized test_number:"<<test_number<<std::endl;
            PHYSBAM_ASSERT(false);
            break;
    }

    // correct number nodes
    for(int i=1;i<=deformable_body_collection.deformable_geometry.structures.m;i++) deformable_body_collection.deformable_geometry.structures(i)->Update_Number_Nodes();
    // correct mass
    binding_list.Distribute_Mass_To_Parents();
    binding_list.Clear_Hard_Bound_Particles(particles.mass);
    particles.Compute_Auxiliary_Attributes(soft_bindings);
    soft_bindings.Set_Mass_From_Effective_Mass();

    if(deformable_body_collection.mpi_solids){
        deformable_body_collection.mpi_solids->KD_Tree_Partition(deformable_body_collection,rigid_geometry_collection,ARRAY<TV>(particles.X));}
}
//#####################################################################
// Function Set_Deformable_Particle_Is_Simulated
//#####################################################################
//#####################################################################
// Function Read_Output_Files_Solids
//#####################################################################
void Read_Output_Files_Solids(const int frame) PHYSBAM_OVERRIDE
{
    BASE::Read_Output_Files_Solids(frame);
    deformable_body_collection.Update_Simulated_Particles();
}
//#####################################################################
// Function Write_Output_Files
//#####################################################################
void Write_Output_Files(const int frame) const PHYSBAM_OVERRIDE
{
    BASE::Write_Output_Files(frame);
    //if(use_proto_debug)PROTO_DEBUG_UTILS<TV>::Write_Output_Files(output_directory,frame);
}
//#####################################################################
// Function Preprocess_Frame
//#####################################################################
void Preprocess_Frame(const int frame) PHYSBAM_OVERRIDE
{
}
//#####################################################################
// Function Postprocess_Frame
//#####################################################################
void Postprocess_Frame(const int frame) PHYSBAM_OVERRIDE
{
}
//#####################################################################
// Function Set_External_Positions
//#####################################################################
void Set_External_Positions(ARRAY_VIEW<TV> X,const T time) PHYSBAM_OVERRIDE
{
}
//#####################################################################
// Function Zero_Out_Enslaved_Position_Nodes
//#####################################################################
void Zero_Out_Enslaved_Position_Nodes(ARRAY_VIEW<TV> X,const T time) PHYSBAM_OVERRIDE
{
    for(int i=1;i<=attached_nodes.m;i++) for(int j=1;j<=attached_nodes(i).m;j++) X(attached_nodes(i)(j))=TV();
}
//#####################################################################
// Function Set_External_Positions
//#####################################################################
void Set_External_Positions(ARRAY_VIEW<TV> X,ARRAY_VIEW<ROTATION<TV> > rotation,const T time) PHYSBAM_OVERRIDE
{
}
//#####################################################################
// Function Clean_Skin_Weights
//#####################################################################
void Clean_Skin_Weights(ARRAY<ARRAY<T> > &skin_weights)
{
    T eps=1e-3;
    int n_particles=skin_weights.m;
    for(int i=1;i<=skin_weights.m;i++){
        for(int j=1;j<=skin_weights(i).m;j++){
            if(skin_weights(i)(j)<eps){
                skin_weights(i)(j)=0;
            }
        }
    }
}
//#####################################################################
// Function Load_Skin_Weights
//#####################################################################
void Load_Skin_Weights(const std::string &skin_weights_path,ARRAY<ARRAY<T> > &skin_weights,const int n_bones)
{
    std::ifstream fin(skin_weights_path);
    if(!fin.good()){
        std::cout<<"Cannot find "<<skin_weights_path<<std::endl;
        PHYSBAM_ASSERT(false);
    }

    ARRAY<ARRAY<T> > transposed_skin_weights;
    int vt_i=1;
    std::string line;std::getline(fin,line);
    for (std::string line; std::getline(fin, line); ) {
        if(line.length()==0)
            continue;
        ARRAY<T> weights(n_bones);ARRAYS_COMPUTATIONS::Fill(weights,0.);
        std::stringstream ss(line);
        double weight_sum = 0;
        while (ss) {
            std::string bone_weight; ss >> bone_weight;
            size_t delim = bone_weight.find(":"); 
            if (delim == std::string::npos) continue;
            int bone_id = std::stoi(bone_weight.substr(0, delim))+1;
            if(bone_id>n_bones){
                std::cout<<"bone_id:"<<bone_id<<",n_bones:"<<n_bones<<std::endl;
            }
            PHYSBAM_ASSERT(bone_id<=n_bones);
            double w = std::stod(bone_weight.substr(delim + 1, std::string::npos));
            if(w<0) w=0;
            weights(bone_id)=w;
            weight_sum += w;
        }
        for (int j = 1; j <= n_bones; j++) weights(j) /= weight_sum;
        transposed_skin_weights.Append(weights);
    }
    int n_particles=transposed_skin_weights.m;

    skin_weights.Clean_Memory();
    skin_weights.Resize(n_bones);
    for(int bone_id=1;bone_id<=n_bones;bone_id++){
        skin_weights(bone_id).Resize(n_particles);
        for(int particle_id=1;particle_id<=n_particles;particle_id++){
            skin_weights(bone_id)(particle_id)=transposed_skin_weights(particle_id)(bone_id);
        }
    }
}
//#####################################################################
// Function Write_Skin_Weights
//#####################################################################
void Write_Skin_Weights(const string &output_path, const ARRAY<ARRAY<T> > &skin_weights, bool normalize=true) {
    ofstream fout(output_path);
    int n_bones=skin_weights.m;
    int n_particles = skin_weights(1).m;
    for (int i = 1; i <= n_particles; i++) {
        // T weight_sum = 0;
        // for (int bone_id = 1; bone_id <= n_bones; bone_id++) {
        //     if (skin_weights(bone_id).m != n_particles) PHYSBAM_FATAL_ERROR();
        //     weight_sum += skin_weights(bone_id)(i);
        // }
        // if (weight_sum < 1e-3) {
        //     PHYSBAM_FATAL_ERROR(STRING_UTILITIES::string_sprintf("particle:%d,weight_sum:%f\n",i,weight_sum));
        // }
        for (int bone_id = 1; bone_id <= n_bones; bone_id++) {
            // T w = (normalize ? skin_weights(bone_id)(i) / weight_sum : skin_weights(bone_id)(i));
            // if (w > 0) fout << bone_id - 1 << ":" << w << " ";
            fout << bone_id - 1 << ":" << skin_weights(bone_id)(i) << " ";
        }
        fout << endl;
    }
    fout.close();
}
//#####################################################################
// Function Write_Skin_Weights_Proto
//#####################################################################
void Write_Skin_Weights_Proto(const string &output_path, const ARRAY<ARRAY<T> > &skin_weights, bool normalize=true) {
    int n_bones=skin_weights.m;
    int n_particles = skin_weights(1).m;
    SparseSkinWeights skin_weights_pb;
    for (int i = 1; i <= n_particles; i++) {
        auto vertex_weights_pb=skin_weights_pb.add_vertex_weights();
        for (int bone_id = 1; bone_id <= n_bones; bone_id++) {
            T w=skin_weights(bone_id)(i);
            if(w!=0){
                auto weight_pair=vertex_weights_pb->add_weight_pairs();
                weight_pair->set_bone_id(bone_id-1);
                weight_pair->set_bone_w(w);                
            }
        }
    }
    std::ofstream fout(output_path,ios::out|ios::trunc|ios::binary);
    if(!skin_weights_pb.SerializeToOstream(&fout)){
        std::cout<<"failed to write to "<<output_path<<std::endl;
    }
}
//#####################################################################
// Function Compute_Tet_Skin_Weights
//#####################################################################
void Compute_Tet_Skin_Weights()
{
    TETRAHEDRALIZED_VOLUME<T> *tet_volume = TETRAHEDRALIZED_VOLUME<T>::Create(); 
    FILE_UTILITIES::Read_From_File(stream_type,input_kdsm_tet_path,*tet_volume);

    TRIANGULATED_SURFACE<T> *body_surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(stream_type,input_body_tri_path,*body_surface);

    ARRAY<ARRAY<T> > body_skin_weights; 
    Load_Skin_Weights(input_body_skin_weights_path,body_skin_weights,n_bones);
    PHYSBAM_ASSERT(body_skin_weights(1).m==body_surface->particles.X.m);
    // exit(0);

    POINT_CLOUD_SUBSET<TV,GEOMETRY_PARTICLES<TV> > particle_subset(tet_volume->particles);
    particle_subset.array_collection->number = tet_volume->particles.array_collection->Size();
    particle_subset.active_indices = IDENTITY_ARRAY<>(tet_volume->particles.array_collection->Size());
    particle_subset.Update_Subset_Index_From_Element_Index();

    int n_particles=tet_volume->particles.X.m;
    ARRAY<T> u(n_particles);
    LAPLACE_SIMPLICIAL_NODE<TV, 3> laplace_simplicial_node(particle_subset, *tet_volume, u);
    laplace_simplicial_node.Initialize_Object();
    ARRAYS_COMPUTATIONS::Fill(laplace_simplicial_node.psi_D,false);
    laplace_simplicial_node.Initialize_Dirichlet_Boundary(*body_surface,1e-4,false);
    // const auto &psi_D_edges=laplace_simplicial_node.psi_D_edges;
    // for(int i=1;i<=psi_D_edges.m;i++){
    //     if(psi_D_edges(i)(1).simplex>0){
    //         const TV &weights1=psi_D_edges(i)(1).weights;
    //         if(weights1(1)<0||weights1(2)<0||weights1(3)<0){
    //             std::cout<<"edge:"<<i<<",weights1"<<weights1<<std::endl;
    //         }
    //         const TV &weights2=psi_D_edges(i)(1).weights;
    //         if(weights2(1)<0||weights2(2)<0||weights2(3)<0){
    //             std::cout<<"edge:"<<i<<",weights2"<<weights2<<std::endl;
    //         }
    //     }
    // }
    laplace_simplicial_node.Set_Relative_Tolerance(relative_tol);
    laplace_simplicial_node.Set_Absolute_Tolerance(1e-11);
    laplace_simplicial_node.pcg.Use_Conjugate_Gradient();
    laplace_simplicial_node.pcg.Set_Maximum_Iterations(max_iters);
    laplace_simplicial_node.pcg.show_residual=true;
    laplace_simplicial_node.pcg.show_results=true;

    TETRAHEDRALIZED_VOLUME<T> *tet_copy=Copy_Structure(tet_volume);
    tests.Copy_And_Add_Structure(*tet_copy);
    TRIANGULATED_SURFACE<T> *body_copy=Copy_Structure(body_surface);
    tests.Copy_And_Add_Structure(*body_copy);

    ARRAY<ARRAY<T> > tet_skin_weights(n_bones);
    int test_id=145;
    // for(int bone_id=1;bone_id<=n_bones;bone_id++){
    for(int bone_id=test_id;bone_id<=test_id;bone_id++){
        tet_skin_weights(bone_id).Resize(n_particles);
        ARRAYS_COMPUTATIONS::Fill(u,0.);
        PHYSBAM_ASSERT(laplace_simplicial_node.u_boundary.Size()==body_surface->particles.X.m);
        laplace_simplicial_node.u_boundary=body_skin_weights(bone_id);
        laplace_simplicial_node.Solve();
        tet_skin_weights(bone_id)=u;
        Vlz_Skin_Weights(u,tet_volume->particles.X);
        std::cout<<"Max:"<<ARRAYS_COMPUTATIONS::Max(u)<<",Min:"<<ARRAYS_COMPUTATIONS::Min(u)<<std::endl;
        std::cout<<"finish bone "<<bone_id<<std::endl;
    }

    // Clean_Skin_Weights(tet_skin_weights);
    // std::cout<<"write to "<<output_path<<std::endl;
    // Write_Skin_Weights(output_path, tet_skin_weights, /*normalize*/false);

    delete tet_volume;
    delete body_surface;
}
//#####################################################################
// Function Compute_Grid_Skin_Weights
//#####################################################################
void Compute_Grid_Skin_Weights()
{
    TETRAHEDRALIZED_VOLUME<T> *tet_volume = TETRAHEDRALIZED_VOLUME<T>::Create(); 
    FILE_UTILITIES::Read_From_File(stream_type,input_kdsm_tet_path,*tet_volume);

    TRIANGULATED_SURFACE<T> *body_surface=TRIANGULATED_SURFACE<T>::Create();
    FILE_UTILITIES::Read_From_File(stream_type,input_body_tri_path,*body_surface);
    body_surface->Initialize_Hierarchy();

    TETRAHEDRALIZED_VOLUME<T> *tet_copy=Copy_Structure(tet_volume);
    tests.Copy_And_Add_Structure(*tet_copy);
    TRIANGULATED_SURFACE<T> *body_copy=Copy_Structure(body_surface);
    tests.Copy_And_Add_Structure(*body_copy);

    ARRAY<ARRAY<T> > body_skin_weights; 
    Load_Skin_Weights(input_body_skin_weights_path,body_skin_weights,n_bones);
    PHYSBAM_ASSERT(body_skin_weights(1).m==body_surface->particles.X.m);

    T domain_scale_factor=1.;
    RANGE<TV> mac_grid_domain=RANGE<TV>::Bounding_Box(tet_volume->particles.X);
    mac_grid_domain.Scale_About_Center(domain_scale_factor);
    std::cout<<"cell_size:"<<cell_size<<std::endl;
    GRID<TV> mac_grid=GRID<TV>::Create_Even_Sized_Grid_Given_Cell_Size(mac_grid_domain,cell_size,true);

    POINT_CLOUD_SUBSET<TV,GEOMETRY_PARTICLES<TV> > particle_subset(tet_volume->particles);
    particle_subset.array_collection->number = tet_volume->particles.array_collection->Size();
    particle_subset.active_indices = IDENTITY_ARRAY<>(tet_volume->particles.array_collection->Size());
    particle_subset.Update_Subset_Index_From_Element_Index();

    int n_particles=tet_volume->particles.X.m;

    T_ARRAYS_SCALAR u_grid(mac_grid.Domain_Indices(1));
    ARRAY<T> u_X(n_particles);
    ARRAY<T> u_interface(body_surface->particles.X.m);

    LAPLACE_MORPH_UNIFORM<GRID<TV> > laplace_morph(mac_grid,u_grid,true,false,*body_surface,u_interface);
    laplace_morph.Set_Neumann_Outer_Boundaries();
    laplace_morph.pcg.Enforce_Compatibility(false);

    ARRAY<ARRAY<T> > tet_skin_weights(n_bones);
    // int test_id=145;
    for(int bone_id=1;bone_id<=n_bones;bone_id++){
    // for(int bone_id=test_id;bone_id<=test_id;bone_id++){
        tet_skin_weights(bone_id).Resize(n_particles);
        u_grid.Fill(0.);
        ARRAYS_COMPUTATIONS::Fill(u_X,0.);
        u_interface=body_skin_weights(bone_id);
        laplace_morph.Solve((T)0.,false);
        Interpolate_u_From_X_On_Grid(mac_grid,u_grid,tet_volume->particles.X,u_X);
        tet_skin_weights(bone_id)=u_X;
        // Vlz_Skin_Weights(u,tet_volume->particles.X);
        std::cout<<"Max:"<<ARRAYS_COMPUTATIONS::Max(u_X)<<",Min:"<<ARRAYS_COMPUTATIONS::Min(u_X)<<std::endl;
        std::cout<<"finish bone "<<bone_id<<std::endl;
    }

    // Clean_Skin_Weights(tet_skin_weights);
    std::cout<<"write to "<<output_path<<std::endl;
    Write_Skin_Weights_Proto(output_path, tet_skin_weights, /*normalize*/false);

    delete tet_volume;
    delete body_surface;
}
//#####################################################################
// Function Vlz_Skin_Weights
//#####################################################################
void Vlz_Skin_Weights(const ARRAY<T> &skin_weights,const ARRAY_VIEW<const TV> &X,T range=-0.1)
{
    for(int i=1;i<=skin_weights.m;i++){
        if(skin_weights(i)<0){
            T t=skin_weights(i);
            if(t<range){
                t=range;
            }
            t/=range;
            const TV neutral_color(0.8,0.8,0.8);
            const TV negative_color(1.0,0,0);
            TV color=(1-t)*neutral_color+t*negative_color;
            PROTO_DEBUG_UTILS<TV>::Add_Point(X(i),color);
        }
    }
}
//#####################################################################
// Function Copy_Structure
//#####################################################################
template<class STRUCTURE_T> 
STRUCTURE_T* Copy_Structure(const STRUCTURE_T *structure) const
{
    STRUCTURE_T *copy=STRUCTURE_T::Create();
    int num=structure->particles.array_collection->Size();
    copy->particles.array_collection->Add_Elements(num);copy->particles.X=structure->particles.X;
    copy->mesh.Initialize_Mesh(num,structure->mesh.elements);
    return copy;
}
//#####################################################################
// Function Interpolate_u_From_X_On_Grid
//#####################################################################
void Interpolate_u_From_X_On_Grid(const GRID<TV> &mac_grid,const T_ARRAYS_SCALAR &u_grid,const ARRAY_VIEW<const TV> &X,ARRAY<T> &u_X) const
{
    LINEAR_INTERPOLATION_UNIFORM<GRID<TV>,T> interpolation;
    for(int i=1;i<=X.m;i++){
        u_X(i)+=interpolation.Clamped_To_Array(mac_grid,u_grid,X(i));
    }
}
//#####################################################################
};
}
#endif
