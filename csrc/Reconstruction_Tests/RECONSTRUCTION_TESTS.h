//#####################################################################
// Copyright 2023, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class RECONSTRUCTION_TESTS
//#####################################################################
//#####################################################################
#ifndef __RECONSTRUCTION_TESTS__
#define __RECONSTRUCTION_TESTS__

#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Geometry_Particles/GEOMETRY_PARTICLES.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include <PhysBAM_Tools/Data_Structures/FLOOD_FILL_GRAPH.h>
#include <PhysBAM_Tools/Data_Structures/GRAPH.h>

#include <iostream>
#include <fstream>
#include <ios>
#include <iterator>

namespace PhysBAM{

template<class T, class RW>
class RECONSTRUCTION_TESTS
{
    typedef VECTOR<T,3> TV;
    typedef VECTOR<int,3> TV_INT;

public:

//#####################################################################
// Function Flood_Fill_Mesh
//#####################################################################
void Flood_Fill_Mesh(const std::string tri_path, const std::string out_path)
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
    std::cout<<"Num vertices: "<<triangulated_surface.particles.array_collection->Size()<<std::endl;
    std::cout<<"Num triangles: "<<triangulated_surface.mesh.elements.Size()<<std::endl;

    GRAPH graph(triangulated_surface.particles.array_collection->Size());
    // Add all triangle mesh edges to graph
    for(int t=1;t<=triangulated_surface.mesh.elements.m;t++){
        int i=triangulated_surface.mesh.elements(t)(1);
        int j=triangulated_surface.mesh.elements(t)(2);
        int k=triangulated_surface.mesh.elements(t)(3);
        graph.Add_Undirected_Edge(i,j);
        graph.Add_Undirected_Edge(j,k);
        graph.Add_Undirected_Edge(k,i);
    }
    std::cout<<"Starting flood fill..."<<std::endl;
    FLOOD_FILL_GRAPH flood_fill;
    ARRAY<int> filled_region_colors;
    ARRAY<bool> filled_region_touches_dirichlet;
    filled_region_colors.Resize(triangulated_surface.particles.array_collection->Size(),false,false);
    
    int number_of_regions=flood_fill.Flood_Fill(graph,filled_region_colors,&filled_region_touches_dirichlet);
    std::cout<<"Number of regions: "<<number_of_regions<<std::endl;

    ARRAY<int> color_counts(number_of_regions);
    for(int i=1;i<=number_of_regions;i++)
        color_counts(i)=0;
    for(int i=1;i<=filled_region_colors.Size();i++){
        int color=filled_region_colors(i);
        color_counts(color)+=1;
    }

    // Only keep the largest region
    int max_region=-1;
    int max_count=-1;
    for(int i=1;i<=number_of_regions;i++){
        if(color_counts(i)>max_count){
            max_region=i;
            max_count=color_counts(i);
        }
    }
    std::cout<<"Max region: "<<max_region<<", "<<max_count<<std::endl;

    // Just prune triangles, trimesh loading will prune/reorder vertices
    ARRAY<VECTOR<int,3> > triangles;
    for(int t=1;t<=triangulated_surface.mesh.elements.m;t++){
        int i=triangulated_surface.mesh.elements(t)(1);
        int j=triangulated_surface.mesh.elements(t)(2);
        int k=triangulated_surface.mesh.elements(t)(3);
        if(filled_region_colors(i)==max_region && filled_region_colors(j)==max_region && filled_region_colors(k)==max_region)
            triangles.Append(triangulated_surface.mesh.elements(t));
    }

    triangulated_surface.mesh.elements = triangles;
    FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),out_path,triangulated_surface);
}

//#####################################################################
// Function Remove_Outlier_Triangles
//#####################################################################
void Remove_Outlier_Triangles(const std::string tri_path, const std::string out_path)
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

    ARRAY<VECTOR<int,3> > triangles;
    T threshold = 0.01;
    T avg_edge_length=0;
    T max_edge_length=-1;
    for(int t=1;t<=triangulated_surface.mesh.elements.Size();t++){
        int i=triangulated_surface.mesh.elements(t)(1);
        int j=triangulated_surface.mesh.elements(t)(2);
        int k=triangulated_surface.mesh.elements(t)(3);
        T edge_length = max((triangulated_surface.particles.X(i)-triangulated_surface.particles.X(j)).Magnitude_Squared(),
                            (triangulated_surface.particles.X(j)-triangulated_surface.particles.X(k)).Magnitude_Squared(),
                            (triangulated_surface.particles.X(k)-triangulated_surface.particles.X(i)).Magnitude_Squared());
        edge_length = sqrt(edge_length);
        avg_edge_length += edge_length;
        if(edge_length>max_edge_length) max_edge_length=edge_length;

        //Add triangle if max edge length less than threshold
        if(edge_length<threshold) triangles.Append(triangulated_surface.mesh.elements(t));
    }
    avg_edge_length = avg_edge_length/triangulated_surface.mesh.elements.Size();
    std::cout<<"Average edge length: "<<avg_edge_length<<", "<<triangulated_surface.Average_Edge_Length()<<std::endl;
    std::cout<<"Max edge length: "<<max_edge_length<<", "<<triangulated_surface.Maximum_Edge_Length()<<std::endl;
 
   triangulated_surface.mesh.elements = triangles;
   FILE_UTILITIES::Write_To_File(STREAM_TYPE((RW)1.0),out_path,triangulated_surface);

}
//#####################################################################
};
}
#endif
