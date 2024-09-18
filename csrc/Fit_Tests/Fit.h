//
//  Fit.h
//  fitLevelSet
//
//  Created by Diego Thomas on 2021/03/23.
//

#ifndef Fit_h
#define Fit_h
//#include "Utilities.h"

/*** Include files to manipulate matrices ***/
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Sparse>

#include <fstream>
#include <vector>

namespace fitLevelSet {

    // Copied from Utilities.h
    typedef struct my_int3 {
        int x;
        int y;
        int z;
        
        // Initializer
        my_int3(): x(0), y(0), z(0) {}
        my_int3(int a, int b, int c): x(a), y(b), z(c) {}
        
        //print
        void print() {
            std::cout << x << ", " << y << ", " << z << std::endl;
        }
    } my_int3;

    inline static my_int3 make_my_int3(int X, int Y, int Z) {
        return my_int3(X, Y, Z);
    }

    typedef struct my_float3 {
        float x;
        float y;
        float z;
        
        // Initializer
        my_float3(): x(0.0f), y(0.0f), z(0.0f) {}
        
        my_float3(float a): x(a), y(a), z(a) {}
        
        my_float3(float X, float Y, float Z): x(X), y(Y), z(Z) {}
        
        // Scalar Multiplication
        my_float3 operator* (float s) {
            return my_float3(s*x, s*y, s*z);
        }
        
        // outer product
        Eigen::Matrix3f operator* (my_float3 v) {
            Eigen::Matrix3f res;
            res <<  x*v.x, x*v.y, x*v.z,
                    y*v.x, y*v.y, y*v.z,
                    z*v.x, z*v.y, z*v.z;
            return res;
        }
        
        //matrix product
        my_float3 operator* (Eigen::Matrix3f M) {
            return my_float3(x*M(0,0) + y*M(0,1) + z*M(0,2),
                          x*M(1,0) + y*M(1,1) + z*M(1,2),
                          x*M(2,0) + y*M(2,1) + z*M(2,2));
        }
        
        // Addition
        my_float3 operator+ (my_float3 a) {
            return my_float3(a.x + x, a.y + y, a.z + z);
        }
        
        // Subtraction
        my_float3 operator- (my_float3 a) {
            return my_float3(x - a.x, y - a.y, z - a.z);
        }
        
        //print
        void print() {
            std::cout << x << ", " << y << ", " << z << std::endl;
        }
        
    } my_float3;

    inline static my_float3 make_my_float3(float X, float Y, float Z) {
        return my_float3(X, Y, Z);
    }

    inline static my_float3 make_my_float3(float X) {
        return my_float3(X);
    }

    static my_float3 cross (my_float3 a, my_float3 b) {
        return my_float3(a.y*b.z - a.z*b.y,
                      -a.x*b.z + a.z*b.x,
                      a.x*b.y - a.y*b.x);
    }

    static float dot (my_float3 a, my_float3 b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    static float norm (my_float3 a) {
        return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    }

    static Eigen::Matrix3f outer_prod_tri_order(my_float3 a, my_float3 b) {
        Eigen::Matrix3f res;
        res << a.x*b.x, a.x*b.y, a.x*b.z,
               a.y*b.x, a.y*b.y, a.y*b.z,
               a.z*b.x, a.z*b.y, a.z*b.z;
        
        return res;
    }

    float AreaFace(float *Nodes, int *Faces, int face) {
        // get the three summits
        my_float3 s1 = make_my_float3(Nodes[3*Faces[3*face]], Nodes[3*Faces[3*face]+1], Nodes[3*Faces[3*face]+2]);
        my_float3 s2 = make_my_float3(Nodes[3*Faces[3*face+1]], Nodes[3*Faces[3*face+1]+1], Nodes[3*Faces[3*face+1]+2]);
        my_float3 s3 = make_my_float3(Nodes[3*Faces[3*face+2]], Nodes[3*Faces[3*face+2]+1], Nodes[3*Faces[3*face+2]+2]);
        
        // compute area of base s1,s2,s3
        my_float3 n_vec = cross(s2-s1, s3-s1);
        return norm(n_vec)/2.0f;
    }
    
    float AreaFace(float *Nodes, int i1, int i2, int i3) {
        // get the three summits
        my_float3 s1 = make_my_float3(Nodes[3*i1], Nodes[3*i1+1], Nodes[3*i1+2]);
        my_float3 s2 = make_my_float3(Nodes[3*i2], Nodes[3*i2+1], Nodes[3*i2+2]);
        my_float3 s3 = make_my_float3(Nodes[3*i3], Nodes[3*i3+1], Nodes[3*i3+2]);
        
        // compute area of base s1,s2,s3
        my_float3 n_vec = cross(s2-s1, s3-s1);
        return norm(n_vec)/2.0f;
    }

    float ScTP(my_float3 a, my_float3 b, my_float3 c)
    {
        // computes scalar triple product
        return dot(a, cross(b, c));
    }


    void UpdateNormals(float *Vertices, float *Normals, int *Faces, int nbVertices, int nbFaces) {
        
        for (int f = 0; f < nbVertices; f++) {
            Normals[3*f] = 0.0f;
            Normals[3*f+1] = 0.0f;
            Normals[3*f+2] = 0.0f;
        }
        
        for (int f = 0; f < nbFaces; f++) {
            // compute face normal
            my_float3 p1 = make_my_float3(Vertices[3*Faces[3*f]], Vertices[3*Faces[3*f]+1], Vertices[3*Faces[3*f]+2]);
            my_float3 p2 = make_my_float3(Vertices[3*Faces[3*f+1]], Vertices[3*Faces[3*f+1]+1], Vertices[3*Faces[3*f+1]+2]);
            my_float3 p3 = make_my_float3(Vertices[3*Faces[3*f+2]], Vertices[3*Faces[3*f+2]+1], Vertices[3*Faces[3*f+2]+2]);
            
            my_float3 nml = cross(p2-p1, p3-p1);
            Normals[3*Faces[3*f]] = Normals[3*Faces[3*f]] + nml.x;
            Normals[3*Faces[3*f]+1] = Normals[3*Faces[3*f]+1] + nml.y;
            Normals[3*Faces[3*f]+2] = Normals[3*Faces[3*f]+2] + nml.z;
            
            Normals[3*Faces[3*f+1]] = Normals[3*Faces[3*f+1]] + nml.x;
            Normals[3*Faces[3*f+1]+1] = Normals[3*Faces[3*f+1]+1] + nml.y;
            Normals[3*Faces[3*f+1]+2] = Normals[3*Faces[3*f+1]+2] + nml.z;
            
            Normals[3*Faces[3*f+2]] = Normals[3*Faces[3*f+2]] + nml.x;
            Normals[3*Faces[3*f+2]+1] = Normals[3*Faces[3*f+2]+1] + nml.y;
            Normals[3*Faces[3*f+2]+2] = Normals[3*Faces[3*f+2]+2] + nml.z;
        }
        
        for (int f = 0; f < nbVertices; f++) {
            my_float3 nml = make_my_float3(Normals[3*f], Normals[3*f+1], Normals[3*f+2]);
            float mag = sqrt(nml.x*nml.x + nml.y*nml.y + nml.z*nml.z);
            
            if (mag > 0.0f) {
                Normals[3*f] = Normals[3*f]/mag;
                Normals[3*f+1] = Normals[3*f+1]/mag;
                Normals[3*f+2] = Normals[3*f+2]/mag;
            }
        }
    }


    void ComputeRBF(float *RBF, int *RBF_id, float *Vertices, int *Faces, int Nb_Faces, int Nb_Vertices) {
        // 1. for each vertex get the list of triangles that is attached to the vertex
        std::vector<std::vector<int>> neigh_face;
        std::vector<std::vector<int>> neigh_n;
        for (int n = 0; n < Nb_Vertices; n++) {
            std::vector<int> tmp;
            tmp.clear();
            neigh_face.push_back(tmp);
            std::vector<int> tmp2;
            tmp2.clear();
            neigh_n.push_back(tmp2);
        }
        
        for (int f = 0; f < Nb_Faces; f++) {
            neigh_face[Faces[3*f]].push_back(f);
            neigh_face[Faces[3*f+1]].push_back(f);
            neigh_face[Faces[3*f+2]].push_back(f);
            
            bool valid1 = true;
            bool valid2 = true;
            for (std::vector<int>::iterator it_n = neigh_n[Faces[3*f]].begin(); it_n != neigh_n[Faces[3*f]].end(); it_n++) {
                if (Faces[3*f+1] == (*it_n))
                    valid1 = false;
                
                if (Faces[3*f+2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3*f]].push_back(Faces[3*f+1]);
            
            if (valid2)
                neigh_n[Faces[3*f]].push_back(Faces[3*f+2]);
            
            valid1 = true;
            valid2 = true;
            for (std::vector<int>::iterator it_n = neigh_n[Faces[3*f+1]].begin(); it_n != neigh_n[Faces[3*f+1]].end(); it_n++) {
                if (Faces[3*f] == (*it_n))
                    valid1 = false;
                
                if (Faces[3*f+2] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3*f+1]].push_back(Faces[3*f]);
            
            if (valid2)
                neigh_n[Faces[3*f+1]].push_back(Faces[3*f+2]);
            
            valid1 = true;
            valid2 = true;
            for (std::vector<int>::iterator it_n = neigh_n[Faces[3*f+2]].begin(); it_n != neigh_n[Faces[3*f+2]].end(); it_n++) {
                if (Faces[3*f+1] == (*it_n))
                    valid1 = false;
                
                if (Faces[3*f] == (*it_n))
                    valid2 = false;
            }
            if (valid1)
                neigh_n[Faces[3*f+2]].push_back(Faces[3*f+1]);
            
            if (valid2)
                neigh_n[Faces[3*f+2]].push_back(Faces[3*f]);
        }
        
        // 2. Go through each vertex
        for (int n = 0; n < Nb_Vertices; n++) {
            if (neigh_n[n].size() > 16)
                std::cout << "nb tet in vicinity S: " << n << ", " << neigh_n[n].size() << std::endl;
            
            // compute the total volume of the surrounding faces.
            float area_tot = 1.0f;
            for (std::vector<int>::iterator it = neigh_face[n].begin(); it != neigh_face[n].end(); it++) {
                int face = (*it);
                area_tot *= 100.0f*AreaFace(Vertices, Faces[3*face], Faces[3*face+1], Faces[3*face+2]);
            }
            
            // for each neighborhing summit
            float tot_weight = 0.0f;
            int count = 0;
            for (std::vector<int>::iterator it_n = neigh_n[n].begin(); it_n != neigh_n[n].end(); it_n++) {
                float area_cur = area_tot;
                int curr_n = (*it_n);
                int s1 = -1;
                int s2 = -1;
        
                for (std::vector<int>::iterator it = neigh_face[n].begin(); it != neigh_face[n].end(); it++) {
                    int face = (*it);
                    assert(Faces[3*face] == n || Faces[3*face+1] == n || Faces[3*face+2] == n);
                    if (Faces[3*face] == curr_n || Faces[3*face+1] == curr_n || Faces[3*face+2] == curr_n) {
                        area_cur = area_cur/(100.0f*AreaFace(Vertices, Faces[3*face], Faces[3*face+1], Faces[3*face+2]));
                        if (s1 == -1) {
                            if (Faces[3*face] != n && Faces[3*face] != curr_n) {
                                s1 = Faces[3*face];
                            } else if (Faces[3*face+1] != n && Faces[3*face+1] != curr_n) {
                                s1 = Faces[3*face+1];
                            } else if (Faces[3*face+2] != n && Faces[3*face+2] != curr_n) {
                                s1 = Faces[3*face+2];
                            }
                        } else {
                            if (Faces[3*face] != n && Faces[3*face] != curr_n) {
                                s2 = Faces[3*face];
                            } else if (Faces[3*face+1] != n && Faces[3*face+1] != curr_n) {
                                s2 = Faces[3*face+1];
                            } else if (Faces[3*face+2] != n && Faces[3*face+2] != curr_n) {
                                s2 = Faces[3*face+2];
                            }
                        }
                    }
                }
                
                if (s1 == -1) {
                    //std::cout << "s1 == -1" << std::endl;
                    continue;
                }
                if (s2 == -1) {
                    //std::cout << "s2 == -1" << std::endl;
                    continue;
                    std::cout << n << std::endl;
                    std::cout << curr_n << std::endl;
                    std::cout << neigh_face[n].size() << std::endl;
                    for (std::vector<int>::iterator it = neigh_face[n].begin(); it != neigh_face[n].end(); it++) {
                        int face = (*it);
                        std::cout << Faces[3*face] << ", " << Faces[3*face+1] << ", " << Faces[3*face+2] << std::endl;
                    }
                }
                assert(s1 != -1);
                assert(s2 != -1);
                
                RBF[16*n + count] = 100.0f*AreaFace(Vertices, s1, curr_n, s2) * area_cur;
                RBF_id[16*n + count] = curr_n;
                //std::cout << n << ", " << curr_n << std::endl;
                tot_weight += RBF[16*n + count];
                count++;
            }
            
            for (int k = 0; k < 16; k++) {
                if (RBF_id[16*n + k] == -1)
                    break;
                RBF[16*n + k] = RBF[16*n + k]/tot_weight;
                //std::cout << n << ", " << _RBF[16*_Surface[n] + k] << std::endl;
            }
        }
        
    }

    // This function creates a hash table that associate to each voxel a list of tetrahedras for quick search
    std::vector<int> *GenerateHashTable(float *Nodes, int *Tets, int nb_tets, my_int3 size_grid, my_float3 center_grid, float res) {
        
        std::vector<int> *HashTable = new std::vector<int>[size_grid.x*size_grid.y*size_grid.z];
        // Go through each tetrahedra
        for (int t = 0; t < nb_tets; t++) {
            my_float3 s1 = make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t] + 1], Nodes[3*Tets[4*t] + 2]);
            my_float3 s2 = make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1] + 1], Nodes[3*Tets[4*t+1] + 2]);
            my_float3 s3 = make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2] + 1], Nodes[3*Tets[4*t+2] + 2]);
            my_float3 s4 = make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3] + 1], Nodes[3*Tets[4*t+3] + 2]);
            
            // compute min max
            float min_x = std::min(s1.x,std::min(s2.x,std::min(s3.x, s4.x)));
            float min_y = std::min(s1.y,std::min(s2.y,std::min(s3.y, s4.y)));
            float min_z = std::min(s1.z,std::min(s2.z,std::min(s3.z, s4.z)));
            
            float max_x = std::max(s1.x,std::max(s2.x,std::max(s3.x, s4.x)));
            float max_y = std::max(s1.y,std::max(s2.y,std::max(s3.y, s4.y)));
            float max_z = std::max(s1.z,std::max(s2.z,std::max(s3.z, s4.z)));
                                    
            // Convert to grid coordinates
            int l_x = int(std::floor((min_x - center_grid.x)/res)) + size_grid.x/2;
            int l_y = int(std::floor((min_y - center_grid.y)/res)) + size_grid.y/2;
            int l_z = int(std::floor((min_z - center_grid.z)/res)) + size_grid.z/2;
            
            int u_x = int(std::ceil((max_x - center_grid.x)/res)) + size_grid.x/2;
            int u_y = int(std::ceil((max_y - center_grid.y)/res)) + size_grid.y/2;
            int u_z = int(std::ceil((max_z - center_grid.z)/res)) + size_grid.z/2;
                        
            for (int a = l_x; a < u_x+1; a++) {
                for (int b = l_y; b < u_y+1; b++) {
                    for (int c = l_z; c < u_z+1; c++) {
                        HashTable[a*size_grid.y*size_grid.z + b*size_grid.z + c].push_back(t);
                    }
                }
            }
        }
        return HashTable;
        
    }

    float VolumeTetrahedra(my_float3 s1, my_float3 s2, my_float3 s3, my_float3 s4) {
        // compute area of base s1,s2,s3
        my_float3 n_vec = cross(s2-s1, s3-s1);
        float area_base = norm(n_vec)/2.0f;
        
        // compute height
        // project s4 on the base plan
        my_float3 n_unit = n_vec * (1.0f/area_base);
        float dot_prod = dot(s4-s1, n_unit);
        
        return (area_base * fabs(dot_prod))/3.0f;
    }

    float *bary_tet(my_float3 a, my_float3 b, my_float3 c, my_float3 d, my_float3 p)
    {
        my_float3 vap = p - a;
        my_float3 vbp = p - b;

        my_float3 vab = b - a;
        my_float3 vac = c - a;
        my_float3 vad = d - a;

        my_float3 vbc = c - b;
        my_float3 vbd = d - b;
        // ScTP computes the scalar triple product
        float va6 = ScTP(vbp, vbd, vbc);
        float vb6 = ScTP(vap, vac, vad);
        float vc6 = ScTP(vap, vad, vab);
        float vd6 = ScTP(vap, vab, vac);
        float v6 = 1.0f / ScTP(vab, vac, vad);
        
        float *res = new float[4];
        res[0] = va6*v6;
        res[1] = vb6*v6;
        res[2] = vc6*v6;
        res[3] = vd6*v6;
        return res;
    }

    float *BarycentricCoordinates(my_float3 v, float *Nodes, int *Tets, int t) {
        float vol_tot = VolumeTetrahedra(make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t]+1], Nodes[3*Tets[4*t]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1]+1], Nodes[3*Tets[4*t+1]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2]+1], Nodes[3*Tets[4*t+2]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3]+1], Nodes[3*Tets[4*t+3]+2]));
        
        //Compute the volumes of inside tetras
        float vol_1 = VolumeTetrahedra(v,
                                         make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1]+1], Nodes[3*Tets[4*t+1]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2]+1], Nodes[3*Tets[4*t+2]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3]+1], Nodes[3*Tets[4*t+3]+2]));
        
        float vol_2 = VolumeTetrahedra(make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t]+1], Nodes[3*Tets[4*t]+2]),
                                         v,
                                         make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2]+1], Nodes[3*Tets[4*t+2]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3]+1], Nodes[3*Tets[4*t+3]+2]));
    
        float vol_3 = VolumeTetrahedra(make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t]+1], Nodes[3*Tets[4*t]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1]+1], Nodes[3*Tets[4*t+1]+2]),
                                         v,
                                         make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3]+1], Nodes[3*Tets[4*t+3]+2]));
    
        float vol_4 = VolumeTetrahedra(make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t]+1], Nodes[3*Tets[4*t]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1]+1], Nodes[3*Tets[4*t+1]+2]),
                                         make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2]+1], Nodes[3*Tets[4*t+2]+2]),
                                         v);
        
        float *res = new float[4];
        res[0] = vol_1/vol_tot;
        res[1] = vol_2/vol_tot;
        res[2] = vol_3/vol_tot;
        res[3] = vol_4/vol_tot;
        return res;
    }

    bool IsInside(my_float3 v, float *Nodes, int *Tets, int t) {
        // Get four summits:
        my_float3 s1 = make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t]+1], Nodes[3*Tets[4*t]+2]);
        my_float3 s2 = make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1]+1], Nodes[3*Tets[4*t+1]+2]);
        my_float3 s3 = make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2]+1], Nodes[3*Tets[4*t+2]+2]);
        my_float3 s4 = make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3]+1], Nodes[3*Tets[4*t+3]+2]);
        
        // Compute normal of each face
        my_float3 n1 = cross(s2-s1, s3-s1);
        n1 = n1 * (1.0f/norm(n1));
        my_float3 n2 = cross(s2-s1, s4-s1);
        n2 = n2 * (1.0f/norm(n2));
        my_float3 n3 = cross(s3-s2, s4-s2);
        n3 = n3 * (1.0f/norm(n3));
        my_float3 n4 = cross(s3-s1, s4-s1);
        n4 = n4 * (1.0f/norm(n4));
        
        // Orient all faces outward
        if (dot(n1, s4-s1) > 0.0f)
            n1 = n1 * (-1.0f);
        if (dot(n2, s1-s2) > 0.0f)
            n2 = n2 * (-1.0f);
        if (dot(n3, s1-s3) > 0.0f)
            n3 = n3 * (-1.0f);
        if (dot(n4, s2-s1) > 0.0f)
            n4 = n4 * (-1.0f);
        
        // Test orientation of vertex
        float o1 = dot(n1, v-s1);
        float o2 = dot(n2, v-s2);
        float o3 = dot(n3, v-s3);
        float o4 = dot(n4, v-s1);
        
        return o1 < 0.0f && o2 < 0.0f && o3 < 0.0f && o4 < 0.0f;
    }

    int GetTet(my_float3 v, float *Nodes, int *Tets, std::vector<int> *HashTable, my_int3 dim_grid, my_float3 center_grid, float res) {
        //compute the key for the vertex
        my_int3 v_id = make_my_int3(int((v.x - center_grid.x)/res) + dim_grid.x/2,
                                      int((v.y - center_grid.y)/res) + dim_grid.y/2,
                                      int((v.z - center_grid.z)/res) + dim_grid.z/2);
        
        if (v_id.x < 0 || v_id.x >= dim_grid.x - 1 || v_id.y < 0 || v_id.y >= dim_grid.y - 1 || v_id.z < 0 || v_id.z >= dim_grid.z - 1) {
            return -1;
        }
        
        std::vector<int> listTets = HashTable[v_id.x*dim_grid.y*dim_grid.z + v_id.y*dim_grid.z + v_id.z];
                
        int best_t = -1;
        for (std::vector<int>::iterator it = listTets.begin(); it != listTets.end(); it++) {
            int t = (*it);
        //for (int t = 0; t < 400110; t++) {
            
            // test if v is inside the tetrahedron
            if (IsInside(v, Nodes, Tets, t)) {
                best_t = t;
                break;
            }
        }
        
        return best_t;
        
    }

    float InterpolateField(my_float3 v, float *sdf, float *Nodes, int *Tets, std::vector<int> *HashTable, my_int3 dim_grid, my_float3 center, float res) {
        
        int t = GetTet(v, Nodes, Tets, HashTable, dim_grid, center, res);
        if (t == -1) {
            // point ouside the field
            //v.print();
            //std::cout << "point outside" << std::endl;
            return NAN;
        }
        
        //Get barycentric coordinates
        //float *B_coo = BarycentricCoordinates(v, Nodes, Tets, t);
        float *B_coo = bary_tet(make_my_float3(Nodes[3*Tets[4*t]], Nodes[3*Tets[4*t]+1], Nodes[3*Tets[4*t]+2]),
                                make_my_float3(Nodes[3*Tets[4*t+1]], Nodes[3*Tets[4*t+1]+1], Nodes[3*Tets[4*t+1]+2]),
                                make_my_float3(Nodes[3*Tets[4*t+2]], Nodes[3*Tets[4*t+2]+1], Nodes[3*Tets[4*t+2]+2]),
                                make_my_float3(Nodes[3*Tets[4*t+3]], Nodes[3*Tets[4*t+3]+1], Nodes[3*Tets[4*t+3]+2]), v);
        
        assert(B_coo[0] <= 1.0f && B_coo[1] <= 1.0f && B_coo[2] <= 1.0f && B_coo[3] <= 1.0f);
        
        /*std::cout << "BarycentricCoordinates" << B_coo[0] << ", " << B_coo[1] << ", " << B_coo[2] << ", " << B_coo[3] << std::endl;
        v.print();
        std::cout << "s1: " << Nodes[Tets[4*t]] << ", " << Nodes[Tets[4*t]+1] << ", " << Nodes[Tets[4*t]+2] << ", " << std::endl;
        std::cout << "sdf1: " << sdf[Tets[4*t]] << ", " << std::endl;
        std::cout << "s2: " << Nodes[Tets[4*t+1]] << ", " << Nodes[Tets[4*t+1]+1] << ", " << Nodes[Tets[4*t+1]+2] << ", " << std::endl;
        std::cout << "sdf3: " << sdf[Tets[4*t+1]] << ", " << std::endl;
        std::cout << "s3: " << Nodes[Tets[4*t+2]] << ", " << Nodes[Tets[4*t+2]+1] << ", " << Nodes[Tets[4*t+2]+2] << ", " << std::endl;
        std::cout << "sdf4: " << sdf[Tets[4*t+2]] << ", " << std::endl;
        std::cout << "s4: " << Nodes[Tets[4*t+3]] << ", " << Nodes[Tets[4*t+3]+1] << ", " << Nodes[Tets[4*t+3]+2] << ", " << std::endl;
        std::cout << "sdf5: " << sdf[Tets[4*t+3]] << ", " << std::endl;*/
        
        float out_res = (B_coo[0]*sdf[Tets[4*t]] + B_coo[1]*sdf[Tets[4*t+1]] + B_coo[2]*sdf[Tets[4*t+2]] + B_coo[3]*sdf[Tets[4*t+3]]) /
                        (B_coo[0] + B_coo[1] + B_coo[2] + B_coo[3]);
        delete[] B_coo;
        
        return out_res;
    }

    my_float3 Gradient(my_float3 v, float *sdf, float *Nodes, int *Tets, std::vector<int> *HashTable, my_int3 dim_grid, my_float3 center, float res) {
        
        my_float3 grad = make_my_float3(0.0f, 0.0f, 0.0f);
        
        my_float3 v_x_p = v + make_my_float3(0.001f, 0.0f, 0.0f);
        float sdf_x_p = InterpolateField(v_x_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        my_float3 v_x_m = v + make_my_float3(-0.001f, 0.0f, 0.0f);
        float sdf_x_m = InterpolateField(v_x_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        if (sdf_x_p != sdf_x_p) {
            if (sdf_x_m != sdf_x_m) {
                grad.x = 0.0f;
            } else {
                grad.x = 1.0f;
            }
        } else {
            if (sdf_x_m != sdf_x_m) {
                grad.x = -1.0f;
            } else {
                grad.x = (sdf_x_p - sdf_x_m);
            }
        }
        //grad.x = (sdf_x_p - sdf_x_m);// / 0.002f;
        
        my_float3 v_y_p = v + make_my_float3(0.0f, 0.001f, 0.0f);
        float sdf_y_p = InterpolateField(v_y_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        my_float3 v_y_m = v + make_my_float3(0.0f, -0.001f, 0.0f);
        float sdf_y_m = InterpolateField(v_y_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        if (sdf_y_p != sdf_y_p) {
            if (sdf_y_m != sdf_y_m) {
                grad.y = 0.0f;
            } else {
                grad.y = 1.0f;
            }
        } else {
            if (sdf_y_m != sdf_y_m) {
                grad.y = -1.0f;
            } else {
                grad.y = (sdf_y_p - sdf_y_m);
            }
        }
        //grad.y = (sdf_y_p - sdf_y_m);// / 0.002f;
        
        my_float3 v_z_p = v + make_my_float3(0.0f, 0.0f, 0.001f);
        float sdf_z_p = InterpolateField(v_z_p, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        my_float3 v_z_m = v + make_my_float3(0.0f, 0.0f, -0.001f);
        float sdf_z_m = InterpolateField(v_z_m, sdf, Nodes, Tets, HashTable, dim_grid, center, res);
        if (sdf_z_p != sdf_z_p) {
            if (sdf_z_m != sdf_z_m) {
                grad.z = 0.0f;
            } else {
                grad.z = 1.0f;
            }
        } else {
            if (sdf_z_m != sdf_z_m) {
                grad.z = -1.0f;
            } else {
                grad.z = (sdf_z_p - sdf_z_m);
            }
        }
        //grad.z = (sdf_z_p - sdf_z_m);// / 0.002f;
        
        /*if (grad.x != grad.x || grad.y != grad.y || grad.z != grad.z) {
            //std::cout << "NAN" << std::endl;
            return make_my_float3(0.0f, 0.0f, 0.0f);
        }*/
        
        return grad;
    }

    void FitToLevelSet(float *sdf, float *Nodes, int *Tets, int nb_tets, float *Vertices, float *Normals, int *Faces, int Nb_Vertices, int Nb_Faces, int outerloop_maxiter = 1, int innerloop_maxiter = 10, float alpha = 0.1f, float delta = 0.1f, float delta_elastic = 0.01f) {
        //TODO: One time save
        /* 
        float *RBF = new float[16*Nb_Vertices];
        int *RBF_id = new int[16*Nb_Vertices];
        for (int n = 0; n < Nb_Vertices; n++) {
            for (int i = 0; i < 16; i++)
            RBF_id[16*n + i] = -1;
        }

        ComputeRBF(RBF, RBF_id, Vertices, Faces, Nb_Faces, Nb_Vertices);
        //Save RBF and RBF_id
        cnpy::npy_save("RBF.npy",RBF,{16*Nb_Vertices},"w");
        cnpy::npy_save("RBF_id.npy",RBF_id,{16*Nb_Vertices},"w");
        */

        cnpy::NpyArray RBF_arr=cnpy::npy_load("RBF.npy");
        std::vector<float> RBF=RBF_arr.as_vec<float>();
        cnpy::NpyArray RBF_id_arr=cnpy::npy_load("RBF_id.npy");
        std::vector<int> RBF_id=RBF_id_arr.as_vec<int>();

        std::cout<<"RBF check: "<<RBF[100]<<" "<<RBF_id[100]<<std::endl;
        
        std::vector<int> *HashTable = GenerateHashTable(Nodes, Tets, nb_tets, make_my_int3(100,100,100), make_my_float3(0.0f,-0.1f,0.0f), 0.03f);
        std::cout<<"Done generating hashtable"<<std::endl;
        /*
        std::ofstream HT_File;
        HT_File.open("HashTable.csv");
        std::ofstream HTIDX_File;
        HTIDX_File.open("HashTable_idx.csv");

        for (int i = 0; i < 100*100*100; i++){
            if (HashTable[i].size() == 0)
                continue;
            std::cout<<i<<": "<<HashTable[i].size()<<std::endl;
            HTIDX_File << i;
            HTIDX_File << "\n";
            for(int t=0; t < HashTable[i].size();t++){
                HT_File << HashTable[i][t];
                HT_File << " ";
            }
            HT_File << "\n";
        }
        HT_File.close();
        HTIDX_File.close();
        */
        /*
        std::vector<int> *HashTable = new std::vector<int>[100*100*100];
        std::ifstream HT_File("HashTable.csv");
        std::string line;
        std::cout<<"Reading file"<<std::endl;
        for (int i = 0; i < 100*100*100; i++){
            int temp;
            std::getline(HT_File, line);
            if(line.size() == 0){
                std::cout<<"Empty: "<<i<<std::endl;
                continue;
            }
            std::cout<<i<<": "<<line<<std::endl;

            std::stringstream ss(line);
            while(ss >> temp)
                HashTable[i].push_back(temp);
        }
        */
        //cnpy::NpyArray HashTable_arr=cnpy::npy_load("HashTable.npy");
        //std::vector<int> *HashTable=&HashTable_arr.as_vec<int>();
        //std::cout<<"HashTable check: "<<(*HashTable)[100]<<std::endl;

        // Copy of vertex position
        float *vtx_copy;
        vtx_copy = new float [3*Nb_Vertices];
        memcpy(vtx_copy, Vertices, 3*Nb_Vertices*sizeof(float));
                
        //############## Outer loop ##################
        float alpha_curr = alpha;
        for (int OuterIter = 0; OuterIter < outerloop_maxiter; OuterIter++) {
            std::cout<<"OuterIter: "<<OuterIter<<std::endl;
            // Copy current state
            memcpy(vtx_copy, Vertices, 3*Nb_Vertices*sizeof(float));
            
            // Increment deplacement of surface vertices
            for (int s = 0; s < Nb_Vertices; s++) {
                my_float3 vertex = make_my_float3(Vertices[3*s], Vertices[3*s+1], Vertices[3*s+2]);
                my_float3 normal = make_my_float3(Normals[3*s], Normals[3*s+1], Normals[3*s+2]);
                my_float3 grad = Gradient(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100,100,100), make_my_float3(0.0f,-0.1f,0.0f), 0.03f);
                float sdf_val = InterpolateField(vertex, sdf, Nodes, Tets, HashTable, make_my_int3(100,100,100), make_my_float3(0.0f,-0.1f,0.0f), 0.03f);
                if (sdf_val != sdf_val) {
                    vertex = vertex - normal * 0.001f;
                } else {
                    //    std::cout << "sdf: " << sdf_val << std::endl;
                    if (norm(grad) > 0.0f)
                        grad = grad * (sdf_val/norm(grad));
                    
                    //grad.print();
                    if (dot(grad, normal) < 0.0f)
                        vertex = vertex - grad * delta + normal * norm(grad) * alpha_curr;
                    else
                        vertex = vertex - grad * delta - normal * norm(grad) * alpha_curr;
                }
                Vertices[3*s] = vertex.x;
                Vertices[3*s+1] = vertex.y;
                Vertices[3*s+2] = vertex.z;
            }
            
            for (int inner_iter = 0; inner_iter < innerloop_maxiter; inner_iter++) {
                // Copy current state of voxels
                memcpy(vtx_copy, Vertices, 3*Nb_Vertices*sizeof(float));
                for (int s = 0; s < Nb_Vertices; s++) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    
                    my_float3 vtx = make_my_float3(vtx_copy[3*s], vtx_copy[3*s+1], vtx_copy[3*s+2]);
                    my_float3 nmle = make_my_float3(Normals[3*s], Normals[3*s+1], Normals[3*s+2]);
                    
                    for (int k = 0; k < 16; k++) {
                        if (RBF_id[16*s + k] == -1)
                            break;
                        
                        // Project points into tangent plane
                        my_float3 curr_v = make_my_float3(vtx_copy[3*RBF_id[16*s + k]], vtx_copy[3*RBF_id[16*s + k]+1], vtx_copy[3*RBF_id[16*s + k]+2]);
                        
                        float ps = dot(curr_v - vtx, nmle);
                        
                        my_float3 v_proj = curr_v - nmle * ps;
                        
                        x += v_proj.x*RBF[16*s + k];
                        y += v_proj.y*RBF[16*s + k];
                        z += v_proj.z*RBF[16*s + k];
                        /*x += vtx_copy[3*RBF_id[16*s + k]]*RBF[16*s + k];
                        y += vtx_copy[3*RBF_id[16*s + k]+1]*RBF[16*s + k];
                        z += vtx_copy[3*RBF_id[16*s + k]+2]*RBF[16*s + k];*/
                    }
                    
                    if (x != 0.0f && y != 0.0f && z != 0.0f) {
                        Vertices[3*s] = x;
                        Vertices[3*s+1] = y;
                        Vertices[3*s+2] = z;
                    }
                }
                
            }
            
            UpdateNormals(Vertices, Normals, Faces, Nb_Vertices, Nb_Faces);
            //alpha_curr = alpha_curr/1.5f;
        }
        
        for (int t = 0; t < 100*100*100; t++)
            HashTable[t].clear();
        delete[] HashTable;
        //delete[] RBF;
        //delete[] RBF_id;
        delete []vtx_copy;
    }

}

#endif /* Fit_h */
