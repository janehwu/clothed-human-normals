######################################################################
# Copyright 2019. Jenny Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os

class Obj:
    # 0-based indexing
    def __init__(self, v=None, f=None, vn=None):
        self.v = v
        self.f = f
        self.vn = vn

def read_obj(obj_filename):
    if not os.path.exists(obj_filename):
        return None
    verts = []
    faces = []
    vert_normals = []
    with open(obj_filename) as fin:
        for line in fin:
            if line.startswith('v '):
                fields = line.strip().split()
                assert(len(fields) == 4)
                verts.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if line.startswith('vn '):
                fields = line.strip().split()
                assert(len(fields) == 4)
                vert_normals.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if line.startswith('f '):
                fields = line.strip().split()
                assert(len(fields) == 4)
                if "//" in fields[1]:
                    a = fields[1].split("//")[0]
                    b = fields[2].split("//")[0]
                    c = fields[3].split("//")[0]
                    faces.append([int(a)-1, int(b)-1, int(c)-1])
                if "/" in fields[1]:
                    a = fields[1].split("/")[0]
                    b = fields[2].split("/")[0]
                    c = fields[3].split("/")[0]
                    faces.append([int(a)-1, int(b)-1, int(c)-1])
                else:
                    faces.append([int(fields[1])-1, int(fields[2])-1, int(fields[3])-1])
        verts = np.array(verts)
        faces = np.array(faces)
        vert_normals = np.array(vert_normals)
        nv = len(verts)
#         assert(np.amax(faces) == nv - 1)
#         assert(np.amin(faces) == 0)
    return Obj(verts, faces, vert_normals)

def write_obj(obj, obj_filename, vert_normals=None):
    assert(obj.v is not None)
    with open(obj_filename, 'w') as fout:
        for i in range(obj.v.shape[0]):
            fout.write('v %f %f %f\n' %(obj.v[i][0], obj.v[i][1], obj.v[i][2]))
        if vert_normals is not None:
            for i in range(obj.v.shape[0]):
                fout.write('vn %f %f %f\n' % (vert_normals[i,0], vert_normals[i,1], vert_normals[i,2]))
        if obj.f is not None:
            for i in range(obj.f.shape[0]):
                fout.write('f %d %d %d\n' %(obj.f[i][0]+1, obj.f[i][1]+1, obj.f[i][2]+1))
