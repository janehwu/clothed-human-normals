import numpy as np
from scipy.sparse import csr_matrix

def LoadPLY_Tet(filename):
    file = open(filename, 'r')
    line = file.readline()
    words = line.split(" ")

    while(not words[0] == "end_header\n"):
        if words[0] == "element":
            if words[1] == "vertex":
                nb_verts = int(words[2])
            elif words[1] == "face":
                nb_faces = int(words[2])
            elif words[1] == "voxel":
                nb_tets = int(words[2])
            elif words[1] == "res":
                tet_size = float(words[2])/1000.0

        line = file.readline()
        words = line.split(" ")

    verts = np.zeros((nb_verts,3))
    faces = np.zeros((nb_faces,3))
    tets = np.zeros((nb_tets,4), dtype = int)
    
    for i in range(nb_verts):
        line = file.readline()
        words = line.split(" ")
        verts[i,0] = float(words[0])
        verts[i,1] = float(words[1])
        verts[i,2] = float(words[2])

    
    for i in range(nb_faces):
        line = file.readline()
        words = line.split(" ")
        if not int(words[0]) == 3:
            print("Error non triangular face")
        faces[i,0] = int(words[1])
        faces[i,1] = int(words[2])
        faces[i,2] = int(words[3])

        
    for i in range(nb_tets):
        line = file.readline()
        words = line.split(" ")
        if not int(words[0]) == 4:
            print("Error non tetrahedra")
        tets[i,0] = int(words[1])
        tets[i,1] = int(words[2])
        tets[i,2] = int(words[3])
        tets[i,3] = int(words[4])

    file.close()

    return [verts, faces, tets]

def loadOuterShell(filename):
    nodes, faces, tets = LoadPLY_Tet(filename)
    print("nodes: {}".format(nodes.shape))
    print("faces: {}".format(faces.shape))
    print("tets: {}".format(tets.shape))

    # Create List of edges
    nb_nodes = nodes.shape[0]
    nb_tets = tets.shape[0]
    rows = []
    cols = []
    data = []
    for vox_id in range(nb_tets):
        for sum_1 in range(4):
            for sum_2 in range(sum_1+1,4):
                if tets[vox_id,sum_1] < tets[vox_id, sum_2]:
                    rows.append(tets[vox_id,sum_1])
                    cols.append(tets[vox_id,sum_2])
                    data.append(len(rows))
                else:
                    rows.append(tets[vox_id,sum_2])
                    cols.append(tets[vox_id,sum_1])
                    data.append(len(rows))

    edges_sp = csr_matrix((data,(rows,cols)), shape=(nb_nodes, nb_nodes))
    edges = edges_sp.nonzero()

    print("nb_edges: {}".format(edges[0].shape))
    print("nb_edges: {}".format(edges_sp.count_nonzero()))   

    return [nodes, faces, tets, edges_sp]
