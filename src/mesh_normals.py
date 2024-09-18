# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union
import torch


def list_to_packed(x: List[torch.Tensor]):
    r"""
    Transforms a list of N tensors each of shape (Mi, K, ...) into a single
    tensor of shape (sum(Mi), K, ...).

    Args:
      x: list of tensors.

    Returns:
        4-element tuple containing

        - **x_packed**: tensor consisting of packed input tensors along the
          1st dimension.
        - **num_items**: tensor of shape N containing Mi for each element in x.
        - **item_packed_first_idx**: tensor of shape N indicating the index of
          the first item belonging to the same element in the original list.
        - **item_packed_to_list_idx**: tensor of shape sum(Mi) containing the
          index of the element in the list the item belongs to.
    """
    N = len(x)
    num_items = torch.zeros(N, dtype=torch.int64, device=x[0].device)
    item_packed_first_idx = torch.zeros(N, dtype=torch.int64, device=x[0].device)
    item_packed_to_list_idx = []
    cur = 0
    for i, y in enumerate(x):
        num = len(y)
        num_items[i] = num
        item_packed_first_idx[i] = cur
        item_packed_to_list_idx.append(
            torch.full((num,), i, dtype=torch.int64, device=y.device)
        )
        cur += num

    x_packed = torch.cat(x, dim=0)
    item_packed_to_list_idx = torch.cat(item_packed_to_list_idx, dim=0)

    return x_packed, num_items, item_packed_first_idx, item_packed_to_list_idx


def packed_to_list(x: torch.Tensor, split_size: Union[list, int]):
    r"""
    Transforms a tensor of shape (sum(Mi), K, L, ...) to N set of tensors of
    shape (Mi, K, L, ...) where Mi's are defined in split_size

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: A list of Tensors
    """
    return x.split(split_size, dim=0)


def verts_normals_list(verts, faces, device):
    """
    Get the list representation of the vertex normals.

    Returns:
        list of tensors of normals of shape (V_n, 3).
    """
    # Preprocess verts and faces
    verts_list = verts
    faces_list = [
        f[f.gt(-1).all(1)].to(torch.int64) if len(f) > 0 else f for f in faces
    ]
    verts_packed, faces_packed, num_verts_per_mesh = compute_packed(verts_list, faces_list, device)
    verts_normals_packed = compute_vertex_normals(verts_packed, faces_packed)
    split_size = num_verts_per_mesh.tolist()
    return packed_to_list(verts_normals_packed, split_size)

def compute_vertex_normals(verts_packed, faces_packed):
    """Computes the packed version of vertex normals from the packed verts
    and faces. This assumes verts are shared between faces. The normal for
    a vertex is computed as the sum of the normals of all the faces it is
    part of weighed by the face areas.

    Args:
        refresh: Set to True to force recomputation of vertex normals.
            Default: False.
    """

    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    # NOTE: this is already applying the area weighting as the magnitude
    # of the cross product is 2 x area of the triangle.
    # pyre-fixme[16]: `Tensor` has no attribute `index_add`.
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 1],
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 2],
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 0],
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
            dim=1,
        ),
    )

    verts_normals_packed = torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )
    return verts_normals_packed


# TODO(nikhilar) Improve performance of _compute_packed.
def compute_packed(verts_list, faces_list, device):
    """
    Computes the packed version of the meshes from verts_list and faces_list
    and sets the values of auxiliary tensors.

    Args:
        refresh: Set to True to force recomputation of packed representations.
            Default: False.
    """
    num_verts_per_mesh = torch.tensor(
        [len(v) for v in verts_list], device=device
    )
    num_faces_per_mesh = torch.tensor(
        [len(f) for f in faces_list], device=device
    )

    verts_list_to_packed = list_to_packed(verts_list)
    verts_packed = verts_list_to_packed[0]
    if not torch.allclose(num_verts_per_mesh, verts_list_to_packed[1]):
        raise ValueError("The number of verts per mesh should be consistent.")
    mesh_to_verts_packed_first_idx = verts_list_to_packed[2]
    verts_packed_to_mesh_idx = verts_list_to_packed[3]

    faces_list_to_packed = list_to_packed(faces_list)
    faces_packed = faces_list_to_packed[0]
    if not torch.allclose(num_faces_per_mesh, faces_list_to_packed[1]):
        raise ValueError("The number of faces per mesh should be consistent.")
    mesh_to_faces_packed_first_idx = faces_list_to_packed[2]
    faces_packed_to_mesh_idx = faces_list_to_packed[3]

    faces_packed_offset = mesh_to_verts_packed_first_idx[
        faces_packed_to_mesh_idx
    ]
    faces_packed = faces_packed + faces_packed_offset.view(-1, 1)

    return verts_packed, faces_packed, num_verts_per_mesh

