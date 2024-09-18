#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import numpy as np
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.camera_transforms import Rotate, Transform3d, Translate


# Default values for rotation and translation matrices.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)

Device = Union[str, torch.device]


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:  # pyre-ignore[16]
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


class TensorAccessor(nn.Module):
    """
    A helper class to be used with the __getitem__ method. This can be used for
    getting/setting the values for an attribute of a class at one particular
    index.  This is useful when the attributes of a class are batched tensors
    and one element in the batch needs to be modified.
    """

    def __init__(self, class_object, index: Union[int, slice]) -> None:
        """
        Args:
            class_object: this should be an instance of a class which has
                attributes which are tensors representing a batch of
                values.
            index: int/slice, an index indicating the position in the batch.
                In __setattr__ and __getattr__ only the value of class
                attributes at this index will be accessed.
        """
        self.__dict__["class_object"] = class_object
        self.__dict__["index"] = index

    def __setattr__(self, name: str, value: Any):
        """
        Update the attribute given by `name` to the value given by `value`
        at the index specified by `self.index`.

        Args:
            name: str, name of the attribute.
            value: value to set the attribute to.
        """
        v = getattr(self.class_object, name)
        if not torch.is_tensor(v):
            msg = "Can only set values on attributes which are tensors; got %r"
            raise AttributeError(msg % type(v))

        # Convert the attribute to a tensor if it is not a tensor.
        if not torch.is_tensor(value):
            value = torch.tensor(
                value, device=v.device, dtype=v.dtype, requires_grad=v.requires_grad
            )

        # Check the shapes match the existing shape and the shape of the index.
        if v.dim() > 1 and value.dim() > 1 and value.shape[1:] != v.shape[1:]:
            msg = "Expected value to have shape %r; got %r"
            raise ValueError(msg % (v.shape, value.shape))
        if (
            v.dim() == 0
            and isinstance(self.index, slice)
            and len(value) != len(self.index)
        ):
            msg = "Expected value to have len %r; got %r"
            raise ValueError(msg % (len(self.index), len(value)))
        self.class_object.__dict__[name][self.index] = value

    def __getattr__(self, name: str):
        """
        Return the value of the attribute given by "name" on self.class_object
        at the index specified in self.index.

        Args:
            name: string of the attribute name
        """
        if hasattr(self.class_object, name):
            return self.class_object.__dict__[name][self.index]
        else:
            msg = "Attribute %s not found on %r"
            return AttributeError(msg % (name, self.class_object.__name__))


BROADCAST_TYPES = (float, int, list, tuple, torch.Tensor, np.ndarray)


class TensorProperties(nn.Module):
    """
    A mix-in class for storing tensors as properties with helper methods.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        **kwargs,
    ) -> None:
        """
        Args:
            dtype: data type to set for the inputs
            device: Device (as str or torch.device)
            kwargs: any number of keyword arguments. Any arguments which are
                of type (float/int/list/tuple/tensor/array) are broadcasted and
                other keyword arguments are set as attributes.
        """
        super().__init__()
        self.device = make_device(device)
        self._N = 0
        if kwargs is not None:

            # broadcast all inputs which are float/int/list/tuple/tensor/array
            # set as attributes anything else e.g. strings, bools
            args_to_broadcast = {}
            for k, v in kwargs.items():
                if v is None or isinstance(v, (str, bool)):
                    setattr(self, k, v)
                elif isinstance(v, BROADCAST_TYPES):
                    args_to_broadcast[k] = v
                else:
                    msg = "Arg %s with type %r is not broadcastable"
                    warnings.warn(msg % (k, type(v)))

            names = args_to_broadcast.keys()
            # convert from type dict.values to tuple
            values = tuple(v for v in args_to_broadcast.values())

            if len(values) > 0:
                broadcasted_values = convert_to_tensors_and_broadcast(
                    *values, device=device
                )

                # Set broadcasted values as attributes on self.
                for i, n in enumerate(names):
                    setattr(self, n, broadcasted_values[i])
                    if self._N == 0:
                        self._N = broadcasted_values[i].shape[0]

    def __len__(self) -> int:
        return self._N

    def isempty(self) -> bool:
        return self._N == 0

    def __getitem__(self, index: Union[int, slice]) -> TensorAccessor:
        """

        Args:
            index: an int or slice used to index all the fields.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original camera.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(class_object=self, index=index)

        msg = "Expected index of type int or slice; got %r"
        raise ValueError(msg % type(index))

    def to(self, device: Device = "cpu") -> "TensorProperties":
        """
        In place operation to move class properties which are tensors to a
        specified device. If self has a property "device", update this as well.
        """
        device_ = make_device(device)
        for k in dir(self):
            v = getattr(self, k)
            if k == "device":
                setattr(self, k, device_)
            if torch.is_tensor(v) and v.device != device_:
                setattr(self, k, v.to(device_))
        return self

    def cpu(self) -> "TensorProperties":
        return self.to("cpu")

    def cuda(self, device: Optional[int] = None) -> "TensorProperties":
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    def clone(self, other) -> "TensorProperties":
        """
        Update the tensor properties of other with the cloned properties of self.
        """
        for k in dir(self):
            v = getattr(self, k)
            if inspect.ismethod(v) or k.startswith("__"):
                continue
            if torch.is_tensor(v):
                v_clone = v.clone()
            else:
                v_clone = copy.deepcopy(v)
            setattr(other, k, v_clone)
        return other

    def gather_props(self, batch_idx) -> "TensorProperties":
        """
        This is an in place operation to reformat all tensor class attributes
        based on a set of given indices using torch.gather. This is useful when
        attributes which are batched tensors e.g. shape (N, 3) need to be
        multiplied with another tensor which has a different first dimension
        e.g. packed vertices of shape (V, 3).

        Example

        .. code-block:: python

            self.specular_color = (N, 3) tensor of specular colors for each mesh

        A lighting calculation may use

        .. code-block:: python

            verts_packed = meshes.verts_packed()  # (V, 3)

        To multiply these two tensors the batch dimension needs to be the same.
        To achieve this we can do

        .. code-block:: python

            batch_idx = meshes.verts_packed_to_mesh_idx()  # (V)

        This gives index of the mesh for each vertex in verts_packed.

        .. code-block:: python

            self.gather_props(batch_idx)
            self.specular_color = (V, 3) tensor with the specular color for
                                     each packed vertex.

        torch.gather requires the index tensor to have the same shape as the
        input tensor so this method takes care of the reshaping of the index
        tensor to use with class attributes with arbitrary dimensions.

        Args:
            batch_idx: shape (B, ...) where `...` represents an arbitrary
                number of dimensions

        Returns:
            self with all properties reshaped. e.g. a property with shape (N, 3)
            is transformed to shape (B, 3).
        """
        # Iterate through the attributes of the class which are tensors.
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                if v.shape[0] > 1:
                    # There are different values for each batch element
                    # so gather these using the batch_idx.
                    # First clone the input batch_idx tensor before
                    # modifying it.
                    _batch_idx = batch_idx.clone()
                    idx_dims = _batch_idx.shape
                    tensor_dims = v.shape
                    if len(idx_dims) > len(tensor_dims):
                        msg = "batch_idx cannot have more dimensions than %s. "
                        msg += "got shape %r and %s has shape %r"
                        raise ValueError(msg % (k, idx_dims, k, tensor_dims))
                    if idx_dims != tensor_dims:
                        # To use torch.gather the index tensor (_batch_idx) has
                        # to have the same shape as the input tensor.
                        new_dims = len(tensor_dims) - len(idx_dims)
                        new_shape = idx_dims + (1,) * new_dims
                        expand_dims = (-1,) + tensor_dims[1:]
                        _batch_idx = _batch_idx.view(*new_shape)
                        _batch_idx = _batch_idx.expand(*expand_dims)

                    v = v.gather(0, _batch_idx)
                    setattr(self, k, v)
        return self


class FoVPerspectiveCameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices using the OpenGL convention for a perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> screen.

    The `transform_points` method calculates the full world -> screen transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.
    """

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0),),
        K=None,
        R=r,
        T=t,
        znear=1.0,
        zfar=100.0,
        x0=0,
        y0=0,
        w=256,
        h=256,
        device="cpu",
    ):
        """
        __init__(self, znear, zfar, R, T, device) -> None  # noqa

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            K=K,
            R=R,
            T=T,
            znear=znear,
            zfar=zfar,
            x0=x0,
            y0=y0,
            h=h,
            w=w,
        )

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the OpenGL perpective projection matrix with a symmetric
        viewing frustrum. Use column major order.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            P: a Transform3d object which represents a batch of projection
            matrices of shape (N, 3, 3)

        .. code-block:: python
            q = -(far + near)/(far - near)
            qn = -2*far*near/(far-near)

            P.T = [
                    [2*fx/w,     0,           0,  0],
                    [0,          -2*fy/h,     0,  0],
                    [(2*px-w)/w, (-2*py+h)/h, -q, 1],
                    [0,          0,           qn, 0],
                ]
                sometimes P[2,:] *= -1, P[1, :] *= -1
        """
        if self.K is not None:
            P = self.K
            # OpenGL uses column vectors so need to transpose the projection matrix
            # as torch3d uses row vectors.
            transform = Transform3d(device=self.device)
            transform._matrix = P.transpose(1, 2).contiguous()
            return transform

        znear = kwargs.get("znear", self.znear)  # pyre-ignore[16]
        zfar = kwargs.get("zfar", self.zfar)  # pyre-ignore[16]
        x0 = kwargs.get("x0", self.x0)  # pyre-ignore[16]
        y0 = kwargs.get("y0", self.y0)  # pyre-ignore[16]
        w = kwargs.get("w", self.w)  # pyre-ignore[16]
        h = kwargs.get("h", self.h)  # pyre-ignore[16]
        principal_point = kwargs.get(
            "principal_point", self.principal_point
        )  # pyre-ignore[16]
        focal_length = kwargs.get(
            "focal_length", self.focal_length
        )  # pyre-ignore[16]

        if not torch.is_tensor(focal_length):
            focal_length = torch.tensor(focal_length, device=self.device)

        if len(focal_length.shape) in (0, 1) or focal_length.shape[1] == 1:
            fx = fy = focal_length
        else:
            fx, fy = focal_length.unbind(1)

        if not torch.is_tensor(principal_point):
            principal_point = torch.tensor(principal_point, device=self.device)
        px, py = principal_point.unbind(1)

        P = torch.zeros(
            (self._N, 4, 4), device=self.device, dtype=torch.float32
        )
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space postive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0
        # define P.T directly
        P[:, 0, 0] = 2.0 * fx / w
        P[:, 1, 1] = -2.0 * fy / h
        P[:, 2, 0] = -(-2 * px + w + 2 * x0) / w
        P[:, 2, 1] = -(+2 * py - h + 2 * y0) / h
        P[:, 2, 3] = z_sign * ones

        # NOTE: This part of the matrix is for z renormalization in OpenGL
        # which maps the z to [-1, 1]. This won't work yet as the torch3d
        # rasterizer ignores faces which have z < 0.
        # P[:, 2, 2] = z_sign * (far + near) / (far - near)
        # P[:, 2, 3] = -2.0 * far * near / (far - near)
        # P[:, 2, 3] = z_sign * torch.ones((N))

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane. This replaces the OpenGL z normalization to [-1, 1]
        # until rasterization is changed to clip at z = -1.
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 3, 2] = -(zfar * znear) / (zfar - znear)

        # OpenGL uses column vectors so need to transpose the projection matrix
        # as torch3d uses row vectors.
        transform = Transform3d(device=self.device)
        transform._matrix = P
        return transform

    def clone(self):
        other = OpenGLRealPerspectiveCameras(device=self.device)
        return super().clone(other)

    def get_camera_center(self, **kwargs):
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        R = self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        T = self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        if T.shape[0] != R.shape[0]:
            msg = "Expected R, T to have the same batch dimension; got %r, %r"
            raise ValueError(msg % (R.shape[0], T.shape[0]))
        if T.dim() != 2 or T.shape[1:] != (3,):
            msg = "Expected T to have shape (N, 3); got %r"
            raise ValueError(msg % repr(T.shape))
        if R.dim() != 3 or R.shape[1:] != (3, 3):
            msg = "Expected R to have shape (N, 3, 3); got %r"
            raise ValueError(msg % R.shape)

        # Create a Transform3d object
        T = Translate(T, device=T.device)
        R = Rotate(R, device=R.device)
        world_to_view_transform = R.compose(T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-screen transform composing the
        world-to-view and view-to-screen transforms.

        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]

        world_to_view_transform = self.get_world_to_view_transform(
            R=self.R, T=self.T
        )
        view_to_screen_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_screen_transform)

    def transform_points(self, points, **kwargs) -> torch.Tensor:
        """
        Transform input points from world to screen space.

        Args:
            points: torch tensor of shape (..., 3).

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_screen_transform = self.get_full_projection_transform(**kwargs)
        return world_to_screen_transform.transform_points(points)

    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        For cameras that can be specified in screen space, this transform
        allows points to be converted from screen to NDC space.
        The default transform scales the points from [0, W-1]x[0, H-1] to [-1, 1].
        This function should be modified per camera definitions if need be,
        e.g. for Perspective/Orthographic cameras we provide a custom implementation.
        This transform assumes PyTorch3D coordinate system conventions for
        both the NDC space and the input points.

        This transform interfaces with the PyTorch3D renderer which assumes
        input points to the renderer to be in NDC space.
        """
        if self.in_ndc():
            return Transform3d(device=self.device, dtype=torch.float32)
        else:
            # For custom cameras which can be defined in screen space,
            # users might might have to implement the screen to NDC transform based
            # on the definition of the camera parameters.
            # See PerspectiveCameras/OrthographicCameras for an example.
            # We don't flip xy because we assume that world points are in
            # PyTorch3D coordinates, and thus conversion from screen to ndc
            # is a mere scaling from image to [-1, 1] scale.
            image_size = kwargs.get("image_size", self.get_image_size())
            return get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size
            )

    def transform_points_ndc(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to NDC space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in NDC space: +X left, +Y up, origin at image center.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3D.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform)

        return world_to_ndc_transform.transform_points(points, eps=eps)

    def transform_points_screen(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to screen space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3D.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        points_ndc = self.transform_points_ndc(points, eps=eps, **kwargs)
        image_size = kwargs.get("image_size", self.get_image_size())
        return get_ndc_to_screen_transform(
            self, with_xyflip=True, image_size=image_size
        ).transform_points(points_ndc, eps=eps)

    def is_perspective(self):
        return True

    def in_ndc(self):
        return True

    def get_image_size(self):
        """
        Returns the image size, if provided, expected in the form of (height, width)
        The image size is used for conversion of projected points to screen coordinates.
        """
        return self.image_size if hasattr(self, "image_size") else None


def get_ndc_to_screen_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
    PyTorch3D NDC to screen conversion.
    Conversion from PyTorch3D's NDC space (+X left, +Y up) to screen/image space
    (+X right, +Y down, origin top left).

    Args:
        cameras
        with_xyflip: flips x- and y-axis if set to True.
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.

    We represent the NDC to screen conversion as a Transform3d
    with projection matrix

    K = [
            [s,   0,    0,  cx],
            [0,   s,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]

    """
    # We require the image size, which is necessary for the transform
    if image_size is None:
        msg = "For NDC to screen conversion, image_size=(height, width) needs to be specified."
        raise ValueError(msg)

    K = torch.zeros((cameras._N, 4, 4), device=cameras.device, dtype=torch.float32)
    if not torch.is_tensor(image_size):
        image_size = torch.tensor(image_size, device=cameras.device)
    image_size = image_size.view(-1, 2)  # of shape (1 or B)x2
    height, width = image_size.unbind(1)

    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer
    scale = (image_size.min(dim=1).values - 1.0) / 2.0

    K[:, 0, 0] = scale
    K[:, 1, 1] = scale
    K[:, 0, 3] = -1.0 * (width - 1.0) / 2.0
    K[:, 1, 3] = -1.0 * (height - 1.0) / 2.0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0

    # Transpose the projection matrix as PyTorch3D transforms use row vectors.
    transform = Transform3d(
        matrix=K.transpose(1, 2).contiguous(), device=cameras.device
    )

    if with_xyflip:
        # flip x, y axis
        xyflip = torch.eye(4, device=cameras.device, dtype=torch.float32)
        xyflip[0, 0] = -1.0
        xyflip[1, 1] = -1.0
        xyflip = xyflip.view(1, 4, 4).expand(cameras._N, -1, -1)
        xyflip_transform = Transform3d(
            matrix=xyflip.transpose(1, 2).contiguous(), device=cameras.device
        )
        transform = transform.compose(xyflip_transform)
    return transform


def get_screen_to_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
    Screen to PyTorch3D NDC conversion.
    Conversion from screen/image space (+X right, +Y down, origin top left)
    to PyTorch3D's NDC space (+X left, +Y up).

    Args:
        cameras
        with_xyflip: flips x- and y-axis if set to True.
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.

    We represent the screen to NDC conversion as a Transform3d
    with projection matrix

    K = [
            [1/s,    0,    0,  cx/s],
            [  0,  1/s,    0,  cy/s],
            [  0,    0,    1,     0],
            [  0,    0,    0,     1],
    ]

    """
    transform = get_ndc_to_screen_transform(
        cameras,
        with_xyflip=with_xyflip,
        image_size=image_size,
    ).inverse()
    return transform


def format_tensor(
    input,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    device_ = make_device(device)
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device_)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device_:
        return input

    input = input.to(device=device)
    return input


def convert_to_tensors_and_broadcast(
    *args,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    if len(args) == 1:
        args_Nd = args_Nd[0]  # Return the first element

    return args_Nd

