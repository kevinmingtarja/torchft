# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers shared by checkpoint transports that send a state dict as raw tensor
storages plus a picklable description of its structure.

The sender flattens the state dict with :func:`_prepare_state_dict`, which
returns a :class:`_StateDictMeta` and the ``uint8`` storage of every tensor.
The receiver obtains one ``uint8`` buffer per tensor leaf, in order, and calls
:func:`_unflatten_state_dict` to rebuild the original tensors, DTensors and
non-tensor leaves from them.
"""

from dataclasses import dataclass
from typing import Callable, cast, Union

import torch
from torch.distributed.tensor import _DTensorSpec, DTensor
from torch.utils._pytree import (
    KeyPath,
    tree_flatten_with_path,
    tree_unflatten,
    TreeSpec,
)


@dataclass
class _TensorMeta:
    """
    This is the metadata for a tensor that is used to transfer checkpoints.
    It contains the shape, the dtype, the storage offset and the stride of the
    tensor.

    This must be pickleable so that it can be sent over the wire.
    """

    shape: torch.Size
    dtype: torch.dtype
    storage_offset: int
    stride: tuple[int, ...]
    nbytes: int


@dataclass
class _DTensorMeta:
    """
    This is the metadata for a DTensor that is used to transfer checkpoints.
    It contains the metadata for the local tensor and the spec of the DTensor.

    This must be pickleable so that it can be sent over the wire.
    """

    local: _TensorMeta
    spec: _DTensorSpec


@dataclass
class _StateDictMeta:
    """
    This is the metadata for a state dict that is used to transfer checkpoints.
    It contains the step, the pytree spec of the state dict and the metadata for
    each tensor in the state dict.

    This must be pickleable so that it can be sent over the wire.

    Args:
        step: the step of the checkpoint to verify consistency
        treespec: the pytree spec of the state dict
        paths: the path of each leaf in the state dict
        non_tensor_leaves: the metadata for each tensor in the state dict and any
            non-tensor leaves in the state dict
    """

    step: int
    treespec: TreeSpec
    paths: list[KeyPath]
    non_tensor_leaves: list[Union[object, _TensorMeta, _DTensorMeta]]


def _prepare_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, _TensorMeta]:
    return (
        _cast_tensor(tensor, torch.uint8),
        _TensorMeta(
            shape=tensor.shape,
            dtype=tensor.dtype,
            storage_offset=cast(int, tensor.storage_offset()),
            stride=tensor.stride(),
            nbytes=tensor.untyped_storage().nbytes(),
        ),
    )


def _prepare_state_dict(
    state_dict: object,
    step: int,
) -> tuple[_StateDictMeta, list[torch.Tensor]]:
    leaves: list[tuple[KeyPath, object]]
    leaves, treespec = tree_flatten_with_path(state_dict)

    paths: list[KeyPath] = []
    non_tensor_leaves: list[Union[object, _TensorMeta, _DTensorMeta]] = []
    tensors: list[torch.Tensor] = []
    for key_path, v in leaves:
        paths.append(key_path)

        if isinstance(v, DTensor):
            tensor, tensor_meta = _prepare_tensor(v._local_tensor)

            tensors.append(tensor)

            non_tensor_leaves.append(
                _DTensorMeta(
                    local=tensor_meta,
                    spec=v._spec,
                )
            )
        elif isinstance(v, torch.Tensor):
            tensor, tensor_meta = _prepare_tensor(v)
            tensors.append(tensor)
            non_tensor_leaves.append(tensor_meta)
        else:
            non_tensor_leaves.append(v)

    return (
        _StateDictMeta(
            step=step,
            treespec=treespec,
            paths=paths,
            non_tensor_leaves=non_tensor_leaves,
        ),
        tensors,
    )


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Casts the underlying storage to a tensor of the given dtype.

    The returned tensor will be of size ``storage.nbytes``.

    This works for all datatypes and supports strided/offset tensors with the
    caveat that the cast tensor may be larger than the original tensor due to
    the differences in striding.
    """
    assert type(tensor) is torch.Tensor, (
        f"can only cast standard tensors not {type(tensor)}"
    )
    storage = tensor.untyped_storage()
    ret = torch.tensor(storage, dtype=dtype, device=tensor.device)
    assert ret.untyped_storage() is storage, "storage should be the same"
    return ret


def _inplace_buffer(inplace: torch.Tensor, meta: _TensorMeta) -> torch.Tensor:
    """
    Returns the ``uint8`` storage of ``inplace`` so a received tensor can be
    written directly into it. DTensors are unwrapped to their local tensor.
    The storage must be the same size as the sender's.
    """
    if isinstance(inplace, DTensor):
        inplace = inplace._local_tensor
    t = _cast_tensor(inplace, torch.uint8)
    assert t.nbytes == meta.nbytes, "inplace tensor storage must be the same size"
    return t


def _rebuild_tensor(buf: torch.Tensor, meta: _TensorMeta) -> torch.Tensor:
    """
    Reinterprets a received ``uint8`` storage buffer as the sender's tensor,
    restoring its dtype, shape, stride and storage offset.
    """
    return torch.as_strided(
        buf.view(meta.dtype),
        size=meta.shape,
        stride=meta.stride,
        storage_offset=meta.storage_offset,
    )


def _unflatten_state_dict(
    meta: _StateDictMeta,
    recv: Callable[[KeyPath, _TensorMeta], torch.Tensor],
) -> object:
    """
    Rebuilds a state dict from its metadata.

    ``recv(path, tensor_meta)`` is called for each tensor leaf, in the order
    the sender prepared them, and returns that tensor (for example via
    :func:`_rebuild_tensor`). DTensor leaves are rewrapped with their spec, and
    non-tensor leaves are taken from ``meta``.

    If ``recv`` starts transfers asynchronously, the caller must wait for them
    before using the returned state dict.
    """
    values = []
    for path, v in zip(meta.paths, meta.non_tensor_leaves):
        if isinstance(v, _TensorMeta):
            values.append(recv(path, v))
        elif isinstance(v, _DTensorMeta):
            tensor = recv(path, v.local)
            values.append(DTensor(tensor, v.spec, requires_grad=False))
        else:
            # pyrefly: ignore [bad-argument-type]
            values.append(v)

    return tree_unflatten(values, meta.treespec)
