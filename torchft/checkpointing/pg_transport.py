# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable, cast, Generator, Optional, TypeVar

import torch
from torch.distributed import Work
from torch.utils._pytree import KeyPath, tree_flatten_with_path
from torchft.checkpointing._state_dict import (  # noqa: F401
    # Metadata pickled by a torchft from before these classes moved names
    # this module for them, so all three must be importable from here.
    _DTensorMeta,
    _inplace_buffer,
    _prepare_state_dict,
    _rebuild_tensor,
    _StateDictMeta,
    _TensorMeta,
    _unflatten_state_dict,
)
from torchft.checkpointing.transport import CheckpointTransport
from torchft.process_group import ProcessGroup

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def _timeit(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    logger.info(f"{name} took {dur}s")


class PGTransport(CheckpointTransport[T]):
    """
    This is a checkpoint transport that uses the process group to transfer checkpoints.
    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        pg: the process group to use for communication
        timeout: the default timeout for each communication wait. An explicit
            timeout passed to send_checkpoint or recv_checkpoint overrides this
            value. This is not a deadline for the entire checkpoint transfer and
            does not change the process group's own timeout.
        device: the device to use for tensors
        state_dict: if specified this function will be called to do an inplace
            receive into the returned state_dict. This is much faster than
            having to allocate new tensors and transferring them to the CPU.
    """

    def __init__(
        self,
        pg: ProcessGroup,
        timeout: timedelta,
        device: torch.device,
        state_dict: Optional[Callable[[], object]] = None,
    ) -> None:
        self._work: list[Work] = []
        self._pg = pg
        self._timeout = timeout
        self._device = device
        self._state_dict = state_dict

    def metadata(self) -> str:
        return "<n/a>"

    def disallow_checkpoint(self) -> None:
        pass

    def send_checkpoint(
        self,
        dst_ranks: list[int],
        step: int,
        state_dict: T,
        timeout: Optional[timedelta] = None,
    ) -> None:
        """Send a checkpoint, using the constructor timeout if timeout is None."""
        wait_timeout: timedelta = self._timeout if timeout is None else timeout

        with _timeit("preparing state_dict"):
            meta, tensors = _prepare_state_dict(state_dict, step)

        work = []

        with _timeit("send pickle"):
            buf = pickle.dumps(meta)
            len_t = torch.tensor([len(buf)], dtype=torch.int64, device=self._device)
            buf_t = torch.frombuffer(buf, dtype=torch.uint8).to(self._device)
            for dst_rank in dst_ranks:
                work.append(self._pg.send([len_t], dst_rank, tag=1))
                work.append(self._pg.send([buf_t], dst_rank, tag=2))

        with _timeit("send tensors"):
            for i, t in enumerate(tensors):
                original_device = t.device
                t = t.to(self._device)
                for dst_rank in dst_ranks:
                    work.append(self._pg.send([t], dst_rank, tag=3 + i))

                # if we did a copy we should wait for the work to complete so we
                # can free the memory to avoid OOMs
                if original_device == torch.device("cpu"):
                    for w in work:
                        w.wait(wait_timeout)
                    work = []

            for w in work:
                w.wait(wait_timeout)

    def recv_checkpoint(
        self,
        src_rank: int,
        metadata: str,
        step: int,
        timeout: Optional[timedelta] = None,
    ) -> T:
        """Receive a checkpoint, using the constructor timeout if timeout is None."""
        wait_timeout: timedelta = self._timeout if timeout is None else timeout

        state_dict = self._state_dict() if self._state_dict else {}
        state_dict_leaves, _ = tree_flatten_with_path(state_dict)

        dst_tensors: dict[KeyPath, object] = dict(state_dict_leaves)

        len_t = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._pg.recv([len_t], src_rank, tag=1).wait(wait_timeout)
        length = cast(int, len_t.item())

        assert length > 0, f"invalid metadata length {length=}"

        buf = torch.empty(length, dtype=torch.uint8, device=self._device)
        self._pg.recv([buf], src_rank, tag=2).wait(wait_timeout)

        meta: _StateDictMeta = pickle.loads(buf.cpu().numpy().tobytes())
        assert meta.step == step

        i: int = 0
        works: list[Work] = []

        def recv(path: KeyPath, v: _TensorMeta) -> torch.Tensor:
            nonlocal i

            inplace = dst_tensors.get(path)
            if (
                isinstance(inplace, torch.Tensor)
                and inplace.device.type == self._device.type
            ):
                t = _inplace_buffer(inplace, v)
            else:
                t = torch.empty(v.nbytes, dtype=torch.uint8, device=self._device)

            work = self._pg.recv([t], src_rank, tag=3 + i)
            i += 1

            if inplace is None:
                # if not inplace we need to copy it to CPU to avoid OOMing
                work.wait(wait_timeout)
                t = t.cpu()
            else:
                works.append(work)

            return _rebuild_tensor(t, v)

        received = _unflatten_state_dict(meta, recv)

        for work in works:
            work.wait(wait_timeout)

        return cast(T, received)
