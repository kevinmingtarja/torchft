# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import pickle
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, cast, List, Optional, TYPE_CHECKING, TypeVar

import torch
from torch.distributed import PrefixStore, Store, TCPStore, Work
from torch.utils._pytree import KeyPath, tree_flatten_with_path
from torchft.checkpointing._state_dict import (
    _inplace_buffer,
    _prepare_state_dict,
    _rebuild_tensor,
    _StateDictMeta,
    _TensorMeta,
    _unflatten_state_dict,
)
from torchft.checkpointing.transport import CheckpointTransport
from torchft.process_group import create_store_client

if TYPE_CHECKING:
    from torch.distributed._transport import RemoteBuffer, Transport

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

# A send's keys in the control store.
_STATE_KEY = "state"
# How many receivers have counted themselves in and not out again. The source
# cannot see one-sided reads, so this is all it knows of its readers: one that
# died mid-read stays counted.
_READERS_KEY = "readers"
# The pickled _Rendezvous, or a pickled None once the send has ended.
_RENDEZVOUS_KEY = "rendezvous"

# Values of a send's state key.
_OPEN = "open"  # readers may start
_CLOSED = "closed"  # no new readers; ones that already started may finish
# The send failed or could not be ended cleanly, or the source stopped waiting
# for its readers: what a receiver read may be torn.
_ABORTED = "aborted"

_DRAIN_POLL_INTERVAL = 0.01


# The backends, with their options, that have been seen to work. A failure is
# not remembered: it may be transient.
_AVAILABLE: set[tuple[str, str]] = set()


def torch_transport_available(backend: str = "nixl", **options: Any) -> bool:
    """Return whether the given torch.distributed transport backend works.

    Creates and closes one transport with ``options``, since registered
    backends and their plugins may lack their native dependencies. A backend
    that works is remembered: loading one can take seconds.
    """

    def works() -> bool:
        try:
            from torch.distributed._transport import available_transports, new_transport
        except ImportError:
            return False
        if backend.lower() not in available_transports():
            return False
        try:
            new_transport(backend, **options).close()
        except Exception:
            logger.warning(
                "torch.distributed transport backend %r is unusable with options %r",
                backend,
                options,
                exc_info=True,
            )
            return False
        return True

    # Options need not be hashable, NIXL takes a dict of plugin parameters.
    key = (backend, repr(sorted(options.items())))
    if key not in _AVAILABLE and works():
        _AVAILABLE.add(key)
    return key in _AVAILABLE


def _endpoint_plugin(transport: "Transport", options: dict[str, Any]) -> Optional[str]:
    """
    Returns the plugin that ``transport`` runs on: the one the backend resolved
    its options to if it says so, as NIXL does, else the one the options name.
    Two replicas that name NIXL's default differently then still agree.
    """
    plugin = getattr(transport, "_plugin", None) or options.get("plugin")
    return None if plugin is None else str(plugin).upper()


def _publish_rendezvous(send_store: Store, rendezvous: "Optional[_Rendezvous]") -> None:
    """
    Publishes a send's rendezvous record, or None once the send has ended, so
    that a receiver waiting for the record fails at once.
    """
    # Store.set accepts bytes; the stub only declares str.
    # pyrefly: ignore [bad-argument-type]
    send_store.set(_RENDEZVOUS_KEY, pickle.dumps(rendezvous))


@dataclass
class _Rendezvous:
    """
    What the source publishes in the store for one send: how to reach its
    endpoint and where to read the checkpoint's metadata.

    Args:
        backend: the source's transport backend
        bind: the source endpoint's connection bytes
        meta: the serialized descriptor of the buffer that holds the pickled
            :class:`_CheckpointMeta`
        meta_nbytes: the size of that buffer
        plugin: the backend plugin the source's endpoint runs on, if known.
            Descriptors from one plugin mean nothing to another.
    """

    backend: str
    bind: bytes
    meta: bytes
    meta_nbytes: int
    plugin: Optional[str]


@dataclass
class _CheckpointMeta:
    """
    The metadata of one send. The receiver reads it from the source's memory,
    as it does the tensors: it grows with the number of tensors, and a store
    value cannot exceed 8 MiB.

    Args:
        state_dict: the structure of the state dict and its tensor leaves
        buffers: one serialized descriptor per tensor leaf, in order, or
            ``None`` for an empty storage. Leaves that share a storage have
            the same descriptor.
    """

    state_dict: _StateDictMeta
    buffers: list[Optional[bytes]]


@dataclass
class _ActiveSend:
    """
    The send a source is currently serving.

    Args:
        step: the step being served
        send_store: this send's keys in the control store
        transport: the endpoint holding the memory registrations
        meta: the buffer holding the pickled metadata
    """

    step: int
    send_store: Store
    transport: "Transport"
    meta: torch.Tensor


class RDMATransport(CheckpointTransport[T]):
    """
    This is a checkpoint transport that uses RDMA to transfer checkpoints,
    through a ``torch.distributed._transport`` backend. This allows for fast
    recovery of workers by reading the current weights directly from the
    memory of an existing worker.

    With NIXL's UCX plugin, UCX can take host memory for CUDA memory that the
    process has freed (``torch.cuda.empty_cache()``), and fail to register it
    (``NIXL_ERR_BACKEND``). To rule that out, pass
    ``transport_options={"backend_options": {"engine_config": "MEMTYPE_CACHE=n"}}``,
    or start the process with ``UCX_MEMTYPE_CACHE=n``.

    Args:
        timeout: the default timeout for control-plane operations, and how long
            disallow_checkpoint waits for in-flight readers
        backend: the ``torch.distributed._transport`` backend to use
        state_dict: if specified this function will be called to do an inplace
            receive into the returned state_dict. Tensors without a match are
            received into newly allocated host memory.
        transport_options: keyword options for the backend's ``new_transport``.
            On AWS EFA pass ``{"plugin": "LIBFABRIC"}``: NIXL's default UCX
            plugin only finds TCP there. Every replica must use the same
            plugin.
    """

    def __init__(
        self,
        timeout: timedelta,
        backend: str = "nixl",
        state_dict: Optional[Callable[[], object]] = None,
        transport_options: Optional[dict[str, Any]] = None,
    ) -> None:
        options = dict(transport_options or {})
        if not torch_transport_available(backend, **options):
            raise RuntimeError(
                f"torch.distributed transport backend {backend!r} is unavailable"
                + (f" with options {options!r}. " if options else ". ")
                + "It requires a PyTorch build with torch.distributed._transport "
                "and a backend listed by available_transports() whose native "
                "dependencies are installed (for nixl: pip install nixl; its "
                "LIBFABRIC plugin also needs libfabric and hwloc)."
            )

        self._timeout = timeout
        self._backend: str = backend.lower()
        self._transport_options = options
        self._state_dict = state_dict
        self._store = TCPStore(
            host_name=socket.gethostname(),
            port=0,
            is_master=True,
            wait_for_workers=False,
            timeout=timeout,
        )

        # send_checkpoint runs on the manager's quorum thread and
        # disallow_checkpoint on the training thread.
        self._lock = threading.Lock()
        # Set by shutdown(wait=False), to end a wait for readers that another
        # thread is in while holding the lock.
        self._skip_drain = threading.Event()
        # Identifies the next send: its keys in the store are under it.
        self._session_id: str = uuid.uuid4().hex
        self._active_send: Optional[_ActiveSend] = None
        logger.info(f"Started RDMATransport control store on {self.address()}...")

    def address(self) -> str:
        """
        Returns the address of this transport's control store.

        Format: host:port

        Returns:
            a TCPStore address
        """
        return f"{socket.gethostname()}:{self._store.port}"

    def metadata(self) -> str:
        """
        Returns where the next send will publish its checkpoint.

        This is only valid for that one send, so fetch it again for every
        recovery, before the send starts, as the manager does.

        Format: host:port/session_id
        """
        return f"{self.address()}/{self._session_id}"

    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        """
        Registers the state dict's tensor storages and publishes their
        descriptors. Returns without waiting: receivers read the memory
        directly, so it must not change until :meth:`disallow_checkpoint`.
        """
        from torch.distributed._transport import new_transport

        timeout_seconds = timeout.total_seconds()
        with self._lock:
            # Normally a no-op: disallow_checkpoint should have ended the previous send.
            self._stop_serving(wait=True)

            send_store = PrefixStore(self._session_id, self._store)
            self._session_id = uuid.uuid4().hex

            transport = None
            try:
                meta, storages = _prepare_state_dict(state_dict, step)

                # Reads don't follow CUDA streams, so pending writes must land
                # first.
                for device in {t.device for t in storages if t.is_cuda}:
                    torch.cuda.synchronize(device)

                # Opened for this send only: an idle endpoint keeps a core busy
                # (NIXL's progress thread). No device restriction, so both host
                # and GPU tensors can be registered.
                transport = new_transport(self._backend, **self._transport_options)
                # Taken before anything is registered. Afterwards it would
                # describe every registration, which each descriptor already
                # does for its own.
                bind = transport.bind(timeout=timeout_seconds)

                # One descriptor per storage: leaves can share one, as tied
                # weights do.
                descriptors: dict[tuple[torch.device, int, int], bytes] = {}
                buffers: list[Optional[bytes]] = []
                for storage in storages:
                    if storage.nbytes == 0:
                        buffers.append(None)
                        continue
                    key = (storage.device, storage.data_ptr(), storage.nbytes)
                    if key not in descriptors:
                        memory = transport.register_memory(
                            storage, timeout=timeout_seconds
                        )
                        descriptors[key] = memory.to_remote_buffer(
                            timeout=timeout_seconds
                        ).serialize()
                    buffers.append(descriptors[key])

                published = _CheckpointMeta(state_dict=meta, buffers=buffers)
                meta_buf = torch.frombuffer(
                    bytearray(pickle.dumps(published)), dtype=torch.uint8
                )
                meta_memory = transport.register_memory(
                    meta_buf, timeout=timeout_seconds
                )
                rendezvous = _Rendezvous(
                    backend=self._backend,
                    bind=bind,
                    meta=meta_memory.to_remote_buffer(
                        timeout=timeout_seconds
                    ).serialize(),
                    meta_nbytes=meta_buf.nbytes,
                    plugin=_endpoint_plugin(transport, self._transport_options),
                )
                send_store.set(_STATE_KEY, _OPEN)
                # Published last: a receiver that sees it can rely on the rest.
                _publish_rendezvous(send_store, rendezvous)
            except BaseException:
                # So that receivers of this send fail at once instead of waiting
                # for it, if the store still takes these writes.
                with contextlib.suppress(Exception):
                    send_store.set(_STATE_KEY, _ABORTED)
                with contextlib.suppress(Exception):
                    _publish_rendezvous(send_store, None)
                if transport is not None:
                    self._close_endpoint(transport)
                raise

            self._active_send = _ActiveSend(
                step=step, send_store=send_store, transport=transport, meta=meta_buf
            )

    def disallow_checkpoint(self) -> None:
        """
        Stops serving the checkpoint and closes the endpoint that holds its
        memory registrations. A failure to close it is only logged.

        New readers are refused, and readers that already started are given the
        constructor ``timeout`` to finish. If they don't, the send is marked
        aborted, which makes a receiver discard what it read, and this returns
        anyway: nothing writes to the source's memory, so the source can always
        continue.

        Once this returns, the state_dict may be mutated.
        """
        with self._lock:
            self._stop_serving(wait=True)

    def _stop_serving(self, wait: bool) -> None:
        """Ends the send being served, if any. Must be called with the lock held."""
        active = self._active_send
        if active is None:
            return
        send_store = active.send_store

        try:
            # Close before counting the readers. A receiver adds itself to the
            # count first and only then checks that the state is open, so one
            # that this count misses is sure to see the state closed.
            send_store.set(_STATE_KEY, _CLOSED)
            _publish_rendezvous(send_store, None)

            deadline = time.monotonic() + (
                self._timeout.total_seconds() if wait else 0.0
            )
            while (readers := send_store.add(_READERS_KEY, 0)) > 0:
                if self._skip_drain.is_set() or time.monotonic() >= deadline:
                    send_store.set(_STATE_KEY, _ABORTED)
                    logger.warning(
                        f"stopped waiting for {readers} reader(s) of the checkpoint "
                        f"for step {active.step}; their recovery will fail"
                    )
                    break
                time.sleep(_DRAIN_POLL_INTERVAL)
        except Exception:
            # The source continues either way. Readers must not trust this
            # send: the state says so, if the store still takes the write below.
            logger.exception(
                f"failed to end the checkpoint send for step {active.step} cleanly"
            )
            with contextlib.suppress(Exception):
                send_store.set(_STATE_KEY, _ABORTED)
        finally:
            self._close_endpoint(active.transport)
            self._active_send = None

    def _close_endpoint(
        self, transport: "Transport", timeout: Optional[float] = None
    ) -> None:
        """
        Closes an endpoint. Logs instead of raising if that fails, and does not
        try again: an endpoint whose close timed out stays open.
        """
        if timeout is None:
            timeout = self._timeout.total_seconds()
        try:
            transport.close(timeout=timeout)
        except Exception:
            logger.exception(
                "failed to close a checkpoint transport endpoint, which may "
                "keep its buffers registered until this process exits"
            )

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        """
        Reads the checkpoint for ``step`` from the source at ``metadata``.

        ``timeout`` bounds each wait, for the send and for each read. It is
        not a deadline for the whole receive, and it does not cancel a read:
        one still in flight when this raises can write to an in-place target
        afterwards.

        The calls to the source's store are plain TCPStore calls. While the
        source's process is stopped or its host is unreachable they block,
        whatever their timeout, and so does this.
        """
        from torch.distributed._transport import new_transport

        timeout_seconds = timeout.total_seconds()

        store = create_store_client(metadata, timeout=timeout)
        store.wait([_RENDEZVOUS_KEY], timeout)
        rendezvous: Optional[_Rendezvous] = pickle.loads(store.get(_RENDEZVOUS_KEY))
        if rendezvous is None:
            raise RuntimeError(
                f"the checkpoint for step {step} is no longer being served"
            )
        if rendezvous.backend != self._backend:
            raise ValueError(
                f"source uses transport {rendezvous.backend!r}, "
                f"this replica uses {self._backend!r}"
            )

        targets: dict[KeyPath, object] = {}
        if self._state_dict is not None:
            targets = dict(tree_flatten_with_path(self._state_dict())[0])

        try:
            # Count in before checking the state, so the source either refuses
            # this reader or waits for it, for as long as its timeout.
            store.add(_READERS_KEY, 1)
            if store.get(_STATE_KEY).decode() != _OPEN:
                raise RuntimeError(
                    f"the checkpoint for step {step} is no longer being served"
                )

            transport = new_transport(self._backend, **self._transport_options)
            deadline = time.monotonic() + timeout_seconds
            try:
                plugin = _endpoint_plugin(transport, self._transport_options)
                if rendezvous.plugin != plugin:
                    raise ValueError(
                        f"source uses plugin {rendezvous.plugin!r} of transport "
                        f"{self._backend!r}, this replica uses {plugin!r}"
                    )
                transport.connect(rendezvous.bind, timeout=timeout_seconds)

                meta_buf = torch.empty(rendezvous.meta_nbytes, dtype=torch.uint8)
                meta_memory = transport.register_memory(
                    meta_buf, timeout=timeout_seconds
                )
                # The class of this endpoint's own descriptors, so that the
                # peer's bytes do not choose the class that parses them. The
                # peer is trusted all the same: the rendezvous record and the
                # metadata are unpickled.
                remote_buffer_cls = cast(
                    "type[RemoteBuffer]",
                    type(meta_memory.to_remote_buffer(timeout=timeout_seconds)),
                )
                transport.read(
                    meta_memory.to_mutable_view(),
                    remote_buffer_cls.deserialize(rendezvous.meta),
                    timeout=timeout_seconds,
                )
                published: _CheckpointMeta = pickle.loads(meta_buf.numpy().tobytes())
                meta: _StateDictMeta = published.state_dict
                assert meta.step == step, f"expected step {step}, got {meta.step}"

                # As on the source: writes pending on a target must land first.
                for device in {
                    t.device
                    for t in targets.values()
                    if isinstance(t, torch.Tensor) and t.is_cuda
                }:
                    torch.cuda.synchronize(device)

                remotes = iter(published.buffers)
                works: list[Work] = []

                def recv(path: KeyPath, v: _TensorMeta) -> torch.Tensor:
                    remote = next(remotes)
                    inplace = targets.get(path)
                    if isinstance(inplace, torch.Tensor):
                        buf = _inplace_buffer(inplace, v)
                    else:
                        buf = torch.empty(v.nbytes, dtype=torch.uint8)
                    # Nothing to read for an empty storage.
                    if remote is not None:
                        memory = transport.register_memory(buf, timeout=timeout_seconds)
                        work: Work | int = transport.read(
                            memory.to_mutable_view(),
                            remote_buffer_cls.deserialize(remote),
                            async_op=True,
                            timeout=timeout_seconds,
                        )
                        works.append(cast(Work, work))
                    return _rebuild_tensor(buf, v)

                received = _unflatten_state_dict(meta, recv)

                for work in works:
                    work.wait()
            except BaseException:
                # Closing waits for the reads still in flight, so it gets only
                # what is left of the timeout: such a read may never end.
                self._close_endpoint(transport, max(deadline - time.monotonic(), 0.0))
                raise
            self._close_endpoint(transport, timeout_seconds)

            # The reads are done, but the source may have given up on them and
            # changed its state while they were in flight.
            if store.get(_STATE_KEY).decode() == _ABORTED:
                raise RuntimeError(
                    f"the source stopped waiting for the read of step {step}; "
                    "the received data may be inconsistent"
                )
        finally:
            try:
                store.add(_READERS_KEY, -1)
            except Exception:
                # Raising here would replace the read's own result or error.
                logger.warning(
                    "failed to report the end of a checkpoint read", exc_info=True
                )

        return cast(T, received)

    def shutdown(self, wait: bool = True) -> None:
        """
        Stops serving and closes the endpoint that holds the memory
        registrations.

        Args:
            wait: whether to give in-flight readers the constructor ``timeout``
                to finish. Otherwise the send is marked aborted at once, and
                their receive fails.
        """
        if not wait:
            self._skip_drain.set()
        try:
            with self._lock:
                self._stop_serving(wait=wait)
        finally:
            self._skip_drain.clear()
