# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import multiprocessing as mp
import signal
import sys
import threading
import time
from datetime import timedelta
from typing import Any, List

import torch
from torchft.checkpointing.pg_transport import _timeit
from torchft.checkpointing.rdma_transport import RDMATransport

logger: logging.Logger = logging.getLogger(__name__)

TIMEOUT: timedelta = timedelta(seconds=60)


def _make_state_dict(
    device: torch.device, value: float, chunk_size: int, total_size: int
) -> dict[str, torch.Tensor]:
    return {
        f"chunk/{i}": torch.full(
            (chunk_size // 4,), value, dtype=torch.float32, device=device
        )
        for i in range(0, total_size, chunk_size)
    }


def _send(
    device: str,
    backend: str,
    chunk_size: int,
    total_size: int,
    transport_options: dict[str, Any],
) -> tuple[RDMATransport[dict[str, torch.Tensor]], str, dict[str, torch.Tensor]]:
    """
    The source replica: publishes a checkpoint. Returns its transport, the
    metadata a receiver needs, and the state dict, which must stay alive while
    it is served.
    """
    with _timeit("create state_dict"):
        state_dict = _make_state_dict(torch.device(device), 1.0, chunk_size, total_size)

    source = RDMATransport[dict[str, torch.Tensor]](
        timeout=TIMEOUT, backend=backend, transport_options=transport_options
    )
    metadata = source.metadata()
    with _timeit("send_checkpoint"):
        source.send_checkpoint(
            dst_ranks=[1], step=1, state_dict=state_dict, timeout=TIMEOUT
        )
    return source, metadata, state_dict


def _stop(source: RDMATransport[dict[str, torch.Tensor]]) -> None:
    with _timeit("disallow_checkpoint"):
        source.disallow_checkpoint()
    source.shutdown()


def _serve(
    device: str,
    backend: str,
    chunk_size: int,
    total_size: int,
    transport_options: dict[str, Any],
    ready: Any,
    done: Any,
) -> None:
    """Serves a checkpoint from a child process until the receiver has read it."""
    logging.basicConfig(level=logging.INFO)

    source, metadata, _state_dict = _send(
        device, backend, chunk_size, total_size, transport_options
    )
    ready.put(metadata)
    done.wait()
    _stop(source)


def _receive(
    metadata: str,
    device: torch.device,
    backend: str,
    chunk_size: int,
    total_size: int,
    inplace: bool,
    transport_options: dict[str, Any],
) -> None:
    """The recovering replica: reads the checkpoint published at ``metadata``."""
    # A replica has its model on the GPU before it recovers. The backend only
    # finds its fast paths for GPU memory if this process has used CUDA by the
    # time the transport connects.
    if device.type == "cuda":
        torch.zeros(1, device=device)

    target = None
    if inplace:
        with _timeit("create state_dict"):
            target = _make_state_dict(device, 0.0, chunk_size, total_size)

    receiver = RDMATransport[dict[str, torch.Tensor]](
        timeout=TIMEOUT,
        backend=backend,
        state_dict=(lambda: target) if inplace else None,
        transport_options=transport_options,
    )

    logger.info(
        f"fetching from {metadata=} {device=} {inplace=} {chunk_size=} {total_size=}"
    )

    start = time.perf_counter()
    with _timeit("recv_checkpoint"):
        received = receiver.recv_checkpoint(
            src_rank=0, metadata=metadata, step=1, timeout=TIMEOUT
        )
    dur = time.perf_counter() - start
    logger.info(f"recv_checkpoint throughput: {total_size / dur / 1e9:.2f} GB/s")

    chunks = list(received.values())
    for chunk in (chunks[0], chunks[-1]):
        assert bool((chunk == 1.0).all()), "received the wrong data"

    receiver.shutdown()


def main(argv: List[str]) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=3_000_000)  # 3MB
    parser.add_argument("--total-size", type=int, default=12_000_000_000)  # 12GB
    parser.add_argument("--backend", type=str, default="nixl")
    # The backend's plugin, e.g. LIBFABRIC for nixl on AWS EFA.
    parser.add_argument("--plugin", type=str)
    # "both" runs the two replicas on this host. To run them on two hosts,
    # start a "source", which prints "metadata: ..." and serves until it reads
    # a line from stdin or is interrupted, and pass that metadata to a
    # "receiver".
    parser.add_argument(
        "--role", choices=["both", "source", "receiver"], default="both"
    )
    parser.add_argument("--metadata", type=str)
    args = parser.parse_args(argv)

    CHUNK_SIZE: int = args.chunk_size
    TOTAL_SIZE: int = args.total_size
    transport_options: dict[str, Any] = {}
    if args.plugin is not None:
        transport_options["plugin"] = args.plugin

    if args.role == "source":
        source, metadata, _state_dict = _send(
            args.device, args.backend, CHUNK_SIZE, TOTAL_SIZE, transport_options
        )
        # The backend logs to stdout too, so mark the line.
        print(f"metadata: {metadata}", flush=True)
        # Without a stdin, as under nohup or kubectl exec without -i, the read
        # returns at once: then serve until interrupted or terminated.
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        with contextlib.suppress(KeyboardInterrupt):
            if not sys.stdin.readline():
                threading.Event().wait()
        _stop(source)
        return

    if args.role == "receiver":
        if args.metadata is None:
            parser.error("--role receiver needs --metadata")
        _receive(
            args.metadata,
            torch.device(args.device),
            args.backend,
            CHUNK_SIZE,
            TOTAL_SIZE,
            inplace=args.inplace,
            transport_options=transport_options,
        )
        return

    # The source serves from one device and the receiver reads into another,
    # as two replicas would.
    src_device = dst_device = torch.device(args.device)
    if src_device.type == "cuda" and src_device.index is None:
        src_device = torch.device("cuda", 0)
        dst_device = torch.device("cuda", 1 if torch.cuda.device_count() > 1 else 0)

    # The source runs in its own process, as a replica does: with both ends in
    # one process the backend cannot use its fast paths between GPUs.
    context = mp.get_context("spawn")
    ready, done = context.Queue(), context.Event()
    process = context.Process(
        target=_serve,
        args=(
            str(src_device),
            args.backend,
            CHUNK_SIZE,
            TOTAL_SIZE,
            transport_options,
            ready,
            done,
        ),
    )
    process.start()

    try:
        _receive(
            ready.get(timeout=300),
            dst_device,
            args.backend,
            CHUNK_SIZE,
            TOTAL_SIZE,
            inplace=args.inplace,
            transport_options=transport_options,
        )
    finally:
        done.set()
        process.join(120)

    assert process.exitcode == 0, f"source exited with {process.exitcode}"


if __name__ == "__main__":
    main(sys.argv[1:])
