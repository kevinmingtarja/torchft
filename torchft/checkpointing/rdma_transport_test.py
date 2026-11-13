# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
from datetime import timedelta
from typing import Any, Callable, Optional
from unittest import SkipTest, TestCase
from unittest.mock import patch

import torch
import torch.distributed as dist
from parameterized import parameterized
from torchft.checkpointing.rdma_transport import (
    RDMATransport,
    torch_transport_available,
)
from torchft.checkpointing.rdma_transport_bench import main as bench_main
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import (
    assertStateDictEqual,
    run_multi_recovery_test,
)
from torchft.process_group import create_store_client

TIMEOUT = timedelta(seconds=10)


def make_views(base: torch.Tensor) -> dict[str, Any]:
    """Leaves that view one storage, as tied weights do, beside other kinds."""
    return {
        "strided": base[1::2],
        "offset": base[3:7],
        "empty": torch.empty(0, device=base.device),
        "host": base.cpu() * 2,
        "int": 1234,
    }


class TestRDMATransport(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Checked here and not at import: it loads the backend and opens an
        # endpoint.
        if not torch_transport_available():
            raise SkipTest("needs torch.distributed._transport + nixl")

    def make_transport(
        self,
        timeout: timedelta = TIMEOUT,
        state_dict: Optional[Callable[[], object]] = None,
    ) -> RDMATransport[dict[str, Any]]:
        transport: RDMATransport[dict[str, Any]] = RDMATransport(
            timeout=timeout, state_dict=state_dict
        )
        self.addCleanup(transport.shutdown, wait=False)
        return transport

    def send(
        self, source: RDMATransport[dict[str, Any]], step: int, state_dict: Any
    ) -> str:
        """Sends as the manager does, and returns the metadata a receiver gets."""
        metadata = source.metadata()
        source.send_checkpoint(
            dst_ranks=[1], step=step, state_dict=state_dict, timeout=TIMEOUT
        )
        return metadata

    def recv(
        self, receiver: RDMATransport[dict[str, Any]], metadata: str, step: int
    ) -> dict[str, Any]:
        return receiver.recv_checkpoint(
            src_rank=0, metadata=metadata, step=step, timeout=TIMEOUT
        )

    def control_store(self, metadata: str) -> dist.Store:
        """The keys of the send that ``metadata`` was handed out for."""
        return create_store_client(metadata, timeout=TIMEOUT)

    @parameterized.expand([("cpu",), ("cuda",)])
    def test_multi_recovery(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("needs CUDA")

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            return RDMATransport(timeout=TIMEOUT)

        run_multi_recovery_test(self, init, device=torch.device(device))

    @parameterized.expand(
        [("cpu", False), ("cpu", True), ("cuda", False), ("cuda", True)]
    )
    def test_storage_layouts(self, device: str, inplace: bool) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        expected = make_views(torch.arange(10, dtype=torch.float32, device=device))
        target = make_views(torch.zeros(10, device=device))
        source = self.make_transport()
        receiver = self.make_transport(state_dict=(lambda: target) if inplace else None)

        metadata = self.send(source, step=1, state_dict=expected)
        got = self.recv(receiver, metadata, step=1)

        assertStateDictEqual(self, got, expected)
        if inplace:
            # Read straight into the target's storage, on its device.
            assertStateDictEqual(self, target, expected)
            self.assertEqual(
                got["strided"].untyped_storage().data_ptr(),
                target["strided"].untyped_storage().data_ptr(),
            )
        else:
            self.assertEqual(got["strided"].device.type, "cpu")

    def test_send_lifecycle(self) -> None:
        source = self.make_transport()
        receiver = self.make_transport()
        expected: dict[str, Any] = {"t": torch.rand(4)}

        metadata = self.send(source, step=1, state_dict=expected)
        assertStateDictEqual(self, self.recv(receiver, metadata, step=1), expected)
        with self.assertRaisesRegex(AssertionError, "expected step 2, got 1"):
            self.recv(receiver, metadata, step=2)

        # Once disallowed, the send's address refuses readers at once.
        source.disallow_checkpoint()
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "no longer being served"):
            self.recv(receiver, metadata, step=1)
        self.assertLess(time.monotonic() - start, 2)
        # No reader is left counted, whether its receive failed or not.
        self.assertEqual(self.control_store(metadata).add("readers", 0), 0)

        # A failed commit leaves the step unchanged, so it can be sent again.
        expected = {"t": torch.rand(4)}
        metadata = self.send(source, step=1, state_dict=expected)
        assertStateDictEqual(self, self.recv(receiver, metadata, step=1), expected)

    def test_disallow_waits_for_reader(self) -> None:
        source = self.make_transport()
        metadata = self.send(source, step=1, state_dict={"t": torch.rand(4)})
        store = self.control_store(metadata)
        # A reader that has started and not finished.
        store.add("readers", 1)

        thread = threading.Thread(target=source.disallow_checkpoint)
        thread.start()
        thread.join(timeout=0.3)
        self.assertTrue(thread.is_alive(), "returned while a reader was active")

        store.add("readers", -1)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        # The reader finished in time, so its data is valid.
        self.assertEqual(store.get("state"), b"closed")

    def test_disallow_gives_up_on_stuck_reader(self) -> None:
        source = self.make_transport(timeout=timedelta(seconds=0.3))
        metadata = self.send(source, step=1, state_dict={"t": torch.rand(4)})
        store = self.control_store(metadata)
        # A reader that died mid-read never counts itself out.
        store.add("readers", 1)

        start = time.monotonic()
        source.disallow_checkpoint()

        self.assertGreaterEqual(time.monotonic() - start, 0.3)
        self.assertEqual(store.get("state"), b"aborted")

    def test_shutdown_without_wait(self) -> None:
        source = self.make_transport()
        metadata = self.send(source, step=1, state_dict={"t": torch.rand(4)})
        store = self.control_store(metadata)
        store.add("readers", 1)

        start = time.monotonic()
        source.shutdown(wait=False)

        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(store.get("state"), b"aborted")

    def test_recv_fails_if_source_stops_waiting(self) -> None:
        source = self.make_transport(timeout=timedelta(seconds=0.1))
        receiver = self.make_transport()
        metadata = self.send(source, step=1, state_dict={"t": torch.rand(4)})
        close = receiver._close_endpoint

        # The source reaches disallow_checkpoint while this reader is counted
        # in, waits out its timeout and continues. The reader must not return
        # data that the source may since have changed.
        def close_then_source_gives_up(*args: Any) -> None:
            close(*args)
            source.disallow_checkpoint()

        with patch.object(
            receiver, "_close_endpoint", side_effect=close_then_source_gives_up
        ):
            with self.assertRaisesRegex(RuntimeError, "stopped waiting"):
                self.recv(receiver, metadata, step=1)
        # The reader still counted itself out.
        self.assertEqual(self.control_store(metadata).add("readers", 0), 0)

    def test_recv_refuses_another_plugin(self) -> None:
        source = self.make_transport()
        receiver = self.make_transport()

        # Descriptors from one plugin mean nothing to another.
        with patch(
            "torchft.checkpointing.rdma_transport._endpoint_plugin",
            side_effect=["UCX", "OTHER"],
        ):
            metadata = self.send(source, step=1, state_dict={"t": torch.rand(4)})
            with self.assertRaisesRegex(
                ValueError, "source uses plugin 'UCX'.*this replica uses 'OTHER'"
            ):
                self.recv(receiver, metadata, step=1)

    def test_unavailable_plugin(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unavailable with options"):
            RDMATransport(
                timeout=TIMEOUT, transport_options={"plugin": "NO_SUCH_PLUGIN"}
            )

    def test_benchmark(self) -> None:
        bench_main(["--chunk-size=10", "--total-size=100", "--device=cpu"])
