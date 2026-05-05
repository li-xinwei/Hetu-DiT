"""NIXL-accelerated weight loading at cold start (D1).

Two strategies, gated by ``runtime_config.init_strategy``:

* ``nixl_broadcast``: rank 0 fully completes ``model.to('cuda')``, registers
  every top-level submodule with NIXL, then signals all peers to fetch every
  block. Peers wait for "all ready" before issuing their NIXL pulls.

* ``nixl_pipelined``: rank 0 copies block-by-block, signalling each ready
  immediately, so peers fetch block N while rank 0 is copying block N+1.

Preconditions
-------------
* ``Worker.nixl_manager`` already created and peers connected before any call
  here. ``async_serving_engine._init_workers_ray`` does this when
  ``init_strategy != 'default'``, before ``init_instance_model``.
* All ranks have already called ``_load_serving_pipeline`` so the module
  *structure* is identical across ranks (singleton CPU pipeline). Only the
  parameter *data* differs at this point.

Note on serialization: this module uses ``pickle`` for NIXL transfer
descriptor exchange between ranks. This matches the existing pattern in
``hetu_dit/core/distributed/nixl_manager.py`` (line 74). The data flows
inside a trusted Ray actor cluster, never from external sources.
"""

from __future__ import annotations

import asyncio
import pickle as _pickle
from typing import List, Tuple

import torch
import torch.nn as nn

from hetu_dit.cstrace import cst_print
from hetu_dit.logger import init_logger

logger = init_logger(__name__)

_PIPELINE_SUBMODULE_NAMES: Tuple[str, ...] = (
    "transformer",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "vae",
    "unet",
)


def _enumerate_pipeline_blocks(model) -> List[Tuple[str, nn.Module]]:
    blocks: List[Tuple[str, nn.Module]] = []
    for name in _PIPELINE_SUBMODULE_NAMES:
        sub = getattr(model, name, None)
        if isinstance(sub, nn.Module) and any(p.numel() > 0 for p in sub.parameters()):
            blocks.append((name, sub))
    if not blocks:
        raise RuntimeError(
            "no recognised submodules with parameters on the pipeline; "
            "extend _PIPELINE_SUBMODULE_NAMES if a new model adds new components"
        )
    return blocks


def _allocate_gpu_skeleton(blocks) -> None:
    """Replace every CPU parameter / buffer on each pipeline submodule with an
    empty CUDA tensor of identical shape and dtype. Pipelines (StableDiffusion3Pipeline,
    FluxPipeline, ...) aren't nn.Module — they don't expose ``.modules()`` —
    so iterate over the enumerated nn.Module blocks instead."""
    if hasattr(blocks, "modules"):
        # called with a single nn.Module (legacy)
        modules_iter = blocks.modules()
    else:
        modules_iter = (m for _, sub in blocks for m in sub.modules())
    for module in modules_iter:
        for pname, p in list(module._parameters.items()):
            if p is None or p.is_cuda:
                continue
            module._parameters[pname] = nn.Parameter(
                torch.empty(p.shape, dtype=p.dtype, device="cuda"),
                requires_grad=p.requires_grad,
            )
        for bname, b in list(module._buffers.items()):
            if b is None or b.is_cuda:
                continue
            module._buffers[bname] = torch.empty(b.shape, dtype=b.dtype, device="cuda")


async def init_instance_model_via_nixl(worker, *, model, init_strategy: str) -> None:
    assert init_strategy in ("nixl_broadcast", "nixl_pipelined"), init_strategy
    rank = worker.rank
    nixl_manager = getattr(worker, "nixl_manager", None)
    if nixl_manager is None:
        raise RuntimeError(
            f"rank {rank}: NIXL manager not initialised — engine should have "
            "called create_nixl_manager + init_nixl_peers before init_instance_model "
            "when init_strategy != default"
        )

    blocks = _enumerate_pipeline_blocks(model)
    cst_print("nixl_load_blocks_enumerated", rank=rank, count=len(blocks))

    if rank == 0:
        await _rank0_load_and_publish(worker, model, blocks, nixl_manager, init_strategy)
    else:
        await _peer_skeleton_and_fetch(worker, model, blocks, nixl_manager, init_strategy)


async def _rank0_load_and_publish(worker, model, blocks, nixl_manager, init_strategy: str) -> None:
    rank = worker.rank
    if init_strategy == "nixl_broadcast":
        cst_print("block_load_start", rank=rank, block="all")
        model.to("cuda")
        cst_print("block_load_done", rank=rank, block="all")
        for idx, (name, mod) in enumerate(blocks):
            nixl_manager.register_block(idx, mod)
            cst_print("block_registered", rank=rank, block=name, idx=idx)
        for idx, (name, _) in enumerate(blocks):
            await _broadcast_block_ready(worker, idx, name)
    else:
        for idx, (name, mod) in enumerate(blocks):
            cst_print("block_load_start", rank=rank, block=name, idx=idx)
            mod.to("cuda")
            cst_print("block_load_done", rank=rank, block=name, idx=idx)
            nixl_manager.register_block(idx, mod)
            await _broadcast_block_ready(worker, idx, name)


async def _peer_skeleton_and_fetch(worker, model, blocks, nixl_manager, init_strategy: str) -> None:
    rank = worker.rank
    cst_print("skeleton_alloc_start", rank=rank)
    _allocate_gpu_skeleton(blocks)
    cst_print("skeleton_alloc_done", rank=rank)

    for idx, (name, mod) in enumerate(blocks):
        nixl_manager.register_block(idx, mod)

    for idx, (name, mod) in enumerate(blocks):
        await _wait_block_ready(worker, idx)
        cst_print("block_load_start", rank=rank, block=name, idx=idx)
        await _nixl_pull_block(worker, idx, name, mod, nixl_manager)
        cst_print("block_load_done", rank=rank, block=name, idx=idx)


async def _broadcast_block_ready(worker, idx: int, name: str) -> None:
    """Rank 0 -> all peers: mark block ``idx`` ready to fetch."""
    refs = []
    for r, handle in worker.all_worker_handles.items():
        if r == worker.rank:
            continue
        refs.append(handle.coldstart_mark_block_ready.remote(idx, name))
    if refs:
        await asyncio.gather(*refs)


async def _wait_block_ready(worker, idx: int) -> None:
    ev = worker.coldstart_ready_events.setdefault(idx, asyncio.Event())
    await ev.wait()


async def _nixl_pull_block(worker, idx: int, name: str, mod: nn.Module, nixl_manager) -> None:
    """Receiver-side: prepare destination NIXL descriptors + RPC rank 0 to push.

    Protocol:
      1. Build destination address descriptors from this rank's already-
         registered tensors (registration done in _peer_skeleton_and_fetch).
      2. Get this agent's partial metadata so rank 0 can identify us.
      3. Build the receiver-side xfer descriptors (sorted, cuda mem).
      4. RPC rank 0's coldstart_send_block, which performs the actual NIXL
         WRITE-mode transfer and waits for completion.
    """
    agent = nixl_manager.agent

    tensors = nixl_manager.registered_blocks.get(idx, [])
    if not tensors:
        raise RuntimeError(
            f"rank {worker.rank}: block idx={idx} ({name}) has no NIXL-registered "
            "tensors on receiver side; register_block must run before pull"
        )

    dest_descs = [
        (t.data_ptr(), t.element_size() * t.numel(), t.device.index)
        for t in tensors
    ]

    reg_desc = agent.get_reg_descs(tensors)
    partial_md = agent.get_partial_agent_metadata(reg_desc, inc_conn_info=True)
    remote_xfer_desc = agent.get_xfer_descs(dest_descs, mem_type="cuda")
    remote_xfer_desc_bytes = _pickle.dumps(remote_xfer_desc)

    rank0_handle = worker.all_worker_handles[0]
    await rank0_handle.coldstart_send_block.remote(
        block_idx=idx,
        remote_xfer_desc_bytes=remote_xfer_desc_bytes,
        remote_partial_md=partial_md,
        receiver_rank=worker.rank,
    )
