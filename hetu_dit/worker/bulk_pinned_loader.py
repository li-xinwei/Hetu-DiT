"""Bulk Pinned Staging (BPS) for cold-start weight load.

Replaces ``model = model.to("cuda")`` (which issues ~500-1000 individual
cudaMemcpyAsync calls and is dominated by per-tensor driver launch overhead)
with a single bulk H2D transfer:

    1. Allocate one pinned host slab sized to fit all leaf parameters/buffers
    2. Copy each tensor into its slot in the slab; immediately drop the CPU ref
    3. Allocate one GPU slab and issue a single cudaMemcpyAsync host -> device
    4. Re-point every parameter/buffer to a typed view over its slab slice

Measurement on RunPod 4x H100 SXM 80GB (2026-05-06): the per-tensor variant
takes ~2.34s for SD3 (17GB FP16); the physical lower bound at PCIe Gen5 64
GB/s is ~266ms. BPS targets <=500ms by amortizing driver launches.

Compatibility:
- SD3 only in this initial version; gated on ``model_class is
  StableDiffusion3Pipeline`` at the call site.
- ``adjust_strategy=p2p`` works because the existing ``_refresh_nixl_for_cache``
  path (worker.py) detects data_ptr changes and re-registers with NIXL.
- Tied/shared parameters are detected via ``untyped_storage().data_ptr()`` and
  share a single slab slice (no double-copy, no double-free).
- Pinned alloc failure falls back to legacy ``model.to("cuda")``.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Tuple

import torch
from torch import nn

from hetu_dit.cstrace import cst_print

logger = logging.getLogger(__name__)


# Align each tensor's slab offset to 256 bytes; matches typical HBM cache line
# and keeps unaligned-load micro-benchmarks honest. Cheap (~1 KB total padding
# for 1000 tensors) relative to a 22 GB slab.
_ALIGN = 256


def _align_up(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


def _collect_items(module: nn.Module):
    """Walk ``module.named_modules()`` and gather every leaf parameter and
    buffer that is currently on CPU. Returns (items, passthrough, total_bytes,
    n_aliases) where:

        items: list of (submod, name, kind, dtype, shape, offset, nbytes,
                        nbytes_aligned, alias_of)
            kind in {"param", "buffer"}; alias_of is the index into items of
            the master entry when this tensor shares storage, else None.
        passthrough: tensors already on the target device (skipped, returned
            so the caller can decide what to do).
        total_bytes: sum of nbytes_aligned over non-alias items.
        n_aliases: count of alias entries (for cstrace).
    """
    items = []
    passthrough = []
    seen_storage: Dict[int, int] = {}  # storage data_ptr -> items index
    cursor = 0
    n_aliases = 0

    for _, submod in module.named_modules():
        for kind, store in (("param", submod._parameters),
                             ("buffer", submod._buffers)):
            for name, t in list(store.items()):
                if t is None or t.numel() == 0:
                    continue
                if t.device.type == "cuda":
                    passthrough.append((submod, name, kind, t))
                    continue

                # Detect tied params via storage identity, not tensor identity
                # (two Parameter objects can wrap the same storage).
                storage_key = t.untyped_storage().data_ptr()
                if storage_key in seen_storage:
                    master_idx = seen_storage[storage_key]
                    master = items[master_idx]
                    items.append((submod, name, kind, t.dtype, tuple(t.shape),
                                  master[5], master[6], master[7], master_idx))
                    n_aliases += 1
                    continue

                nbytes = t.numel() * t.element_size()
                nbytes_aligned = _align_up(nbytes)
                items.append((submod, name, kind, t.dtype, tuple(t.shape),
                              cursor, nbytes, nbytes_aligned, None))
                seen_storage[storage_key] = len(items) - 1
                cursor += nbytes_aligned

    return items, passthrough, cursor, n_aliases


def bulk_pinned_to_cuda(module: nn.Module, device: str = "cuda") -> torch.Tensor:
    """BPS for a single ``nn.Module``.

    Returns the GPU slab tensor; the caller is responsible for keeping a
    reference (e.g. via ``pipeline._bps_gpu_slabs.append(slab)``) so the slab
    is not garbage-collected while parameter views still reference it.

    Raises ``RuntimeError`` only on pinned-memory exhaustion or CUDA OOM; the
    caller (``bulk_pinned_pipeline_to_cuda``) catches and falls back.
    """
    cst_print("bps_collect_start")
    items, passthrough, total_bytes, n_aliases = _collect_items(module)
    cst_print("bps_collect_done", n_items=len(items), n_aliases=n_aliases,
              n_passthrough=len(passthrough), total_bytes=total_bytes)

    if total_bytes == 0:
        # Module already entirely on GPU (or empty). Nothing to slab.
        return torch.empty(0, dtype=torch.uint8, device=device)

    # Step 1: pinned host slab
    cst_print("bps_pinned_alloc_start", bytes=total_bytes)
    pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    cst_print("bps_pinned_alloc_done")

    # Step 2: copy each tensor into pinned, then drop CPU refs
    cst_print("bps_pinned_fill_start")
    for (submod, name, kind, dtype, shape, offset, nbytes, _nbytes_aligned,
         alias_of) in items:
        if alias_of is not None:
            continue  # master already filled
        store = submod._parameters if kind == "param" else submod._buffers
        t = store[name]
        # Force contiguous bytes view (handles non-contig + dtype variety).
        src_bytes = t.contiguous().view(torch.uint8).reshape(-1)
        pinned[offset:offset + nbytes].copy_(src_bytes)
        # Drop the CPU tensor immediately so peak host RAM stays bounded
        # (CPU original + pinned + GPU = 3x peak otherwise; this drops it to 2x).
        store[name] = None
    cst_print("bps_pinned_fill_done")
    gc.collect()  # encourage CPU original release before GPU alloc

    # Step 3: GPU slab + single bulk async copy
    cst_print("bps_gpu_alloc_start", bytes=total_bytes)
    gpu_slab = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    cst_print("bps_gpu_alloc_done")

    cst_print("bps_bulk_copy_start")
    gpu_slab.copy_(pinned, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    cst_print("bps_bulk_copy_done")

    # Free pinned now that data is on GPU
    del pinned
    gc.collect()

    # Step 4: re-point parameters/buffers to typed views over the GPU slab
    cst_print("bps_repoint_start")
    for (submod, name, kind, dtype, shape, offset, nbytes, _nbytes_aligned,
         alias_of) in items:
        slice_u8 = gpu_slab.narrow(0, offset, nbytes)
        # Reinterpret bytes as the original dtype + shape. The slab is
        # contiguous-packed so view(dtype) is always legal regardless of the
        # original tensor's stride pattern.
        slice_typed = slice_u8.view(dtype).view(shape)
        store = submod._parameters if kind == "param" else submod._buffers
        if kind == "param":
            store[name] = nn.Parameter(slice_typed, requires_grad=False)
        else:
            store[name] = slice_typed
    cst_print("bps_repoint_done")

    return gpu_slab


def bulk_pinned_pipeline_to_cuda(pipeline, device: str, model_class) -> None:
    """BPS for a diffusers Pipeline.

    Walks ``pipeline.components`` (a dict of named sub-models, e.g. transformer,
    vae, text_encoder, text_encoder_2, text_encoder_3 for SD3), invokes
    ``bulk_pinned_to_cuda`` on each ``nn.Module``, and stores the resulting
    GPU slabs on the pipeline so they outlive function scope. Tokenizers and
    schedulers are not ``nn.Module`` and are skipped.

    Falls back to ``pipeline.to(device)`` on RuntimeError (pinned alloc OOM,
    CUDA OOM) — the caller can then re-issue the request without the BPS flag.
    """
    pipeline._bps_gpu_slabs: List[torch.Tensor] = []
    total_bytes = 0
    n_components = 0

    components = getattr(pipeline, "components", None) or {}
    try:
        for comp_name, comp in components.items():
            if not isinstance(comp, nn.Module):
                continue
            slab = bulk_pinned_to_cuda(comp, device)
            pipeline._bps_gpu_slabs.append(slab)
            total_bytes += slab.numel()
            n_components += 1
            logger.info(
                "BPS: %s -> %s (%d bytes)", comp_name, device, slab.numel()
            )
    except RuntimeError as exc:
        # Pinned alloc or CUDA OOM. Drop any partial slabs and fall back.
        logger.warning(
            "BPS aborted (%s); falling back to legacy pipeline.to(%s)",
            exc, device,
        )
        cst_print("bps_fallback_pinned_oom", reason=str(exc)[:120])
        pipeline._bps_gpu_slabs.clear()
        pipeline.to(device)
        return

    # Defeat any Accelerate / device-tracking hooks that still think pipeline
    # is on CPU. ``_execution_device`` is a diffusers-internal property used
    # by Pipeline.__call__ to decide where to put generated latents.
    if hasattr(pipeline, "_execution_device"):
        pipeline._execution_device = torch.device(device)
    if hasattr(pipeline, "hf_device_map"):
        pipeline.hf_device_map = None

    cst_print("bps_pipeline_done", n_components=n_components,
              total_bytes=total_bytes)
