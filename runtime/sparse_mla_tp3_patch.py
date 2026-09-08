"""Exact sparse-MLA dispatch support for TP3.

The GLM TP3 runtime pads 64 semantic query heads to 66 and assigns 22 heads
to each rank.  FlashInfer's SM120 sparse kernels are instantiated for selected
head counts (including 32, but not 22) on both the short decode and longer
prefill dispatches.  This patch pads the independent local head axis from 22
to 32 for ordinary TP3 calls and drops the ten synthetic outputs.  With
decode context parallelism (DCP), vLLM first gathers the query heads across
the DCP group; that gathered axis is passed through unchanged, while sparse
index rows are filtered to the local KV shard and the FlashInfer LSE is
returned for the DCP reducer.  No semantic query, KV, index, checkpoint, or
model value is changed.

An opt-in one-shot shadow gate compares the padded 32-head decode result with
the supported 32-head generic kernel by adding independent dummy query rows
until the generic (>64 token) path is selected.  Only the original rows and
the 22 semantic heads are scored.
"""

from __future__ import annotations

import atexit
from functools import wraps
import math
import os
import threading

import torch


LOCAL_TP3_HEADS = 22
KERNEL_HEADS = 32
DECODE_MAX_TOKENS = 64
GENERIC_MIN_TOKENS = 65

_LOCK = threading.RLock()
_CALLS = 0
_SHORT_CALLS = 0
_PREFILL_CALLS = 0
_SHADOW_DONE = False
_SHADOW_RELATIVE_L2: float | None = None
_SHADOW_MAX_ABS: float | None = None


def _enabled() -> bool:
    return os.environ.get("VLLM_GLM53_EXL3_TP3_FULL_MODEL", "0") == "1"


def _pad_head_axis(tensor: torch.Tensor, target: int = KERNEL_HEADS) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"expected [tokens, heads, width], got {tuple(tensor.shape)}")
    heads = int(tensor.shape[1])
    if heads == target:
        return tensor
    if heads > target:
        raise ValueError(f"refusing to truncate {heads} heads to {target}")
    shape = (tensor.shape[0], target - heads, tensor.shape[2])
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)


def _pad_token_axis(tensor: torch.Tensor, target: int, value: int | float = 0):
    if tensor.shape[0] >= target:
        return tensor
    shape = (target - tensor.shape[0], *tensor.shape[1:])
    fill = tensor.new_full(shape, value)
    return torch.cat((tensor, fill), dim=0)


def _requires_head_adapter(module_heads: int, query: torch.Tensor) -> bool:
    """Return whether this query uses the unsupported TP3-local head count."""

    return int(module_heads) == LOCAL_TP3_HEADS and query.ndim == 3


def _normalize_lse(
    lse: torch.Tensor,
    num_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """Normalize FlashInfer's SM120 LSE variants for the DCP combiner."""

    if lse.dim() == 3:
        if lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        elif lse.shape[1] == 1:
            lse = lse.squeeze(1)
        elif lse.shape[0] * lse.shape[1] == num_tokens:
            lse = lse.reshape(num_tokens, lse.shape[-1])
    if lse.shape != (num_tokens, num_heads):
        raise RuntimeError(
            "Unexpected FlashInfer SM120 sparse MLA LSE shape: "
            f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
        )
    return lse


def _emit_stats() -> None:
    if not _enabled() or _CALLS == 0:
        return
    print(
        "GLM53_TP3_SPARSE_MLA_PAD_STATS "
        f"calls={_CALLS} short_calls={_SHORT_CALLS} "
        f"prefill_calls={_PREFILL_CALLS} shadow_done={int(_SHADOW_DONE)} "
        f"shadow_relative_l2={_SHADOW_RELATIVE_L2} "
        f"shadow_max_abs={_SHADOW_MAX_ABS}",
        flush=True,
    )


def apply_patches() -> None:
    """Install the 22-to-32 sparse-MLA head adapter once."""

    if not _enabled() or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True

    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _get_workspace_buffer,
        FlashInferMLASparseMetadata,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index,
        triton_filter_and_convert_dcp_index,
    )

    # The installed SM120 implementation predates DCP support even though the
    # FlashInfer 0.6.17 kernel already accepts return_lse=True.  Set the
    # capability before any attention implementation is instantiated so its
    # base __new__ derives need_to_return_lse_for_decode correctly.
    FlashInferMLASparseSM120Impl.can_return_lse_for_decode = True
    FlashInferMLASparseSM120Impl.lse_base_on_e = False
    # The image's sparse metadata predates the generic MLA DCP combiner, which
    # probes ``attn_metadata.decode`` before falling back to the existing flat
    # ``seq_lens`` field.  Supply the missing optional attribute without
    # changing the dataclass layout or scheduler semantics.
    if not hasattr(FlashInferMLASparseMetadata, "decode"):
        FlashInferMLASparseMetadata.decode = None

    # vLLM's generic DCP reducer uses ``tl.arange(0, N)`` for the gathered
    # rank dimension.  Triton requires that compile-time range to be a
    # power-of-two, while this homelab topology has N=3.  Pad only the LSE
    # gather with -inf so the extra lane contributes exactly zero to the
    # log-sum-exp; the attention output and the reduce-scatter head geometry
    # remain unchanged.
    from vllm.v1.attention.ops import common as dcp_common

    original_correct_attn_out = dcp_common.correct_attn_out
    if not getattr(original_correct_attn_out, "_glm53_dcp_lse_pad", False):

        @wraps(original_correct_attn_out)
        def correct_attn_out(*args, **kwargs):
            if len(args) >= 2:
                lses = args[1]
                arg_prefix = args[:1]
                arg_suffix = args[2:]
            else:
                lses = kwargs.get("lses")
                arg_prefix = ()
                arg_suffix = ()
            if isinstance(lses, torch.Tensor) and lses.ndim >= 3:
                count = int(lses.shape[0])
                rounded = 1 << max(count - 1, 0).bit_length()
                if rounded != count:
                    pad_shape = (rounded - count, *lses.shape[1:])
                    padding = lses.new_full(pad_shape, float("-inf"))
                    lses = torch.cat((lses, padding), dim=0)
                    if len(args) >= 2:
                        args = (*arg_prefix, lses, *arg_suffix)
                    else:
                        kwargs = dict(kwargs)
                        kwargs["lses"] = lses
                    print(
                        "GLM53_DCP_LSE_ARANGE_PAD_APPLIED "
                        f"ranks={count}->{rounded}",
                        flush=True,
                    )
            return original_correct_attn_out(*args, **kwargs)

        correct_attn_out._glm53_dcp_lse_pad = True
        dcp_common.correct_attn_out = correct_attn_out

    original = FlashInferMLASparseSM120Impl.forward_mqa

    @wraps(original)
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        global _CALLS, _SHORT_CALLS, _PREFILL_CALLS
        global _SHADOW_DONE, _SHADOW_RELATIVE_L2, _SHADOW_MAX_ABS

        q_head_source = q[0] if isinstance(q, tuple) else q
        dcp_enabled = int(getattr(self, "dcp_world_size", 1)) > 1
        if not dcp_enabled and not _requires_head_adapter(self.num_heads, q_head_source):
            return original(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if not dcp_enabled and q_head_source.shape[1] != LOCAL_TP3_HEADS:
            raise RuntimeError(
                "TP3 sparse-MLA local head geometry differs: "
                f"module={self.num_heads} query={tuple(q_head_source.shape)}"
            )

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.rope_pad:
            q = torch.nn.functional.pad(q, (0, self.rope_pad))

        num_actual_toks = int(q.shape[0])
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if dcp_enabled:
            topk_indices_physical, topk_lengths = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical, topk_lengths = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        sparse_topk_capacity = int(topk_indices_physical.shape[1])
        empty_rows = topk_lengths == 0
        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(
            empty_rows, 0
        )
        topk_lengths = topk_lengths.clamp(min=1)

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        def run(query, indices, lengths, heads):
            output = query.new_empty(
                (query.shape[0], heads, self.kv_lora_rank), dtype=query.dtype
            )
            result = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                query=query.unsqueeze(1),
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
                workspace_buffer=self._workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.kernel_qk_rope_head_dim,
                block_tables=indices.unsqueeze(1),
                seq_lens=lengths,
                max_seq_len=sparse_topk_capacity,
                out=output.unsqueeze(1),
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                sparse_mla_top_k=sparse_topk_capacity,
                kv_scale_format=self.kv_scale_format,
                return_lse=bool(getattr(self, "need_to_return_lse_for_decode", False)),
            )
            if isinstance(result, tuple):
                result_out, result_lse = result
                return result_out.squeeze(1), result_lse
            return result.squeeze(1), None

        if dcp_enabled:
            # query_gather has already made the complete physical head axis
            # visible to this rank (66 for GLM TP3).  The SM120 decode
            # dispatch, however, is instantiated only for head counts
            # 8/16/32/64/128.  Run the first 64 heads and the remaining two
            # heads in an 8-head padded call, then concatenate before the DCP
            # reducer.  This preserves the required 66-head output while
            # keeping each kernel invocation on a supported shape.
            if int(q.shape[1]) == 66:
                first_out, first_lse = run(
                    q[:, :64, :].contiguous(),
                    topk_indices_physical,
                    topk_lengths,
                    64,
                )
                tail_q = _pad_head_axis(q[:, 64:, :].contiguous(), 8)
                tail_out, tail_lse = run(
                    tail_q,
                    topk_indices_physical,
                    topk_lengths,
                    8,
                )
                run_out = torch.cat((first_out, tail_out[:, :2, :]), dim=1)
                if first_lse is None or tail_lse is None:
                    raise RuntimeError("SM120 DCP head chunks did not return LSE")
                run_lse = torch.cat((first_lse, tail_lse[:, :2]), dim=1)
            else:
                run_q = q
                run_heads = int(q.shape[1])
                run_out, run_lse = run(
                    run_q, topk_indices_physical, topk_lengths, run_heads
                )
        else:
            run_q = _pad_head_axis(q)
            run_heads = KERNEL_HEADS
            run_out, run_lse = run(
                run_q, topk_indices_physical, topk_lengths, run_heads
            )
        out = run_out if dcp_enabled else run_out[:, :LOCAL_TP3_HEADS, :]
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)

        with _LOCK:
            _CALLS += 1
            if num_actual_toks <= DECODE_MAX_TOKENS:
                _SHORT_CALLS += 1
            else:
                _PREFILL_CALLS += 1
            do_shadow = (
                not dcp_enabled
                and not _SHADOW_DONE
                and num_actual_toks <= DECODE_MAX_TOKENS
                and os.environ.get("GLM53_TP3_SPARSE_MLA_SHADOW_COMPARE", "0") == "1"
            )
            if do_shadow:
                # The generic SM120 kernel requires >64 query rows and also
                # has a finite head-instantiation set. Use the same supported
                # 32-head execution geometry as decode; dummy rows and heads
                # are independent and cannot affect the semantic submatrix.
                reference_q = _pad_token_axis(run_q, GENERIC_MIN_TOKENS, 0)
                reference_indices = _pad_token_axis(
                    topk_indices_physical, GENERIC_MIN_TOKENS, 0
                )
                reference_lengths = _pad_token_axis(
                    topk_lengths, GENERIC_MIN_TOKENS, 1
                )
                reference_full, _ = run(
                    reference_q,
                    reference_indices,
                    reference_lengths,
                    KERNEL_HEADS,
                )
                reference = reference_full[:num_actual_toks, :LOCAL_TP3_HEADS]
                reference.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
                delta = out.float() - reference.float()
                denominator = max(float(reference.float().norm().item()), 1e-30)
                relative = float(delta.norm().item()) / denominator
                maximum = float(delta.abs().max().item())
                if not math.isfinite(relative) or relative > 0.02:
                    raise RuntimeError(
                        "TP3 sparse-MLA padded-head shadow parity failed: "
                        f"relative_l2={relative} max_abs={maximum}"
                    )
                _SHADOW_RELATIVE_L2 = relative
                _SHADOW_MAX_ABS = maximum
                _SHADOW_DONE = True
                print(
                    "GLM53_TP3_SPARSE_MLA_SHADOW_OK "
                    f"tokens={num_actual_toks} heads=22->32 "
                    f"reference_tokens={GENERIC_MIN_TOKENS} "
                    f"relative_l2={relative:.10g} max_abs={maximum:.10g}",
                    flush=True,
                )
        if dcp_enabled:
            if run_lse is None:
                raise RuntimeError("SM120 DCP path did not return FlashInfer LSE")
            run_lse = _normalize_lse(run_lse, out.shape[0], out.shape[1])
            run_lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
            return out, run_lse
        return out, None

    FlashInferMLASparseSM120Impl.forward_mqa = forward_mqa
    atexit.register(_emit_stats)
    print(
        "GLM53_TP3_SPARSE_MLA_HEAD_ADAPTER_APPLIED "
        "tokens=all local_heads=22 kernel_heads=32 dcp_lse=1",
        flush=True,
    )


__all__ = [
    "DECODE_MAX_TOKENS",
    "GENERIC_MIN_TOKENS",
    "KERNEL_HEADS",
    "LOCAL_TP3_HEADS",
    "_pad_head_axis",
    "_pad_token_axis",
    "_requires_head_adapter",
    "apply_patches",
]
