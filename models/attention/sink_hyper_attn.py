import torch
import torch.nn as nn

from .hyper_attn import HyperAttention
from .utils import add_self_attentions, exact_attention


class SinkAwareHyperAttention(nn.Module):
    """Sink-aware single-layer attention.

    Inputs use shape [batch, heads, seq_len, dim]. The attention is decomposed into:
    1) Exact attention from all queries to sink keys/values [:sink_size]
    2) HyperAttention on the causal tail subproblem [sink_size:]
    The two terms are merged with add_self_attentions using their LSEs.
    """

    def __init__(
        self,
        input_dim: int = 64,
        sink_size: int = 32,
        lsh_num_projs: int = 7,
        block_size: int = 256,
        sample_size: int = 256,
        min_seq_len: int = 0,
        cuda: bool = False,
    ):
        super().__init__()
        self.sink_size = sink_size
        self.hyper_attn = HyperAttention(
            input_dim=input_dim,
            lsh_num_projs=lsh_num_projs,
            block_size=block_size,
            sample_size=sample_size,
            min_seq_len=min_seq_len,
            cuda=cuda,
        )

    def forward(self, query, key, value, scale=None, causal=True, return_lse=False):
        if not causal:
            raise ValueError("SinkAwareHyperAttention currently supports causal=True only.")

        batch_size, n_heads, seq_len, dim = query.shape
        sink_size = min(self.sink_size, seq_len)
        scale = dim ** (-0.5) if scale is None else scale

        # Exact contribution over sink keys/values for all queries.
        prefix_attn, prefix_lse = exact_attention(
            query, key[:, :, :sink_size, :], value[:, :, :sink_size, :], softmax_scale=scale, causal=True
        )

        # If there is no tail, the exact sink part is already the full attention.
        if sink_size >= seq_len:
            return (prefix_attn, prefix_lse) if return_lse else prefix_attn

        # HyperAttention on the tail subproblem only.
        tail_attn, tail_lse = self.hyper_attn(
            query[:, :, sink_size:, :],
            key[:, :, sink_size:, :],
            value[:, :, sink_size:, :],
            scale=scale,
            causal=True,
            return_lse=True,
        )

        # Pad tail outputs back to full sequence length.
        padded_tail_attn = torch.zeros_like(prefix_attn)
        padded_tail_attn[:, :, sink_size:, :] = tail_attn

        neg_inf = torch.full_like(prefix_lse[:, :, :sink_size, :], torch.finfo(prefix_lse.dtype).min)
        padded_tail_lse = torch.cat([neg_inf, tail_lse], dim=2)

        # Merge in log-space using provided helper (no naive averaging).
        attn, lse = add_self_attentions(prefix_attn, prefix_lse, padded_tail_attn, padded_tail_lse)
        return (attn, lse) if return_lse else attn
