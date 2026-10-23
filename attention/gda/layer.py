"""Gated delta attention (GDA) layer.

This is a standalone adaptation of the ``Qwen3_5GatedDeltaNet`` block from
``transformers`` (which in turn follows the FLA ``chunk_gated_delta_rule``
kernel).  The layer keeps the original parameter names so that Qwen3.5
checkpoints can be mapped onto it, but it no longer depends on the
``transformers`` package: projections, causal convolution, decay/gate
computation, the delta-rule kernel and the gated RMSNorm are all plain
PyTorch.

The block is a linear-attention layer: instead of a quadratic attention matrix
it keeps a per-head matrix state and applies the gated delta rule (see
:mod:`attention.gda.reference`).  A short depthwise causal convolution mixes
neighbouring tokens before the recurrence, and a SiLU output gate modulates the
normalized output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .torch import gated_delta_rule, recurrent_gated_delta_rule

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNetState",
    "GatedRMSNorm",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5RMSNormGated",
]

_ACTIVATIONS = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


@dataclass
class GatedDeltaNetState:
    """Incremental-decoding state of :class:`GatedDeltaNet`.

    Attributes:
        conv_state: Raw (pre-activation) inputs of the causal convolution for
            the last ``conv_kernel_size - 1`` positions, shape
            ``(batch, conv_dim, conv_kernel_size - 1)``.  It is empty when the
            kernel size is one.
        recurrent_state: Delta-rule state after the processed tokens, shape
            ``(batch, num_v_heads, head_k_dim, head_v_dim)``, float32.
    """

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class GatedRMSNorm(nn.Module):
    """RMSNorm followed by a SiLU output gate.

    ``out = weight * rms_norm(x) * silu(gate)``.  This is the normalization used
    by Qwen3.5's linear-attention blocks (``Qwen3_5RMSNormGated``); unlike
    :class:`attention.norm.RMSNorm`, the scale is a plain ``weight`` initialized
    to one and the gate is applied *after* the normalization.

    Args:
        dim: Size of the normalized (last) dimension.
        eps: Numerical stability constant.
        activation: Gate activation; ``"silu"`` (default) matches Qwen3.5.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unsupported activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        # Match the original implementation: scale in the input dtype, apply
        # the gate in float32, then cast back once.
        hidden_states = self.weight * hidden_states.to(input_dtype)
        gate = _ACTIVATIONS[self.activation](gate.float())
        return (hidden_states * gate).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
            f"activation={self.activation}"
        )


class GatedDeltaNet(nn.Module):
    """Gated delta attention block used by Qwen3.5 linear-attention layers.

    The forward pass is::

        qkv = in_proj_qkv(x)                    # key | key | value
        qkv = causal_conv1d(qkv)                # depthwise, kernel 4
        q, k, v = split(qkv)
        z = in_proj_z(x)                        # output gate
        beta = sigmoid(in_proj_b(x))
        g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
        o = gated_delta_rule(q, k, v, g, beta)  # linear attention
        o = gated_rmsnorm(o, z)
        y = out_proj(o)

    Args:
        hidden_size: Input and output hidden dimension.
        num_v_heads: Number of value (and state) heads.
        num_k_heads: Number of key/query heads.  Must divide
            ``num_v_heads``; query/key heads are repeated when the value side
            has more heads.
        head_k_dim: Key/query head dimension.
        head_v_dim: Value head dimension.
        conv_kernel_size: Depthwise causal convolution kernel size.
        rms_norm_eps: Epsilon of :class:`GatedRMSNorm`.
        activation: Activation used by the convolution and the output gate.
        use_qk_l2norm: L2-normalize query/key before the delta rule.  Qwen3.5
            enables this.
        scale: Optional query scale passed to the delta rule.  Defaults to
            ``head_k_dim ** -0.5``.

    Note:
        The layer processes one sequence at a time.  Pass ``use_cache=True`` to
        obtain a :class:`GatedDeltaNetState` and feed it back through
        ``past_state`` for token-by-token decoding; the cached path is
        numerically equivalent to a full-sequence forward.
    """

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        num_k_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        activation: str = "silu",
        use_qk_l2norm: bool = True,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_v_heads % num_k_heads != 0:
            raise ValueError(
                "num_v_heads must be divisible by num_k_heads, got "
                f"{num_v_heads} and {num_k_heads}"
            )
        if conv_kernel_size < 1:
            raise ValueError(
                f"conv_kernel_size must be positive, got {conv_kernel_size}"
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unsupported activation {activation!r}; "
                f"choose from {sorted(_ACTIVATIONS)}"
            )

        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.use_qk_l2norm = use_qk_l2norm
        self.scale = scale
        self.num_key_value_groups = num_v_heads // num_k_heads

        # key | key | value, convolved jointly with a depthwise convolution.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        # ``forward`` prepends the cached history and runs a valid (unpadded)
        # convolution, so the module itself needs no padding.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
        )

        # Discretization of the decay: g = -exp(A_log) * softplus(a + dt_bias).
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_v_heads).uniform_(0.01, 16))
        )

        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation=activation)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(
            hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_v_heads, bias=False)

    def _apply_padding_mask(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                "GatedDeltaNet expects a 2D padding mask of shape (batch, seq_len); "
                f"got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "attention_mask must match (batch, seq_len) = "
                f"{tuple(hidden_states.shape[:2])}, got "
                f"{tuple(attention_mask.shape)}"
            )
        mask = attention_mask
        if mask.dtype != torch.bool:
            # Accept both the hub's additive convention (0 = keep, negative =
            # drop) and a plain 0/1 multiplicative mask.
            if bool((mask < 0).any()):
                mask = mask == 0
            mask = mask.to(hidden_states.dtype)
        else:
            mask = mask.to(hidden_states.dtype)
        return hidden_states * mask.unsqueeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_state: GatedDeltaNetState | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, GatedDeltaNetState]:
        """Run the gated delta attention block.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` input.
            attention_mask: Optional 2D padding mask ``(batch, seq_len)``.
                Boolean with ``True`` = keep, a float 0/1 mask, or a 2D
                additive float mask (0 = keep, negative = drop).  Unlike the
                softmax layers, GDA is recurrent, so a per-query attention mask
                is not meaningful; padding positions are zeroed before the
                projections.
            past_state: State returned by a previous call, for incremental
                decoding.  ``None`` starts from a zero state.
            use_cache: Return the updated :class:`GatedDeltaNetState` together
                with the output.

        Returns:
            ``output`` of shape ``(batch, seq_len, hidden_size)``, or
            ``(output, state)`` when ``use_cache=True``.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape (batch, seq_len, hidden_size), "
                f"got {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len == 0:
            raise ValueError("seq_len must be positive")
        if attention_mask is not None:
            hidden_states = self._apply_padding_mask(hidden_states, attention_mask)

        # ---- Projections -------------------------------------------------
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # ---- Depthwise causal convolution --------------------------------
        # Prepend the cached (or zero) history so that a plain valid
        # convolution yields exactly ``seq_len`` outputs.  ``conv_state`` keeps
        # the raw inputs, not the activated outputs, so it can be reused by the
        # next chunk.
        if past_state is not None and past_state.conv_state is not None:
            conv_history = past_state.conv_state.to(mixed_qkv)
        else:
            conv_history = mixed_qkv.new_zeros(
                batch_size, self.conv_dim, self.conv_kernel_size - 1
            )
        conv_input = torch.cat([conv_history, mixed_qkv], dim=-1)
        conv_out = F.conv1d(
            conv_input.to(self.conv1d.weight.dtype),
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.conv_dim,
        )[..., -seq_len:]
        conv_out = _ACTIVATIONS[self.activation](conv_out).to(hidden_states.dtype)
        new_conv_state = (
            conv_input[..., -(self.conv_kernel_size - 1) :]
            if self.conv_kernel_size > 1
            else conv_input[..., :0]
        )

        mixed_qkv = conv_out.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(
            batch_size, seq_len, self.num_k_heads, self.head_k_dim
        )
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )

        # ---- Gates and decay ---------------------------------------------
        beta = b.sigmoid()
        # A_log may be fp16 when the model is loaded in half precision; compute
        # the decay in float32 to avoid -inf.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_key_value_groups > 1:
            query = query.repeat_interleave(self.num_key_value_groups, dim=2)
            key = key.repeat_interleave(self.num_key_value_groups, dim=2)

        # ---- Gated delta rule --------------------------------------------
        initial_state = (
            None if past_state is None else past_state.recurrent_state
        )
        # A single token has no intra-chunk work; the token-wise kernel is
        # both cheaper and the natural decoding path.
        use_recurrent = seq_len == 1
        kernel = recurrent_gated_delta_rule if use_recurrent else gated_delta_rule
        core_attn_out, new_recurrent_state = kernel(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            scale=self.scale,
        )

        # ---- Gated normalization and output projection -------------------
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(core_attn_out)

        if not use_cache:
            return output
        if new_recurrent_state is None:
            # ``output_final_state=True`` always returns a tensor, but keep a
            # defensive fallback so the state type is well-defined.
            new_recurrent_state = torch.zeros(
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        return output, GatedDeltaNetState(
            conv_state=new_conv_state,
            recurrent_state=new_recurrent_state,
        )


#: Names used by the original ``transformers`` module.
Qwen3_5GatedDeltaNet = GatedDeltaNet
Qwen3_5RMSNormGated = GatedRMSNorm
