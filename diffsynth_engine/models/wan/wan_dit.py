import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from einops import rearrange

from diffsynth_engine.models.base import StateDictConverter, PreTrainedModel
from diffsynth_engine.models.utils import no_init_weights
from diffsynth_engine.utils.constants import (
    WAN_DIT_1_3B_T2V_CONFIG_FILE,
    WAN_DIT_14B_I2V_CONFIG_FILE,
    WAN_DIT_14B_T2V_CONFIG_FILE,
)
from diffsynth_engine.utils.fp8_linear import fp8_inference


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    q, k, v = (rearrange(t, "b s (n d) -> b n s d ", n=num_heads) for t in (q, k, v))

    from sageattention import sageattn

    x = sageattn(q, k, v)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-5,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.norm(x.float()).to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.k = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.v = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.o = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.norm_k = RMSNorm(dim, eps=eps, device=device, dtype=dtype)

    def get_feta_scores(self, query, key, num_heads, weight, num_frames):
        img_q, img_k = query, key

        # After rope_apply, tensors have shape [B, S, flattened_dimension]
        B, S, flattened_dim = img_q.shape

        # We need to reshape back to [B, S, N, C]
        N = num_heads  # Use the number of heads from the class
        C = flattened_dim // N  # Calculate head dimension

        # Calculate spatial dimension
        spatial_dim = S // num_frames

        # Reshape to 4D tensors
        img_q = img_q.reshape(B, S, N, C)
        img_k = img_k.reshape(B, S, N, C)

        # Add time dimension between spatial and head dims
        query_image = img_q.reshape(B, spatial_dim, num_frames, N, C)
        key_image = img_k.reshape(B, spatial_dim, num_frames, N, C)

        # Expand time dimension
        query_image = query_image.expand(-1, -1, num_frames, -1, -1)  # [B, spatial_dim, T, N, C]
        key_image = key_image.expand(-1, -1, num_frames, -1, -1)      # [B, spatial_dim, T, N, C]

        # Reshape to match feta_score input format: [(B spatial_dim) N T C]
        query_image = rearrange(query_image, "b s t n c -> (b s) n t c")
        key_image = rearrange(key_image, "b s t n c -> (b s) n t c")

        return self.feta_score(query_image, key_image, C, weight, num_frames)

    @torch.compiler.disable()
    def feta_score(self, query_image, key_image, head_dim, weight, num_frames):
        scale = head_dim**-0.5
        query_image = query_image * scale
        attn_temp = query_image @ key_image.transpose(-2, -1)
        attn_temp = attn_temp.to(torch.float32)
        attn_temp = attn_temp.softmax(dim=-1)

        # Reshape to [batch_size * num_tokens, num_frames, num_frames]
        attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

        # Create a mask for diagonal elements
        diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

        # Zero out diagonal elements
        attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

        # Calculate mean for each token's attention matrix
        # Number of off-diagonal elements per matrix is n*n - n
        num_off_diag = num_frames * num_frames - num_frames
        mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

        enhance_scores = mean_scores.mean() * (num_frames + weight)
        enhance_scores = enhance_scores.clamp(min=1)
        return enhance_scores

    def forward(self, x, freqs, num_frames):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        num_heads = q.shape[2] // self.head_dim
        q = rope_apply(q, freqs, num_heads)
        k = rope_apply(k, freqs, num_heads)
        # feta_scores = self.get_feta_scores(q, k, num_heads, 2.0, (num_frames - 1) // 4 + 1)  # WARNING: Don't forget to modify in case of FLF2V-14B
        x = attention(q=q, k=k, v=v, num_heads=num_heads)
        # x *= feta_scores
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.k = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.v = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.o = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.norm_k = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim, device=device, dtype=dtype)
            self.v_img = nn.Linear(dim, dim, device=device, dtype=dtype)
            self.norm_k_img = RMSNorm(dim, eps=eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        num_heads = q.shape[2] // self.head_dim
        x = attention(q, k, v, num_heads=num_heads)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = attention(q, k_img, v_img, num_heads=num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps, device=device, dtype=dtype)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input, device=device, dtype=dtype
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(dim, eps=eps, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim, device=device, dtype=dtype),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim, device=device, dtype=dtype) / dim**0.5)

    def forward(self, x, context, t_mod, freqs, num_frames):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs, num_frames)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim, device=device, dtype=dtype),
            nn.Linear(in_dim, in_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, device=device, dtype=dtype),
            nn.LayerNorm(out_dim, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size), device=device, dtype=dtype)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim, device=device, dtype=dtype) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation + t_mod).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanDiTStateDictConverter(StateDictConverter):
    def convert(self, state_dict):
        return state_dict


class WanDiT(PreTrainedModel):
    converter = WanDiTStateDictConverter()

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, device=device, dtype=dtype),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps, device=device, dtype=dtype)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, device=device, dtype=dtype)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)  # b c f h w -> b 4c f h/2 w/2
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        num_frames: int,
        clip_feature: Optional[torch.Tensor] = None,  # clip_vision_encoder(img)
        y: Optional[torch.Tensor] = None,  # vae_encoder(img)
        slg_layers: Optional[list[int]] = [],
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)  # (b, s1 + s2, d)
        x, (f, h, w) = self.patchify(x)
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        # https://github.com/ali-vilab/TeaCache
        modulated_input = t_mod if self.use_ref_steps else t
        if self.cnt % 2 == 0:  # Even -> Conditional
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_input - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_input.clone()

        else:  # Odd -> Unconditional
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_input - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_input.clone()

        if self.is_even:
            if not should_calc_even:
                x += self.previous_residual_even
            else:
                ori_x = x.clone()

                with fp8_inference():
                    for block_idx, block in enumerate(self.blocks):
                        if block_idx in slg_layers:
                            continue
                        x = block(x, context, t_mod, freqs, num_frames)
                self.previous_residual_even = x - ori_x
        else:
            if not should_calc_odd:
                x += self.previous_residual_odd
            else:
                ori_x = x.clone()

                with fp8_inference():
                    for block_idx, block in enumerate(self.blocks):
                        if block_idx in slg_layers:
                            continue
                        x = block(x, context, t_mod, freqs, num_frames)
                self.previous_residual_odd = x - ori_x
        #

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))

        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0

        return x

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
        num_inference_steps: int,
        teacache_thresh: float,
        model_type: str = "1.3b-t2v",
    ):
        if model_type == "1.3b-t2v":
            config = json.load(open(WAN_DIT_1_3B_T2V_CONFIG_FILE, "r"))
        elif model_type == "14b-t2v":
            config = json.load(open(WAN_DIT_14B_T2V_CONFIG_FILE, "r"))
        elif model_type == "14b-i2v":
            config = json.load(open(WAN_DIT_14B_I2V_CONFIG_FILE, "r"))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        with no_init_weights():
            model = torch.nn.utils.skip_init(cls, **config, device=device, dtype=dtype)
        model.load_state_dict(state_dict, assign=True, strict=False)
        model.to(device=device, dtype=dtype, non_blocking=True)

        # Initialize TeaCache
        model.__class__.cnt = 0
        model.__class__.num_steps = num_inference_steps * 2
        model.__class__.teacache_thresh = teacache_thresh
        model.__class__.accumulated_rel_l1_distance_even = 0
        model.__class__.accumulated_rel_l1_distance_odd = 0
        model.__class__.previous_e0_even = None
        model.__class__.previous_e0_odd = None
        model.__class__.previous_residual_even = None
        model.__class__.previous_residual_odd = None
        model.__class__.use_ref_steps = True
        if model_type == "14b-t2v":
            model.__class__.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]  # 480P
        elif model_type == "14b-i2v":
            model.__class__.coefficients = [ 2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]  # 480P
        model.__class__.ret_steps = 5 * 2
        model.__class__.cutoff_steps = model.__class__.num_steps

        return model

    def get_tp_plan(self):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            SequenceParallel,
            PrepareModuleOutput,
        )
        from torch.distributed.tensor import Replicate, Shard

        tp_plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_embedding.0": ColwiseParallel(),
            "time_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
        }
        for idx in range(len(self.blocks)):
            tp_plan.update(
                {
                    f"blocks.{idx}.norm1": SequenceParallel(use_local_output=True),
                    f"blocks.{idx}.norm2": SequenceParallel(use_local_output=True),
                    f"blocks.{idx}.norm3": SequenceParallel(use_local_output=True),
                    f"blocks.{idx}.ffn.0": ColwiseParallel(),
                    f"blocks.{idx}.ffn.2": RowwiseParallel(),
                    f"blocks.{idx}.self_attn.q": ColwiseParallel(output_layouts=Replicate()),
                    f"blocks.{idx}.self_attn.k": ColwiseParallel(output_layouts=Replicate()),
                    f"blocks.{idx}.self_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.self_attn.o": RowwiseParallel(),
                    f"blocks.{idx}.self_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Replicate(),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.self_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Replicate(),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.q": ColwiseParallel(output_layouts=Replicate()),
                    f"blocks.{idx}.cross_attn.k": ColwiseParallel(output_layouts=Replicate()),
                    f"blocks.{idx}.cross_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.o": RowwiseParallel(),
                    f"blocks.{idx}.cross_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Replicate(),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Replicate(),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.k_img": ColwiseParallel(output_layouts=Replicate()),
                    f"blocks.{idx}.cross_attn.v_img": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.norm_k_img": PrepareModuleOutput(
                        output_layouts=Replicate(),
                        desired_output_layouts=Shard(-1),
                    ),
                }
            )
        return tp_plan
