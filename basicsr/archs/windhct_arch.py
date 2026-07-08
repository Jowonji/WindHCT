"""
WindHCT

Attributions:
- Coordinate Attention (CoordAtt): Hou et al., CVPR 2021.
  Paper: https://arxiv.org/abs/2103.02907
  Repo:  https://github.com/houqb/CoordAttention
- Swin Transformer (shifted window attention / relative position bias): Liu et al., ICCV 2021.
  Paper: https://arxiv.org/abs/2103.14030
  Repo:  https://github.com/microsoft/Swin-Transformer

Note:
- References are provided for attribution of ideas and implementation patterns.
- If any third-party code is directly reused, ensure license compliance and include corresponding license texts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# =============================================================================
# Stochastic Depth (DropPath)
# =============================================================================
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping the path.
        training: Whether currently in training mode.

    Returns:
        Tensor with stochastic depth applied.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Module wrapper for stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# =============================================================================
# MLP block used in Transformer-style layers
# =============================================================================
class Mlp(nn.Module):
    """Standard Transformer MLP block."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# =============================================================================
# Coordinate Attention (CoordAtt)
# =============================================================================
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    """
    Coordinate Attention (CoordAtt), CVPR 2021.

    Produces separate attention maps along height and width via pooled context,
    then reweights the input feature map.
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()

        x_h = self.pool_h(x)                      # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)          # (B, C, H+W, 1)

        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


# =============================================================================
# Swin-style relative position bias (window attention)
# =============================================================================
class RelativePositionBias(nn.Module):
    """
    Relative position bias for window attention, following Swin Transformer.

    Produces:
        rel_pos_bias: (1, num_heads, T, T) where T = window_size * window_size
    """
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.register_buffer("relative_position_index", self._create_position_index())

    def _create_position_index(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = coords.flatten(1)                                       # (2, T)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, T, T)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (T, T, 2)
        relative_coords += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1)
        return relative_coords.sum(-1)  # (T, T)

    def forward(self):
        T = self.window_size * self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(T, T, -1)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, T, T)


# =============================================================================
# Window partitioning / merging utilities (non-overlapping windows)
# =============================================================================
class WindowPartition:
    """
    Convert feature maps to/from non-overlapping window tokens.

    - partition: (B, C, H, W) -> (B*N, T, C)
    - merge:     (B*N, T, C) -> (B, C, H, W)
    """
    @staticmethod
    def partition(x, window_size):
        B, C, H, W = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[2:]

        x_unfold = F.unfold(x, kernel_size=window_size, stride=window_size)  # (B, C*T, N)
        N = x_unfold.shape[-1]
        x_windows = x_unfold.transpose(1, 2).reshape(B * N, window_size * window_size, C)
        return x_windows, N, (Hp, Wp)

    @staticmethod
    def merge(x_windows, window_size, padded_size, original_size, batch_size, channels):
        Hp, Wp = padded_size
        H, W = original_size
        N = x_windows.shape[0] // batch_size

        x_fold = x_windows.reshape(batch_size, N, channels * window_size * window_size).transpose(1, 2)
        x = F.fold(x_fold, output_size=(Hp, Wp), kernel_size=window_size, stride=window_size)

        # Defensive normalization.
        ones = torch.ones((batch_size, 1, Hp, Wp), device=x.device, dtype=x.dtype)
        divisor = F.fold(
            F.unfold(ones, kernel_size=window_size, stride=window_size),
            output_size=(Hp, Wp),
            kernel_size=window_size,
            stride=window_size
        )
        x = x / divisor.clamp(min=1e-6)

        return x[:, :, :H, :W]


# =============================================================================
# Shifted window attention mask (for Swin-style shifted windows)
# =============================================================================
class ShiftedWindowAttentionMask(nn.Module):
    """
    Create attention masks for shifted window attention.

    This mask prevents tokens from attending across window boundaries after a cyclic shift.
    """
    def __init__(self, window_size, shift_size):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self._cached_masks = {}

    @staticmethod
    def _partition_mask(mask, window_size):
        B, H, W, C = mask.shape
        nH = H // window_size
        nW = W // window_size
        mask = mask.view(B, nH, window_size, nW, window_size, C)
        windows = mask.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size, window_size, C)

    def _create_mask(self, H, W, device):
        if self.shift_size == 0:
            return None

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        Hp, Wp = H + pad_h, W + pad_w

        cache_key = f"{Hp}_{Wp}_{str(device)}"
        if cache_key in self._cached_masks:
            return self._cached_masks[cache_key]

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)

        s = self.shift_size
        h_slices = (slice(0, -s), slice(-s, None))
        w_slices = (slice(0, -s), slice(-s, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self._partition_mask(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        shift_attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0))
        shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask == 0, float(0.0))

        if len(self._cached_masks) > 10:
            self._cached_masks.clear()
        self._cached_masks[cache_key] = shift_attn_mask
        return shift_attn_mask

    def get_mask(self, H, W, device):
        return self._create_mask(H, W, device)


# =============================================================================
# Cyclic shift helper (Swin-style)
# =============================================================================
class CyclicShift:
    """Apply / reverse cyclic shift used by shifted window attention."""
    @staticmethod
    def apply_shift(x, shift_size):
        if shift_size > 0:
            return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        return x

    @staticmethod
    def reverse_shift(x, shift_size):
        if shift_size > 0:
            return torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x


# =============================================================================
# Window attention
# =============================================================================
class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.

    Supports:
      - Relative position bias (Swin-style)
      - Shifted-window attention mask (Swin-style)
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rel_pos_bias=None, shift_attn_mask=None):
        """
        Args:
            x: (BN, T, C), where T = window_size*window_size and BN = B*num_windows
            rel_pos_bias: (1, num_heads, T, T) relative position bias (optional)
            shift_attn_mask: (num_windows, T, T) shifted-window attention mask (optional)

        Returns:
            out: (BN, T, C)
        """
        BN, T, C = x.shape

        qkv = self.qkv(x).reshape(BN, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # (BN, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BN, H, T, T)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        if shift_attn_mask is not None:
            num_windows = shift_attn_mask.shape[0]
            B = BN // num_windows
            attn = attn.view(B, num_windows, self.num_heads, T, T)
            attn = attn + shift_attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(BN, self.num_heads, T, T)

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(BN, T, C)
        return self.proj(out)


# =============================================================================
# Context branch: window attention + optional coordinate attention
# =============================================================================
class ContextBranch(nn.Module):
    """
    Context branch:
      - Swin-style window attention (relative position bias + optional shifted window mask)
      - Optional coordinate attention after merging windows
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0,
                 drop_path_rate=0.1, use_coord_attn=True,
                 mlp_drop=0.0, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_coord_attn = use_coord_attn
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.window_attn = WindowAttention(dim, window_size, num_heads)
        self.rel_pos_bias = RelativePositionBias(window_size, num_heads)
        self.shift_mask = ShiftedWindowAttentionMask(window_size, shift_size)

        self.drop_path = DropPath(drop_path_rate)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=mlp_drop)

        if use_coord_attn:
            self.coord_attn = CoordAttention(dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) Cyclic shift for shifted window attention.
        x_shifted = CyclicShift.apply_shift(x, self.shift_size)

        # 2) Partition into non-overlapping windows.
        x_windows, _, padded_size = WindowPartition.partition(x_shifted, self.window_size)

        # 3) Pre-norm and window attention.
        x_norm = self.norm1(x_windows)
        rel_pos_bias = self.rel_pos_bias()
        shift_attn_mask = self.shift_mask.get_mask(H, W, x.device)

        attn_out = self.window_attn(
            x_norm,
            rel_pos_bias=rel_pos_bias,
            shift_attn_mask=shift_attn_mask
        )

        # 4) Residual + MLP.
        x_windows = x_windows + attn_out
        x_windows = x_windows + self.drop_path(self.mlp(self.norm2(x_windows)))

        # 5) Merge windows and reverse shift.
        out = WindowPartition.merge(x_windows, self.window_size, padded_size, (H, W), B, C)
        out = CyclicShift.reverse_shift(out, self.shift_size)

        # 6) Optional coordinate attention.
        if self.use_coord_attn:
            out = self.coord_attn(out)

        return out


# =============================================================================
# Local branch: local CNN feature extractor (large kernels)
# =============================================================================
class LocalBranch(nn.Module):
    """Local branch: local convolutional feature extraction with larger kernels."""
    def __init__(self, dim, k1=5, k2=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, k1, padding=k1 // 2),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, k2, padding=k2 // 2),
            nn.GroupNorm(8, dim),
        )

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# Context–Local fusion block
# =============================================================================
class ContextLocalFusionBlock(nn.Module):
    """
    Fuse context (window-attention) and local (CNN) branches using a learned mixing gate.

    Shift pattern alternates per block index (Swin-style):
        even -> no shift, odd -> half-window shift.
    """
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.0,
                 drop_prob=0.1, layer_idx=0,
                 use_coord_attn=True, debug=False,
                 fusion_mode='gate',
                 lb_k1=5, lb_k2=7):
        super().__init__()
        shift_size = 0 if (layer_idx % 2 == 0) else window_size // 2
        self.fusion_mode = fusion_mode

        self.context_branch = ContextBranch(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_prob,
            use_coord_attn=use_coord_attn,
            shift_size=shift_size,
        )
        self.local_branch = LocalBranch(dim, k1=lb_k1, k2=lb_k2)

        if fusion_mode == 'gate':
            self.gate_conv = nn.Conv2d(dim * 2, 1, 1)
        elif fusion_mode == 'proj_cat':
            self.proj_conv = nn.Conv2d(dim * 2, dim, 1)
        # 'sum' mode needs no extra modules

        self.drop_path = DropPath(drop_prob)

        self.debug = debug
        self._printed_gate_stats = False

    def forward(self, x):
        context_feat = self.context_branch(x)
        local_feat = self.local_branch(x)

        if self.fusion_mode == 'gate':
            gate = torch.sigmoid(self.gate_conv(torch.cat([context_feat, local_feat], dim=1)))

            if self.debug and (not self._printed_gate_stats):
                with torch.no_grad():
                    g = gate
                    print(
                        f"[FusionGate] mean={g.mean().item():.4f} std={g.std().item():.4f} "
                        f"min={g.min().item():.4f} max={g.max().item():.4f}"
                    )
                self._printed_gate_stats = True

            fused = gate * context_feat + (1.0 - gate) * local_feat
        elif self.fusion_mode == 'sum':
            fused = 0.5 * (context_feat + local_feat)
        elif self.fusion_mode == 'proj_cat':
            fused = self.proj_conv(torch.cat([context_feat, local_feat], dim=1))
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return x + self.drop_path(fused)


# =============================================================================
# Residual group
# =============================================================================
class ResidualGroup(nn.Module):
    """
    ResidualGroup:
      - Applies multiple ContextLocalFusionBlocks.
      - Applies a group-level residual fusion.
    """
    def __init__(self, dim, num_blocks, window_size, num_heads,
                 mlp_ratio=4.0, drop_prob=0.1,
                 use_coord_attn=True,
                 debug=False,
                 fusion_mode='gate',
                 lb_k1=5, lb_k2=7):
        super().__init__()
        self.debug = debug

        self.blocks = nn.ModuleList([
            ContextLocalFusionBlock(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                layer_idx=i,
                use_coord_attn=use_coord_attn,
                debug=debug,
                fusion_mode=fusion_mode,
                lb_k1=lb_k1,
                lb_k2=lb_k2,
            )
            for i in range(num_blocks)
        ])

        self.fusion = nn.Conv2d(dim, dim, 3, 1, 1)
        self.final_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)

        out = x + self.drop_path(self.fusion(out))
        out = self.final_conv(out)
        return out


# =============================================================================
# WindHCT (top-level)
# =============================================================================
@ARCH_REGISTRY.register()
class WindHCT(nn.Module):
    """
    WindHCT:
      - Head conv
      - Stack of ResidualGroups
      - Group feature aggregation + tail conv
      - Upsampling via PixelShuffle
      - Residual prediction added to bicubic-upsampled input
    """
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=64, num_group=6,
                 num_block=6, upscale=5, img_range=1., rgb_mean=(0,),
                 drop_prob=0.1, window_size=5, num_heads=4, mlp_ratio=4.0,
                 use_coord_attn=True, fusion_mode='gate',
                 lb_k1=5, lb_k2=7, debug=False):
        super().__init__()
        self.img_range = img_range
        self.register_buffer("mean", torch.Tensor(rgb_mean).view(1, num_in_ch, 1, 1))

        self.upsample_scale = upscale
        self.base_proj = nn.Identity() if num_in_ch == num_out_ch else nn.Conv2d(num_in_ch, num_out_ch, 1)

        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList([
            ResidualGroup(
                dim=num_feat,
                num_blocks=num_block,
                window_size=window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                use_coord_attn=use_coord_attn,
                debug=debug,
                fusion_mode=fusion_mode,
                lb_k1=lb_k1,
                lb_k2=lb_k2,
            )
            for _ in range(num_group)
        ])

        self.fusion_conv = nn.Conv2d(num_feat * num_group, num_feat, 1)
        self.body_tail = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x_head = self.head(x)

        feats = []
        out = x_head
        for g in self.body:
            out = g(out)
            feats.append(out)

        out = self.fusion_conv(torch.cat(feats, dim=1))
        out = self.body_tail(out) + x_head

        out = self.upsample(out)
        res = self.tail(out)

        base = F.interpolate(x, scale_factor=self.upsample_scale, mode="bicubic", align_corners=False)
        base = self.base_proj(base)

        return res + base
