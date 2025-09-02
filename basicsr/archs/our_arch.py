import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# ------------------------------------
# DropPath: Stochastic Depth 구현 (Residual 연결을 확률적으로 Drop)
# ------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ------------------------------------
# Sobel 필터 초기화 함수
# ------------------------------------
def init_sobel_weights(conv_layer):
    sobel_kernel = torch.tensor([[[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]])
    C = conv_layer.in_channels
    weight = sobel_kernel.repeat(C, 1, 1, 1)
    with torch.no_grad():
        conv_layer.weight.copy_(weight)

# ------------------------------------
# Window 파티션 및 복원 함수들 (Shifted Window용)
# ------------------------------------
def window_partition(x, window_size):
    """
    이미지를 window로 분할
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Window를 다시 이미지로 복원
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H: int
        W: int
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# ------------------------------------
# window_unfold: 입력 feature를 패치 단위로 분할
# ------------------------------------
def window_unfold(x, window_size, stride):
    B, C, H, W = x.shape
    unfold = F.unfold(x, kernel_size=window_size, stride=stride)
    N = (H - window_size) // stride + 1
    M = (W - window_size) // stride + 1
    return unfold.transpose(1, 2).reshape(B * N * M, C, window_size, window_size), N, M

# ------------------------------------
# FrequencyAwareSpatialAttention (FSA) - stride 제거
# ------------------------------------
class FrequencyAwareSpatialAttention(nn.Module):
    def __init__(self, window_size=17, channel=64, use_learnable_mask=True, debug=False):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.use_learnable_mask = use_learnable_mask
        self.debug = debug

        # Learnable depthwise Sobel filter
        self.freq_filter = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        init_sobel_weights(self.freq_filter)

        if use_learnable_mask:
            self.mask_generator = nn.Sequential(
                nn.Conv2d(channel, 16, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))  # [B, C, H_pad, W_pad]
        Hp, Wp = x.shape[2:]

        # 1. Patch unfolding (stride = window_size for non-overlapping)
        x_unfold = F.unfold(x, kernel_size=self.window_size, stride=self.window_size)  # [B, C*K*K, N]
        N = x_unfold.shape[-1]
        x_windows = x_unfold.transpose(1, 2).reshape(B * N, C, self.window_size, self.window_size)

        # 2. Frequency filtering
        freq_feat = self.freq_filter(x_windows)  # [B*N, C, K, K]

        # 3. Mask generation
        if self.use_learnable_mask:
            mask = self.mask_generator(freq_feat)  # [B*N, 1, K, K]
            mask = torch.clamp(mask, min=0.05, max=0.95)
        else:
            avg_feat = freq_feat.mean(dim=1, keepdim=True)
            mask = self.sigmoid(avg_feat)

        # 4. Fold back to full map
        mask = mask.view(B, N, -1).transpose(1, 2)  # [B, K*K, N]
        attn_mask = F.fold(mask, output_size=(Hp, Wp), kernel_size=self.window_size, stride=self.window_size)

        # 5. Normalize (no overlap for stride=window_size, but keep for robustness)
        ones = torch.ones((B, 1, Hp, Wp), device=x.device)
        divisor = F.fold(F.unfold(ones, self.window_size, stride=self.window_size),
                         output_size=(Hp, Wp), kernel_size=self.window_size, stride=self.window_size)
        attn_mask = attn_mask / divisor.clamp(min=1e-8)
        attn_mask = torch.nan_to_num(attn_mask, nan=0.0)

        # 6. Crop back to original size
        attn_mask = attn_mask[:, :, :H, :W]

        if self.debug and torch.isnan(attn_mask).any():
            print("❗ NaN detected in FSA mask")

        return attn_mask

class CoordAttention(nn.Module):
    """Coordinate Attention 모듈"""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()

        x_h = self.pool_h(x)               # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W]

        y = torch.cat([x_h, x_w], dim=2)   # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


# ------------------------------------
# SwinStyleBlockWithFSA - Shifted Window 지원, stride 제거
# ------------------------------------
class SwinStyleBlockWithFSA(nn.Module):
    def __init__(self, dim, window_size=17, num_heads=4, use_fsa=True, use_coord_attn=True,
                 debug=False, mlp_drop=0., shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_fsa = use_fsa
        self.use_coord_attn = use_coord_attn
        self.debug = debug
        self.scale = self.head_dim ** -0.5
        self.shift_size = shift_size

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_prob=0.1)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, drop=mlp_drop)

        if use_fsa:
            self.fsa = FrequencyAwareSpatialAttention(
                window_size=window_size, channel=dim, use_learnable_mask=True
            )

        if use_coord_attn:
            self.coord_attn = CoordAttention(dim)

        # Relative Position Bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # Create relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [T, T]

        self.register_buffer("relative_position_index", relative_position_index)

        # Shifted Window Attention을 위한 마스크 생성
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._create_mask())
        else:
            self.attn_mask = None

    def _create_mask(self):
        """Shifted Window Attention을 위한 마스크 생성"""
        H, W = self.window_size, self.window_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.shift_size),
                   slice(-self.shift_size, -self.shift_size//2),
                   slice(-self.shift_size//2, None))
        w_slices = (slice(0, -self.shift_size),
                   slice(-self.shift_size, -self.shift_size//2),
                   slice(-self.shift_size//2, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, fsa_mask=None):
        B, C, H, W = x.shape
        shortcut = x

        # Apply cyclic shift for Shifted Window Attention
        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            x_shifted = x

        # Padding for window partitioning
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x_shifted, (0, pad_w, 0, pad_h))
        Hp, Wp = x_padded.shape[2:]

        # Window unfolding (non-overlapping: stride = window_size)
        x_unfold = F.unfold(x_padded, kernel_size=self.window_size, stride=self.window_size)
        N = x_unfold.shape[-1]
        x_unfold = x_unfold.transpose(1, 2).reshape(B * N, self.window_size * self.window_size, C)

        # Apply FSA mask if enabled
        if self.use_fsa:
            # 2) no_grad() 제거
            if fsa_mask is None:
                fsa_mask = self.fsa(shortcut.detach())  # 입력만 detach (선택)
            fsa_mask = F.interpolate(fsa_mask, size=(Hp, Wp), mode='bilinear', align_corners=False)
            fsa_mask = torch.clamp(fsa_mask, 0.0, 1.0)
            fsa_mask = torch.nan_to_num(fsa_mask, nan=0.0)

            mask_unfold = F.unfold(fsa_mask, kernel_size=self.window_size, stride=self.window_size)
            mask_unfold = mask_unfold.transpose(1, 2).reshape(B * N, self.window_size * self.window_size, 1)
        else:
            mask_unfold = 1.0

        # Self-Attention computation
        x_norm = self.norm1(x_unfold)
        qkv = self.qkv(x_norm).reshape(B * N, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # [BN, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply FSA mask to queries
        q = q * mask_unfold.unsqueeze(1)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )  # [T, T, H]
        attn += bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, T, T]

        # Apply Shifted Window mask if shifted
        if self.shift_size > 0 and self.attn_mask is not None:
            attn = attn + self.attn_mask.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(B * N, -1, self.dim)
        out = self.proj(out)

        # Residual connection and MLP
        out = x_unfold + out
        out = out + self.drop_path(self.mlp(self.norm2(out)))

        # Fold back to spatial dimensions
        out = out.reshape(B, N, C * self.window_size * self.window_size).transpose(1, 2)
        out = F.fold(out, output_size=(Hp, Wp), kernel_size=self.window_size, stride=self.window_size)

        # Normalize (no overlap for stride=window_size, but keep for robustness)
        ones = torch.ones((B, 1, Hp, Wp), device=x.device)
        divisor = F.fold(F.unfold(ones, self.window_size, stride=self.window_size),
                         output_size=(Hp, Wp), kernel_size=self.window_size, stride=self.window_size)
        out = out / divisor.clamp(min=1e-6)
        out = out[:, :, :H, :W]

        # Reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # Apply Coordinate Attention if enabled
        if self.use_coord_attn:
            out = self.coord_attn(out)

        return out

# ------------------------------------
# CoarseBlock: 일반 CNN 기반 특징 추출 (큰 receptive field)
# ------------------------------------
class CoarseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 7, padding=3),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        return self.conv(x)


# ------------------------------------
# DBFB: Dual-Branch Fusion Block - Shifted Window 지원
# ------------------------------------
class DBFB(nn.Module):
    def __init__(self, dim, drop_prob=0.1, window_size=17, layer_idx=0):
        super().__init__()
        # Alternating shift pattern: even layers no shift, odd layers shift
        shift_size = 0 if (layer_idx % 2 == 0) else window_size // 2

        self.fine = SwinStyleBlockWithFSA(
            dim=64, window_size=window_size,
            use_fsa=True, use_coord_attn=True,
            shift_size=shift_size
        )
        self.coarse = CoarseBlock(dim)
        self.mix_conv = nn.Conv2d(dim * 2, 1, 1)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x, fsa_mask=None):
        fine_feat = self.fine(x, fsa_mask=fsa_mask)
        coarse_feat = self.coarse(x)
        mix = torch.cat([fine_feat, coarse_feat], dim=1)
        mask = torch.sigmoid(self.mix_conv(mix))
        fused = mask * fine_feat + (1 - mask) * coarse_feat
        out = x + self.drop_path(fused)
        return out

# ------------------------------------
# FARG: Frequency-Aware Reasoning Group
# ------------------------------------
class FARG(nn.Module):
    def __init__(self, dim, num_blocks, drop_prob=0.1, window_size=17):
        super().__init__()
        self.fsa_shared = FrequencyAwareSpatialAttention(
            window_size=window_size, channel=dim, use_learnable_mask=True
        )

        # Create blocks with layer indices for alternating shifts
        self.blocks = nn.ModuleList([
            DBFB(dim, drop_prob=drop_prob, window_size=window_size, layer_idx=i)
            for i in range(num_blocks)
        ])

        self.fusion = nn.Conv2d(dim, dim, 3, 1, 1)
        self.final_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.drop_path = DropPath(drop_prob)

    # FARG.forward
    def forward(self, x):
        # 1) no_grad() 제거
        fsa_mask = self.fsa_shared(x.detach())  # 입력만 detach하여 안정성↑ (선택)
        fsa_mask = torch.clamp(fsa_mask, 0.0, 1.0)
        fsa_mask = torch.nan_to_num(fsa_mask, nan=0.0)

        out = x
        for block in self.blocks:
            out = block(out, fsa_mask=fsa_mask)

        out = x + self.drop_path(self.fusion(out))
        out = self.final_conv(out)
        return out


# ------------------------------------
# WaSRNet: 전체 네트워크 정의 - Shifted Window 지원
# ------------------------------------
@ARCH_REGISTRY.register()
class WaSRNet(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=64, num_group=6,
                 num_block=6, upscale=5, img_range=1., rgb_mean=(0,),
                 drop_prob=0.1, window_size=17):
        super().__init__()
        self.img_range = img_range
        self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, num_in_ch, 1, 1))
        self.upsample_scale = upscale  # ← 스케일 저장
        # (선택) 입력/출력 채널이 다를 수도 있으니 베이스를 맞춰주는 프로젝션
        self.base_proj = nn.Identity() if num_in_ch == num_out_ch else nn.Conv2d(num_in_ch, num_out_ch, 1)

        # Head convolution
        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Body: Multiple FARG groups
        self.body = nn.ModuleList()
        for _ in range(num_group):
            self.body.append(
                FARG(num_feat, num_block, drop_prob=drop_prob, window_size=window_size)
            )

        # Feature fusion and tail
        self.fusion_conv = nn.Conv2d(num_feat * num_group, num_feat, 1)
        self.body_tail = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Output tail
        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        # Head
        x_head = self.head(x)

        # Body: Feature extraction through FARG groups
        feats = []
        out = x_head
        for block in self.body:
            out = block(out)
            feats.append(out)

        # Feature fusion
        out = self.fusion_conv(torch.cat(feats, dim=1))
        out = self.body_tail(out) + x_head

        # Upsampling (feature)
        out = self.upsample(out)

        # Predicted residual
        res = self.tail(out)   # [B, num_out_ch, H*scale, W*scale]

        # Base image (bicubic upsample of input)
        base = F.interpolate(x, scale_factor=self.upsample_scale, mode='bicubic', align_corners=False)
        base = self.base_proj(base)  # 채널 수 맞추기 (num_in_ch != num_out_ch 대비)

        # Output = base + residual
        return res + base