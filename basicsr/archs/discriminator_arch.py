from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    def __init__(self, num_in_ch, num_feat, input_size=100):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 100 or self.input_size == 256, (
            f'input size must be 100 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        # Update linear layer for 100x100 input
        self.linear1 = nn.Linear(num_feat * 8 * 3 * 3, 100)  # Adjusted to 4608
        self.linear2 = nn.Linear(100, 1)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator2(nn.Module):
    def __init__(self, num_in_ch, num_feat, input_size=256):
        super().__init__()

        self.input_size = input_size
        assert self.input_size in [100, 256], f'input size must be 100 or 256, but received {input_size}'

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # ✅ Fully Connected Layer 크기 자동 설정
        self.linear1 = nn.Linear(512 * 8 * 8, 100)  # 🔥 32768 → 100으로 줄이기
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        assert x.size(2) == self.input_size, f'Input size must be {self.input_size}, but received {x.size()}'

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))

        feat = feat.view(feat.size(0), -1)  # ✅ Flatten (batch_size, 32768)
        feat = self.lrelu(self.linear1(feat))  # ✅ 32768 → 100으로 변환
        out = self.linear2(feat)
        return out



#@ARCH_REGISTRY.register(suffix='basicsr')
#class UNetDiscriminatorSN(nn.Module):
#    """Defines a U-Net discriminator with spectral normalization (SN)
#
#    Used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
#
#    Args:
#        num_in_ch (int): Number of input channels. Default: 3.
#        num_feat (int): Base intermediate feature channels. Default: 64.
#        skip_connection (bool): Whether to use skip connections. Default: True.
#    """
#
#    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
#        super(UNetDiscriminatorSN, self).__init__()
#        self.skip_connection = skip_connection
#        norm = spectral_norm
#
#        # First convolution
#        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
#
#        # Downsampling layers
#        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
#        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
#        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
#
#        # Upsampling layers
#        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
#        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
#        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
#
#        # Extra convolutions
#        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
#        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
#        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
#
#    def forward(self, x):
#        """Forward pass of the U-Net Discriminator"""
#        # Downsample
#        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
#        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
#        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
#        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
#
#        # Upsample
#        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
#        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
#
#        # Ensure skip connection shape consistency
#        if self.skip_connection:
#            if x4.shape != x2.shape:
#                min_h = min(x4.shape[2], x2.shape[2])
#                min_w = min(x4.shape[3], x2.shape[3])
#                x2 = x2[:, :, :min_h, :min_w]  # Crop x2
#                x4 = x4[:, :, :min_h, :min_w]  # Crop x4
#            x4 = x4 + x2
#
#        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
#        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
#
#        if self.skip_connection:
#            if x5.shape != x1.shape:
#                min_h = min(x5.shape[2], x1.shape[2])
#                min_w = min(x5.shape[3], x1.shape[3])
#                x1 = x1[:, :, :min_h, :min_w]  # Crop x1
#                x5 = x5[:, :, :min_h, :min_w]  # Crop x5
#            x5 = x5 + x1
#
#        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
#        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
#
#        if self.skip_connection:
#            if x6.shape != x0.shape:
#                min_h = min(x6.shape[2], x0.shape[2])
#                min_w = min(x6.shape[3], x0.shape[3])
#                x0 = x0[:, :, :min_h, :min_w]  # Crop x0
#                x6 = x6[:, :, :min_h, :min_w]  # Crop x6
#            x6 = x6 + x0
#
#        # Extra convolutions
#        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
#        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
#        out = self.conv9(out)
#
#        return out

# --------------------------
# Soft Margin Spectral Norm
# --------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import math

# --------------------------
# Adaptive Margin Soft Spectral Norm Conv2d
# --------------------------
class SoftMarginSNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, base_margin=0.90, max_margin=1.1, power_iterations=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.base_margin = base_margin
        self.max_margin = max_margin
        self.power_iterations = power_iterations

        self.u = None
        self.v = None
        self.initialized = False

    def _initialize_params(self, W):
        W_mat = W.view(W.size(0), -1)
        h, w = W_mat.size()

        if not hasattr(self, "u_buffer"):
            self.register_buffer("u_buffer", normalize(W.new_empty(h).normal_(0, 1), dim=0, eps=1e-12))
        else:
            self.u_buffer.copy_(normalize(W.new_empty(h).normal_(0, 1), dim=0, eps=1e-12))

        if not hasattr(self, "v_buffer"):
            self.register_buffer("v_buffer", normalize(W.new_empty(w).normal_(0, 1), dim=0, eps=1e-12))
        else:
            self.v_buffer.copy_(normalize(W.new_empty(w).normal_(0, 1), dim=0, eps=1e-12))

        self.u = self.u_buffer
        self.v = self.v_buffer
        self.initialized = True

    def compute_margin(self, current_iter, total_iter):
        if current_iter is None or total_iter is None:
            return self.base_margin
        ratio = min(current_iter / total_iter, 1.0)
        sigmoid = 1 / (1 + math.exp(-4 * (ratio - 0.5)))  # 조절하고 싶으면 8을 바꾸기
        return self.base_margin + (self.max_margin - self.base_margin) * sigmoid

    def compute_spectral_norm(self, W, current_iter=None, total_iter=None):
        if not self.initialized:
            self._initialize_params(W)

        W_mat = W.view(W.size(0), -1)
        u = self.u
        v = self.v

        for _ in range(self.power_iterations):
            v = normalize(torch.matmul(W_mat.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(W_mat, v), dim=0, eps=1e-12)

        sigma = torch.dot(u, torch.matmul(W_mat, v))
        adaptive_margin = self.compute_margin(current_iter, total_iter)
        sigma = torch.clamp(sigma, min=adaptive_margin)
        W_sn = W / sigma
        return W_sn

    def forward(self, x, current_iter=None, total_iter=None):
        W_sn = self.compute_spectral_norm(self.conv.weight, current_iter, total_iter)
        return nn.functional.conv2d(
            x, W_sn, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding
        )


# --------------------------
# U-Net Discriminator with Adaptive Soft Margin SN
# --------------------------
@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = SoftMarginSNConv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)
        self.conv2 = SoftMarginSNConv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)
        self.conv3 = SoftMarginSNConv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)

        self.conv4 = SoftMarginSNConv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)
        self.conv5 = SoftMarginSNConv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False)
        self.conv6 = SoftMarginSNConv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False)

        self.conv7 = SoftMarginSNConv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.conv8 = SoftMarginSNConv2d(num_feat, num_feat, 3, 1, 1, bias=False)

        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, current_iter=None, total_iter=None):
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0, current_iter, total_iter), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1, current_iter, total_iter), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2, current_iter, total_iter), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = F.leaky_relu(self.conv4(x3, current_iter, total_iter), 0.2, inplace=True)
        if self.skip_connection:
            min_h, min_w = min(x4.shape[2], x2.shape[2]), min(x4.shape[3], x2.shape[3])
            x4 = x4 + x2[:, :, :min_h, :min_w]

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x5 = F.leaky_relu(self.conv5(x4, current_iter, total_iter), 0.2, inplace=True)
        if self.skip_connection:
            min_h, min_w = min(x5.shape[2], x1.shape[2]), min(x5.shape[3], x1.shape[3])
            x5 = x5 + x1[:, :, :min_h, :min_w]

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        x6 = F.leaky_relu(self.conv6(x5, current_iter, total_iter), 0.2, inplace=True)
        if self.skip_connection:
            min_h, min_w = min(x6.shape[2], x0.shape[2]), min(x6.shape[3], x0.shape[3])
            x6 = x6 + x0[:, :, :min_h, :min_w]

        out = F.leaky_relu(self.conv7(x6, current_iter, total_iter), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out, current_iter, total_iter), 0.2, inplace=True)
        out = self.conv9(out)
        return out

@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN_Lite(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # ↓ Downsampling 줄임
        self.conv1 = SoftMarginSNConv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)
        self.conv2 = SoftMarginSNConv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)

        # ↓ Mid
        self.conv3 = SoftMarginSNConv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False)
        self.conv4 = SoftMarginSNConv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False)

        # ↓ Final layers
        self.conv5 = SoftMarginSNConv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.conv6 = SoftMarginSNConv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.conv7 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, current_iter=None, total_iter=None):
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0, current_iter, total_iter), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1, current_iter, total_iter), 0.2, inplace=True)

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = F.leaky_relu(self.conv3(x2, current_iter, total_iter), 0.2, inplace=True)
        if self.skip_connection:
            min_h, min_w = min(x3.shape[2], x1.shape[2]), min(x3.shape[3], x1.shape[3])
            x3 = x3 + x1[:, :, :min_h, :min_w]

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = F.leaky_relu(self.conv4(x3, current_iter, total_iter), 0.2, inplace=True)
        if self.skip_connection:
            min_h, min_w = min(x4.shape[2], x0.shape[2]), min(x4.shape[3], x0.shape[3])
            x4 = x4 + x0[:, :, :min_h, :min_w]

        out = F.leaky_relu(self.conv5(x4, current_iter, total_iter), 0.2, inplace=True)
        out = F.leaky_relu(self.conv6(out, current_iter, total_iter), 0.2, inplace=True)
        out = self.conv7(out)
        return out