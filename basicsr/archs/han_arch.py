from archs import common
import torch
import torch.nn as nn
import pdb
import math
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

def make_model(args, parent=False):
    return HAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class LAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C, H, W = x.size()
        proj_query = x.view(B, N, -1)
        proj_key = x.view(B, N, -1).permute(0, 2, 1)
        scale = math.sqrt(C * H * W)
        energy = torch.bmm(proj_query, proj_key) / scale
        attention = self.softmax(energy)
        proj_value = x.view(B, N, -1)

        out = torch.bmm(attention, proj_value).view(B, N, C, H, W)
        out = self.gamma * out + x
        return out.view(B, -1, H, W)

class CSAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out.view(B, -1, H, W)
        return x * out + x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Holistic Attention Network (HAN)
@ARCH_REGISTRY.register()
class HAN(nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_group=10,
                 num_block=20,
                 reduction=16,
                 upscale=5,
                 res_scale=1,
                 img_range=1.,
                 rgb_mean=(0.0,),
                 rgb_std=(1.0,)):
        super().__init__()
        conv = common.default_conv

        self.mean_shift = common.MeanShift(img_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(img_range, rgb_mean, rgb_std, sign=1)

        self.head = conv(num_in_ch, num_feat, 3)

        self.body = nn.ModuleList([
            ResidualGroup(conv, num_feat, 3, reduction, nn.ReLU(True), res_scale, num_block)
            for _ in range(num_group)
        ])
        self.body_tail = conv(num_feat, num_feat, 3)

        self.csam = CSAM_Module(num_feat)
        self.lam = LAM_Module(num_feat)
        self.last_conv = nn.Conv2d(num_feat * num_group, num_feat, 3, 1, 1)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            conv(num_feat, num_feat * upscale * upscale, 3),
            nn.PixelShuffle(upscale),
            conv(num_feat, num_feat, 3),
            nn.ReLU(inplace=True)
        )
        self.tail = conv(num_feat, num_out_ch, 3)

    def forward(self, x):
        x = self.mean_shift(x)
        x = self.head(x)

        feats = []
        res = x
        for block in self.body:
            res = block(res)
            feats.append(res.unsqueeze(1))

        out1 = res
        stacked = torch.cat(feats, dim=1)
        out2 = self.last_conv(self.lam(stacked))
        out1 = self.csam(out1)

        fusion = torch.cat([out1, out2], dim=1)
        res = self.fusion(fusion) + x

        x = self.upsample(res)
        x = self.tail(x)
        x = self.add_mean(x)
        return x
