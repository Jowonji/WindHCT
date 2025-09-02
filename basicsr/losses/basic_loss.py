import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from .dists_loss import DISTS  # DISTS 클래스가 같은 파일에 있으면 생략 가능

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class DISTSPerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.dists = DISTS()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # DISTS가 배치별 점수 반환할 수 있으므로 평균 취함
        score = 1 - self.dists(pred, target)  # shape: [B]
        if score.ndim > 0:
            score = score.mean()

        return score * self.loss_weight, None

@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    """Gradient-based L1 loss for edge/detail preservation.

    Args:
        loss_weight (float): Loss weight for gradient loss. Default: 1.0.
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: none | mean | sum')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Returns:
            (loss_percep, loss_style) tuple. Style loss is 0 by default.
        """
        # Gradient in x-direction
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient in y-direction
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = F.l1_loss(pred_dx, target_dx, reduction=self.reduction)
        loss_dy = F.l1_loss(pred_dy, target_dy, reduction=self.reduction)
        loss = self.loss_weight * (loss_dx + loss_dy)

        return loss, torch.tensor(0.0, device=loss.device)

#@LOSS_REGISTRY.register()
#class CharbonnierLoss(nn.Module):
#    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
#    variant of L1Loss).
#
#    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
#        Super-Resolution".
#
#    Args:
#        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
#        reduction (str): Specifies the reduction to apply to the output.
#            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#        eps (float): A value used to control the curvature near zero. Default: 1e-12.
#    """
#
#    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
#        super(CharbonnierLoss, self).__init__()
#        if reduction not in ['none', 'mean', 'sum']:
#            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
#
#        self.loss_weight = loss_weight
#        self.reduction = reduction
#        self.eps = eps
#
#    def forward(self, pred, target, weight=None, **kwargs):
#        """
#        Args:
#            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
#        """
#        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """지각적 손실(Perceptual Loss) 및 스타일 손실(Style Loss)을 포함하는 손실 함수 클래스"""
    """
    Args:
        layer_weights (dict): VGG의 특정 계층(layer)에 대해 적용할 가중치 지정.
            예: {'conv5_4': 1.0} -> conv5_4 레이어의 특징을 손실 계산에 반영.
        vgg_type (str): 특징 추출기로 사용할 VGG 네트워크 유형. 기본값: 'vgg19'.
        use_input_norm (bool): 입력 이미지를 VGG 네트워크에서 정규화할지 여부. 기본값: True.
        range_norm (bool): 이미지의 값 범위를 [-1, 1]에서 [0, 1]로 변환할지 여부. 기본값: False.
        perceptual_weight (float): 지각적 손실(Perceptual Loss)에 적용할 가중치. 기본값: 1.0.
        style_weight (float): 스타일 손실(Style Loss)에 적용할 가중치. 기본값: 0.0.
        criterion (str): 손실 함수의 기준 (L1, L2 등). 기본값: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):


        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight # 지각적 손실 가중치 설정
        self.style_weight = style_weight # 스타일 손실 가중치 설정
        self.layer_weights = layer_weights # VGG 계층별 가중치 저장

        # VGG 네트워크에서 특징(feature) 추출
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()), # 사용할 레이어 목록
            vgg_type=vgg_type, # VGG 네트워크 타입 지정(기본: vgg19)
            use_input_norm=use_input_norm, # 입력 정규화 여부
            range_norm=range_norm) # 갑 범위 변환 여부

        # 손실 기준 설정(L1, L2, Frobenius Norm)
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss() # L1 손실 (MAE: Mean Absolute Error)
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss() # L2 손실 (MSE: Mean Squared Error)
        elif self.criterion_type == 'fro':
            self.criterion = None # Frobenius Norm은 별도 계산 필요
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # 1채널 -> 3채널 변환
        if x.size(1) == 1:  # 입력이 1채널인 경우
            x = x.repeat(1, 3, 1, 1)
        if gt.size(1) == 1:  # Ground Truth가 1채널인 경우
            gt = gt.repeat(1, 3, 1, 1)

        # VGG 특징 추출
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # 지각적 손실 계산
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # 스타일 손실 계산
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

@LOSS_REGISTRY.register()
class PatchSimilarityLoss(nn.Module):
    def __init__(self, patch_size=10, l1_weight=0.1, loss_weight=1.0):
        super(PatchSimilarityLoss, self).__init__()
        self.patch_size = patch_size
        self.l1_weight = l1_weight
        self.loss_weight = loss_weight  # ✅ 추가

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        loss = 0.0
        count = 0

        for i in range(0, H - self.patch_size + 1, self.patch_size):
            for j in range(0, W - self.patch_size + 1, self.patch_size):
                pred_patch = pred[:, :, i:i+self.patch_size, j:j+self.patch_size]
                target_patch = target[:, :, i:i+self.patch_size, j:j+self.patch_size]

                if pred_patch.shape[2:] == (self.patch_size, self.patch_size):
                    # (B, patch_size, patch_size, C) → (B, -1, C)
                    pred_flat = pred_patch.permute(0, 2, 3, 1).reshape(B, -1, C)
                    target_flat = target_patch.permute(0, 2, 3, 1).reshape(B, -1, C)

                    sim = F.cosine_similarity(pred_flat, target_flat, dim=2).mean(dim=1)
                    l1_diff = F.l1_loss(pred_patch, target_patch, reduction='mean')

                    loss += torch.mean(1 - sim) + self.l1_weight * l1_diff
                    count += 1

        return (loss / count if count > 0 else loss) * self.loss_weight  # ✅ 적용


@LOSS_REGISTRY.register()
class WaveletHighFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, weight_hl=1.0, weight_lh=1.0, weight_hh=1.0):
        super().__init__()
        self.loss_weight = loss_weight

        # 방향별 Haar 필터 정의
        self.register_buffer('filter_hl', torch.tensor([[1, -1], [1, -1]], dtype=torch.float32).view(1, 1, 2, 2))
        self.register_buffer('filter_lh', torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32).view(1, 1, 2, 2))
        self.register_buffer('filter_hh', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2))

        self.weight_hl = weight_hl
        self.weight_lh = weight_lh
        self.weight_hh = weight_hh

    def extract_hf(self, img, filt):
        B, C, H, W = img.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        return torch.abs(F.conv2d(img, filt.repeat(C, 1, 1, 1), stride=2, groups=C))

    def forward(self, pred, target):
        loss = 0.0
        for filt, weight in zip(
            [self.filter_hl, self.filter_lh, self.filter_hh],
            [self.weight_hl, self.weight_lh, self.weight_hh]
        ):
            pred_hf = self.extract_hf(pred, filt)
            target_hf = self.extract_hf(target, filt)
            loss += weight * F.l1_loss(pred_hf, target_hf)
        return loss * self.loss_weight

#@LOSS_REGISTRY.register()
#class GradientLoss(nn.Module):
#    def __init__(self, loss_weight=1.0):
#        super().__init__()
#        self.loss_weight = loss_weight
#
#    def forward(self, pred, target):
#        pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
#        pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
#        target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
#        target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]
#        loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
#        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, loss_weight=1.0):  # ✅ 추가
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight  # ✅ 저장

    def forward(self, pred, target):
        loss = torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))
        return loss * self.loss_weight  # ✅ 적용