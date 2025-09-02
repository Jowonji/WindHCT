# Modified from https://github.com/JingyunLiang/SwinIR
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import to_2tuple, trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    drop_path 함수는 Residual Block 내부에서 경로를 무작위로 드랍(제거)하는 Stochastic Depth 기법을 구현한 것입니다.
    매 샘플마다 드랍을 적용하여 네트워크의 일반화를 돕습니다.

    매개변수:
      x         : 입력 텐서
      drop_prob : 드랍할 확률 (0이면 드랍하지 않음)
      training  : 학습 모드 여부 (False이면 드랍하지 않고 그대로 반환)
    """
    # 드랍 확률이 0이거나 학습 모드가 아니면 입력을 그대로 반환합니다.
    if drop_prob == 0. or not training:
        return x
    # 남길 확률(keep brobability)
    keep_prob = 1 - drop_prob
    # 입력 텐서와 동일한 배치 차원 유지, 나머지 차원은 1로 만들어 다양한 차원의 텐서를 지원
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # 예: (batch_size, 1, 1, ...)
    # 지정된 shape로 균등분포에서 난수를 생성하고 keep_prob를 더합니다.
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 난수 텐서를 floor 연사을 통해 0 또는 1로 이진화
    random_tensor.floor_()  # binarize: 0또는 1로 만ㄷ름
    # 입력을 keep_prob로 나누어 평균 값을 유지한 후, 이진 마스크를 곱합니다.
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    DropPath 클래스는 nn.Module을 상속받아 drop_path 함수를 모듈 형태로 감싼 것입니다.
    Residual Block의 주 경로에서 경로 드랍(Stochastic Depth)을 쉽게 적용할 수 있도록 합니다.

    속성:
      drop_prob : 드랍할 확률
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob # 드랍 확률 저장

    def forward(self, x):
        # forward 함수에서는 drop_path 함수를 호출하여 입력에 대해 경로 드랍을 적용합니다.
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    Mlp 클래스는 다층 퍼셉트론(MLP)으로, 일반적으로 Transformer의 feed-forward 네트워크에서 사용됩니다.
    두 개의 선형 변환 사이에 활성화 함수와 드랍아웃을 적용합니다.

    매개변수:
      in_features   : 입력 피처의 차원
      hidden_features: 은닉층 피처 차원 (지정하지 않으면 in_features 사용)
      out_features  : 출력 피처의 차원 (지정하지 않으면 in_features 사용)
      act_layer     : 활성화 함수 (기본값: GELU)
      drop          : 드랍아웃 확률
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 출력 피처와 은닉 피처 차원을 지정하거나 기본값(in_features)을 사용합니다.
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 첫번째 선형 레이어: 입력을 은닉 차원으로 변환
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 활성화 함수 초기화 (기본적으로 GELU)
        self.act = act_layer()
        # 두번째 선형 레이어: 은닉 차원을 출력 차원으로 변환
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 드랍아웃 레이어 초기화
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) # 입력 x에 첫 번째 선형 변환 적용
        x = self.act(x) # 활성화 함수 적용
        x = self.drop(x) # 드랍아웃 적용
        x = self.fc2(x) # 두 번째 선형 변환 적용
        x = self.drop(x) # 다시 드랍아웃 적용
        return x

class ContinuousRelativePositionBias(nn.Module):
    """
    연속형 상대 위치 바이어스(CPB).
    윈도 내 상대좌표(Δx,Δy)를 로그 스케일로 매핑 → 작은 MLP → 헤드별 바이어스.
    - 파라미터 소량, 창 크기 변화/해상도 외삽에 강함.
    """
    def __init__(self, num_heads: int, hidden_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads, bias=True)
        )
        # 작은 초기값(과대적합 방지)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        self._cache = {}  # (w, device, dtype) -> [T,T,2] tensor

    @torch.no_grad()
    def _rel_coords(self, w: int, device, dtype):
        key = (w, device, dtype)
        if key in self._cache:
            return self._cache[key]
        coords = torch.stack(torch.meshgrid(
            torch.arange(w, device=device), torch.arange(w, device=device), indexing="ij"
        ), dim=-1).view(-1, 2)                     # [T,2], T=w*w
        rel = coords[:, None, :] - coords[None, :, :]  # [T,T,2], Δx,Δy in [-w+1,w-1]
        # 로그 스케일 정규화(스윈V2 유사): sign * log(1+|Δ|)
        rel = rel.to(dtype)
        rel = torch.sign(rel) * torch.log1p(rel.abs())
        # [-1,1] 근처 분포가 되도록 스케일 정규화
        rel = rel / math.log(1 + (w - 1))
        self._cache[key] = rel
        return rel

    def forward(self, w: int, device, dtype):
        rel = self._rel_coords(w, device, dtype)       # [T,T,2]
        bias = self.mlp(rel)                            # [T,T,Hh]
        return bias.permute(2, 0, 1).contiguous()       # [Hh,T,T]

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
           - 입력 텐서로, 배치 크기(b), 높이(h), 너비(w), 채널 수(c)를 가집니다.
        window_size (int): 윈도우의 크기

    Returns:
        windows: (num_windows * b, window_size, window_size, c)
           - 입력 이미지를 window_size 크기의 작은 윈도우들로 분할한 결과입니다.
           - 각 윈도우는 개별 이미지 조각이며, 전체 윈도우 수는 (h*w)/(window_size^2)입니다.
    """
    # 텐서의 배치, 높이, 너비, 채널 수를 추출
    b, h, w, c = x.shape
    # 이미지를 윈도우 크기별로 나누기 위해 텐서의 형태를 변환합니다.
    # (b, h, w, c)를 (b, h//window_size, window_size, w//window_size, window_size, c)로 reshape합니다.
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    # 차원의 순서를 변경하여 윈도우들이 연속적으로 위치하도록 합니다.
    # 이후 contiguous()를 사용해 메모리 상에서 연속된 텐서로 만든 후,
    # (-1, window_size, window_size, c)로 reshape하여 각 윈도우를 하나의 샘플로 취급합니다.
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows * b, window_size, window_size, c)
           - 분할된 윈도우 텐서입니다.
        window_size (int): 윈도우의 크기
        h (int): 원본 이미지의 높이
        w (int): 원본 이미지의 너비

    Returns:
        x: (b, h, w, c)
           - 분할된 윈도우들을 원본 이미지의 형태로 복원한 텐서입니다.
    """
    # 분할된 윈도우의 총 개수를 이용해 배치 크기 b를 복원합니다.
    # 원본 이미지의 윈도우 개수는 (h * w) / (window_size^2) 이므로,
    # windows.shape[0]를 이 값으로 나누어 배치 크기를 구합니다.
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    # 윈도우 텐서를 (b, h//window_size, w//window_size, window_size, window_size, c) 형태로 reshape합니다.
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    # 차원의 순서를 변경하여 원본 이미지의 형태인 (b, h, w, c)로 복원합니다.
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    윈도우 기반 멀티헤드 셀프 어텐션 (W-MSA) 모듈로, 상대 위치 바이어스(relative position bias)를 포함합니다.
    Shifted 윈도우와 Non-shifted 윈도우 모두를 지원합니다.

    Args:
        dim (int): 입력 채널 수.
        window_size (tuple[int]): 윈도우의 높이와 너비 (예: (Wh, Ww)).
        num_heads (int): 어텐션 헤드 수.
        qkv_bias (bool, optional): True이면 query, key, value에 학습 가능한 bias를 추가합니다. (기본값: True)
        qk_scale (float | None, optional): head_dim ** -0.5의 기본 qk scale을 대체할 값.
        attn_drop (float, optional): 어텐션 가중치 드랍아웃 비율 (기본값: 0.0)
        proj_drop (float, optional): 출력 드랍아웃 비율 (기본값: 0.0)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 윈도우의 높이(Wh)와 너비(Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5 # 스케일 값

        # 상대 위치 바이어스 테이블 정의: (2*Wh-1) * (2*Ww-1) 크기에 num_heads 차원
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 윈도우 내 각 토큰 쌍의 상대 위치 인덱스를 계산
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # 2 x Wh x Ww 형태의 좌표 행렬 생성
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 각 토큰 간의 상대 좌표를 계산 (2, Wh*Ww, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # 차원을 변경하여 (Wh*Ww, Wh*Ww, 2) 형태로 만듦
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 상대 좌표가 음수가 되지 않도록 오프셋 적용 (0부터 시작)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 첫 번째 좌표에 대한 인덱스 스케일링: (2*Ww-1)를 곱함
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 두 좌표를 합산하여 최종 상대 위치 인덱스 계산, shape: (Wh*Ww, Wh*Ww)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 모델 버퍼에 등록 (학습 시 업데이트 되지 않는 고정 텐서)
        self.register_buffer('relative_position_index', relative_position_index)

        # Query, Key, Value를 위한 선형 계층. 차원: dim -> 3*dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 어텐션 드랍아웃
        self.attn_drop = nn.Dropout(attn_drop)
        # 어텐션 결과를 통합하기 위한 선형 계층
        self.proj = nn.Linear(dim, dim)
        # 최종 출력 드랍아웃
        self.proj_drop = nn.Dropout(proj_drop)

        # 상대 위치 바이어스 테이블을 정규분포(truncated normal)로 초기화
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # softmax 함수 정의 (어텐션 가중치 계산 시 사용)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 피처, shape: (num_windows * b, n, c)
               - num_windows: 윈도우 수, b: 배치 크기, n: 각 윈도우 내 토큰 수, c: 채널 수
            mask: (0 또는 -inf) 마스크, shape: (num_windows, Wh*Ww, Wh*Ww) 또는 None

        Returns:
            x: 어텐션 연산을 거친 출력 피처, shape: (num_windows * b, n, c)
        """
        b_, n, c = x.shape
        # qkv 선형 계층 적용 후, (b_, n, 3, num_heads, c // num_heads)로 reshape하고, 차원 순서를 변경
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # query, key, value 분리

        # query에 스케일 적용
        q = q * self.scale
        # 어텐션 스코어 계산: query와 key의 내적 (행렬 곱셈)
        attn = (q @ k.transpose(-2, -1))

        # 상대 위치 바이어스를 가져오기 위한 처리
        # relative_position_bias_table에서 flatten된 relative_position_index를 사용해 값을 추출한 후,
        # 윈도우 크기 (Wh*Ww x Wh*Ww)와 num_heads 차원으로 재구성
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # 차원 순서를 변경하여 (num_heads, Wh*Ww, Wh*Ww)로 만듦
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 어텐션 스코어에 상대 위치 바이어스 추가 (배치 차원 추가)
        attn = attn + relative_position_bias.unsqueeze(0)

        # 만약 마스크가 주어졌다면, 윈도우 별 마스크를 어텐션 스코어에 적용
        if mask is not None:
            nw = mask.shape[0] # 윈도우 수
            # 마스크를 적용하기 위해 어텐션 스코어의 shape을 재구성하고, 마스크를 더한 후 다시 reshape
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            # 마스크가 없으면 바로 softmax 적용
            attn = self.softmax(attn)

        # 어텐션 가중치에 드랍아웃 적용
        attn = self.attn_drop(attn)

        # 어텐션 가중치와 value를 곱하여 출력 계산, 차원 변환 후 reshape
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        # 최종 선형 계층을 통해 출력 차원 통합 후 드랍아웃 적용
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        # 객체의 추가 정보를 문자열로 반환합니다.
        # 모델의 기본 속성(dim, window_size, num_heads)을 요약하여 출력할 때 사용됩니다.
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # 주어진 토큰 길이 n에 대해, 한 개의 윈도우에서 발생하는 FLOPs (부동소수점 연산 수)를 계산합니다.
        flops = 0
        # qkv 연산: 입력 텐서 x에 대해 선형 변환을 수행하여 query, key, value를 계산합니다.
        # 연산량: n * self.dim (입력 차원) * 3 * self.dim (출력 차원 3배)
        flops += n * self.dim * 3 * self.dim

        # 어텐션 스코어 계산: query와 key의 전치 행렬 간의 행렬 곱셈
        # 각 헤드별 연산량: n (query 길이) * (self.dim // self.num_heads) (헤드 차원) * n (key 길이)
        flops += self.num_heads * n * (self.dim // self.num_heads) * n

        # 어텐션 결과 계산: 어텐션 가중치와 value의 행렬 곱셈
        # 각 헤드별 연산량: n * n * (self.dim // self.num_heads)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)

        # 최종 프로젝션: 어텐션 결과에 대해 선형 변환 수행
        # 연산량: n * self.dim * self.dim
        flops += n * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): 입력 채널 수.
        input_resolution (tuple[int]): 입력 해상도 (높이, 너비).
        num_heads (int): 어텐션 헤드 수.
        window_size (int): 윈도우 크기.
        shift_size (int): SW-MSA를 위한 윈도우 시프트 크기.
        mlp_ratio (float): MLP 은닉층 차원의 배수 비율.
        qkv_bias (bool, optional): True이면 query, key, value에 학습 가능한 bias를 추가합니다. (기본값: True)
        qk_scale (float | None, optional): head_dim ** -0.5의 기본 qk scale을 대체할 값.
        drop (float, optional): 드랍아웃 비율 (기본값: 0.0)
        attn_drop (float, optional): 어텐션 드랍아웃 비율 (기본값: 0.0)
        drop_path (float, optional): Stochastic depth 비율 (기본값: 0.0)
        act_layer (nn.Module, optional): 활성화 함수 (기본값: nn.GELU)
        norm_layer (nn.Module, optional): 정규화 계층 (기본값: nn.LayerNorm)
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # 기본 속성 저장: 입력 채널 수, 해상도, 어텐션 헤드 수, 윈도우 크기 및 시프트 크기, MLP 비율
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 입력 해상도가 윈도우 크기보다 작거나 같으면, 윈도우 분할이 불필요하므로 시프트 크기를 0으로 설정하고
        # 윈도우 크기를 입력 해상도의 최소값으로 맞춥니다.
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # shift_size는 0 이상, window_size 미만이어야 합니다.
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        # 첫 번째 정규화 계층 (예: LayerNorm) - 어텐션 전에 피처 정규화
        self.norm1 = norm_layer(dim)
        # WindowAttention 모듈 초기화: 윈도우 내에서 멀티헤드 셀프 어텐션을 수행합니다.
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size), # 윈도우 크기를 튜플 형태로 변환
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        # Stochastic Depth (DropPath) 모듈: drop_path 비율이 0보다 크면 적용, 그렇지 않으면 Identity (변경 없음)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 두 번째 정규화 계층 - MLP 입력 전 정규화
        self.norm2 = norm_layer(dim)
        # MLP 은닉층 차원 설정: 입력 차원에 mlp_ratio를 곱한 값
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP 모듈 초기화: 두 개의 선형 계층과 활성화 함수, 드랍아웃을 포함한 다층 퍼셉트론
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 윈도우 시프트가 적용되는 경우, 어텐션 마스크를 계산하여 설정합니다.
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        # 계산된 어텐션 마스크를 버퍼에 등록합니다.
        # register_buffer를 사용하면 학습 시 업데이트되지 않으며, 모델의 상태에 포함됩니다.
        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w
        # W-MSA/SW-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle' and self.upscale == 5:
            self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)

            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 25, 3, 1, 1),
                nn.PixelShuffle(5),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.ReLU(inplace=True)
            )

            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale in [4, 5], '현재 4배와 5배 업스케일링만 지원됩니다.'  # ✅ 5배 업스케일링 지원 추가
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))

            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # 4배, 5배 업스케일링 지원 추가
            assert self.upscale in [4, 5], '현재 4배와 5배 업스케일링만 지원됩니다.'
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)

            # 5배 업스케일링을 위한 nearest 보간 후 Conv 적용
            if self.upsampler == 'pixelshuffle':
                x = self.conv_first(x)
                x = self.conv_after_body(self.forward_features(x)) + x
                x = self.conv_before_upsample(x)  # Conv2d(embed_dim, num_feat, 3x3)
                x = self.upsample(x)              # PixelShuffle(5) path
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')  # 2배 업스케일링
                x = self.lrelu(self.conv_up1(x))
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')  # 4배 업스케일링
                x = self.lrelu(self.conv_up2(x))

            x = self.conv_last(self.upsample(x))  # 👈 그대로 작동

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(
        upscale=2,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)