import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from basicsr.archs.our_arch import WaSRNet  # ✅ 사용 중인 Generator 클래스
from collections import OrderedDict

# ✅ 중간 feature 저장용 딕셔너리
activations = {}

def register_hooks(model, block_idx=0, block_inner_idx=0):
    # 정확한 submodule 경로 지정
    fine_name = f'body.{block_idx}.blocks.{block_inner_idx}.fine_branch'
    coarse_name = f'body.{block_idx}.blocks.{block_inner_idx}.coarse_branch'

    def get_hook(name):
        return lambda module, input, output: activations.__setitem__(name, output.detach().cpu())

    # Hook 등록
    model.get_submodule(fine_name).register_forward_hook(get_hook('fine'))
    model.get_submodule(coarse_name).register_forward_hook(get_hook('coarse'))

    print(f"✅ Registered hook for: {fine_name}")
    print(f"✅ Registered hook for: {coarse_name}")

    # (옵션) 모델 내 관련 모듈들 목록 출력
    print("\n🔍 Matching modules in model:")
    for name, _ in model.named_modules():
        if 'fine' in name or 'coarse' in name:
            print(f" - {name}")


def visualize_feature_map(feat, title='Feature Map', num_channels=4):
    b, c, h, w = feat.shape

    for i in range(min(num_channels, c)):
        plt.figure()
        plt.imshow(feat[0, i], cmap='viridis')
        plt.title(f'{title} - C{i}')
        plt.axis('off')
        save_path = f'{title.lower().replace(" ", "_")}_c{i}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")


def save_mean_feature(feat, name='fine'):
    plt.figure()
    plt.imshow(feat[0].mean(0), cmap='viridis')
    plt.title(f'{name.capitalize()} Mean Feature')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f'{name}_mean_feature3.png', bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name}_mean_feature3.png")


def load_state(model, ckpt_path):
    print(f"🔄 Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')

    if 'params' in state_dict:  # BasicSR 저장 형식
        state_dict = state_dict['params']

    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('network.'):
            k = k[len('network.'):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    print("✅ Weights loaded successfully.")

def main():
    # ✅ 모델 생성
    model = WaSRNet(
        num_in_ch=1,
        num_out_ch=1,
        num_feat=64,
        num_group=6,
        num_block=6,
        img_range=1.0,
        rgb_mean=[0.0]
    ).to('cuda')
    model.eval()

    # ✅ weight 로드
    ckpt_path = '/home/wj/works/SR-project2/BasicSR/experiments/SCIE/WASR-Net_v3_revise5/models/net_g_48600.pth'
    load_state(model, ckpt_path)

    # ✅ Hook 등록
    register_hooks(model, block_idx=0, block_inner_idx=0)

    # ✅ 입력 데이터 (npy 직접 입력)
    npy_path = '/home/wj/works/Wind_Speed_Data/ERA52CERRA/test_lr_norm.npy'
    npy = np.load(npy_path)
    if npy.ndim == 3:
        npy = npy[0]  # 첫 샘플만 사용
    input_tensor = torch.from_numpy(npy).float().unsqueeze(0).unsqueeze(0).to('cuda')  # [1, 1, H, W]

    # ✅ 추론
    with torch.no_grad():
        _ = model(input_tensor)

    # ✅ 채널별 Feature Map 저장
    visualize_feature_map(activations['fine'], title='Fine Feature3')
    visualize_feature_map(activations['coarse'], title='Coarse Feature3')

    # ✅ 평균 Feature Map 저장
    save_mean_feature(activations['fine'], name='fine')
    save_mean_feature(activations['coarse'], name='coarse')

        # ✅ 융합 feature 시각화도 추가 (body.0.blocks.0 기준으로 접근)
    fused_feat = model.body[0].blocks[0].latest_fused.detach().cpu()
    fusion_mask = model.body[0].blocks[0].latest_mask.detach().cpu()

    visualize_feature_map(fused_feat, title='Fused Feature3')
    save_mean_feature(fused_feat, name='fused')

    # ✅ Fusion Mask는 단일 채널
    plt.figure()
    plt.imshow(fusion_mask[0, 0], cmap='gray')
    plt.colorbar()
    plt.title('Fusion Mask3')
    plt.axis('off')
    plt.savefig('fusion_mask3.png', bbox_inches='tight')
    plt.close()
    print("✅ Saved: fusion_mask3.png")

if __name__ == '__main__':
    main()
    # ✅ Hook 등록 (예: body.0.blocks.0의 fine/coarse 기준)
