import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

@DATASET_REGISTRY.register()
class NumpyPairedDataset(Dataset):
    """정규화된 Numpy 형식의 저해상도(LR) 및 고해상도(HR) 데이터를 로드하는 데이터셋 클래스"""

    def __init__(self, opt):
        super(NumpyPairedDataset, self).__init__()
        self.opt = opt

        # 정규화된 HR 및 LR 데이터 로드
        self.hr_data = np.load(opt['dataroot_gt']).astype(np.float32)
        self.lr_data = np.load(opt['dataroot_lq']).astype(np.float32)

        # 샘플 수 검증
        assert len(self.hr_data) == len(self.lr_data), "HR and LR datasets must have the same number of samples."
        self.scale = opt['scale']

        print(f"[Dataset] Loaded HR shape: {self.hr_data.shape}, LR shape: {self.lr_data.shape}")

    def __getitem__(self, index):
        # HR 및 LR 로드
        hr = self.hr_data[index]
        lr = self.lr_data[index]

        # 채널 차원 추가 (H, W) -> (1, H, W)
        if hr.ndim == 2:
            hr = np.expand_dims(hr, axis=0)
        if lr.ndim == 2:
            lr = np.expand_dims(lr, axis=0)

        # 스케일 자동 확인
        h_gt, w_gt = hr.shape[1], hr.shape[2]
        h_lq, w_lq = lr.shape[1], lr.shape[2]
        computed_scale_h = h_gt / h_lq
        computed_scale_w = w_gt / w_lq
        epsilon = 0.05
        self.scale = round(computed_scale_h, 2)
        if not (abs(computed_scale_h - self.scale) < epsilon and abs(computed_scale_w - self.scale) < epsilon):
            raise ValueError(
                f"Scale mismatches. Computed scale ({computed_scale_h:.3f}, {computed_scale_w:.3f}) does not match expected scale {self.scale}."
            )

        # Tensor로 변환
        hr_tensor = torch.from_numpy(hr)
        lr_tensor = torch.from_numpy(lr)

        # 증강
        if self.opt.get('use_hflip', False) and random.random() < 0.5:
            hr_tensor = TF.hflip(hr_tensor)
            lr_tensor = TF.hflip(lr_tensor)

        if self.opt.get('use_rot', False):
            angle = random.uniform(-15, 15)
            hr_tensor = TF.rotate(hr_tensor, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            lr_tensor = TF.rotate(lr_tensor, angle, interpolation=transforms.InterpolationMode.BILINEAR)

        if self.opt.get('use_blur', False) and random.random() < 0.3:
            sigma = random.uniform(0.1, 0.5)
            hr_tensor = TF.gaussian_blur(hr_tensor, kernel_size=3, sigma=sigma)
            lr_tensor = TF.gaussian_blur(lr_tensor, kernel_size=3, sigma=sigma)

        return {
            'gt': hr_tensor,
            'lq': lr_tensor,
            'gt_path': str(index),
            'lq_path': str(index)
        }

    def __len__(self):
        return len(self.hr_data)