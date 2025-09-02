import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

@DATASET_REGISTRY.register()
class NumpyPairedKorea(Dataset):
    """Numpy 형식의 저해상도(LR) 및 고해상도(HR) 데이터를 로드하는 데이터셋 클래스"""

    def __init__(self, opt):
        super(NumpyPairedKorea, self).__init__()
        self.opt = opt

        # Numpy 파일에서 HR 및 LR 데이터 로드
        self.hr_data = np.load(opt['dataroot_gt'])  # HR 데이터 로드
        self.lr_data = np.load(opt['dataroot_lq'])  # LR 데이터 로드

        assert len(self.hr_data) == len(self.lr_data), "HR and LR datasets must have the same number of samples."
        self.scale = opt['scale']  # 업스케일링 비율 설정

        # Min/Max 정규화 (0으로 채워진 NaN 포함 상태)
        self.hr_min = np.min(self.hr_data)
        self.hr_max = np.max(self.hr_data)
        print(f"HR data min: {self.hr_min}, max: {self.hr_max}")

    def __getitem__(self, index):
        hr_raw = self.hr_data[index].astype(np.float32)
        lr_raw = self.lr_data[index].astype(np.float32)

        # ✅ 마스크 생성: 원래 NaN이었던 영역을 0.0으로 저장했다고 가정
        mask = (hr_raw != 0.0).astype(np.float32)  # 1.0 = 유효한 데이터, 0.0 = NaN이었던 영역

        # 정규화
        hr = (hr_raw - self.hr_min) / (self.hr_max - self.hr_min)
        lr = (lr_raw - self.hr_min) / (self.hr_max - self.hr_min)

        # 채널 차원 추가
        if hr.ndim == 2: hr = np.expand_dims(hr, axis=0)
        if lr.ndim == 2: lr = np.expand_dims(lr, axis=0)
        if mask.ndim == 2: mask = np.expand_dims(mask, axis=0)

        # 스케일 검사
        h_gt, w_gt = hr.shape[1], hr.shape[2]
        h_lq, w_lq = lr.shape[1], lr.shape[2]
        computed_scale_h = h_gt / h_lq
        computed_scale_w = w_gt / w_lq
        epsilon = 0.05
        self.scale = round(computed_scale_h, 2)
        if not (abs(computed_scale_h - self.scale) < epsilon and abs(computed_scale_w - self.scale) < epsilon):
            raise ValueError(f"Scale mismatch: {computed_scale_h}, {computed_scale_w} vs expected {self.scale}")

        # Tensor 변환
        hr_tensor = torch.from_numpy(hr)
        lr_tensor = torch.from_numpy(lr)
        mask_tensor = torch.from_numpy(mask)

        # 데이터 증강 (HR, LR, mask 동일하게 처리)
        if self.opt.get('use_hflip', False) and random.random() < 0.5:
            hr_tensor = TF.hflip(hr_tensor)
            lr_tensor = TF.hflip(lr_tensor)
            mask_tensor = TF.hflip(mask_tensor)

        if self.opt.get('use_rot', False):
            angle = random.uniform(-15, 15)
            hr_tensor = TF.rotate(hr_tensor, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            lr_tensor = TF.rotate(lr_tensor, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask_tensor = TF.rotate(mask_tensor, angle, interpolation=transforms.InterpolationMode.NEAREST)

        if self.opt.get('use_blur', False) and random.random() < 0.3:
            sigma = random.uniform(0.1, 0.5)
            hr_tensor = TF.gaussian_blur(hr_tensor, kernel_size=3, sigma=sigma)
            lr_tensor = TF.gaussian_blur(lr_tensor, kernel_size=3, sigma=sigma)
            # mask에는 blur 적용 안함

        return {
            'gt': hr_tensor,
            'lq': lr_tensor,
            'mask': mask_tensor,
            'gt_path': str(index),
            'lq_path': str(index)
        }

    def __len__(self):
        return len(self.hr_data)
