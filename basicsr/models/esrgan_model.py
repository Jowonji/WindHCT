import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils import get_root_logger
from basicsr.archs import build_network
from basicsr.losses import build_loss

@MODEL_REGISTRY.register()
class ESRGANModel_V2(SRGANModel):
    """풍속 데이터 초해상화 (SR) 최적화 ESRGAN Model"""
    def init_training_settings(self):
        train_opt = self.opt['train']

        # EMA 설정
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            self.model_ema(0)
            self.net_g_ema.eval()

        # 판별기 정의
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # --- ✅ 손실 함수 및 가중치 분리 저장 ---
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if 'pixel_opt' in train_opt else None
        self.pix_weight = train_opt['pixel_opt'].get('loss_weight', 0.0) if 'pixel_opt' in train_opt else 0.0

        self.cri_patch = build_loss(train_opt['patch_opt']).to(self.device) if 'patch_opt' in train_opt else None
        self.patch_weight = train_opt['patch_opt'].get('loss_weight', 0.0) if 'patch_opt' in train_opt else 0.0

        self.cri_wavelet = build_loss(train_opt['wavelet_opt']).to(self.device) if 'wavelet_opt' in train_opt else None
        self.wavelet_weight = train_opt['wavelet_opt'].get('loss_weight', 0.0) if 'wavelet_opt' in train_opt else 0.0

        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        self.gan_weight = train_opt['gan_opt'].get('loss_weight', 0.0)

        # 판별기 학습 스케줄 설정
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.setup_optimizers()
        self.setup_schedulers()

        # 로깅
        logger = get_root_logger()
        logger.info(f'Loss initialized - Pix: {self.pix_weight}, Patch: {self.patch_weight}, '
                    f'Wavelet: {self.wavelet_weight}, GAN: {self.gan_weight}')

    def wavelet_transform(self, img):
        haar_high_filter = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2).to(img.device)
        img = F.pad(img, (1, 1, 1, 1), mode='reflect')
        high_freq = F.conv2d(img, haar_high_filter, stride=2)
        return torch.abs(high_freq)

    def optimize_parameters(self, current_iter):
        # 🔒 판별기 freeze
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        loss_dict = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt) * self.pix_weight
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix.detach()

            if self.cri_patch:
                l_g_patch = self.cri_patch(self.output, self.gt) * self.patch_weight
                l_g_total += l_g_patch
                loss_dict['l_g_patch'] = l_g_patch.detach()

            if self.cri_wavelet:
                l_g_wavelet = self.cri_wavelet(self.output, self.gt) * self.wavelet_weight
                l_g_total += l_g_wavelet
                loss_dict['l_g_wavelet'] = l_g_wavelet.detach()

            #real_hf = self.wavelet_transform(self.gt)
            #fake_hf = self.wavelet_transform(self.output)
            #real_d_patch = self.net_d(real_hf)
            #fake_d_patch = self.net_d(fake_hf)

            real_d_patch = self.net_d(self.gt)
            fake_d_patch = self.net_d(self.output)

            l_g_real = self.cri_gan(real_d_patch.detach() - torch.mean(fake_d_patch), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_d_patch - torch.mean(real_d_patch.detach()), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2 * self.gan_weight

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan.detach()

            # 역전파
            l_g_total.backward()
            self.optimizer_g.step()

        # 🔓 판별기 업데이트
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        fake_d_patch = self.net_d(self.output.detach())
        real_d_patch = self.net_d(self.gt)
        #fake_hf = self.wavelet_transform(self.output.detach())
        #real_hf = self.wavelet_transform(self.gt)
        #fake_d_patch = self.net_d(fake_hf)
        #real_d_patch = self.net_d(real_hf)

        l_d_real = self.cri_gan(real_d_patch - torch.mean(fake_d_patch.detach()), True, is_disc=True) * 0.5
        l_d_fake = self.cri_gan(fake_d_patch - torch.mean(real_d_patch.detach()), False, is_disc=True) * 0.5
        l_d_total = l_d_real + l_d_fake

        l_d_total.backward()
        self.optimizer_d.step()

        # ✅ 로깅
        loss_dict['l_d_real'] = torch.clamp(l_d_real, min=1e-8).detach()
        loss_dict['l_d_fake'] = torch.clamp(l_d_fake, min=1e-8).detach()
        loss_dict['out_d_real'] = torch.mean(real_d_patch.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_patch.detach())

        for key, value in loss_dict.items():
            if isinstance(value, float):
                loss_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device).detach()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA 업데이트
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN 모델: 단일 이미지 초해상도(Single Image Super-Resolution)를 위한 모델."""

    def optimize_parameters(self, current_iter):
        # -------------------------------
        # 생성기 네트워크(net_g) 최적화
        # -------------------------------
        # 판별기 네트워크(net_d)의 모든 매개변수를 고정 (requires_grad=False)
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 생성기 네트워크의 기울기 초기화
        self.optimizer_g.zero_grad()

        # 입력 저해상도 이미지를 사용해 생성기 출력 계산
        self.output = self.net_g(self.lq)

        # 생성기의 총 손실 초기화
        l_g_total = 0
        loss_dict = OrderedDict()

        # 생성기 학습 조건: 판별기의 초기화 단계가 끝났고, 지정된 반복 주기에 해당할 경우
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 1. 픽셀 손실 계산
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)  # 생성된 이미지와 실제 고해상도 이미지 비교
                l_g_total += l_g_pix  # 총 손실에 추가
                loss_dict['l_g_pix'] = l_g_pix  # 손실 정보 저장

            # 2. Perceptual 손실 및 스타일 손실 계산
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)  # 생성된 이미지와 실제 이미지 비교
                if l_g_percep is not None:
                    l_g_total += l_g_percep  # Perceptual 손실 추가
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style  # 스타일 손실 추가
                    loss_dict['l_g_style'] = l_g_style

            # 3. GAN 손실 계산 (Relativistic GAN)
            real_d_pred = self.net_d(self.gt).detach()  # 실제 이미지에 대한 판별기의 출력
            fake_g_pred = self.net_d(self.output)  # 생성된 이미지에 대한 판별기의 출력

            # Relativistic GAN 손실 계산
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan  # 총 손실에 GAN 손실 추가
            loss_dict['l_g_gan'] = l_g_gan

            # 역전파 및 생성기 최적화
            l_g_total.backward()
            self.optimizer_g.step()

        # -------------------------------
        # 판별기 네트워크(net_d) 최적화
        # -------------------------------
        # 판별기의 매개변수 업데이트를 허용 (requires_grad=True)
        for p in self.net_d.parameters():
            p.requires_grad = True

        # 판별기의 기울기 초기화
        self.optimizer_d.zero_grad()

        # Relativistic GAN 손실 계산
        # - 분산 학습 환경에서 발생할 수 있는 오류를 방지하기 위해
        #   실제(real)와 가짜(fake)의 역전파를 분리하여 실행

        # 1. 실제 이미지 손실 계산
        fake_d_pred = self.net_d(self.output).detach()  # 생성된 이미지에 대한 판별기 출력 (역전파 제외)
        real_d_pred = self.net_d(self.gt)  # 실제 이미지에 대한 판별기 출력
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()  # 역전파 실행

        # 2. 생성된 이미지 손실 계산
        fake_d_pred = self.net_d(self.output.detach())  # 생성된 이미지 출력 (역전파 제외)
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()  # 역전파 실행

        # 판별기 최적화
        self.optimizer_d.step()

        # 손실 값 저장
        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())  # 실제 이미지에 대한 평균 출력 저장
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())  # 생성된 이미지에 대한 평균 출력 저장

        # 손실 값을 로그에 기록
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # -------------------------------
        # EMA(Exponential Moving Average) 업데이트
        # -------------------------------
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
