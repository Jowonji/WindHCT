import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp

@MODEL_REGISTRY.register()
class HATModel(SRModel):

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        import numpy as np
        import os
        import imageio
        import matplotlib.cm as cm
        import os.path as osp

        dataset_name = dataloader.dataset.opt['name']
        stats = np.load(self.opt['datasets']['val']['norm_path'])
        # 안전하게 키 확인 후 가져오기
        if 'hr_min' in stats:
            hr_min = stats['hr_min'].item()
            hr_max = stats['hr_max'].item()
        elif 'min_hr' in stats:
            hr_min = stats['min_hr'].item()
            hr_max = stats['max_hr'].item()
        else:
            raise KeyError("npz 파일에 hr_min/hr_max 또는 min_hr/max_hr 키가 없습니다.")

        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)

            # 🔹 HAT 전용 전처리 및 추론
            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()

            if 'gt' not in visuals:
                raise ValueError(f"GT image is missing for {img_name}. Validation cannot proceed.")

            sr_tensor = visuals['result'].cpu().detach().numpy()
            gt_tensor = visuals['gt'].cpu().detach().numpy()

            # 차원 정리
            if sr_tensor.ndim != 2:
                print(f"⚠ Warning: sr_tensor shape {sr_tensor.shape}, reshaping to (H, W)")
                sr_tensor = sr_tensor.reshape((sr_tensor.shape[-2], sr_tensor.shape[-1]))
            if gt_tensor.ndim != 2:
                print(f"⚠ Warning: gt_tensor shape {gt_tensor.shape}, reshaping to (H, W)")
                gt_tensor = gt_tensor.reshape((gt_tensor.shape[-2], gt_tensor.shape[-1]))

            # 역정규화
            sr_img_rescaled = sr_tensor * (hr_max - hr_min) + hr_min
            gt_img_rescaled = gt_tensor * (hr_max - hr_min) + hr_min

            # ✅ 마스크 존재 시 평가 범위 제한
            if 'mask' in val_data:
                mask = val_data['mask'].squeeze().cpu().numpy()  # (H, W)
                # 마스크 적용 (결측 또는 무효 영역 제외)
                sr_img_rescaled = sr_img_rescaled * mask
                gt_img_rescaled = gt_img_rescaled * mask
            else:
                mask = np.ones_like(sr_img_rescaled)  # 평가 전체 영역으로 처리

            if np.isnan(sr_img_rescaled).any() or np.isnan(gt_img_rescaled).any():
                raise ValueError(f"NaN detected in SR or GT image for {img_name}.")
            if np.isinf(sr_img_rescaled).any() or np.isinf(gt_img_rescaled).any():
                raise ValueError(f"Inf detected in SR or GT image for {img_name}.")

            if sr_img_rescaled.shape != gt_img_rescaled.shape:
                raise ValueError(f"Shape mismatch: SR {sr_img_rescaled.shape} vs GT {gt_img_rescaled.shape} for {img_name}")

            metric_data = {
                'img': sr_img_rescaled,
                'img2': gt_img_rescaled
            }

            # 메모리 정리
            del self.lq, self.output
            torch.cuda.empty_cache()

            # 이미지 저장

            # 5. 결과 이미지 저장
            if save_img and sr_img_rescaled.ndim == 2:
                epsilon = 1e-8

                # 🔹 마스크된 영역만 시각화 (mask == 1: 유효 영역)
                sr_vis = np.where(mask == 1, sr_img_rescaled, np.nan)

                # 🔹 NaN 제외 정규화 (범위: 0~1)
                sr_min = np.nanmin(sr_vis)
                sr_max = np.nanmax(sr_vis)
                sr_img_normalized = (sr_vis - sr_min) / (sr_max - sr_min + epsilon)

                # 🔹 Viridis 컬러맵 적용 → RGBA로 반환됨
                sr_colormap = cm.viridis(sr_img_normalized)

                # 🔹 NaN 영역은 회색으로 설정 (R=200, G=200, B=200)
                sr_colormap[np.isnan(sr_img_normalized)] = [0.78, 0.78, 0.78, 1.0]

                # 🔹 RGB만 추출 후 0~255 정수로 변환
                sr_img_rgb = (sr_colormap[:, :, :3] * 255).astype(np.uint8)

                # 🔄 상하 반전 (이미지 좌표계에 맞추기 위해)
                sr_img_rgb = np.flipud(sr_img_rgb)

                # 🔹 저장 경로 생성
                img_folder = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
                os.makedirs(img_folder, exist_ok=True)
                save_img_path = osp.join(img_folder, f'{current_iter}.png')

                # 🔹 이미지 저장
                try:
                    imageio.imwrite(save_img_path, sr_img_rgb)
                    print(f"✅ Image successfully saved at {save_img_path}")
                except Exception as e:
                    print(f"❌ Failed to save image at {save_img_path}. Error: {e}")

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
