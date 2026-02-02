import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import os
import numpy as np
import matplotlib.cm as cm
import imageio


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device) # EMA 네트워크 생성
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            """입력 텐서를 다양하게 변환(좌우/상하 반전, 전치) 수행"""
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy() # 상하 반전
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy() # 좌우 반전
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy() # 전치

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        stats = np.load(self.opt['datasets']['val']['norm_path'])
        # (Numpy dataset specific) Load min/max statistics from an .npz file and support multiple key names
        if 'hr_min' in stats:
            hr_min = stats['hr_min'].item()
            hr_max = stats['hr_max'].item()
        elif 'min_hr' in stats:
            hr_min = stats['min_hr'].item()
            hr_max = stats['max_hr'].item()
        else:
            raise KeyError("npz file does not contain hr_min/hr_max or min_hr/max_hr keys.")

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
            self.test()

            visuals = self.get_current_visuals()

            if 'gt' not in visuals:
                raise ValueError(f"GT image is missing for {img_name}. Validation cannot proceed.")

            # (Numpy dataset specific) Model outputs represent single-channel scalar fields; force (H, W)
            sr_tensor = visuals['result'].cpu().detach().numpy()
            gt_tensor = visuals['gt'].cpu().detach().numpy()

            if sr_tensor.ndim != 2:
                print(f"Warning: sr_tensor has incorrect shape {sr_tensor.shape}, forcing reshape.")
                sr_tensor = sr_tensor.reshape((sr_tensor.shape[-2], sr_tensor.shape[-1]))
            if gt_tensor.ndim != 2:
                print(f"Warning: gt_tensor has incorrect shape {gt_tensor.shape}, forcing reshape.")
                gt_tensor = gt_tensor.reshape((gt_tensor.shape[-2], gt_tensor.shape[-1]))

            # (Numpy dataset specific) Denormalize back to the original physical value range using hr_min/hr_max
            sr_img_rescaled = sr_tensor * (hr_max - hr_min) + hr_min
            gt_img_rescaled = gt_tensor * (hr_max - hr_min) + hr_min

            # (Numpy dataset specific) Optional mask-based evaluation to exclude invalid/missing regions
            if 'mask' in val_data:
                mask = val_data['mask'].squeeze().cpu().numpy()
                sr_img_rescaled = sr_img_rescaled * mask
                gt_img_rescaled = gt_img_rescaled * mask
            else:
                mask = np.ones_like(sr_img_rescaled)

            # (Numpy dataset specific) Extra safety checks for NaN/Inf that can occur in scientific grids
            if np.isnan(sr_img_rescaled).any() or np.isnan(gt_img_rescaled).any():
                raise ValueError(f"NaN detected in SR or GT image for {img_name}. Check normalization or model output.")
            if np.isinf(sr_img_rescaled).any() or np.isinf(gt_img_rescaled).any():
                raise ValueError(f"Infinite values detected in SR or GT image for {img_name}. Check model stability.")

            if sr_img_rescaled.shape != gt_img_rescaled.shape:
                raise ValueError(f"Shape mismatch: SR {sr_img_rescaled.shape} vs GT {gt_img_rescaled.shape} for {img_name}")

            metric_data = {
                'img': sr_img_rescaled,
                'img2': gt_img_rescaled
            }

            del self.lq, self.output
            torch.cuda.empty_cache()

            # (Numpy dataset specific) Save scalar-field outputs with a colormap and visualize masked regions
            if save_img and sr_img_rescaled.ndim == 2:
                epsilon = 1e-8

                sr_vis = np.where(mask == 1, sr_img_rescaled, np.nan)

                sr_min = np.nanmin(sr_vis)
                sr_max = np.nanmax(sr_vis)
                sr_img_normalized = (sr_vis - sr_min) / (sr_max - sr_min + epsilon)

                sr_colormap = cm.viridis(sr_img_normalized)

                # Mark masked/NaN regions as gray
                sr_colormap[np.isnan(sr_img_normalized)] = [0.78, 0.78, 0.78, 1.0]

                sr_img_rgb = (sr_colormap[:, :, :3] * 255).astype(np.uint8)

                # Flip vertically to match the dataset's coordinate convention (grid origin vs image origin)
                sr_img_rgb = np.flipud(sr_img_rgb)

                img_folder = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
                os.makedirs(img_folder, exist_ok=True)
                save_img_path = osp.join(img_folder, f'{current_iter}.png')

                try:
                    imageio.imwrite(save_img_path, sr_img_rgb)
                    print(f"Image successfully saved at {save_img_path}")
                except Exception as e:
                    print(f"Failed to save image at {save_img_path}. Error: {e}")

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

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)