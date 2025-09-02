import logging
import torch
import numpy as np
import os
import imageio
from os import path as osp
from skimage.metrics import structural_similarity as ssim  # SSIM 계산용
from tqdm import tqdm  # ✅ tqdm 추가

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
import matplotlib.cm as cm

# 📌 MAE 계산 함수
def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

# 📌 PSNR 계산 함수
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # MSE가 0이면 PSNR은 무한대
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

# 📌 SSIM 계산 함수
def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=255)

# 📌 RMSE 계산 함수
def calculate_rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # 데이터셋 로드
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model = build_model(opt)
    metric_results = {'psnr': 0, 'ssim': 0, 'rmse': 0, 'mae': 0}
    total_images = 0

    try:
        for test_loader in test_loaders:
            dataset_name = test_loader.dataset.opt['name']

            # ✅ norm_path로부터 min/max 가져오기
            norm_path = test_loader.dataset.opt.get('norm_path', None)
            if norm_path is None or not osp.exists(norm_path):
                raise FileNotFoundError(f"❌ norm_path not found or not specified for dataset {dataset_name}: {norm_path}")
            stats = np.load(norm_path)
            if 'hr_min' in stats:
                hr_min = stats['hr_min'].item()
                hr_max = stats['hr_max'].item()
            elif 'min_hr' in stats:
                hr_min = stats['min_hr'].item()
                hr_max = stats['max_hr'].item()
            logger.info(f"🔹 Starting test for dataset: {dataset_name} with hr_min={hr_min}, hr_max={hr_max}")


            # ✅ tqdm 추가
            for idx, val_data in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Processing {dataset_name}"):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

                model.feed_data(val_data)
                model.test()
                visuals = model.get_current_visuals()

                if 'gt' not in visuals:
                    raise ValueError(f"❌ GT image is missing for {img_name}. Validation cannot proceed.")

                sr_tensor = visuals['result'].cpu().detach().numpy()
                gt_tensor = visuals['gt'].cpu().detach().numpy()

                if sr_tensor.ndim != 2:
                    sr_tensor = sr_tensor.reshape((sr_tensor.shape[-2], sr_tensor.shape[-1]))
                if gt_tensor.ndim != 2:
                    gt_tensor = gt_tensor.reshape((gt_tensor.shape[-2], gt_tensor.shape[-1]))

                sr_img_rescaled = sr_tensor * (hr_max - hr_min) + hr_min
                gt_img_rescaled = gt_tensor * (hr_max - hr_min) + hr_min

                psnr_value = calculate_psnr(
                    ((sr_img_rescaled - hr_min) / (hr_max - hr_min) * 255).astype(np.float32),
                    ((gt_img_rescaled - hr_min) / (hr_max - hr_min) * 255).astype(np.float32)
                )
                ssim_value = calculate_ssim(
                    ((sr_img_rescaled - hr_min) / (hr_max - hr_min) * 255).astype(np.float32),
                    ((gt_img_rescaled - hr_min) / (hr_max - hr_min) * 255).astype(np.float32)
                )
                rmse_value = calculate_rmse(sr_img_rescaled, gt_img_rescaled)
                mae_value = calculate_mae(sr_img_rescaled, gt_img_rescaled)

                metric_results['psnr'] += psnr_value
                metric_results['ssim'] += ssim_value
                metric_results['rmse'] += rmse_value
                metric_results['mae'] += mae_value
                total_images += 1

                if opt['val']['save_img']:
                    save_img_path = osp.join(opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    os.makedirs(osp.dirname(save_img_path), exist_ok=True)

                    epsilon = 1e-8
                    sr_img_normalized = (sr_img_rescaled - sr_img_rescaled.min()) / (
                        max(sr_img_rescaled.max() - sr_img_rescaled.min(), epsilon)
                    )
                    sr_img_colormap = cm.viridis(sr_img_normalized)[:, :, :3]
                    sr_img_colormap = (sr_img_colormap * 255).astype(np.uint8)
                    sr_img_colormap = np.flipud(sr_img_colormap)

                    imageio.imwrite(save_img_path, sr_img_colormap)

                save_gt_path = osp.join(opt['path']['gt_visualization'], dataset_name, f'{img_name}.png')
                os.makedirs(osp.dirname(save_gt_path), exist_ok=True)

                gt_img_normalized = (gt_img_rescaled - gt_img_rescaled.min()) / (
                    max(gt_img_rescaled.max() - gt_img_rescaled.min(), 1e-8)
                )
                gt_img_colormap = cm.viridis(gt_img_normalized)[:, :, :3]
                gt_img_colormap = (gt_img_colormap * 255).astype(np.uint8)
                gt_img_colormap = np.flipud(gt_img_colormap)

                imageio.imwrite(save_gt_path, gt_img_colormap)

                if opt['val'].get('save_npy', False):
                    save_npy_path = osp.join(opt['path']['npy_results'], dataset_name, f'{img_name}.npy')
                    os.makedirs(osp.dirname(save_npy_path), exist_ok=True)
                    np.save(save_npy_path, sr_img_rescaled)

            logger.info(f"✅ Completed dataset: {dataset_name}")

        if total_images > 0:
            for metric in metric_results.keys():
                metric_results[metric] /= total_images
            logger.info(f"📊 Final average metrics: {metric_results}")

    except KeyboardInterrupt:
        if total_images > 0:
            for metric in metric_results.keys():
                metric_results[metric] /= total_images
            logger.info(f"🛑 Test interrupted! Partial average metrics: {metric_results}")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)