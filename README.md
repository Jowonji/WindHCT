# WindHCT

[paper] [arXiv] [pretrained models] — *to be added*

---

> **Abstract:** *WindHCT is a Transformer-based architecture for statistically downscaling low-resolution
> wind-speed fields at 5× super-resolution. It stacks Residual Groups that fuse a window-attention Context
> Branch with a multi-scale convolutional Local Branch through a learned gate. WindHCT is evaluated on
> two reanalysis-to-observation downscaling tasks, ERA5→CERRA and WINDToolkit.*

<!-- teaser figure — to be added -->

---

## Dependencies

- Python 3.9
- PyTorch 2.1 (CUDA 11.8)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
git clone git@github.com:Jowonji/WindHCT.git
cd WindHCT
pip install -r requirements.txt
python setup.py develop
```

## TODO

- [x] ERA5 → CERRA (5×)
- [x] WINDToolkit (5×)
- [ ] Release pretrained model weights
- [ ] Release paper / arXiv

## Contents

1. [Datasets](#datasets)
1. [Models](#models)
1. [Training](#training)
1. [Testing](#testing)
1. [Results](#results)
1. [Citation](#citation)
1. [Acknowledgements](#acknowledgements)

---

## Datasets

- **ERA5 → CERRA** (5×): ERA5 reanalysis wind fields downscaled to CERRA reanalysis resolution.
- **WINDToolkit** (5×): coarsened low-resolution fields downscaled to native WINDToolkit resolution, built
  from the NREL WIND Toolkit via [`datasets/WINDToolkit/prepare_windtoolkit.ipynb`](datasets/WINDToolkit/prepare_windtoolkit.ipynb).

The configs expect paired LR/HR numpy arrays under `datasets/ERA5toCERRA/` and `datasets/WINDToolkit/`
(`{split}_hr_norm.npy`, `{split}_lr_norm.npy`, plus a normalization `.npz`) — update the `dataroot_gt` /
`dataroot_lq` / `norm_path` entries in the yml files if your data lives elsewhere.

ERA5→CERRA data preparation notes — *to be added*.

## Models

Pretrained model weights and quantitative results — *to be added*.

## Training

```bash
# ERA5 → CERRA
python basicsr/train.py -opt options/ERA5toCERRA/train/WindHCT.yml

# WINDToolkit
python basicsr/train.py -opt options/WINDToolkit/train/WindHCT.yml
```

The training experiment is written to `experiments/`.

## Testing

```bash
# ERA5 → CERRA
python basicsr/test_wind.py -opt options/ERA5toCERRA/test/WindHCT.yml

# WINDToolkit
python basicsr/test_wind.py -opt options/WINDToolkit/test/WindHCT.yml
```

The output is written to `results/`.

## Results

<details>
<summary>Quantitative results (click to expand)</summary>

*Coming soon.*

</details>

<details>
<summary>Qualitative results (click to expand)</summary>

*Patch-level SR and error-map comparisons — to be added.*

</details>

## Citation

*Coming soon.*

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR). The architecture draws on ideas from
[Coordinate Attention](https://github.com/houqb/CoordAttention) (Hou et al., CVPR 2021) and
[Swin Transformer](https://github.com/microsoft/Swin-Transformer) (Liu et al., ICCV 2021).
