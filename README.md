# WindHCT

**WindHCT** is a CNN-Transformer hybrid super-resolution model for reconstructing high-resolution wind fields from coarse-resolution inputs.

---

## Overview

WindHCT addresses the challenge of wind field super-resolution by combining global context modeling with local detail reconstruction. The model features:

- **Context-Local Fusion Block (CLFB)** — Parallel branches for global flow patterns and local turbulence structures
- **Edge-Aware Window Mask (EAWM)** — Learnable Sobel-initialized masks to guide attention toward high-frequency regions
- **Adaptive Fusion Gate** — Position-wise weighted combination of context and local features
- **5× Upsampling Module** — Single-stage PixelShuffle-based upsampling for non-standard scale factors

---

## Datasets

We evaluate WindHCT on two complementary datasets with 5× upscaling:

| Attribute | WIND Toolkit | ERA5–CERRA |
|-----------|--------------|------------|
| **Region** | Continental United States | Southern Europe (Italy) |
| **Source** | NREL | ECMWF / C3S |
| **HR Resolution** | ~2 km | ~5.5 km (CERRA) |
| **LR Resolution** | ~10 km (5× downsampled) | ~30 km (ERA5) |
| **Time Interval** | 4-hr (random sampling) | 3-hr (fixed timestamps) |
| **Data Type** | Simulated wind field | Reanalysis-based wind field |
| **Pairs** | 40,000 | 40,912 |

### WIND Toolkit Dataset

- **Source**: [NREL WIND Toolkit](https://www.nrel.gov/grid/wind-toolkit.html) — 100m wind speed variable
- **Period**: 2007–2013, 1,000 randomly sampled timestamps at 4-hour intervals
- **Processing**: 1600×1600 grid → 100×100 HR patches → 20×20 LR patches via average pooling
- **Filtering**: Excluded patches with mean wind speed < 0.5 m/s or missing values
- **Split**: 70% train / 15% validation / 15% test (random)
- **Reference**: [WiSoSuper data.ipynb](https://github.com/RupaKurinchiVendhan/WiSoSuper/blob/main/data.ipynb)

### ERA5–CERRA Dataset

- **Sources**: [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) (~30 km) and [CERRA](https://climate.copernicus.eu/copernicus-regional-reanalysis-europe-cerra) (~5.5 km)
- **Region**: Southern Europe (Italy domain), following [Merizzi et al.](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models)
- **Wind Speed**: Computed as √(u₁₀² + v₁₀²)
- **Processing**: ERA5 52×52 patches → bilinear interpolation to 51×51 (for integer upscaling)
- **Split**: 2010–2019 train / 2020–2021 validation / 2022–2023 test (temporal)

### Normalization

All data is min-max normalized to [0, 1] during training:

```
x_norm = (x - x_min) / (x_max - x_min)
```

---

## Model Architecture

> 📌 *Architecture diagram will be added soon.*


## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Data Preparation

Run the preprocessing notebook to generate datasets:

```
windtoolkit-prep.ipynb
```

### Training

```bash
# WIND Toolkit dataset
python basicsr/train.py -opt options/train/WindHCT_WINDToolkit.yml

# ERA5-CERRA dataset
python basicsr/train.py -opt options/train/WindHCT_ERA5toCEERRA.yml
```

Logs and checkpoints → `experiments/`

### Testing

```bash
# WIND Toolkit dataset
python basicsr/test.py -opt options/test/test_WINDToolkit.yml

# ERA5-CERRA dataset
python basicsr/test.py -opt options/test/test_ERA5toCERRA.yml
```

Results and metrics → `results/`

---

## Repository Structure

```
WindHCT/
├── basicsr/                  # Model architectures and training engine
├── datasets/                 # Dataset metadata
├── experiments/              # Training logs and checkpoints
├── options/                  # Configuration files (YAML)
├── results/                  # Inference outputs
├── windtoolkit-prep.ipynb    # WINDToolkit Dataset generation notebook
├── requirements.txt
└── README.md
```

---

## Acknowledgements

This work builds upon:

- [BasicSR](https://github.com/XPixelGroup/BasicSR) — Training framework
- [WiSoSuper](https://github.com/RupaKurinchiVendhan/WiSoSuper) — HSDS data access patterns
- [ERA5-to-CERRA via Diffusion Models](https://github.com/fmerizzi/ERA5-to-CERRA-via-Diffusion-Models) — ERA5-CERRA preprocessing reference
- [NREL WIND Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) — Window-based attention design

---
