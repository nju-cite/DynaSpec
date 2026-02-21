# Video-Level Spectral Compressive Imaging

[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://arxiv.org/abs/xxxx.xxxxx) [![Dataset](https://img.shields.io/badge/Dataset-Download-green)](#dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation, dataset, and benchmark for the paper:

> **Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark** <br>
> [Conference/Journal CVPR 2026]



---

## Introduction
Welcome to the official repository providing the dataset and benchmark for **video-level compressive spectral reconstruction**. 

In our paper, the proposed baseline (PG-SVRT) primarily evaluates the fusion and reconstruction of multi-frame measurements within a CASSI system utilizing a **fixed mask**. Specifically, we systematically investigate the performance gains achieved by introducing temporal propagation into conventional single-frame reconstruction methodologies.

However, beyond our specific baseline, the open-sourced high-quality dynamic hyperspectral image dataset (DynaSpec) is highly versatile. It can be readily adapted to advance research in a variety of other video-level hyperspectral tasks, such as reconstruction in various snapshot hyperspectral imaging systems with dynamic/static modulation.

If you find this repo or dataset useful, please give it a star ⭐ and consider citing our paper in your research. Thank you!

---

## Benchmark and Results

We comprehensively evaluate our proposed model against state-of-the-art (SOTA) algorithms on our newly proposed dataset. 

* **Metrics:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Spectral Angle Mapper (SAM).

| Method | Venue & Year | PSNR ↑ | SSIM ↑ | SAM ↓ | Params (M) ↓ | FLOPs (G) ↓ | Code |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GAP-TV | *ICIP 2016* | 25.12 | 0.765 | 0.215 | - | - | [Link](#) |
| TSA-Net | *ECCV 2020* | 29.45 | 0.854 | 0.165 | 4.2 | 85.3 | [Link](#) |
| HDNet | *CVPR 2022* | 31.20 | 0.890 | 0.142 | 2.5 | 42.1 | [Link](#) |
| MST | *CVPR 2022* | 31.85 | 0.902 | 0.130 | 3.1 | 55.4 | [Link](#) |
| CST | *CVPR 2023* | 32.45 | 0.915 | 0.125 | 3.0 | 50.2 | [Link](#) |
| **Ours** | **[Venue]** | **34.12** | **0.935** | **0.105** | **2.8** | **45.6** | [Here](#) |

> **Note:** Please replace the dummy numbers above with the actual performance metrics from your paper.



---

## 💾 Dataset

We introduce a large-scale dataset for Video-Level Compressive Spectral Reconstruction. 

### Download Links
* [Google Drive](#) * [Baidu Netdisk](#) (Password: `xxxx`) ### Directory Structure
After downloading and extracting, organize the dataset as follows:
```text
Dataset_Name/
├── train/
│   ├── scene_01.mat
│   ├── scene_02.mat
│   └── ...
├── test/
│   ├── scene_test_01.mat
│   └── ...
└── mask/
    └── mask_video.mat
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/YourUsername/Video-Level-SCI.git](https://github.com/YourUsername/Video-Level-SCI.git)
cd Video-Level-SCI

# Create a conda environment
conda create -n vlsci python=3.9
conda activate vlsci

# Install dependencies
pip install -r requirements.txt
```

### 2. Pre-trained Models
Download our pre-trained checkpoints from [Google Drive](#) or [Baidu Netdisk](#) and place them in the `./checkpoints/` directory.

### 3. Training
To train the model from scratch, run:
```bash
python train.py --data_path ./Dataset_Name --batch_size 4 --epochs 100
```

### 4. Evaluation / Testing
To test the model and reproduce the benchmark results, run:
```bash
python test.py --data_path ./Dataset_Name --checkpoint ./checkpoints/best_model.pth
```

---

## ✒️ Citation
If you find our paper, dataset, or code helpful, please consider citing our work:

```bibtex
@inproceedings{your_bibtex_key,
  title={Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark},
  author={Author1 and Author2 and Author3},
  booktitle={Proceedings of the IEEE/CVF ...},
  year={202X}
}
```

## 🙏 Acknowledgments
This code is built on [TSA-Net](https://github.com/...) and [MST](https://github.com/...). We thank the authors for sharing their codes.

## 📧 Contact
If you have any questions, please feel free to contact `your_email@domain.edu` or open an issue.
