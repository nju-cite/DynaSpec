# Video-Level Spectral Compressive Imaging

[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://arxiv.org/abs/xxxx.xxxxx) [![Dataset](https://img.shields.io/badge/Dataset-Download-green)](#dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation, dataset, and benchmark for the paper:

> **Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark** [CVPR 2026] <br>



---

## Introduction
Welcome to the official repository providing the dataset and benchmark for **video-level compressive spectral reconstruction**. 

In our paper, the proposed baseline (PG-SVRT) primarily evaluates the fusion and reconstruction of multi-frame measurements within a CASSI system utilizing a **fixed mask**. Specifically, we systematically investigate the performance gains achieved by introducing temporal propagation into conventional single-frame reconstruction methodologies.

However, beyond our specific baseline, the open-sourced high-quality dynamic hyperspectral image dataset (DynaSpec) is highly versatile. It can be readily adapted to advance research in a variety of other video-level hyperspectral tasks, such as reconstruction in various snapshot hyperspectral imaging systems with dynamic/static modulation.

If you find this repo or dataset useful, please give it a star ⭐ and consider citing our paper in your research. Thank you!

---

## Benchmark and Results

We comprehensively evaluate our proposed model against state-of-the-art (SOTA) algorithms. 

* **Metrics:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Spectral Angle Mapper (SAM), and Spatio-Temporal Reduced Reference Entropic Differences (ST-RRED).
* **Best results** are highlighted in **bold**.

### 1. Reconstruction Performance for Four SCI Architectures

| Metric | PMVIS | SD-CASSI | NDSSI | DD-CASSI |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR ↑** | 28.45 | 37.78 | 37.84 | **41.52** |
| **SSIM ↑** | 0.8456 | 0.9700 | 0.9825 | **0.9893** |
| **SAM ↓** | 5.4162 | 4.0737 | 5.4091 | **3.9084** |
| **ST-RRED ↓** | 459.49 | **23.21** | 91.80 | 23.25 |

### 2. Quantitative Comparisons on the KAIST Testset

| Method | Venue | PSNR ↑ | SSIM ↑ | SAM ↓ | ST-RRED ↓ | Params | GFLOPs | Code |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MST-L | *CVPR 2022* | 39.99 | 0.9881 | 3.8248 | 30.99 | 2.31 M | 28.23 | [Link](#) |
| CST-L | *ECCV 2022* | 39.93 | 0.9864 | 4.1342 | 35.11 | 3.44 M | 28.53 | [Link](#) |
| DAUHST | *NeurIPS 2022* | 38.98 | 0.9832 | 5.4514 | 37.27 | 3.36 M | 35.93 | [Link](#) |
| GAP-Net | *IJCV 2023* | 36.92 | 0.9755 | 6.1204 | 85.34 | 4.28 M | 58.15 | [Link](#) |
| DADF-Plus-3 | *TMI 2023* | 38.23 | 0.9832 | 4.7676 | 48.45 | 20.25 M | 76.33 | [Link](#) |
| RDLUF | *CVPR 2023* | 39.26 | 0.9860 | 4.2932 | 39.06 | 2.17 M | 59.69 | [Link](#) |
| PADUT | *ICCV 2023* | 38.61 | 0.9828 | 4.7154 | 47.19 | 2.57 M | 32.78 | [Link](#) |
| S²-Transfor. | *TPAMI 2024* | 33.26 | 0.9617 | 8.0837 | 155.82 | **1.33 M** | 56.17 | [Link](#) |
| SSR | *CVPR 2024* | 39.04 | 0.9842 | 5.2201 | 38.29 | 2.06 M | 29.92 | [Link](#) |
| DPU | *CVPR 2024* | 40.02 | 0.9856 | 5.2250 | 25.90 | 1.88 M | 31.04 | [Link](#) |
| DPU* | *CVPR 2024* | 40.50 | 0.9853 | 5.1685 | 26.71 | 15.14 M | 77.36 | [Link](#) |
| **PG-SVRT (Ours)** | **[Venue]** | **41.23** | **0.9882** | **3.8050** | **19.35** | 2.48 M | **28.18** | [Here](#) |

### 3. Quantitative Comparisons on the DynaSpec Testset

| Method | Venue | PSNR ↑ | SSIM ↑ | SAM ↓ | ST-RRED ↓ | Params | GFLOPs | Code |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MST-L | *CVPR 2022* | 39.58 | 0.9873 | 4.2208 | 66.31 | 2.31 M | 28.23 | [Link](#) |
| CST-L | *ECCV 2022* | 40.06 | 0.9876 | 4.4578 | 52.19 | 3.44 M | 28.53 | [Link](#) |
| DAUHST | *NeurIPS 2022* | 40.39 | 0.9883 | 4.7962 | 46.64 | 3.36 M | 35.93 | [Link](#) |
| GAP-Net | *IJCV 2023* | 39.38 | 0.9851 | 5.3402 | 67.54 | 4.28 M | 58.15 | [Link](#) |
| DADF-Plus-3 | *TMI 2023* | 39.00 | 0.9861 | 4.6057 | 73.17 | 20.25 M | 76.33 | [Link](#) |
| RDLUF | *CVPR 2023* | 39.26 | 0.9863 | 4.4429 | 70.64 | 2.17 M | 59.69 | [Link](#) |
| PADUT | *ICCV 2023* | 40.41 | 0.9881 | 4.4372 | 48.88 | 2.57 M | 32.78 | [Link](#) |
| S²-Transfor. | *TPAMI 2024* | 37.10 | 0.9786 | 6.0231 | 114.17 | **1.33 M** | 56.17 | [Link](#) |
| SSR | *CVPR 2024* | 39.66 | 0.9873 | 4.6840 | 59.03 | 2.06 M | 29.92 | [Link](#) |
| DPU | *CVPR 2024* | 41.01 | 0.9893 | 4.4732 | 36.84 | 1.88 M | 31.04 | [Link](#) |
| DPU* | *CVPR 2024* | 41.36 | 0.9889 | 4.5997 | 31.20 | 15.14 M | 77.36 | [Link](#) |
| **PG-SVRT (Ours)** | **[Venue]** | **41.82** | **0.9904** | **4.0118** | **27.14** | 2.48 M | **28.18** | [Here](#) |



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
