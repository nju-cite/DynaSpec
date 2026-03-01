


<div align="center">

# DynaSpec

*Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark*, CVPR 2026


<p align="middle">
  <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Visitors">
  <a href='https://arxiv.org/abs/0000.0000'>
  <img src='https://img.shields.io/badge/Arxiv-0000.0000-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://huggingface.co/datasets/Flipped99/DynaSpec'> 
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>
  <a href="https://visitor-badge.laobi.icu/badge?page_id=nju-cite.DynaSpec">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=nju-cite.DynaSpec&left_text=VISITORS&left_color=gray&right_color=%2342b983" alt="Visitors">
   </a>
  <img src="https://img.shields.io/github/last-commit/nju-cite/DynaSpec">

</p>


</div>

<br>

&emsp;This repository contains the official PyTorch implementation (**PG-SVRT**), dataset (**DynaSpec**), and benchmark for the paper. The primary objective of this work is to advance compressive spectral imaging from traditional image-level reconstruction (i.e., **reconstructing HSIs from a single-frame measurement**) to video-level reconstruction (i.e., **reconstructing HSIs by fusing multi-frame measurements across the temporal domain**). In the paper, the proposed baseline (PG-SVRT) primarily evaluates the reconstruction of multi-frame measurements within a CASSI system utilizing a **fixed mask**. As shown in the video, video-level reconstruction can effectively enhance completeness, improve reconstruction accuracy and temporal consistency, and reduce flickering.

<!-- 进阶版：适配不同屏幕，精准居中+控大小 -->
<div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
  <video 
    style="
      max-width: 90%;  /* 最大宽度占屏幕90%，适配小屏幕 */
      width: 600px;    /* 电脑端固定宽度，超过90%时自动缩小 */
      height: auto;    /* 保持宽高比 */
      border-radius: 8px; /* 可选：给视频加圆角，更美观 */
    "
    controls 
    muted 
    loop 
    autoplay
    src="https://github.com/user-attachments/assets/10cd541a-90e3-4d9e-89df-b5da1439fc26.mp4"
  >
    你的浏览器不支持视频播放
  </video>
</div>

&emsp;However, beyond our specific baseline, the open-sourced high-quality dynamic hyperspectral images dataset (DynaSpec) is highly versatile. For example, it can be readily adapted to advance research in a variety of other video-level hyperspectral tasks, such as reconstruction in various snapshot hyperspectral imaging systems with either adaptive or fixed modulation. It can also serve as approximately clean data for hyperspectral video denoising tasks.

&emsp;If you find this repo or dataset useful, please give it a star ⭐ and consider citing our paper in your research. Thank you!

---
## 💾 Dataset

&emsp;We construct a dynamic hyperspectral image dataset, named DynaSpec. We employ GaiaField push-broom hyperspectral camera to capture controllable objects frame-by-frame, covering the 400-700nm spectral range with a spectral resolution of 2 nm. Diverse motions are then manually introduced to emulate the high degrees of freedom encountered in real-world scenarios.

![image](https://github.com/nju-cite/DynaSpec/blob/main/assets/dynaspec_demo2.gif)

### Download Links
* [Google Drive](#) * [Baidu Netdisk](https://pan.baidu.com/s/19ycdo8N0il_4CInJSzQH4g?pwd=9xze) 

&emsp;After downloading, the datasets are organized as follows. Each subfolder stores a hyperspectral video sequence respectively. For example, '00001.mat' to '00010.mat' represent frames 1 to 10 of the hyperspectral video. Each '.mat' file contains a uint16 hyperspectral image variable named 'img' with dimensions 1024×1024×151, as well as a 1×151 double variable named 'wavelength'.
```text
Dyna_Spec_release/
├── animal_garden/
│   ├── 00001.mat
│   ├── 00002.mat
│   └── ...
├── animal_plus_plus/
│   ├── 00001.mat
│   ├── 00002.mat
│   └── ...
└── ...
```

---

## 🚀 Getting Started
Code and models will be available by the latest, on the camera ready date of CVPR 2026.

---
<!-- 
## Benchmark and Results

We comprehensively evaluate our proposed model against state-of-the-art (SOTA) algorithms. 

* **Metrics:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Spectral Angle Mapper (SAM), and Spatio-Temporal Reduced Reference Entropic Differences (ST-RRED).
* **Best results** are highlighted in **bold**.

### 1. Video-level Reconstruction Performance for Representative SCI Architectures 

| Metric | PMVIS | SD-CASSI | NDSSI | DD-CASSI |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR ↑** | 28.45 | 37.78 | 37.84 | **41.52** |
| **SSIM ↑** | 0.8456 | 0.9700 | 0.9825 | **0.9893** |
| **SAM ↓** | 5.4162 | 4.0737 | 5.4091 | **3.9084** |
| **ST-RRED ↓** | 459.49 | **23.21** | 91.80 | 23.25 |

### 2. Quantitative Comparisons on the KAIST Testset (DD-CASSI, Video-level vs. Image-level)

| Method | Venue | PSNR ↑ | SSIM ↑ | SAM ↓ | ST-RRED ↓ | Params | GFLOPs | Code |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PG-SVRT (Ours)** | **[Venue]** | **41.23** | **0.9882** | **3.8050** | **19.35** | 2.48 M | **28.18** | [Here](#) |
| DPU* | *CVPR 2024* | 40.50 | 0.9853 | 5.1685 | 26.71 | 15.14 M | 77.36 | [Link](#) |
| DPU | *CVPR 2024* | 40.02 | 0.9856 | 5.2250 | 25.90 | 1.88 M | 31.04 | [Link](#) |
| MST-L | *CVPR 2022* | 39.99 | 0.9881 | 3.8248 | 30.99 | 2.31 M | 28.23 | [Link](#) |
| CST-L | *ECCV 2022* | 39.93 | 0.9864 | 4.1342 | 35.11 | 3.44 M | 28.53 | [Link](#) |
| RDLUF | *CVPR 2023* | 39.26 | 0.9860 | 4.2932 | 39.06 | 2.17 M | 59.69 | [Link](#) |
| SSR | *CVPR 2024* | 39.04 | 0.9842 | 5.2201 | 38.29 | 2.06 M | 29.92 | [Link](#) |
| DAUHST | *NeurIPS 2022* | 38.98 | 0.9832 | 5.4514 | 37.27 | 3.36 M | 35.93 | [Link](#) |
| PADUT | *ICCV 2023* | 38.61 | 0.9828 | 4.7154 | 47.19 | 2.57 M | 32.78 | [Link](#) |
| DADF-Plus-3 | *TMI 2023* | 38.23 | 0.9832 | 4.7676 | 48.45 | 20.25 M | 76.33 | [Link](#) |
| GAP-Net | *IJCV 2023* | 36.92 | 0.9755 | 6.1204 | 85.34 | 4.28 M | 58.15 | [Link](#) |
| S²-Transfor. | *TPAMI 2024* | 33.26 | 0.9617 | 8.0837 | 155.82 | **1.33 M** | 56.17 | [Link](#) |

### 3. Quantitative Comparisons on the DynaSpec Testset (DD-CASSI, Video-level vs. Image-level)

| Method | Venue | PSNR ↑ | SSIM ↑ | SAM ↓ | ST-RRED ↓ | Params | GFLOPs | Code |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PG-SVRT (Ours)** | **[Venue]** | **41.82** | **0.9904** | **4.0118** | **27.14** | 2.48 M | **28.18** | [Here](#) |
| DPU* | *CVPR 2024* | 41.36 | 0.9889 | 4.5997 | 31.20 | 15.14 M | 77.36 | [Link](#) |
| DPU | *CVPR 2024* | 41.01 | 0.9893 | 4.4732 | 36.84 | 1.88 M | 31.04 | [Link](#) |
| PADUT | *ICCV 2023* | 40.41 | 0.9881 | 4.4372 | 48.88 | 2.57 M | 32.78 | [Link](#) |
| DAUHST | *NeurIPS 2022* | 40.39 | 0.9883 | 4.7962 | 46.64 | 3.36 M | 35.93 | [Link](#) |
| CST-L | *ECCV 2022* | 40.06 | 0.9876 | 4.4578 | 52.19 | 3.44 M | 28.53 | [Link](#) |
| SSR | *CVPR 2024* | 39.66 | 0.9873 | 4.6840 | 59.03 | 2.06 M | 29.92 | [Link](#) |
| MST-L | *CVPR 2022* | 39.58 | 0.9873 | 4.2208 | 66.31 | 2.31 M | 28.23 | [Link](#) |
| GAP-Net | *IJCV 2023* | 39.38 | 0.9851 | 5.3402 | 67.54 | 4.28 M | 58.15 | [Link](#) |
| RDLUF | *CVPR 2023* | 39.26 | 0.9863 | 4.4429 | 70.64 | 2.17 M | 59.69 | [Link](#) |
| DADF-Plus-3 | *TMI 2023* | 39.00 | 0.9861 | 4.6057 | 73.17 | 20.25 M | 76.33 | [Link](#) |
| S²-Transfor. | *TPAMI 2024* | 37.10 | 0.9786 | 6.0231 | 114.17 | **1.33 M** | 56.17 | [Link](#) |

For more comprehensive experimental results, visual comparisons, and detailed analysis, please refer to our full paper.

---
## 🚀 Getting Started
All code and models will be available by the latest, on the camera ready date of CVPR 2026.
### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/YourUsername/Video-Level-SCI.git](https://github.com/nju-cite/DynaSpec.git)
cd xxx

# Create a conda environment
conda create -n xxx python=3.9
conda activate xxx

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
-->
## 🙏 Acknowledgments
This code is built on [MST](https://github.com/caiyuanhao1998/MST). We thank the authors for sharing their codes.

## 📧 Contact
If you have any questions, please feel free to contact `cailijing@smail.nju.edu.cn` or `shizhan@smail.nju.edu.cn` or open an issue.
