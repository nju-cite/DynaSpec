


<div align="center">

# DynaSpec

*[Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark](https://arxiv.org/abs/2603.00611)* (CVPR 2026)


<p align="middle">
  <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Visitors">
  <a href='https://arxiv.org/abs/2603.00611'>
  <img src='https://img.shields.io/badge/Arxiv-2603.00611-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://huggingface.co/datasets/Flipped99/DynaSpec'> 
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
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

If you find this repo or dataset useful, please give it a star ⭐ and consider citing our paper in your research. Thank you!

---
## 💾 Dataset

&emsp;We construct a dynamic hyperspectral image dataset, named DynaSpec. We employ GaiaField push-broom hyperspectral camera to capture controllable objects frame-by-frame, covering the 400-700nm spectral range with a spectral resolution of 2 nm. Diverse motions are then manually introduced to emulate the high degrees of freedom encountered in real-world scenarios.

![image](https://github.com/nju-cite/DynaSpec/blob/main/assets/dynaspec_demo2.gif)

### Download Links (Raw Data)
* [Hugging Face](https://huggingface.co/datasets/Flipped99/DynaSpec) * [Baidu Netdisk](https://pan.baidu.com/s/19ycdo8N0il_4CInJSzQH4g?pwd=9xze)

&emsp;The raw DynaSpec dataset contains full-resolution hyperspectral video sequences captured by GaiaField push-broom camera. Each subfolder stores one video sequence. For example, `00001.mat` to `00010.mat` represent frames 1 to 10. Each `.mat` file contains a uint16 hyperspectral image variable named `img` with dimensions 1024×1024×151, as well as a 1×151 double variable named `wavelength`.
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

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/nju-cite/DynaSpec.git
cd DynaSpec

# Create a conda environment (Python 3.10, tested)
conda create -n dynaspec python=3.10
conda activate dynaspec

# Install PyTorch (CUDA 12.1, tested)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# For other CUDA versions, see https://pytorch.org/get-started/previous-versions/

# Install other dependencies
cd train_code
pip install -r requirements.txt
```

### 2. Data Preparation

For training and evaluation, we provide pre-processed benchmark data derived from [CAVE](https://cave.cs.columbia.edu/repository/Multispectral), [KAIST](http://vclab.kaist.ac.kr/siggraphasia2017p1/), and DynaSpec. For details on the pre-processing pipeline, please refer to our paper. This is **different** from the raw DynaSpec dataset above.

Download the pre-processed data from [Baidu Netdisk](https://pan.baidu.com/s/19ycdo8N0il_4CInJSzQH4g?pwd=9xze) and organize as follows:

```text
<data_root>/
├── CAVE_512_50030/                              # CAVE static scenes (training)
├── Dyna_Spec_train_320_ref_norm_50030_train/    # DynaSpec dynamic scenes (training)
├── Kaist_Truth_50030/                           # KAIST static scenes (testing)
└── Dyna_Spec_train_320_ref_norm_50030_test/     # DynaSpec dynamic scenes (testing)
```

### 3. Pre-trained Models

Download the pretrained model zoo from [Baidu Netdisk](https://pan.baidu.com/s/19ycdo8N0il_4CInJSzQH4g?pwd=9xze) and place them to `./model_zoo/`:

```text
model_zoo/
└── pgdsrt/
    └── model_epoch_80.pth       # PG-SVRT checkpoint (DD-CASSI)
```

### 4. Training

All training is launched from the `train_code/` directory. The two key arguments `--dispersion` and `--mask_pattern` define the imaging system:

| Imaging System | `--dispersion` | `--mask_pattern` | Model Variant |
| :--- | :---: | :---: | :---: |
| DD-CASSI (default) | `dual` | `random` | PGDSRT_dd |
| NDSSI | `dual` | `notch` | PGDSRT_dd |
| SD-CASSI | `single` | `random` | PGDSRT_sd |
| PMVIS | `single` | `pmvis` | PGDSRT_sd |

**Train DD-CASSI** (default configuration):

```bash
cd train_code
python train.py --data_root /path/to/your/data_root
```

**Train SD-CASSI:**

```bash
python train.py --dispersion single --mask_pattern random --data_root /path/to/your/data_root
```

**Train NDSSI:**

```bash
python train.py --dispersion dual --mask_pattern notch --data_root /path/to/your/data_root
```

**Train PMVIS:**

```bash
python train.py --dispersion single --mask_pattern pmvis --data_root /path/to/your/data_root
```

**Resume training from a checkpoint:**

```bash
python train.py --data_root /path/to/your/data_root --pretrained_model_path ./exp/pgdsrt/xxx/model/model_epoch_40.pth --begin_epoch 41
```

After training, all outputs are saved under `./exp/pgdsrt/<timestamp>/` (or `./exp/pgdsrt_<tag>/<timestamp>/` if `--tag` is specified):

```text
exp/pgdsrt/<timestamp>/
├── model/
│   ├── log.txt                          # Full training log
│   ├── model_epoch_79.pth               # Checkpoint (last 2 epochs are always saved)
│   ├── model_epoch_80.pth
│   ├── model_epoch_<best>.pth           # Best checkpoint (saved when PSNR improves)
│   └── E_epoch_<best>_PSNR_xx.xx.txt   # Per-scene evaluation at best epoch
├── result/
│   └── Test_80_xx.xx_x.xxxx.mat        # Reconstruction results (.mat, last 2 epochs)
└── evaluation.txt                       # Per-scene evaluation (updated every epoch)
```

Key training options (see `option.py` for full list):

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--max_epoch` | 80 | Total training epochs |
| `--learning_rate` | 0.0003 | Initial learning rate |
| `--batch_size_per_gpu` | 1 | Batch size per GPU |
| `--frame_num` | 3 | Number of temporal frames |
| `--gpu_id` | `'0'` | CUDA visible device(s) |
| `--tag` | `''` | Experiment tag for output folder |

**Note:** Training logs are printed with colored text (ANSI escape codes). Colors are auto-detected and only enabled when running in a supported terminal. If your terminal does not display colors correctly, you can disable them by setting the environment variable before launching:

```bash
NO_COLOR=1 python train.py --data_root /path/to/your/data_root
```

### 5. Testing

```bash
cd train_code
python test.py --data_root /path/to/your/data_root
```

This loads the default pretrained model from `../model_zoo/pgdsrt/model_epoch_80.pth`. To test a specific checkpoint:

```bash
python test.py --data_root /path/to/your/data_root --model_test_path /path/to/checkpoint.pth
```

---

## ✒️ Citation
If this repo helps you, please consider citing our works:

```bibtex
@inproceedings{dynaspec,
  title={Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark},
  author={Lijing Cai and Zhan Shi and Chenglong Huang and Jinyao Wu and Qiping Li and Zikang Huo and Linsen Chen and Chongde Zi and Xun Cao},
  booktitle={CVPR},
  year={2026}
}
```
## 🙏 Acknowledgments
This code is built on [MST](https://github.com/caiyuanhao1998/MST). We thank the authors for sharing their codes.

## 📧 Contact
If you have any questions, please feel free to contact `cailijing@smail.nju.edu.cn` or `shizhan@smail.nju.edu.cn` or open an issue.
