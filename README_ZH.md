<div align="center">

# HAANet

### Thinking 3D Object Aesthetics Assessment: Employing Hairstyle Aesthetics Assessment as An Exemplification

**Shuai He · Hongkun Ruan · Anlong Ming<sup>*</sup> · Zhaowen Lin · Haiyang Zhang**

Beijing University of Posts and Telecommunications

[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-8A2BE2.svg)](https://doi.org/10.1145/3767308.3836093)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

[[English](README.md)] · **简体中文**

</div>

HAANet 是面向 Hairstyle Aesthetics Assessment（HAA）的多视图美学评分网络，也是论文提出的 3D Object Aesthetics Assessment（3D-OAA）范式实例。本仓库提供第二阶段训练、单样本推理和 HAA10K 评测代码。

<p align="center">
  <img src="assets/3d-oaa-overview.png" width="96%" alt="3D-OAA overview">
</p>
<p align="center"><em>3D-OAA：将图像、视频或 3D 模型统一转换为多视图图像进行美学评估。</em></p>

## 方法

实际代码每次输入同一对象的 4 个视图：正面、背面、左侧和右侧。

<p align="center">
  <img src="assets/haanet-architecture.png" width="96%" alt="HAANet architecture">
</p>
<p align="center"><em>HAANet 网络结构。</em></p>

- **Hair Encoder**：Swin-B 多尺度特征与双向 Transformer 解码器。
- **Face Encoder**：Swin-B 多尺度特征与人脸任务 Transformer 解码器。
- **Multi-view Fusion**：融合角度分类特征与 4 个可学习视图权重。
- **Cross Attention**：Face-to-Hair 与 Hair-to-Face 双向交叉注意力。
- **Image Encoder**：ConvNeXt-Base 全局图像特征。
- **Rule Layer**：通过语义原型和 3 个兼容性矩阵生成先验分数。
- **Score Head**：融合局部、全局与先验特征，输出 0-10 美学分数。

## 环境安装

```bash
conda create -n haanet python=3.10 -y
conda activate haanet
pip install -r requirements.txt
```

首次运行会由 torchvision 下载 Swin-B 和 ConvNeXt-Base 的 ImageNet-1K V1 权重。

## 数据集与权重

HAA10K 图像和模型权重：

- [Google Drive](https://drive.google.com/drive/folders/1bOKiGFV5QdHfJlhikVhtvoXljOT_N37u?usp=sharing)
- [Baidu Drive](https://pan.baidu.com/s/1W4C91ePORydaB7hTpiENRw?pwd=ace6)

<p align="center">
  <img src="assets/haa10k-samples.png" width="96%" alt="HAA10K samples and attributes">
</p>
<p align="center"><em>HAA10K 多视图样本、分数分布与发型属性。</em></p>

下载后按以下目录组织文件：

```text
HAANet/
├── HAA10K/
│   └── score.json
├── dataset/
│   └── images/
│       ├── 0_0.png
│       ├── 0_1.png
│       └── ...                # 每个样本包含 0-11 共 12 个视图
├── model/
│   ├── hair.pt                # Hair Encoder 预训练权重
│   ├── face.pt                # Face Encoder 预训练权重
│   └── best_aesthetic_model3.pth
└── src/
```

视图索引：正面 `0`、背面 `6`、左侧 `1-5`、右侧 `7-11`。训练时从左右候选视图中各随机选择一张。

## 推理

按 HAA10K 命名格式推理：

```bash
python src/infer_cli.py \
  --sample-dir dataset/images \
  --sample-id 0
```

直接指定 4 张图像：

```bash
python src/infer_cli.py \
  --front front.png \
  --back back.png \
  --left left.png \
  --right right.png
```

输出包括美学分数、4 个视图的角度类别和实际输入路径。自定义权重或设备时使用 `--hair-weights`、`--face-weights`、`--model-weights` 和 `--device`。

## 评测

```bash
python src/evaluate_cli.py \
  --data-root dataset/images \
  --labels HAA10K/score.json \
  --seed 42 \
  --augment-multiplier 5 \
  --batch-size 16
```

结果保存在 `evaluation_results/`，包括 LCC、SRCC、MSE、MAE、RMSE、五级准确率和逐样本预测。快速检查流程：

```bash
python src/evaluate_cli.py \
  --data-root dataset/images \
  --max-samples 32 \
  --augment-multiplier 1
```

## 训练

发布代码执行第二阶段训练：加载 `hair.pt` 和 `face.pt`，联合优化完整 HAANet。

```bash
python src/train_cli.py \
  --data-root dataset/images \
  --labels HAA10K/score.json \
  --hair-weights model/hair.pt \
  --face-weights model/face.pt \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --augment-multiplier 5 \
  --save-path checkpoints/best_model.pth \
  --last-epoch-path checkpoints/last_epoch.pth \
  --log-dir runs
```

训练目标：

```text
L = L1(score) + 0.5 * CrossEntropy(angle) + 0.1 * L1(rule matrices)
```

验证集 MAE 用于学习率调度和最佳权重选择。Rule Layer 原型在第 0、10、20... 个 epoch 使用训练特征重新聚类。不要将 `--save-path` 指向发布的最佳权重。

### 训练冒烟测试

以下命令只使用少量样本，并将权重写入独立目录：

```bash
python src/train_cli.py \
  --data-root dataset/images \
  --epochs 1 \
  --batch-size 2 \
  --augment-multiplier 1 \
  --max-train-samples 12 \
  --max-val-samples 4 \
  --save-path checkpoints/smoke/best.pth \
  --last-epoch-path checkpoints/smoke/last.pth \
  --log-dir runs/smoke
```

只检查模型加载、前向与反向：

```bash
python src/smoke_test.py --train-step
```

## 项目结构

```text
src/
├── model3_enhanced.py     # HAANet 主网络
├── FaceEncoder.py         # Face Encoder
├── HairEncoder.py         # Hair Encoder
├── transformer.py         # 双向 Transformer
├── dataset.py             # HAA10K 数据加载
├── infer_cli.py           # 单样本推理
├── evaluate_cli.py        # 数据集评测
├── train_cli.py           # 第二阶段训练入口
├── training.py            # 训练与验证循环
└── smoke_test.py          # 环境和模型检查
```

## 引用

```bibtex
@inproceedings{he2026thinking,
  title     = {Thinking 3D Object Aesthetics Assessment: Employing Hairstyle Aesthetics Assessment as An Exemplification},
  author    = {He, Shuai and Ruan, Hongkun and Ming, Anlong and Lin, Zhaowen and Zhang, Haiyang},
  booktitle = {Proceedings of the 35th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836093}
}
```

<sup>*</sup> Corresponding author.
