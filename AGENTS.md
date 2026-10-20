# HSLiNets 项目上下文

## 项目概述

HSLiNets 是一个高光谱图像（HSI）与 LiDAR 数据融合分类的深度学习研究项目。项目采用基于 Mamba（状态空间模型，SSM）的创新架构，实现多源遥感数据的高效融合与分类。

### 核心技术特点

- **SpectralSpatialMamba**: 光谱-空间联合状态空间模型，支持选择性扫描
- **CrossAttentionFusion**: HSI-LiDAR 交叉注意力融合模块
- **BandGraphAttentionSelector**: 基于图注意力网络的自适应波段选择（可选）
- **空间划分策略**: 多种训练/测试划分方法，避免 patch 重叠泄露

### 支持的数据集

| 数据集 | 类别数 | HSI波段数 | LiDAR类型 |
|--------|--------|-----------|-----------|
| Trento | 6 | 63 | 单通道 DSM |
| Houston2013 | 15 | 144 | 单通道 DSM |
| MUUFL Gulfport | 11 | 64 | 点云插值 DSM |
| Houston2018 | 20 | 50 | 多光谱 LiDAR (3通道) |

## 项目结构

```
HSI-LiDAR-CLassification/
├── main.py                    # 统一入口脚本
├── model.py                   # 模型定义（SpectralSpatialMamba, CrossAttentionFusion 等）
├── config.py                  # 配置文件（数据集参数、模型超参数）
├── dataset.py                 # 数据加载、预处理、Patch 提取
├── train.py                   # 训练器类和训练循环
├── utils.py                   # 工具函数（评估指标、可视化）
├── spatial_split.py           # 空间划分模块（避免泄露）
├── band_graph_attention_selector.py  # 波段选择模块（可选）
├── adaptive_band_graph_selector.py   # 自适应波段选择（变体）
├── run_experiments.py         # 多随机种子实验脚本
├── requirements.txt           # 依赖包列表
└── test_*.py                  # 各类测试脚本
```

## 构建与运行

### 环境配置

```powershell
# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 可选：安装 mamba_ssm（需要 CUDA）
pip install mamba_ssm causal-conv1d

# 可选：安装 rs-fusion-datasets（自动下载 Houston2018）
pip install rs-fusion-datasets
```

### 训练模型

```powershell
# Trento 数据集训练
python main.py --dataset trento --epochs 100 --patch_size 9

# Houston2013 数据集训练
python main.py --dataset houston --epochs 100 --batch_size 64

# 使用自定义参数
python main.py --dataset trento --hidden_dim 256 --output_dim 128 --lr 0.001

# 启用波段选择模块
python main.py --dataset trento --use_band_selector

# 使用 Early Fusion 模式（默认为 Cross-Attention）
python main.py --dataset trento --no_cross_attention

# 空间划分方法
python main.py --dataset trento --spatial_split_method disjoint
# 可选方法: disjoint, random, grid, block
```

### 多随机种子实验

```powershell
# 运行 3 次实验
python run_experiments.py --dataset houston --seeds 42 123 456 --epochs 100

# 自定义参数
python run_experiments.py --dataset trento --patch_size 7 --epochs 50
```

### 测试与调试

```powershell
# 快速导入测试
python test_import.py

# 快速功能测试
python test_quick.py

# Houston 数据集测试
python test_houston2018.py

# 波段选择模块测试
python test_band_graph_attention_selector.py
```

## 模型架构详解

### HSIClassificationMambaModel（主模型）

```
输入:
  HSI: [Batch, H, W, Bands]
  LiDAR: [Batch, Channels, H, W]

处理流程:
  1. HSI -> SpectralSpatialMamba -> HSI Features [B, H, W, output_dim]
  2. LiDAR -> LiDARFeatureExtractor -> LiDAR Features [B, H, W, output_dim]
  3. HSI + LiDAR Features -> CrossAttentionFusion -> Fused Features
  4. Fused Features -> Global Avg Pool -> Classifier -> Logits [B, num_classes]
```

### 关键组件

#### SpectralSpatialMamba
- 光谱分支：沿波段维度进行选择性扫描
- 空间分支：使用 2D Cross-Scan 捕获空间依赖
- 特征融合 + 残差连接

#### CrossAttentionFusion
- HSI 作为 Query，LiDAR 作为 Key/Value
- 多头交叉注意力机制
- 门控融合策略

#### SelectiveScan（纯 PyTorch 实现）
- 兼容 mamba_ssm 库，无依赖时自动回退
- 支持 d_state, d_conv, expand 参数配置

## 配置参数说明

### 模型参数 (ModelConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| spatial_dim | 1 | Patch 大小 |
| num_bands | 64 | HSI 波段数 |
| hidden_dim | 128 | 隐藏层维度 |
| output_dim | 64 | 输出维度 |
| num_classes | 6 | 分类类别数 |
| d_state | 16 | SSM 状态维度 |
| d_conv | 3 | SSM 卷积核大小 |
| expand | 2 | SSM 扩展因子 |
| fusion_heads | 4 | 交叉注意力头数 |
| use_cross_attention | True | 是否使用交叉注意力融合 |
| use_band_selector | False | 是否使用波段选择 |

### 训练参数 (TrainingConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| batch_size | 32 | 批量大小 |
| learning_rate | 1e-4 | 学习率 |
| weight_decay | 1e-4 | 权重衰减 |
| scheduler | step | 学习率调度器 |
| early_stop_patience | 20 | 早停耐心值 |

### 数据参数 (DataConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| patch_size | 9 | 空间 Patch 大小 |
| normalize_method | minmax | 归一化方法 |
| spatial_split_method | grid | 空间划分方法 |

## 数据格式要求

### 数据目录结构

```
Dataset/
├── TrentoDataset/
│   ├── HSI_Trento.mat      # [H, W, 63]
│   ├── Lidar_Trento.mat    # [H, W]
│   └── GT_Trento.mat       # [H, W]
├── Houston2013/
│   ├── HSI.mat             # [Bands, H, W]
│   ├── LiDAR_1.mat         # [1, H, W]
│   ├── TRLabel.mat         # 训练标签
│   └── TSLabel.mat         # 测试标签
├── muuflgulfport/
│   └── muufl_gulfport_campus_1_hsi_220_label.mat
└── Houston2018/            # 或使用 rs-fusion-datasets 自动下载
```

### 数据格式说明

- **HSI 格式**: 
  - Trento/MUUFL: [H, W, Bands] (height, width, bands)
  - Houston2013: [Bands, H, W] 需要转置
  
- **LiDAR 格式**:
  - 单通道: [H, W] 或 [1, H, W]
  - 多通道 (Houston2018): [3, H, W]

- **标签格式**: [H, W], 0 表示无标签，1-N 表示类别

## 开发约定

### 代码风格

- 使用 Python 3.8+ 语法
- 遵循 PEP 8 编码规范
- 类型注解：关键函数使用 typing 模块
- 文档字符串：NumPy 风格

### 模块设计原则

- **配置分离**: 所有超参数集中在 `config.py`
- **数据处理独立**: 数据加载、预处理、增强在 `dataset.py`
- **训练逻辑封装**: `Trainer` 类封装完整训练流程
- **兼容性设计**: 支持有/无 mamba_ssm 库两种模式

### 测试约定

- 每个模块底部包含 `if __name__ == "__main__":` 测试代码
- 测试脚本命名: `test_*.py`
- 使用模拟数据验证模块功能

## 常见问题

### 1. CUDA 内存不足
```powershell
# 减小 batch_size 或 hidden_dim
python main.py --dataset houston --batch_size 16 --hidden_dim 64
```

### 2. 数据集路径问题
- 修改 `config.py` 中的 `DATASET_ROOT` 变量
- 或使用命令行参数 `--data_dir`

### 3. mamba_ssm 安装失败
- 项目提供纯 PyTorch 实现的 SelectiveScan
- 无需 mamba_ssm 也能正常运行

### 4. 训练不收敛
- 检查学习率（推荐 1e-4）
- 增加训练轮数
- 启用数据增强 `--augmentation`

## 输出文件

训练完成后，输出目录结构：

```
experiments/
└── {dataset}_patch{size}/
    └── hd{hidden}_od{output}/
        ├── best_model.pth       # 最佳模型检查点
        ├── training_log.json    # 训练日志
        └── training_curves.png  # 训练曲线图
```

## 相关论文与引用

本项目基于 Mamba 架构实现高光谱图像分类，融合 LiDAR 数据提升分类精度。如需引用，请参考相关论文。
