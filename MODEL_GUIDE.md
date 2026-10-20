# HSLiNets 模型指南

## 模型架构

HSLiNets 是基于 **FusionMamba** 架构的高光谱图像（HSI）与 LiDAR 数据融合分类模型。核心创新是使用双向状态空间模型（Bidirectional SSM）捕获光谱序列的前向和后向依赖关系。

### 核心组件

#### 1. HSIVimBlock（双向 SSM 处理块）

```
输入: [Batch, H, W, Bands]
输出: [Batch, output_dim]
```

**处理流程：**
1. **LayerNorm** - 对光谱维度进行归一化
2. **线性投影** - 投影到隐藏维度 (hidden_dim)
3. **前向路径** - Conv1D + SSM 处理原始光谱序列
4. **后向路径** - Conv1D + SSM 处理反转光谱序列
5. **特征融合** - 前向和后向特征相加融合

**关键参数：**
- `hidden_dim`: 隐藏层维度（默认 512）
- `output_dim`: 输出维度（默认 256）
- `delta_param_init`: SSM delta 参数初始值（默认 0.01）

#### 2. SpatialFeatureProcessing（空间特征处理）

```
输入: [Batch, input_channels, H, W]
输出: [Batch, 512]
```

使用空洞卷积（dilated convolution）提取空间特征：
- Conv2D (dilation=1) → 256 通道
- Conv2D (dilation=2) → 512 通道
- 全局平均池化

#### 3. Classifier（分类器）

```
输入: [Batch, in_features]
输出: [Batch, num_classes]
```

两层全连接网络：
- Linear(in_features → 1024) → ReLU → Dropout(0.4)
- Linear(1024 → num_classes)

#### 4. BandGraphAttentionSelector（可选波段选择模块）

基于图注意力网络的波段选择模块，用于自适应选择重要波段。

### 数据融合策略

**Early Fusion（早期融合）**：LiDAR 数据作为额外波段与 HSI 数据在输入层拼接。

```
融合前: HSI [H, W, Bands] + LiDAR [H, W, 1] 或 [H, W, 3]
融合后: [H, W, Bands + LiDAR_bands]
```

---

## 支持的数据集

| 数据集 | HSI 波段数 | LiDAR 通道 | 类别数 | 空间分辨率 | 来源 |
|--------|-----------|-----------|--------|-----------|------|
| Trento | 63 | 1 (DSM) | 6 | 1m | 意大利 Trento |
| Houston2013 | 144 | 1 (DSM) | 15 | 2.5m | Houston 2013 比赛 |
| Houston2018 | 50 | 3 (DSM+MSLiDAR) | 20 | 1m | Houston 2018 比赛 |
| MUUFL | 64 | 2 (DSM+Intensity) | 11 | 1m | 美国密西西比 |

---

## 命令行参数

### 数据集选择
```bash
--dataset {trento,houston,muuhfl,houston2018}
```

### 数据参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--patch_size` | Patch 大小 | 数据集默认值 |
| `--augmentation` | 启用数据增强 | False |

### 模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hidden_dim` | 隐藏层维度 | 512 |
| `--output_dim` | 输出维度 | 256 |
| `--delta_param_init` | Delta 参数初始值 | 0.01 |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批量大小 | 32 |
| `--lr` | 学习率 | 1e-4 |
| `--weight_decay` | 权重衰减 | 1e-4 |
| `--scheduler` | 学习率调度器 | step |
| `--early_stop` | 早停耐心值 | 20 |

### 波段选择参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_band_selector` | 启用波段选择 | False |
| `--band_selector_hidden` | 波段选择隐藏维度 | 64 |
| `--band_selector_out` | 波段选择输出维度 | 32 |

### 空间划分参数
| 参数 | 说明 | 默认值 | 适用方法 |
|------|------|--------|----------|
| `--spatial_split_method` | 空间划分方法 | grid | 全部 |
| `--spatial_split_seed` | 空间划分随机种子 | 42 | 全部 |
| `--min_test_ratio` | 测试集最小比例 | 0.2 | disjoint |
| `--min_test_samples` | 测试集最小样本数 | 50 | disjoint, grid |

### 其他参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--seed` | 随机种子 | 42 |
| `--device` | 计算设备 | cuda |
| `--output_dir` | 输出目录 | experiments |

---

## 运行示例

### 1. Trento 数据集

```bash
# 基础训练（使用默认参数）
python main.py --dataset trento

# 使用 patch_size=9 和数据增强
python main.py --dataset trento --patch_size 9 --augmentation

# 自定义模型参数
python main.py --dataset trento --patch_size 9 --hidden_dim 512 --output_dim 256

# 使用 grid 空间划分，指定最小测试样本数（推荐）
python main.py --dataset trento --patch_size 9 --spatial_split_method grid --min_test_samples 50

# 使用 disjoint 空间划分，指定测试集比例和最小样本数
python main.py --dataset trento --patch_size 9 --spatial_split_method disjoint --min_test_ratio 0.2 --min_test_samples 50

# 启用波段选择模块
python main.py --dataset trento \
    --patch_size 9 \
    --use_band_selector \
    --band_selector_hidden 64 \
    --band_selector_out 32

# 完整训练配置（含波段选择和空间划分）
python main.py --dataset trento \
    --patch_size 9 \
    --hidden_dim 512 \
    --output_dim 256 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --augmentation \
    --use_band_selector \
    --spatial_split_method grid \
    --min_test_samples 50 \
    --seed 42
```

### 2. Houston2013 数据集

```bash
# 基础训练
python main.py --dataset houston

# 使用 patch_size=9 和数据增强
python main.py --dataset houston --patch_size 9 --augmentation

# 使用 grid 划分并指定最小测试样本数
python main.py --dataset houston --patch_size 9 --spatial_split_method grid --min_test_samples 50

# 启用波段选择模块
python main.py --dataset houston \
    --patch_size 9 \
    --use_band_selector \
    --band_selector_hidden 64 \
    --band_selector_out 32

# 完整配置（含波段选择和空间划分参数）
python main.py --dataset houston \
    --patch_size 9 \
    --hidden_dim 512 \
    --output_dim 256 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --augmentation \
    --use_band_selector \
    --spatial_split_method grid \
    --min_test_samples 50 \
    --output_dir experiments/houston_patch9
```

### 3. Houston2018 数据集

```bash
# 基础训练（需先下载数据）
python main.py --dataset houston2018

# 使用 grid 划分并指定最小测试样本数
python main.py --dataset houston2018 --patch_size 9 --spatial_split_method grid --min_test_samples 50

# 启用波段选择模块
python main.py --dataset houston2018 \
    --patch_size 9 \
    --use_band_selector \
    --band_selector_hidden 64 \
    --band_selector_out 32

# 完整配置（含波段选择和空间划分参数）
python main.py --dataset houston2018 \
    --patch_size 9 \
    --hidden_dim 512 \
    --output_dim 256 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --augmentation \
    --use_band_selector \
    --spatial_split_method grid \
    --min_test_samples 50 \
    --output_dir experiments/houston2018_patch9
```

### 4. MUUFL 数据集

```bash
# 基础训练
python main.py --dataset muufl

# 使用 patch_size=9 和数据增强
python main.py --dataset muufl --patch_size 9 --augmentation

# 使用 grid 划分并指定最小测试样本数
python main.py --dataset muufl --patch_size 9 --spatial_split_method grid --min_test_samples 50

# 启用波段选择模块
python main.py --dataset muufl \
    --patch_size 9 \
    --use_band_selector \
    --band_selector_hidden 64 \
    --band_selector_out 32

# 完整配置（含波段选择和空间划分参数）
python main.py --dataset muufl \
    --patch_size 9 \
    --hidden_dim 512 \
    --output_dim 256 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --augmentation \
    --use_band_selector \
    --spatial_split_method grid \
    --min_test_samples 50 \
    --output_dir experiments/muufl_patch9
```

---

## 不同配置对比实验

### Patch Size 实验

```bash
# Patch Size = 1
python main.py --dataset trento --patch_size 1 --output_dir experiments/trento_patch1

# Patch Size = 5
python main.py --dataset trento --patch_size 5 --output_dir experiments/trento_patch5

# Patch Size = 9
python main.py --dataset trento --patch_size 9 --output_dir experiments/trento_patch9

# Patch Size = 11
python main.py --dataset trento --patch_size 11 --output_dir experiments/trento_patch11

# Patch Size = 13
python main.py --dataset trento --patch_size 13 --output_dir experiments/trento_patch13
```

### 隐藏层维度实验

```bash
# hidden_dim=256, output_dim=128
python main.py --dataset trento --patch_size 9 \
    --hidden_dim 256 --output_dim 128 \
    --output_dir experiments/trento_hd256_od128

# hidden_dim=512, output_dim=256 (默认)
python main.py --dataset trento --patch_size 9 \
    --hidden_dim 512 --output_dim 256 \
    --output_dir experiments/trento_hd512_od256

# hidden_dim=768, output_dim=384
python main.py --dataset trento --patch_size 9 \
    --hidden_dim 768 --output_dim 384 \
    --output_dir experiments/trento_hd768_od384
```

### 空间划分方法对比

```bash
# Disjoint 划分（推荐），指定测试集比例和最小样本数
python main.py --dataset trento --patch_size 9 \
    --spatial_split_method disjoint \
    --min_test_ratio 0.2 \
    --min_test_samples 50 \
    --output_dir experiments/trento_disjoint

# Grid 划分，指定最小测试样本数
python main.py --dataset trento --patch_size 9 \
    --spatial_split_method grid \
    --min_test_samples 50 \
    --output_dir experiments/trento_grid

# Random 划分
python main.py --dataset trento --patch_size 9 \
    --spatial_split_method random \
    --output_dir experiments/trento_random

# Block 划分
python main.py --dataset trento --patch_size 9 \
    --spatial_split_method block \
    --output_dir experiments/trento_block
```

### 学习率调度对比

```bash
# StepLR (默认)
python main.py --dataset trento --patch_size 9 --scheduler step

# CosineAnnealingLR
python main.py --dataset trento --patch_size 9 --scheduler cosine

# ReduceLROnPlateau
python main.py --dataset trento --patch_size 9 --scheduler plateau
```

### 随机种子对比（稳定性测试）

```bash
python main.py --dataset trento --patch_size 9 --seed 42 --output_dir experiments/trento_seed42
python main.py --dataset trento --patch_size 9 --seed 123 --output_dir experiments/trento_seed123
python main.py --dataset trento --patch_size 9 --seed 456 --output_dir experiments/trento_seed456
```

### 波段选择模块对比

```bash
# 不使用波段选择（默认）
python main.py --dataset trento --patch_size 9 \
    --output_dir experiments/trento_no_band_selector

# 使用波段选择（默认参数）
python main.py --dataset trento --patch_size 9 \
    --use_band_selector \
    --output_dir experiments/trento_band_selector_default

# 使用波段选择（自定义参数）
python main.py --dataset trento --patch_size 9 \
    --use_band_selector \
    --band_selector_hidden 128 \
    --band_selector_out 64 \
    --output_dir experiments/trento_band_selector_large
```

---

## 输出文件

训练完成后，输出目录包含：

```
experiments/{experiment_name}/
├── best_model.pth       # 最佳模型权重
├── training_curves.png  # 训练曲线图
└── training_log.json    # 训练日志
```

### training_log.json 内容示例

```json
{
    "dataset": "trento",
    "patch_size": 9,
    "hidden_dim": 512,
    "output_dim": 256,
    "best_epoch": 45,
    "best_val_loss": 0.1234,
    "best_val_acc": 0.9876,
    "test_acc": 0.9812,
    "test_kappa": 0.9756,
    "total_params": 1234567
}
```

---

## 注意事项

1. **空间划分方法**：推荐使用 `grid` 或 `disjoint` 方法，避免 patch 重叠导致的数据泄露。

2. **Houston2018 数据**：该数据集约 500MB，首次运行会自动下载到 `Dataset/Houston2018/` 目录。

3. **GPU 内存**：较大的 patch_size 和 hidden_dim 需要更多 GPU 内存。如果遇到 OOM 错误，尝试减小 batch_size。

4. **数据增强**：对于小样本数据集，建议启用 `--augmentation`。

5. **波段选择模块**：启用 `--use_band_selector` 可能增加训练时间，但可能提高模型性能。
