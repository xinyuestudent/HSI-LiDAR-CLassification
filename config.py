"""
HSLiNets 配置文件
包含数据集参数、模型超参数、训练配置等
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class DatasetType(Enum):
    """支持的数据集类型"""
    TRENTO = "trento"
    HOUSTON = "houston"
    MUUFL = "muufl"
    HOUSTON2018 = "houston2018"


# =============================================================================
# 数据集配置
# =============================================================================

# Trento 数据集配置
TRENTO_CONFIG = {
    "name": "Trento",
    "hsi_file": "HSI_Trento.mat",
    "lidar_file": "Lidar_Trento.mat",
    "gt_file": "GT_Trento.mat",
    # 数据维度
    "hsi_shape": (166, 600, 63),  # (H, W, Bands)
    "num_bands": 63,
    "num_classes": 6,
    # 类别信息: (class_id, class_name, training_samples, test_samples, total_samples)
    "class_info": [
        (1, "Apple trees", 129, 3905, 4034),
        (2, "Buildings", 125, 2778, 2903),
        (3, "Ground", 105, 374, 479),
        (4, "Wood", 154, 8969, 9123),
        (5, "Vineyard", 184, 10317, 10501),
        (6, "Roads", 122, 3052, 3174),
    ],
    # 数据格式
    "hsi_format": "H,W,Bands",  # HSI 数据维度顺序
    "lidar_format": "H,W",       # LiDAR 数据维度顺序
}

# Houston2013 数据集配置
HOUSTON_CONFIG = {
    "name": "Houston2013",
    "hsi_file": "HSI.mat",
    "lidar_file": "LiDAR_1.mat",
    "gt_train_file": "TRLabel.mat",
    "gt_test_file": "TSLabel.mat",
    # 数据维度
    "hsi_shape": (144, 349, 1905),  # (Bands, H, W)
    "num_bands": 144,
    "num_classes": 15,
    # 类别信息
    "class_info": [
        (1, "Healthy grass", 198, 1053, 1251),
        (2, "Stressed grass", 190, 1064, 1254),
        (3, "Synthetic grass", 192, 505, 697),
        (4, "Trees", 188, 1058, 1244),
        (5, "Soil", 186, 1056, 1242),
        (6, "Water", 182, 141, 325),
        (7, "Residential", 196, 1072, 1268),
        (8, "Commercial", 191, 1053, 1244),
        (9, "Road", 193, 1059, 1252),
        (10, "Highway", 191, 1036, 1227),
        (11, "Railway", 181, 1054, 1235),
        (12, "Parking lot 1", 192, 1041, 1233),
        (13, "Parking lot 2", 184, 285, 469),
        (14, "Tennis court", 181, 247, 428),
        (15, "Running track", 187, 473, 660),
    ],
    # 数据格式
    "hsi_format": "Bands,H,W",  # HSI 数据维度顺序
    "lidar_format": "Channel,H,W",  # LiDAR 数据维度顺序
}

# MUUFL Gulfport 数据集配置
MUUFL_CONFIG = {
    "name": "MUUFL",
    "hsi_file": "muufl_gulfport_campus_1_hsi_220_label.mat",
    # 数据维度
    "hsi_shape": (325, 220, 64),  # (H, W, Bands)
    "num_bands": 64,
    "num_classes": 11,
    # 类别信息: (class_id, class_name, training_samples, test_samples, total_samples)
    # training_samples 根据类别样本数合理分配，确保小类别有足够训练样本
    "class_info": [
        (1, "Trees", 150, 23096, 23246),
        (2, "Grass (pure)", 80, 4190, 4270),
        (3, "Grass + ground", 80, 6802, 6882),
        (4, "Dirt & Sand", 50, 1776, 1826),
        (5, "Road Materials", 80, 6607, 6687),
        (6, "Water", 50, 416, 466),      # 增加
        (7, "Shadow", 50, 2183, 2233),   # 增加
        (8, "Buildings", 80, 6160, 6240),
        (9, "Sidewalk", 50, 1335, 1385), # 增加
        (10, "Yellow Curb", 40, 143, 183),  # 大幅增加
        (11, "Cloth Panels", 50, 219, 269), # 大幅增加
    ],
    # 数据格式
    "hsi_format": "H,W,Bands",  # HSI 数据维度顺序
    "lidar_format": "point_cloud",  # LiDAR 数据维度顺序 (点云格式)
}

# Houston 2018 数据集配置 (多光谱 LiDAR)
HOUSTON2018_CONFIG = {
    "name": "Houston2018",
    # 数据维度 (来自 rs-fusion-datasets)
    "hsi_shape": (None, None, 50),  # (H, W, Bands) - 尺寸从数据中获取
    "hsi_bands": 50,           # 实际波段数
    "lidar_bands": 1,          # DSM 单通道
    "hsi_gsd": 1.0,   # meters
    "lidar_gsd": 0.5,  # meters
    "num_classes": 20,         # 20 个地物类别 (类别 0 为背景)
    # 类别信息: (class_id, class_name, training_samples, test_samples, total_samples)
    # 训练样本数设为较小值，实际数据通过 rs-fusion-datasets 获取
    "class_info": [
        (1, "Healthy grass", 100, 0, 100),
        (2, "Stressed grass", 100, 0, 100),
        (3, "Artificial turf", 100, 0, 100),
        (4, "Evergreen trees", 100, 0, 100),
        (5, "Deciduous trees", 100, 0, 100),
        (6, "Bare earth", 100, 0, 100),
        (7, "Water", 100, 0, 100),
        (8, "Residential buildings", 100, 0, 100),
        (9, "Non-residential buildings", 100, 0, 100),
        (10, "Roads", 100, 0, 100),
        (11, "Sidewalks", 100, 0, 100),
        (12, "Crosswalks", 100, 0, 100),
        (13, "Major thoroughfares", 100, 0, 100),
        (14, "Highways", 100, 0, 100),
        (15, "Railways", 100, 0, 100),
        (16, "Paved parking lots", 100, 0, 100),
        (17, "Unpaved parking lots", 100, 0, 100),
        (18, "Cars", 100, 0, 100),
        (19, "Trains", 100, 0, 100),
        (20, "Stadium seats", 100, 0, 100),
    ],
    # 数据格式
    "hsi_format": "H,W,Bands",
    "lidar_format": "multi_spectral",  # 多光谱 LiDAR (3 通道)
    # 数据来源
    "data_source": "rs-fusion-datasets",  # 使用 rs-fusion-datasets 库加载
}

def get_dataset_config(dataset_type: str) -> dict:
    """
    获取数据集配置
    
    Args:
        dataset_type: 数据集类型 ('trento', 'houston', 'muufl' 或 'houston2018')
        
    Returns:
        数据集配置字典
    """
    configs = {
        "trento": TRENTO_CONFIG,
        "houston": HOUSTON_CONFIG,
        "muufl": MUUFL_CONFIG,
        "houston2018": HOUSTON2018_CONFIG,
    }
    
    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower not in configs:
        raise ValueError(f"Unknown dataset: {dataset_type}. "
                        f"Available: {list(configs.keys())}")
    
    return configs[dataset_type_lower]


def get_class_dict(class_info: List) -> Dict:
    """
    将类别信息列表转换为字典格式
    
    Args:
        class_info: 类别信息列表
        
    Returns:
        类别字典 {class_id: {name, training, test, total}}
    """
    return {
        class_id: {
            "class_name": class_name,
            "training": training,
            "test": test,
            "samples": total
        }
        for class_id, class_name, training, test, total in class_info
    }


# =============================================================================
# 模型超参数配置
# =============================================================================

@dataclass
class ModelConfig:
    """模型超参数配置"""
    # HSIVimBlock 参数
    spatial_dim: int = 1          # 空间维度 (patch size)
    num_bands: int = 64           # HSI 波段数（不含 LiDAR）
    hidden_dim: int = 128         # 隐藏层维度（减小以提高效率）
    output_dim: int = 64          # 输出维度
    delta_param_init: float = 0.01  # SSM delta 参数初始化（保留兼容性）
    
    # 分类器参数
    num_classes: int = 6          # 分类类别数
    classifier_hidden: int = 256  # 分类器隐藏层维度
    dropout: float = 0.2          # Dropout 概率
    
    # 波段选择模块 (可选)
    use_band_selector: bool = False
    band_selector_hidden: int = 64
    band_selector_out: int = 32
    
    # ============ 新增参数 ============
    # LiDAR 参数
    lidar_channels: int = 1       # LiDAR 通道数（1 或 3）
    
    # SSM 参数
    d_state: int = 16             # SSM 状态维度
    d_conv: int = 3               # SSM 卷积核大小
    expand: int = 2               # SSM 扩展因子
    
    # 融合参数
    fusion_heads: int = 4         # 交叉注意力头数
    use_cross_attention: bool = True  # 是否使用交叉注意力融合


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 学习率调度
    scheduler: str = "step"       # 'step', 'cosine', 'plateau'
    step_size: int = 30           # StepLR 步长
    gamma: float = 0.1            # StepLR 衰减系数
    
    # 早停
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    
    # 数据增强
    augment: bool = True
    augment_rotations: List[int] = field(default_factory=lambda: [45, 90, 135])
    augment_flip: bool = True
    
    # 验证集划分
    val_ratio: float = 0.1
    
    # 空间划分配置（避免 patch 重叠泄露）
    spatial_split_method: str = "disjoint"  # 'disjoint', 'random', 'grid', 'block'
    spatial_split_seed: int = 42
    
    # 其他
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"          # 'cuda' 或 'cpu'


@dataclass
class DataConfig:
    """数据处理配置"""
    # Patch 提取
    patch_size: int = 9           # 空间 patch 大小
    
    # 归一化
    normalize_method: str = "minmax"  # 'minmax' 或 'zscore'
    
    # 数据路径
    data_root: str = ""
    
    # 输出路径
    output_dir: str = "checkpoints"


# =============================================================================
# 默认配置获取函数
# =============================================================================

def get_default_model_config(dataset_type: str) -> ModelConfig:
    """
    根据数据集获取默认模型配置
    
    Args:
        dataset_type: 数据集类型
        
    Returns:
        ModelConfig 实例
    """
    dataset_config = get_dataset_config(dataset_type)
    
    # 获取 HSI 波段数（不含 LiDAR）
    num_bands = dataset_config.get("num_bands", dataset_config.get("hsi_bands", 64))
    
    # 获取 LiDAR 通道数
    if dataset_type.lower() == "houston2018":
        lidar_channels = dataset_config.get("lidar_bands", 3)
    else:
        lidar_channels = 1
    
    return ModelConfig(
        num_bands=num_bands,  # 仅 HSI 波段数
        num_classes=dataset_config["num_classes"],
        lidar_channels=lidar_channels,
    )


def get_default_training_config() -> TrainingConfig:
    """获取默认训练配置"""
    return TrainingConfig()


def get_default_data_config(dataset_type: str, data_root: str = "") -> DataConfig:
    """
    获取默认数据配置
    
    Args:
        dataset_type: 数据集类型
        data_root: 数据根目录
        
    Returns:
        DataConfig 实例
    """
    return DataConfig(data_root=data_root)


# =============================================================================
# 数据集配置类
# =============================================================================

@dataclass
class DatasetConfig:
    """数据集配置类"""
    name: str
    data_path: str
    hsi_file: str
    lidar_file: str
    gt_file: str
    hsi_shape: tuple
    num_bands: int
    num_classes: int
    class_info: List
    class_names: List[str]
    hsi_format: str
    lidar_format: str
    patch_size: int = 9
    train_samples: Dict = field(default_factory=dict)


def _create_dataset_config(config: dict, data_path: str) -> DatasetConfig:
    """从配置字典创建 DatasetConfig 实例"""
    class_names = [info[1] for info in config["class_info"]]
    train_samples = {info[0]: info[2] for info in config["class_info"]}
    
    # 获取波段数，兼容不同配置格式
    num_bands = config.get("num_bands", config.get("hsi_bands", 0))
    
    return DatasetConfig(
        name=config["name"],
        data_path=data_path,
        hsi_file=config.get("hsi_file", ""),
        lidar_file=config.get("lidar_file", ""),
        gt_file=config.get("gt_file", ""),
        hsi_shape=config.get("hsi_shape", (None, None, num_bands)),
        num_bands=num_bands,
        num_classes=config["num_classes"],
        class_info=config["class_info"],
        class_names=class_names,
        hsi_format=config["hsi_format"],
        lidar_format=config["lidar_format"],
        patch_size=9,
        train_samples=train_samples
    )


# =============================================================================
# 数据集配置字典
# =============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "Dataset")

# 创建数据集配置实例
DATASET_CONFIGS = {
    "trento": _create_dataset_config(
        TRENTO_CONFIG, 
        os.path.join(DATASET_ROOT, "TrentoDataset")
    ),
    "houston": _create_dataset_config(
        HOUSTON_CONFIG,
        os.path.join(DATASET_ROOT, "Houston2013")
    ),
    "muufl": _create_dataset_config(
        MUUFL_CONFIG,
        os.path.join(DATASET_ROOT, "muuflgulfport")
    ),
    "houston2018": _create_dataset_config(
        HOUSTON2018_CONFIG,
        os.path.join(DATASET_ROOT, "Houston2018")
    ),
}


# =============================================================================
# 配置打印
# =============================================================================

def print_config(model_config: ModelConfig, 
                 training_config: TrainingConfig,
                 data_config: DataConfig,
                 dataset_config: dict):
    """打印所有配置信息"""
    print("=" * 60)
    print("配置信息")
    print("=" * 60)
    
    print(f"\n[数据集]")
    print(f"  名称: {dataset_config['name']}")
    print(f"  波段数: {dataset_config['num_bands']}")
    print(f"  类别数: {dataset_config['num_classes']}")
    
    print(f"\n[模型]")
    print(f"  spatial_dim: {model_config.spatial_dim}")
    print(f"  num_bands: {model_config.num_bands}")
    print(f"  hidden_dim: {model_config.hidden_dim}")
    print(f"  output_dim: {model_config.output_dim}")
    print(f"  num_classes: {model_config.num_classes}")
    print(f"  use_band_selector: {model_config.use_band_selector}")
    
    print(f"\n[训练]")
    print(f"  epochs: {training_config.epochs}")
    print(f"  batch_size: {training_config.batch_size}")
    print(f"  learning_rate: {training_config.learning_rate}")
    print(f"  augment: {training_config.augment}")
    print(f"  device: {training_config.device}")
    
    print(f"\n[数据]")
    print(f"  patch_size: {data_config.patch_size}")
    print(f"  normalize_method: {data_config.normalize_method}")
    print(f"  data_root: {data_config.data_root}")
    
    print("=" * 60)
