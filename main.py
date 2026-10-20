"""
HSLiNets 统一入口脚本
支持命令行参数选择数据集和调整超参数

使用示例:
    # 使用 Trento 数据集训练
    python main.py --dataset trento --epochs 100
    
    # 使用 Houston2013 数据集训练
    python main.py --dataset houston --epochs 100 --lr 0.001
    
    # 使用自定义参数
    python main.py --dataset trento --patch_size 7 --batch_size 64 --hidden_dim 512
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

from config import DATASET_CONFIGS, ModelConfig, TrainingConfig
from model import HSIClassificationMambaModel
from dataset import prepare_dataloaders
from train import Trainer
from utils import Timer


# =============================================================================
# 参数解析
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="HSLiNets: 高光谱与LiDAR融合分类模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集选择
    parser.add_argument(
        "--dataset", type=str, default="trento",
        choices=["trento", "houston", "muufl", "houston2018"],
        help="选择数据集: trento, houston, muufl 或 houston2018"
    )
    
    # 数据参数
    parser.add_argument(
        "--patch_size", type=int, default=None,
        help="Patch 大小 (默认使用数据集配置)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=None,
        help="训练集比例 (默认使用数据集配置)"
    )
    parser.add_argument(
        "--augmentation", action="store_true",
        help="是否使用数据增强"
    )
    
    # 模型参数
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="隐藏层维度"
    )
    parser.add_argument(
        "--output_dim", type=int, default=None,
        help="输出维度"
    )
    parser.add_argument(
        "--delta_param_init", type=float, default=None,
        help="Delta 参数初始值"
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="批量大小"
    )
    parser.add_argument(
        "--lr", "--learning_rate", type=float, default=None,
        dest="learning_rate",
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=None,
        help="权重衰减"
    )
    parser.add_argument(
        "--scheduler", type=str, default=None,
        choices=["step", "cosine", "plateau"],
        help="学习率调度器"
    )
    parser.add_argument(
        "--early_stop", type=int, default=10,
        help="早停耐心值"
    )
    
    # 波段选择参数
    parser.add_argument(
        "--use_band_selector", action="store_true",
        help="是否使用自适应波段选择模块"
    )
    parser.add_argument(
        "--band_selector_hidden", type=int, default=64,
        help="波段选择模块隐藏层维度"
    )
    parser.add_argument(
        "--band_selector_out", type=int, default=32,
        help="波段选择模块输出维度"
    )
    
    # 融合参数
    parser.add_argument(
        "--use_cross_attention", action="store_true", default=True,
        help="是否使用交叉注意力融合（默认开启）"
    )
    parser.add_argument(
        "--no_cross_attention", dest="use_cross_attention", action="store_false",
        help="禁用交叉注意力融合，使用 Early Fusion"
    )
    
    # 其他参数
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="计算设备"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="数据加载器工作进程数"
    )
    
    # 空间划分参数
    parser.add_argument(
        "--spatial_split_method", type=str, default="grid",
        choices=["disjoint", "random", "grid", "block"],
        help="空间划分方法，避免patch重叠泄露"
    )
    parser.add_argument(
        "--spatial_split_seed", type=int, default=42,
        help="空间划分随机种子"
    )
    parser.add_argument(
        "--min_test_ratio", type=float, default=0.2,
        help="测试集最小比例（相对于类别总样本数）"
    )
    parser.add_argument(
        "--min_test_samples", type=int, default=50,
        help="测试集最小样本数"
    )
    
    return parser.parse_args()


# =============================================================================
# 设置随机种子
# =============================================================================

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取数据集配置
    dataset_name = args.dataset.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    print("\n" + "=" * 70)
    print("HSLiNets - 高光谱与LiDAR融合分类模型")
    print("=" * 70)
    print(f"数据集: {dataset_config.name}")
    print(f"数据路径: {dataset_config.data_path}")
    print(f"类别数: {dataset_config.num_classes}")
    print(f"波段数: {dataset_config.num_bands}")
    print("=" * 70)
    
    # 创建模型配置
    # Houston 2018 有 3 通道多光谱 LiDAR
    if dataset_name == "houston2018":
        lidar_bands = 3
    else:
        lidar_bands = 1
    
    # 根据是否使用交叉注意力决定 num_bands
    # use_cross_attention=True: num_bands 只包含 HSI 波段，LiDAR 单独处理
    # use_cross_attention=False: num_bands 包含 HSI + LiDAR (Early Fusion)
    use_cross_attention = args.use_cross_attention
    if use_cross_attention:
        num_bands = dataset_config.num_bands  # 只有 HSI 波段
    else:
        num_bands = dataset_config.num_bands + lidar_bands  # HSI + LiDAR
    
    model_config = ModelConfig(
        spatial_dim=args.patch_size or dataset_config.patch_size,
        num_bands=num_bands,
        lidar_channels=lidar_bands,  # LiDAR 通道数
        hidden_dim=args.hidden_dim or 128,
        output_dim=args.output_dim or 64,
        delta_param_init=args.delta_param_init or 0.01,
        num_classes=dataset_config.num_classes,
        # 波段选择参数
        use_band_selector=args.use_band_selector,
        band_selector_hidden=args.band_selector_hidden,
        band_selector_out=args.band_selector_out,
        # 融合参数
        use_cross_attention=use_cross_attention
    )
    
    print("\n[模型配置]")
    print(f"  spatial_dim: {model_config.spatial_dim}")
    print(f"  num_bands: {model_config.num_bands}")
    print(f"  hidden_dim: {model_config.hidden_dim}")
    print(f"  output_dim: {model_config.output_dim}")
    print(f"  delta_param_init: {model_config.delta_param_init}")
    print(f"  num_classes: {model_config.num_classes}")
    print(f"  use_band_selector: {model_config.use_band_selector}")
    if model_config.use_band_selector:
        print(f"  band_selector_hidden: {model_config.band_selector_hidden}")
        print(f"  band_selector_out: {model_config.band_selector_out}")
    
    # 创建训练配置
    training_config = TrainingConfig(
        epochs=args.epochs or 100,
        batch_size=args.batch_size or 32,
        learning_rate=args.learning_rate or 0.001,
        weight_decay=args.weight_decay or 0.01,
        scheduler=args.scheduler or "step",
        early_stop_patience=args.early_stop or 20,
        device=args.device
    )
    
    print("\n[训练配置]")
    print(f"  epochs: {training_config.epochs}")
    print(f"  batch_size: {training_config.batch_size}")
    print(f"  learning_rate: {training_config.learning_rate}")
    print(f"  weight_decay: {training_config.weight_decay}")
    print(f"  scheduler: {training_config.scheduler}")
    print(f"  early_stop_patience: {training_config.early_stop_patience}")
    print(f"  device: {training_config.device}")
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(
            "experiments",
            f"{dataset_name}_patch{model_config.spatial_dim}",
            f"hd{model_config.hidden_dim}_od{model_config.output_dim}"
        )
    else:
        output_dir = args.output_dir
    
    print(f"\n[输出目录]")
    print(f"  {output_dir}")
    
    # 加载数据
    print("\n[加载数据]")
    timer = Timer()
    timer.start()
    
    # 获取类别字典
    from config import get_class_dict
    class_dict = get_class_dict(dataset_config.class_info)
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        dataset_type=dataset_name,
        data_dir=dataset_config.data_path,
        class_dict=class_dict,
        patch_size=model_config.spatial_dim,
        batch_size=training_config.batch_size,
        val_ratio=training_config.val_ratio,
        augment=args.augmentation,
        num_workers=args.num_workers,
        spatial_split_method=args.spatial_split_method,
        spatial_split_seed=args.spatial_split_seed,
        min_test_ratio=args.min_test_ratio,
        min_test_samples=args.min_test_samples,
        use_cross_attention=model_config.use_cross_attention
    )
    
    print(f"数据加载完成，耗时: {timer}")
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  测试样本: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n[创建模型]")
    from model import create_model
    model = create_model(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 创建训练器并训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
        class_names=dataset_config.class_names
    )
    
    test_metrics = trainer.train()
    
    # 输出最终结果
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"测试集 OA: {test_metrics['OA']:.2f}%")
    print(f"测试集 AA: {test_metrics['AA']:.2f}%")
    print(f"测试集 Kappa: {test_metrics['Kappa']:.2f}%")
    print(f"模型保存至: {output_dir}")
    print("=" * 70)
    
    return test_metrics


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    main()
