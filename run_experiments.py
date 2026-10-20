"""
多次随机种子实验脚本
运行3次训练，计算 OA/AA/Kappa 的 mean ± std
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
import torch
import json
from datetime import datetime

from config import DATASET_CONFIGS, ModelConfig, TrainingConfig, get_class_dict
from model import create_model
from dataset import prepare_dataloaders
from train import Trainer


def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_experiment(dataset_name: str, seed: int, patch_size: int = 9,
                          epochs: int = 100, use_augmentation: bool = True,
                          use_band_selector: bool = False):
    """
    运行单次实验
    
    Args:
        dataset_name: 数据集名称
        seed: 随机种子
        patch_size: Patch 大小
        epochs: 训练轮数
        use_augmentation: 是否使用数据增强
        use_band_selector: 是否使用波段选择器
        
    Returns:
        results: 包含 OA, AA, Kappa 的字典
    """
    print(f"\n{'='*70}")
    print(f"实验种子: {seed}")
    print(f"{'='*70}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 获取数据集配置
    dataset_config = DATASET_CONFIGS[dataset_name]
    class_dict = get_class_dict(dataset_config.class_info)
    
    # 准备数据
    print(f"\n加载数据...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        dataset_type=dataset_name,
        data_dir=dataset_config.data_path,
        class_dict=class_dict,
        patch_size=patch_size,
        batch_size=32,
        val_ratio=0.1,
        augment=use_augmentation,
        num_workers=0
    )
    
    # 创建模型配置
    model_config = ModelConfig(
        spatial_dim=patch_size,
        num_bands=dataset_config.num_bands + 1,
        hidden_dim=512,
        output_dim=256,
        delta_param_init=0.01,
        num_classes=dataset_config.num_classes,
        use_band_selector=use_band_selector
    )
    
    # 创建训练配置
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        scheduler="step",
        early_stop_patience=20,
        device="cuda"
    )
    
    # 输出目录
    output_dir = os.path.join(
        "experiments",
        f"{dataset_name}_seed{seed}",
        f"hd{model_config.hidden_dim}_od{model_config.output_dim}"
    )
    
    # 创建模型
    model = create_model(model_config)
    
    # 创建训练器
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
    
    # 训练并获取测试结果
    test_metrics = trainer.train(epochs=epochs)
    
    return {
        'OA': test_metrics['OA'],
        'AA': test_metrics['AA'],
        'Kappa': test_metrics['Kappa'],
        'seed': seed
    }


def run_multiple_experiments(dataset_name: str, seeds: list = [42, 123, 456],
                             patch_size: int = 9, epochs: int = 100,
                             use_augmentation: bool = True,
                             use_band_selector: bool = False):
    """
    运行多次实验并计算统计量
    
    Args:
        dataset_name: 数据集名称
        seeds: 随机种子列表
        patch_size: Patch 大小
        epochs: 训练轮数
        use_augmentation: 是否使用数据增强
        use_band_selector: 是否使用波段选择器
    """
    print("\n" + "=" * 70)
    print(f"HSLiNets 多次随机种子实验")
    print(f"数据集: {dataset_name}")
    print(f"随机种子: {seeds}")
    print(f"Patch 大小: {patch_size}")
    print(f"训练轮数: {epochs}")
    print(f"数据增强: {use_augmentation}")
    print(f"波段选择器: {use_band_selector}")
    print("=" * 70)
    
    # 存储结果
    all_results = []
    
    # 运行每次实验
    for i, seed in enumerate(seeds):
        print(f"\n\n{'#'*70}")
        print(f"# 实验 {i+1}/{len(seeds)} - 种子: {seed}")
        print(f"{'#'*70}")
        
        result = run_single_experiment(
            dataset_name=dataset_name,
            seed=seed,
            patch_size=patch_size,
            epochs=epochs,
            use_augmentation=use_augmentation,
            use_band_selector=use_band_selector
        )
        all_results.append(result)
        
        print(f"\n实验 {i+1} 结果:")
        print(f"  OA: {result['OA']:.2f}%")
        print(f"  AA: {result['AA']:.2f}%")
        print(f"  Kappa: {result['Kappa']:.4f}")
    
    # 计算统计量
    oa_values = [r['OA'] for r in all_results]
    aa_values = [r['AA'] for r in all_results]
    kappa_values = [r['Kappa'] for r in all_results]
    
    oa_mean, oa_std = np.mean(oa_values), np.std(oa_values)
    aa_mean, aa_std = np.mean(aa_values), np.std(aa_values)
    kappa_mean, kappa_std = np.mean(kappa_values), np.std(kappa_values)
    
    # 打印最终结果
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    
    print(f"\n{'Seed':<10} {'OA (%)':<15} {'AA (%)':<15} {'Kappa':<15}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['seed']:<10} {r['OA']:<15.2f} {r['AA']:<15.2f} {r['Kappa']:<15.4f}")
    print("-" * 55)
    
    print(f"\n最终结果 (Mean ± Std):")
    print(f"  OA:    {oa_mean:.2f} ± {oa_std:.2f}%")
    print(f"  AA:    {aa_mean:.2f} ± {aa_std:.2f}%")
    print(f"  Kappa: {kappa_mean:.4f} ± {kappa_std:.4f}")
    
    # 保存结果到txt文件
    output_file = f"{dataset_name}_results.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"HSLiNets 实验结果\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"随机种子: {seeds}\n")
        f.write(f"Patch 大小: {patch_size}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"数据增强: {use_augmentation}\n")
        f.write(f"波段选择器: {use_band_selector}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("各次实验结果:\n")
        f.write(f"{'Seed':<10} {'OA (%)':<15} {'AA (%)':<15} {'Kappa':<15}\n")
        f.write("-" * 55 + "\n")
        for r in all_results:
            f.write(f"{r['seed']:<10} {r['OA']:<15.2f} {r['AA']:<15.2f} {r['Kappa']:<15.4f}\n")
        f.write("-" * 55 + "\n\n")
        
        f.write("最终结果 (Mean ± Std):\n")
        f.write(f"  OA:    {oa_mean:.2f} ± {oa_std:.2f}%\n")
        f.write(f"  AA:    {aa_mean:.2f} ± {aa_std:.2f}%\n")
        f.write(f"  Kappa: {kappa_mean:.4f} ± {kappa_std:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n结果已保存至: {output_file}")
    
    # 同时保存JSON格式
    json_output = {
        'dataset': dataset_name,
        'seeds': seeds,
        'patch_size': patch_size,
        'epochs': epochs,
        'use_augmentation': use_augmentation,
        'use_band_selector': use_band_selector,
        'individual_results': all_results,
        'final_results': {
            'OA_mean': oa_mean,
            'OA_std': oa_std,
            'AA_mean': aa_mean,
            'AA_std': aa_std,
            'Kappa_mean': kappa_mean,
            'Kappa_std': kappa_std
        }
    }
    
    json_file = f"{dataset_name}_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"JSON 结果已保存至: {json_file}")
    
    return json_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HSLiNets 多次随机种子实验")
    parser.add_argument("--dataset", type=str, default="houston",
                        choices=["trento", "houston"],
                        help="数据集名称")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456],
                        help="随机种子列表")
    parser.add_argument("--patch_size", type=int, default=9,
                        help="Patch 大小")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--augmentation", action="store_true", default=True,
                        help="是否使用数据增强")
    parser.add_argument("--use_band_selector", action="store_true",
                        help="是否使用波段选择器")
    
    args = parser.parse_args()
    
    run_multiple_experiments(
        dataset_name=args.dataset,
        seeds=args.seeds,
        patch_size=args.patch_size,
        epochs=args.epochs,
        use_augmentation=args.augmentation,
        use_band_selector=args.use_band_selector
    )
