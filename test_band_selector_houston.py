"""
使用 Houston2013 数据集验证 AdaptiveBandGraphSelector 模块
"""

import sys
import os
import torch
import numpy as np
import scipy.io as sio

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_band_graph_selector import AdaptiveBandGraphSelector


def main():
    print("=" * 70)
    print("使用 Houston2013 数据集验证 AdaptiveBandGraphSelector 模块")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载 Houston2013 数据集
    # =========================================================================
    print("\n[1] 加载 Houston2013 数据集...")
    
    data_dir = r"E:\项目\HSLiNets\Dataset\Houston2013"
    
    # 加载 HSI 数据
    hsi_data = sio.loadmat(os.path.join(data_dir, "HSI.mat"))
    hsi_key = list(hsi_data.keys())[-1]
    hsi = hsi_data[hsi_key]
    print(f"  HSI 数据变量名: {hsi_key}")
    print(f"  HSI shape: {hsi.shape}")
    
    # 加载 LiDAR 数据
    lidar_data = sio.loadmat(os.path.join(data_dir, "LiDAR_1.mat"))
    lidar_key = list(lidar_data.keys())[-1]
    lidar = lidar_data[lidar_key]
    print(f"  LiDAR 数据变量名: {lidar_key}")
    print(f"  LiDAR shape: {lidar.shape}")
    
    # 加载训练标签
    tr_label_data = sio.loadmat(os.path.join(data_dir, "TRLabel.mat"))
    tr_label_key = list(tr_label_data.keys())[-1]
    tr_label = tr_label_data[tr_label_key]
    print(f"  训练标签变量名: {tr_label_key}")
    print(f"  训练标签 shape: {tr_label.shape}")
    print(f"  训练标签范围: [{tr_label.min()}, {tr_label.max()}]")
    print(f"  训练样本数: {np.sum(tr_label > 0)}")
    
    # =========================================================================
    # 2. 数据预处理
    # =========================================================================
    print("\n[2] 数据预处理...")
    
    # 归一化 HSI 数据
    hsi_min = hsi.min()
    hsi_max = hsi.max()
    hsi_normalized = (hsi - hsi_min) / (hsi_max - hsi_min + 1e-8)
    print(f"  HSI 归一化后范围: [{hsi_normalized.min():.4f}, {hsi_normalized.max():.4f}]")
    
    # 归一化 LiDAR 数据
    lidar_min = lidar.min()
    lidar_max = lidar.max()
    lidar_normalized = (lidar - lidar_min) / (lidar_max - lidar_min + 1e-8)
    print(f"  LiDAR 归一化后范围: [{lidar_normalized.min():.4f}, {lidar_normalized.max():.4f}]")
    
    # =========================================================================
    # 3. 提取单个 patch 进行测试
    # =========================================================================
    print("\n[3] 提取测试 patch...")
    
    # Houston2013 数据格式:
    # HSI: (144, 349, 1905) -> (Bands, H, W)
    # LiDAR: (1, 349, 1905) -> (Channel, H, W)
    # TRLabel: (349, 1905) -> (H, W)
    
    num_bands, h_height, h_width = hsi.shape
    print(f"  HSI 维度解释: Bands={num_bands}, Height={h_height}, Width={h_width}")
    
    # 找到一个有效像素（非0标签）
    valid_indices = np.argwhere(tr_label > 0)
    if len(valid_indices) == 0:
        print("  错误: 未找到有效像素!")
        return
    
    # 随机选择一个像素
    np.random.seed(42)
    random_idx = np.random.randint(0, len(valid_indices))
    center_i, center_j = valid_indices[random_idx]
    
    print(f"  选择像素位置: ({center_i}, {center_j})")
    print(f"  该像素标签: {tr_label[center_i, center_j]}")
    
    # 提取 HSI patch [num_bands, H, W]
    patch_size = 11
    half_size = patch_size // 2
    
    # 计算 patch 边界
    i_start = max(0, center_i - half_size)
    i_end = min(h_height, center_i + half_size + 1)
    j_start = max(0, center_j - half_size)
    j_end = min(h_width, center_j + half_size + 1)
    
    # 提取 HSI patch: (Bands, H_patch, W_patch)
    hsi_patch = hsi_normalized[:, i_start:i_end, j_start:j_end]
    
    # 提取 LiDAR patch: (1, H_patch, W_patch)
    lidar_patch = lidar_normalized[:, i_start:i_end, j_start:j_end]
    
    print(f"  原始 HSI patch shape: {hsi_patch.shape}")
    print(f"  原始 LiDAR patch shape: {lidar_patch.shape}")
    
    # 如果 patch 不够大，进行 padding
    if hsi_patch.shape[1] < patch_size or hsi_patch.shape[2] < patch_size:
        pad_i = patch_size - hsi_patch.shape[1]
        pad_j = patch_size - hsi_patch.shape[2]
        hsi_patch = np.pad(hsi_patch, ((0, 0), (0, pad_i), (0, pad_j)), mode='reflect')
        lidar_patch = np.pad(lidar_patch, ((0, 0), (0, pad_i), (0, pad_j)), mode='reflect')
    
    print(f"  处理后 HSI patch shape: {hsi_patch.shape}")
    print(f"  处理后 LiDAR patch shape: {lidar_patch.shape}")
    
    # =========================================================================
    # 4. 实例化 AdaptiveBandGraphSelector
    # =========================================================================
    print("\n[4] 实例化 AdaptiveBandGraphSelector...")
    
    num_bands = hsi_patch.shape[0]
    
    model = AdaptiveBandGraphSelector(
        node_feature_dim=3,
        gat_hidden_dim=64,
        gat_out_dim=32,
        score_hidden_dim=64,
        n_heads=4,
        dropout=0.1
    )
    
    # 设置为评估模式
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数总数: {total_params:,}")
    
    # =========================================================================
    # 5. 运行模型
    # =========================================================================
    print("\n[5] 运行模型...")
    
    # 转换为 torch tensor
    hsi_tensor = torch.from_numpy(hsi_patch).float()
    lidar_tensor = torch.from_numpy(lidar_patch).float()
    
    print(f"  输入 HSI tensor: {hsi_tensor.shape}")
    print(f"  输入 LiDAR tensor: {lidar_tensor.shape}")
    
    # 模式 1: 仅返回波段权重
    print("\n  --- 模式 1: 返回波段权重 ---")
    with torch.no_grad():
        band_weights = model(hsi_tensor, lidar_tensor, mode='weights')
    
    print(f"  Band weights shape: {band_weights.shape}")
    print(f"  Band weights 范围: [{band_weights.min().item():.4f}, {band_weights.max().item():.4f}]")
    print(f"  Band weights 均值: {band_weights.mean().item():.4f}")
    
    # 显示前10个波段权重
    print(f"\n  前10个波段权重:")
    for i in range(min(10, num_bands)):
        print(f"    Band {i}: {band_weights[i].item():.4f}")
    
    # 显示最重要和最不重要的波段
    top5_idx = torch.argsort(band_weights, descending=True)[:5]
    bottom5_idx = torch.argsort(band_weights)[:5]
    
    print(f"\n  最重要5个波段: {top5_idx.tolist()}")
    print(f"  对应权重: {[f'{band_weights[i].item():.4f}' for i in top5_idx]}")
    print(f"\n  最不重要5个波段: {bottom5_idx.tolist()}")
    print(f"  对应权重: {[f'{band_weights[i].item():.4f}' for i in bottom5_idx]}")
    
    # 模式 2: 返回加权后的 HSI
    print("\n  --- 模式 2: 返回加权后的 HSI ---")
    with torch.no_grad():
        weighted_hsi = model(hsi_tensor, lidar_tensor, mode='weighted')
    
    print(f"  Weighted HSI shape: {weighted_hsi.shape}")
    print(f"  Weighted HSI 范围: [{weighted_hsi.min().item():.4f}, {weighted_hsi.max().item():.4f}]")
    
    # =========================================================================
    # 6. 验证输出正确性
    # =========================================================================
    print("\n[6] 验证输出正确性...")
    
    # 手动计算加权 HSI
    manual_weighted = hsi_tensor * band_weights.view(-1, 1, 1)
    diff = torch.abs(weighted_hsi - manual_weighted).max()
    
    print(f"  手动加权与模型输出最大差异: {diff.item():.10f}")
    print(f"  验证结果: {'通过 ✓' if diff < 1e-6 else '失败 ✗'}")
    
    # =========================================================================
    # 7. 完成总结
    # =========================================================================
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)
    print(f"\n总结:")
    print(f"  - 数据集: Houston2013")
    print(f"  - 波段数: {num_bands}")
    print(f"  - Patch 大小: {patch_size} x {patch_size}")
    print(f"  - 模型参数: {total_params:,}")
    print(f"  - Band weights shape: {band_weights.shape}")
    print(f"  - Weighted HSI shape: {weighted_hsi.shape}")
    print(f"  - 模块运行状态: 正常 ✓")


if __name__ == "__main__":
    main()
