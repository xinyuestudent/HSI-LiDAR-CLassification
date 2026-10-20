"""
测试脚本：使用 MUUFL Gulfport 数据集验证 AdaptiveBandGraphSelector 模块
"""

import os
import sys
import numpy as np
import torch
import scipy.io as sio

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_band_graph_selector import AdaptiveBandGraphSelector


def load_muufl_data(data_dir: str):
    """
    加载 MUUFL Gulfport 数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        hsi: 高光谱数据, shape = (H, W, Bands)
        lidar: LiDAR 数据 (DEM), shape = (H, W)
    """
    data_path = os.path.join(data_dir, 'muufl_gulfport_campus_w_lidar_1.mat')
    
    print(f"加载数据文件: {data_path}")
    data = sio.loadmat(data_path)
    
    # 提取 HSI 数据
    hsi_struct = data['hsi'][0, 0]
    hsi = hsi_struct['Data']  # shape: (325, 337, 72)
    
    # 提取 LiDAR 数据 (z 字段包含 DEM 和 intensity)
    lidar_struct = hsi_struct['Lidar'][0, 0]
    lidar_z = lidar_struct['z'][0, 0]  # shape: (325, 337, 2)
    
    # 使用 DEM (第一个通道) 作为 LiDAR 数据
    # lidar_z[:,:,0] 是 DEM, lidar_z[:,:,1] 是 intensity
    lidar = lidar_z[:, :, 0]  # shape: (325, 337)
    
    return hsi, lidar


def normalize_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    归一化数据
    
    Args:
        data: 输入数据
        method: 归一化方法
        
    Returns:
        归一化后的数据
    """
    if method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    elif method == "zscore":
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    return data


def extract_patch(hsi: np.ndarray, lidar: np.ndarray, 
                  center_i: int, center_j: int, 
                  patch_size: int = 11):
    """
    提取指定位置的 patch
    
    Args:
        hsi: HSI 数据
        lidar: LiDAR 数据
        center_i: 中心行索引
        center_j: 中心列索引
        patch_size: patch 大小
        
    Returns:
        hsi_patch: shape = (Bands, patch_size, patch_size)
        lidar_patch: shape = (1, patch_size, patch_size)
    """
    half = patch_size // 2
    h, w = hsi.shape[:2]
    
    # 边界检查
    i_start = max(0, center_i - half)
    i_end = min(h, center_i + half + 1)
    j_start = max(0, center_j - half)
    j_end = min(w, center_j + half + 1)
    
    # 如果超出边界，用零填充
    hsi_patch = np.zeros((patch_size, patch_size, hsi.shape[2]), dtype=hsi.dtype)
    lidar_patch = np.zeros((patch_size, patch_size), dtype=lidar.dtype)
    
    hsi_patch[i_start - (center_i - half):i_end - (center_i - half),
              j_start - (center_j - half):j_end - (center_j - half)] = hsi[i_start:i_end, j_start:j_end]
    lidar_patch[i_start - (center_i - half):i_end - (center_i - half),
                j_start - (center_j - half):j_end - (center_j - half)] = lidar[i_start:i_end, j_start:j_end]
    
    # 转换为 格式
    hsi_patch = np.transpose(hsi_patch, (2, 0, 1))  # (Bands, H, W)
    lidar_patch = lidar_patch[np.newaxis, :, :]     # (1, H, W)
    
    return hsi_patch, lidar_patch


def main():
    print("=" * 70)
    print("使用 MUUFL Gulfport 数据集验证 AdaptiveBandGraphSelector 模块")
    print("=" * 70)
    
    # 数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "Dataset", "muuflgulfport")
    
    # 1. 加载数据
    print("\n[1] 加载 MUUFL Gulfport 数据集...")
    hsi, lidar = load_muufl_data(data_dir)
    
    print(f"  HSI shape: {hsi.shape}")
    print(f"  LiDAR shape: {lidar.shape}")
    print(f"  HSI 波段数: {hsi.shape[2]}")
    print(f"  HSI 数据范围: [{hsi.min():.4f}, {hsi.max():.4f}]")
    print(f"  LiDAR 数据范围: [{lidar.min():.4f}, {lidar.max():.4f}]")
    
    # 2. 归一化
    print("\n[2] 数据归一化...")
    hsi_normalized = normalize_data(hsi, method="minmax")
    lidar_normalized = normalize_data(lidar, method="minmax")
    
    print(f"  HSI 归一化范围: [{hsi_normalized.min():.4f}, {hsi_normalized.max():.4f}]")
    print(f"  LiDAR 归一化范围: [{lidar_normalized.min():.4f}, {lidar_normalized.max():.4f}]")
    
    # 3. 提取测试 patch
    print("\n[3] 提取测试 patch...")
    patch_size = 11
    # 选择中心位置
    center_i, center_j = hsi.shape[0] // 2, hsi.shape[1] // 2
    
    hsi_patch, lidar_patch = extract_patch(
        hsi_normalized, lidar_normalized, center_i, center_j, patch_size
    )
    
    print(f"  选择像素位置: ({center_i}, {center_j})")
    print(f"  HSI patch shape: {hsi_patch.shape}")
    print(f"  LiDAR patch shape: {lidar_patch.shape}")
    
    # 4. 实例化模型
    print("\n[4] 实例化 AdaptiveBandGraphSelector...")
    model = AdaptiveBandGraphSelector(
        node_feature_dim=3,
        gat_hidden_dim=64,
        gat_out_dim=32,
        score_hidden_dim=64,
        n_heads=4,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数总数: {total_params:,}")
    
    # 5. 转换为 tensor
    hsi_tensor = torch.from_numpy(hsi_patch).float()
    lidar_tensor = torch.from_numpy(lidar_patch).float()
    
    print(f"\n[5] 运行模型...")
    print(f"  输入 HSI tensor: {hsi_tensor.shape}")
    print(f"  输入 LiDAR tensor: {lidar_tensor.shape}")
    
    # 6. 模式 1: 返回波段权重
    print(f"\n  --- 模式 1: 返回波段权重 ---")
    band_weights = model(hsi_tensor, lidar_tensor, mode='weights')
    print(f"  Band weights shape: {band_weights.shape}")
    print(f"  Band weights 范围: [{band_weights.min().item():.4f}, {band_weights.max().item():.4f}]")
    print(f"  Band weights 均值: {band_weights.mean().item():.4f}")
    
    # 显示前 10 个波段权重
    print(f"\n  前10个波段权重:")
    for i in range(min(10, len(band_weights))):
        print(f"    Band {i}: {band_weights[i].item():.4f}")
    
    # 显示最重要和最不重要的波段
    sorted_indices = torch.argsort(band_weights, descending=True)
    print(f"\n  最重要5个波段: {sorted_indices[:5].tolist()}")
    print(f"  对应权重: {[f'{band_weights[i].item():.4f}' for i in sorted_indices[:5]]}")
    print(f"  最不重要5个波段: {sorted_indices[-5:].tolist()}")
    print(f"  对应权重: {[f'{band_weights[i].item():.4f}' for i in sorted_indices[-5:]]}")
    
    # 7. 模式 2: 返回加权后的 HSI
    print(f"\n  --- 模式 2: 返回加权后的 HSI ---")
    weighted_hsi = model(hsi_tensor, lidar_tensor, mode='weighted')
    print(f"  Weighted HSI shape: {weighted_hsi.shape}")
    print(f"  Weighted HSI 范围: [{weighted_hsi.min().item():.4f}, {weighted_hsi.max().item():.4f}]")
    
    # 8. 验证正确性
    print(f"\n[6] 验证输出正确性...")
    manual_weighted = hsi_tensor * band_weights.view(-1, 1, 1)
    diff = torch.abs(weighted_hsi - manual_weighted).max()
    print(f"  手动加权与模型输出最大差异: {diff.item():.10f}")
    
    if diff.item() < 1e-6:
        print(f"  验证结果: 通过 ✓")
    else:
        print(f"  验证结果: 失败 ✗")
    
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)
    
    # 总结
    print(f"\n总结:")
    print(f"  - 数据集: MUUFL Gulfport")
    print(f"  - 波段数: {hsi.shape[2]}")
    print(f"  - Patch 大小: {patch_size} x {patch_size}")
    print(f"  - 模型参数: {total_params:,}")
    print(f"  - Band weights shape: {band_weights.shape}")
    print(f"  - Weighted HSI shape: {weighted_hsi.shape}")
    print(f"  - 模块运行状态: 正常 ✓")


if __name__ == "__main__":
    main()
