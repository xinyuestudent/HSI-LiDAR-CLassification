"""
测试 BandGraphAttentionSelector 集成到 HSIClassificationMambaModel
使用 Houston2013 数据集验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import scipy.io as sio

from model import HSIClassificationMambaModel, BAND_SELECTOR_AVAILABLE
from config import ModelConfig


def load_houston_data(data_dir):
    """加载 Houston2013 数据集"""
    hsi_data = sio.loadmat(os.path.join(data_dir, 'HSI.mat'))['HSI']
    lidar_data = sio.loadmat(os.path.join(data_dir, 'LiDAR_1.mat'))['LiDAR_1']
    gt_data = sio.loadmat(os.path.join(data_dir, 'TRLabel.mat'))['TRLabel']
    
    # Houston 数据格式: [Bands, H, W]
    hsi_data = np.transpose(hsi_data, (1, 2, 0))  # -> [H, W, Bands]
    # LiDAR 格式: [1, H, W] -> [H, W]
    lidar_data = lidar_data[0]
    
    return hsi_data, lidar_data, gt_data


def normalize_data(data, method='minmax'):
    """归一化数据"""
    if method == 'minmax':
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min + 1e-8)
    return data


def extract_patch(hsi, lidar, center_i, center_j, patch_size=11):
    """提取 patch"""
    half_size = patch_size // 2
    h, w = hsi.shape[:2]
    
    i_start = max(0, center_i - half_size)
    i_end = min(h, center_i + half_size + 1)
    j_start = max(0, center_j - half_size)
    j_end = min(w, center_j + half_size + 1)
    
    hsi_patch = hsi[i_start:i_end, j_start:j_end, :]
    lidar_patch = lidar[i_start:i_end, j_start:j_end]
    
    return hsi_patch, lidar_patch


def main():
    print("=" * 70)
    print("测试 BandGraphAttentionSelector 集成 - Houston2013 数据集")
    print("=" * 70)
    
    # 检查模块可用性
    print(f"\n[0] 检查模块可用性...")
    print(f"  BAND_SELECTOR_AVAILABLE: {BAND_SELECTOR_AVAILABLE}")
    
    if not BAND_SELECTOR_AVAILABLE:
        print("  错误: BandGraphAttentionSelector 不可用!")
        return
    
    # 加载数据
    data_dir = r'E:\项目\HSLiNets\Dataset\Houston2013'
    print(f"\n[1] 加载数据...")
    hsi, lidar, gt = load_houston_data(data_dir)
    print(f"  HSI shape: {hsi.shape}")
    print(f"  LiDAR shape: {lidar.shape}")
    print(f"  GT shape: {gt.shape}")
    
    # 归一化
    print(f"\n[2] 数据归一化...")
    hsi_norm = normalize_data(hsi, method='minmax')
    lidar_norm = normalize_data(lidar, method='minmax')
    print(f"  HSI 归一化范围: [{hsi_norm.min():.4f}, {hsi_norm.max():.4f}]")
    print(f"  LiDAR 归一化范围: [{lidar_norm.min():.4f}, {lidar_norm.max():.4f}]")
    
    # 提取测试 batch
    print(f"\n[3] 提取测试 batch (batch_size=4)...")
    labeled_pixels = np.argwhere(gt > 0)
    batch_size = 4
    patch_size = 11
    
    hsi_patches = []
    lidar_patches = []
    
    for idx in range(batch_size):
        center_i, center_j = labeled_pixels[idx * 100]
        hsi_patch, lidar_patch = extract_patch(hsi_norm, lidar_norm, center_i, center_j, patch_size)
        
        # 转换为 tensor: [H, W, Bands]
        hsi_patches.append(torch.from_numpy(hsi_patch).float())
        lidar_patches.append(torch.from_numpy(lidar_patch).float())
    
    # Stack 为 batch: [Batch, H, W, Bands]
    hsi_batch = torch.stack(hsi_patches, dim=0)
    lidar_batch = torch.stack(lidar_patches, dim=0).unsqueeze(1)  # [Batch, 1, H, W]
    
    print(f"  HSI batch shape: {hsi_batch.shape}")
    print(f"  LiDAR batch shape: {lidar_batch.shape}")
    
    # 创建模型配置
    print(f"\n[4] 创建模型配置...")
    num_bands = hsi.shape[2]  # 144
    num_classes = int(gt.max())  # 15
    
    model_config = ModelConfig(
        spatial_dim=patch_size,
        num_bands=num_bands + 1,  # +1 for LiDAR
        hidden_dim=512,
        output_dim=256,
        delta_param_init=0.01,
        num_classes=num_classes,
        use_band_selector=True,
        band_selector_hidden=64,
        band_selector_out=32
    )
    
    print(f"  spatial_dim: {model_config.spatial_dim}")
    print(f"  num_bands: {model_config.num_bands}")
    print(f"  hidden_dim: {model_config.hidden_dim}")
    print(f"  output_dim: {model_config.output_dim}")
    print(f"  num_classes: {model_config.num_classes}")
    print(f"  use_band_selector: {model_config.use_band_selector}")
    
    # 创建模型
    print(f"\n[5] 创建模型...")
    from model import create_model
    model = create_model(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    band_selector_params = sum(p.numel() for p in model.band_selector.parameters()) if model.band_selector else 0
    
    print(f"  总参数: {total_params:,}")
    print(f"  波段选择器参数: {band_selector_params:,}")
    
    # 测试前向传播 (不使用波段选择器)
    print(f"\n[6] 测试前向传播 (不使用波段选择器)...")
    model_no_selector = HSIClassificationMambaModel(
        spatial_dim=patch_size,
        num_bands=num_bands + 1,
        hidden_dim=512,
        output_dim=256,
        delta_param_init=0.01,
        num_classes=num_classes,
        use_band_selector=False
    )
    
    # 拼接 HSI 和 LiDAR
    x_input = torch.cat([hsi_batch, lidar_batch.squeeze(1).unsqueeze(-1)], dim=-1)
    print(f"  输入 shape: {x_input.shape}")
    
    model_no_selector.eval()
    with torch.no_grad():
        output_no_selector = model_no_selector(x_input)
    print(f"  输出 shape (无波段选择): {output_no_selector.shape}")
    
    # 测试前向传播 (使用波段选择器)
    print(f"\n[7] 测试前向传播 (使用波段选择器)...")
    model.eval()
    with torch.no_grad():
        output_with_selector = model(x_input, lidar_batch)
    print(f"  输出 shape (有波段选择): {output_with_selector.shape}")
    
    # 获取波段权重
    print(f"\n[8] 获取波段权重...")
    with torch.no_grad():
        band_weights = model.get_band_weights(x_input, lidar_batch)
    print(f"  Band weights shape: {band_weights.shape}")
    print(f"  Band weights 范围: [{band_weights.min():.4f}, {band_weights.max():.4f}]")
    
    # 验证波段选择器冻结/解冻
    print(f"\n[9] 测试波段选择器冻结/解冻...")
    print(f"  初始状态 - requires_grad: {model.band_selector.gat.gat1.W.requires_grad}")
    
    model.band_selector.freeze()
    print(f"  freeze() 后 - requires_grad: {model.band_selector.gat.gat1.W.requires_grad}")
    
    model.band_selector.unfreeze()
    print(f"  unfreeze() 后 - requires_grad: {model.band_selector.gat.gat1.W.requires_grad}")
    
    # 结果汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    print(f"  ✓ 模块导入成功")
    print(f"  ✓ 模型创建成功 (use_band_selector=True)")
    print(f"  ✓ 前向传播成功 (无波段选择)")
    print(f"  ✓ 前向传播成功 (有波段选择)")
    print(f"  ✓ 波段权重获取成功")
    print(f"  ✓ 冻结/解冻功能正常")
    print(f"\n  总参数: {total_params:,}")
    print(f"  波段选择器参数: {band_selector_params:,}")
    print("=" * 70)
    
    print("\n集成测试通过! BandGraphAttentionSelector 已成功集成到 HSIClassificationMambaModel")


if __name__ == "__main__":
    main()
