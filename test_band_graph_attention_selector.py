"""
测试 BandGraphAttentionSelector 模块
支持 Trento、Houston2013、MUUFL Gulfport 三个数据集
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import scipy.io as sio

from band_graph_attention_selector import BandGraphAttentionSelector


def load_trento_data(data_dir):
    """加载 Trento 数据集"""
    hsi_data = sio.loadmat(os.path.join(data_dir, 'HSI_Trento.mat'))['HSI_Trento']
    lidar_data = sio.loadmat(os.path.join(data_dir, 'Lidar_Trento.mat'))['Lidar_Trento']
    gt_data = sio.loadmat(os.path.join(data_dir, 'GT_Trento.mat'))['GT_Trento']
    return hsi_data, lidar_data, gt_data


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


def load_muufl_data(data_dir):
    """加载 MUUFL Gulfport 数据集"""
    data = sio.loadmat(os.path.join(data_dir, 'muufl_gulfport_campus_w_lidar_1.mat'))
    hsi_struct = data['hsi'][0, 0]
    hsi_data = hsi_struct['Data']
    lidar_struct = hsi_struct['Lidar'][0, 0]
    lidar_z = lidar_struct['z'][0, 0]
    lidar_data = lidar_z[:, :, 0]  # DEM 高程数据
    gt_data = np.zeros((hsi_data.shape[0], hsi_data.shape[1]))  # 无 GT 标签
    return hsi_data, lidar_data, gt_data


def normalize_data(data, method='minmax'):
    """归一化数据"""
    if method == 'minmax':
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min + 1e-8)
    return data


def extract_patch(hsi, lidar, gt, center_i, center_j, patch_size=11):
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


def test_dataset(dataset_name, data_dir, load_func, center_i=None, center_j=None):
    """测试单个数据集"""
    print("\n" + "=" * 70)
    print(f"测试 BandGraphAttentionSelector - {dataset_name} 数据集")
    print("=" * 70)
    
    # 加载数据
    print(f"\n[1] 加载数据...")
    hsi, lidar, gt = load_func(data_dir)
    print(f"  HSI shape: {hsi.shape}")
    print(f"  LiDAR shape: {lidar.shape}")
    print(f"  GT shape: {gt.shape}")
    
    # 归一化
    print(f"\n[2] 数据归一化...")
    hsi_norm = normalize_data(hsi, method='minmax')
    lidar_norm = normalize_data(lidar, method='minmax')
    print(f"  HSI 归一化范围: [{hsi_norm.min():.4f}, {hsi_norm.max():.4f}]")
    print(f"  LiDAR 归一化范围: [{lidar_norm.min():.4f}, {lidar_norm.max():.4f}]")
    
    # 选择一个像素
    print(f"\n[3] 提取测试 patch...")
    if center_i is None or center_j is None:
        if gt.max() > 0:
            labeled_pixels = np.argwhere(gt > 0)
            center_i, center_j = labeled_pixels[len(labeled_pixels) // 2]
        else:
            h, w = hsi.shape[:2]
            center_i, center_j = h // 2, w // 2
    
    print(f"  选择像素位置: ({center_i}, {center_j})")
    if gt.max() > 0:
        print(f"  该像素标签: {gt[center_i, center_j]}")
    
    hsi_patch, lidar_patch = extract_patch(hsi_norm, lidar_norm, gt, center_i, center_j, patch_size=11)
    
    # 转换为 torch tensor: [B, H, W]
    hsi_tensor = torch.from_numpy(hsi_patch).permute(2, 0, 1).float()
    lidar_tensor = torch.from_numpy(lidar_patch).unsqueeze(0).float()
    
    print(f"  HSI patch shape: {hsi_tensor.shape}")
    print(f"  LiDAR patch shape: {lidar_tensor.shape}")
    
    # 实例化模型
    print(f"\n[4] 实例化 BandGraphAttentionSelector...")
    model = BandGraphAttentionSelector(
        node_feature_dim=3,
        gat_hidden_dim=64,
        gat_out_dim=32,
        score_hidden_dim=64,
        n_heads=4,
        dropout=0.1,
        verbose=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数总数: {total_params:,}")
    
    # 运行模型
    print(f"\n[5] 运行模型...")
    weighted_hsi = model(hsi_tensor, lidar_tensor, mode='weighted')
    
    # 验证输出
    print(f"\n[6] 验证输出...")
    band_weights = model(hsi_tensor, lidar_tensor, mode='weights')
    
    # 验证 residual weighting
    manual_weighted = hsi_tensor * (1 + band_weights).view(-1, 1, 1)
    diff = torch.abs(weighted_hsi - manual_weighted).max()
    
    if diff < 1e-5:
        print(f"  Residual weighting 验证通过 ✓")
        success = True
    else:
        print(f"  Residual weighting 验证失败 ✗ (diff={diff.item():.10f})")
        success = False
    
    print("\n" + "-" * 70)
    print(f"{dataset_name} 测试结果:")
    print(f"  - 波段数: {hsi_tensor.shape[0]}")
    print(f"  - Patch 大小: {hsi_tensor.shape[1]} x {hsi_tensor.shape[2]}")
    print(f"  - Band weights shape: {band_weights.shape}")
    print(f"  - Weighted HSI shape: {weighted_hsi.shape}")
    print(f"  - 状态: {'正常 ✓' if success else '失败 ✗'}")
    print("-" * 70)
    
    return success, hsi_tensor.shape[0], hsi_tensor.shape[1], hsi_tensor.shape[2], band_weights.shape, weighted_hsi.shape


def main():
    print("=" * 70)
    print("BandGraphAttentionSelector 多数据集测试")
    print("=" * 70)
    
    datasets = [
        {
            'name': 'Trento',
            'data_dir': r'E:\项目\HSLiNets\Dataset\TrentoDataset',
            'load_func': load_trento_data,
            'center_i': None,
            'center_j': None
        },
        {
            'name': 'Houston2013',
            'data_dir': r'E:\项目\HSLiNets\Dataset\Houston2013',
            'load_func': load_houston_data,
            'center_i': None,
            'center_j': None
        },
        {
            'name': 'MUUFL Gulfport',
            'data_dir': r'E:\项目\HSLiNets\Dataset\muuflgulfport',
            'load_func': load_muufl_data,
            'center_i': None,
            'center_j': None
        }
    ]
    
    results = []
    for ds in datasets:
        try:
            success, bands, h, w, weights_shape, weighted_shape = test_dataset(
                ds['name'], ds['data_dir'], ds['load_func'], ds['center_i'], ds['center_j']
            )
            results.append({
                'name': ds['name'],
                'success': success,
                'bands': bands,
                'patch_size': f"{h}x{w}",
                'weights_shape': weights_shape,
                'weighted_shape': weighted_shape
            })
        except Exception as e:
            print(f"\n{ds['name']} 测试失败: {e}")
            results.append({
                'name': ds['name'],
                'success': False,
                'error': str(e)
            })
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    print(f"\n{'数据集':<20} {'波段数':<10} {'Patch大小':<12} {'状态':<10}")
    print("-" * 52)
    for r in results:
        if r['success']:
            print(f"{r['name']:<20} {r['bands']:<10} {r['patch_size']:<12} {'通过 ✓':<10}")
        else:
            print(f"{r['name']:<20} {'N/A':<10} {'N/A':<12} {'失败 ✗':<10}")
    
    # 总体结果
    all_passed = all(r['success'] for r in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("所有数据集测试通过! ✓")
    else:
        print("部分数据集测试失败!")
    print("=" * 70)


if __name__ == "__main__":
    main()