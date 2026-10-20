"""
HSLiNets 诊断脚本
逐步检查数据加载、预处理、模型前向传播等环节
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. 数据加载检查
# =============================================================================

def check_data_loading():
    """检查数据加载"""
    print("\n" + "=" * 70)
    print("1. 数据加载检查")
    print("=" * 70)
    
    data_dir = r'E:\项目\HSLiNets\Dataset\Houston2013'
    
    # 加载原始数据
    print("\n[1.1] 加载原始 .mat 文件...")
    hsi_data = sio.loadmat(os.path.join(data_dir, 'HSI.mat'))
    lidar_data = sio.loadmat(os.path.join(data_dir, 'LiDAR_1.mat'))
    train_gt_data = sio.loadmat(os.path.join(data_dir, 'TRLabel.mat'))
    test_gt_data = sio.loadmat(os.path.join(data_dir, 'TSLabel.mat'))
    
    # 获取实际数据
    hsi = hsi_data['HSI']
    lidar = lidar_data['LiDAR_1']
    train_gt = train_gt_data['TRLabel']
    test_gt = test_gt_data['TSLabel']
    
    print(f"  HSI raw shape: {hsi.shape}")
    print(f"  LiDAR raw shape: {lidar.shape}")
    print(f"  Train GT shape: {train_gt.shape}")
    print(f"  Test GT shape: {test_gt.shape}")
    
    # 检查数据格式
    print("\n[1.2] 检查数据格式...")
    
    # HSI 格式判断
    if hsi.shape[0] < hsi.shape[2]:
        print(f"  HSI 格式: [Bands, H, W] = [{hsi.shape[0]}, {hsi.shape[1]}, {hsi.shape[2]}]")
        hsi_hwc = np.transpose(hsi, (1, 2, 0))  # 转为 H, W, Bands
        print(f"  转换后 HSI 格式: [H, W, Bands] = {hsi_hwc.shape}")
    else:
        print(f"  HSI 格式: [H, W, Bands]")
        hsi_hwc = hsi
    
    # LiDAR 格式判断
    if lidar.ndim == 3:
        print(f"  LiDAR 格式: [Channel, H, W] = {lidar.shape}")
        lidar_2d = lidar[0]  # 取第一个通道
        print(f"  转换后 LiDAR 格式: [H, W] = {lidar_2d.shape}")
    else:
        print(f"  LiDAR 格式: [H, W] = {lidar.shape}")
        lidar_2d = lidar
    
    # 检查标签分布
    print("\n[1.3] 检查标签分布...")
    print("  训练集标签分布:")
    train_classes, train_counts = np.unique(train_gt[train_gt > 0], return_counts=True)
    for c, cnt in zip(train_classes, train_counts):
        print(f"    Class {int(c)}: {cnt} samples")
    print(f"  总训练样本: {np.sum(train_gt > 0)}")
    
    print("\n  测试集标签分布:")
    test_classes, test_counts = np.unique(test_gt[test_gt > 0], return_counts=True)
    for c, cnt in zip(test_classes, test_counts):
        print(f"    Class {int(c)}: {cnt} samples")
    print(f"  总测试样本: {np.sum(test_gt > 0)}")
    
    return hsi_hwc, lidar_2d, train_gt, test_gt


# =============================================================================
# 2. 数据预处理检查
# =============================================================================

def check_preprocessing(hsi, lidar, train_gt, test_gt):
    """检查数据预处理"""
    print("\n" + "=" * 70)
    print("2. 数据预处理检查")
    print("=" * 70)
    
    # 归一化
    print("\n[2.1] 归一化...")
    hsi_min, hsi_max = hsi.min(), hsi.max()
    lidar_min, lidar_max = lidar.min(), lidar.max()
    
    hsi_norm = (hsi - hsi_min) / (hsi_max - hsi_min + 1e-8)
    lidar_norm = (lidar - lidar_min) / (lidar_max - lidar_min + 1e-8)
    
    print(f"  HSI 原始范围: [{hsi_min:.4f}, {hsi_max:.4f}]")
    print(f"  HSI 归一化后: [{hsi_norm.min():.4f}, {hsi_norm.max():.4f}]")
    print(f"  LiDAR 原始范围: [{lidar_min:.4f}, {lidar_max:.4f}]")
    print(f"  LiDAR 归一化后: [{lidar_norm.min():.4f}, {lidar_norm.max():.4f}]")
    
    # 检查 NaN/Inf
    print("\n[2.2] 检查 NaN/Inf...")
    print(f"  HSI NaN count: {np.isnan(hsi_norm).sum()}")
    print(f"  HSI Inf count: {np.isinf(hsi_norm).sum()}")
    print(f"  LiDAR NaN count: {np.isnan(lidar_norm).sum()}")
    print(f"  LiDAR Inf count: {np.isinf(lidar_norm).sum()}")
    
    return hsi_norm, lidar_norm


# =============================================================================
# 3. Patch 提取检查
# =============================================================================

def check_patch_extraction(hsi_norm, lidar_norm, train_gt, test_gt, patch_size=9):
    """检查 patch 提取"""
    print("\n" + "=" * 70)
    print("3. Patch 提取检查")
    print("=" * 70)
    
    half_size = patch_size // 2
    h, w = train_gt.shape
    
    # 找一个训练样本
    train_coords = np.argwhere(train_gt > 0)
    print(f"\n[3.1] 训练样本总数: {len(train_coords)}")
    
    # 提取一个样本的 patch
    center_i, center_j = train_coords[0]
    print(f"\n[3.2] 提取测试 patch (center: {center_i}, {center_j})...")
    print(f"  标签: {train_gt[center_i, center_j]}")
    
    i_start = max(0, center_i - half_size)
    i_end = min(h, center_i + half_size + 1)
    j_start = max(0, center_j - half_size)
    j_end = min(w, center_j + half_size + 1)
    
    hsi_patch = hsi_norm[i_start:i_end, j_start:j_end, :]
    lidar_patch = lidar_norm[i_start:i_end, j_start:j_end]
    
    print(f"  HSI patch shape: {hsi_patch.shape}")
    print(f"  LiDAR patch shape: {lidar_patch.shape}")
    print(f"  HSI patch range: [{hsi_patch.min():.4f}, {hsi_patch.max():.4f}]")
    print(f"  LiDAR patch range: [{lidar_patch.min():.4f}, {lidar_patch.max():.4f}]")
    
    # 检查拼接
    print("\n[3.3] 检查 HSI+LiDAR 拼接...")
    lidar_patch_3d = lidar_patch[..., np.newaxis]
    combined = np.concatenate([hsi_patch, lidar_patch_3d], axis=-1)
    print(f"  Combined shape: {combined.shape}")
    print(f"  Combined range: [{combined.min():.4f}, {combined.max():.4f}]")
    
    return combined


# =============================================================================
# 4. 模型输入格式检查
# =============================================================================

def check_model_input(patch_data):
    """检查模型输入格式"""
    print("\n" + "=" * 70)
    print("4. 模型输入格式检查")
    print("=" * 70)
    
    # 转换为 tensor
    print("\n[4.1] 转换为 tensor...")
    
    # 方式1: [H, W, Bands] -> [1, H, W, Bands] (batch)
    tensor_batch = torch.from_numpy(patch_data.astype(np.float32)).unsqueeze(0)
    print(f"  Tensor shape (batch): {tensor_batch.shape}")
    print(f"  Tensor dtype: {tensor_batch.dtype}")
    print(f"  Tensor range: [{tensor_batch.min():.4f}, {tensor_batch.max():.4f}]")
    
    # 检查 NaN/Inf
    print(f"  Tensor NaN count: {torch.isnan(tensor_batch).sum().item()}")
    print(f"  Tensor Inf count: {torch.isinf(tensor_batch).sum().item()}")
    
    return tensor_batch


# =============================================================================
# 5. 模型前向传播检查
# =============================================================================

def check_model_forward(tensor_batch):
    """检查模型前向传播"""
    print("\n" + "=" * 70)
    print("5. 模型前向传播检查")
    print("=" * 70)
    
    from model import HSIClassificationMambaModel
    
    num_bands = tensor_batch.shape[-1]  # 145 (144 HSI + 1 LiDAR)
    num_classes = 15
    
    print(f"\n[5.1] 创建模型...")
    print(f"  num_bands: {num_bands}")
    print(f"  num_classes: {num_classes}")
    
    model = HSIClassificationMambaModel(
        spatial_dim=9,
        num_bands=num_bands,
        hidden_dim=512,
        output_dim=256,
        delta_param_init=0.01,
        num_classes=num_classes,
        use_band_selector=False
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
    
    # 前向传播
    print("\n[5.2] 前向传播...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(tensor_batch)
            print(f"  输出 shape: {output.shape}")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"  输出 NaN count: {torch.isnan(output).sum().item()}")
            
            # 检查输出分布
            probs = torch.softmax(output, dim=-1)
            print(f"\n  Softmax 概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
            print(f"  概率和: {probs.sum():.4f}")
            
            # 预测类别
            pred = torch.argmax(output, dim=-1)
            print(f"  预测类别: {pred.item()}")
            
        except Exception as e:
            print(f"  前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    return model


# =============================================================================
# 6. 训练流程检查
# =============================================================================

def check_training_flow():
    """检查训练流程"""
    print("\n" + "=" * 70)
    print("6. 训练流程检查")
    print("=" * 70)
    
    from dataset import prepare_dataloaders
    from config import DATASET_CONFIGS, get_class_dict
    
    dataset_config = DATASET_CONFIGS["houston"]
    class_dict = get_class_dict(dataset_config.class_info)
    
    print("\n[6.1] 准备数据加载器...")
    print(f"  类别数: {len(class_dict)}")
    
    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(
            dataset_type="houston",
            data_dir=dataset_config.data_path,
            class_dict=class_dict,
            patch_size=9,
            batch_size=32,
            val_ratio=0.1,
            augment=False,
            num_workers=0
        )
        
        print(f"\n[6.2] 检查数据加载器...")
        print(f"  训练 batches: {len(train_loader)}")
        print(f"  验证 batches: {len(val_loader)}")
        print(f"  测试 batches: {len(test_loader)}")
        
        # 检查一个 batch
        print(f"\n[6.3] 检查第一个 batch...")
        data, labels = next(iter(train_loader))
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Labels unique: {torch.unique(labels).tolist()}")
        print(f"  Labels range: [{labels.min()}, {labels.max()}]")
        
        # 检查标签是否正确 (应该是 0-14)
        if labels.min() < 0 or labels.max() >= 15:
            print(f"  ⚠️ 标签范围异常! 应该是 0-14")
        else:
            print(f"  ✓ 标签范围正确")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"  数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =============================================================================
# 7. 模型训练一步检查
# =============================================================================

def check_one_training_step(train_loader):
    """检查一步训练"""
    print("\n" + "=" * 70)
    print("7. 单步训练检查")
    print("=" * 70)
    
    if train_loader is None:
        print("  跳过: train_loader 为空")
        return
    
    from model import HSIClassificationMambaModel
    import torch.nn as nn
    
    # 创建模型
    model = HSIClassificationMambaModel(
        spatial_dim=9,
        num_bands=145,
        hidden_dim=512,
        output_dim=256,
        delta_param_init=0.01,
        num_classes=15,
        use_band_selector=False
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 获取一个 batch
    data, labels = next(iter(train_loader))
    
    print(f"\n[7.1] 前向传播...")
    model.train()
    optimizer.zero_grad()
    
    output = model(data)
    print(f"  Output shape: {output.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    print(f"\n[7.2] 计算损失...")
    loss = criterion(output, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    print(f"\n[7.3] 反向传播...")
    loss.backward()
    
    # 检查梯度
    print(f"\n[7.4] 检查梯度...")
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
    
    # 打印最大和最小梯度
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    print(f"  最大梯度:")
    for name, norm in grad_norms[:3]:
        print(f"    {name}: {norm:.6f}")
    print(f"  最小梯度:")
    for name, norm in grad_norms[-3:]:
        print(f"    {name}: {norm:.6f}")
    
    print(f"\n[7.5] 更新参数...")
    optimizer.step()
    print(f"  ✓ 参数更新成功")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("HSLiNets 诊断脚本 - Houston 数据集")
    print("=" * 70)
    
    # 1. 数据加载检查
    hsi, lidar, train_gt, test_gt = check_data_loading()
    
    # 2. 数据预处理检查
    hsi_norm, lidar_norm = check_preprocessing(hsi, lidar, train_gt, test_gt)
    
    # 3. Patch 提取检查
    patch_data = check_patch_extraction(hsi_norm, lidar_norm, train_gt, test_gt)
    
    # 4. 模型输入格式检查
    tensor_batch = check_model_input(patch_data)
    
    # 5. 模型前向传播检查
    model = check_model_forward(tensor_batch)
    
    # 6. 训练流程检查
    train_loader, val_loader, test_loader = check_training_flow()
    
    # 7. 单步训练检查
    check_one_training_step(train_loader)
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
