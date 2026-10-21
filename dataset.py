"""
HSLiNets 数据集模块
包含数据加载、预处理、Patch 提取、数据增强等功能
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.ndimage import rotate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict, List, Optional
import scipy.io as sio

try:
    import mat73
    MAT73_AVAILABLE = True
except ImportError:
    MAT73_AVAILABLE = False

# rs-fusion-datasets 库支持
try:
    from rs_fusion_datasets import fetch_houston2018_ouc, Houston2018Ouc
    RS_FUSION_AVAILABLE = True
except ImportError:
    RS_FUSION_AVAILABLE = False
    Houston2018Ouc = None

# 导入空间划分模块
from spatial_split import (
    spatial_train_test_split,
    print_split_statistics,
    check_overlap
)


# =============================================================================
# 数据归一化
# =============================================================================

def normalize_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    归一化数据
    
    Args:
        data: 输入数据
        method: 归一化方法 ('minmax' 或 'zscore')
        
    Returns:
        归一化后的数据
    """
    if method == "minmax":
        data_min = np.min(data)
        data_max = np.max(data)
        return (data - data_min) / (data_max - data_min + 1e-8)
    elif method == "zscore":
        scaler = StandardScaler()
        original_shape = data.shape
        if data.ndim == 3:
            # 对于 3D 数据，在最后一个维度上归一化
            data_flat = data.reshape(-1, data.shape[-1])
            data_normalized = scaler.fit_transform(data_flat)
            return data_normalized.reshape(original_shape)
        else:
            return scaler.fit_transform(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# 数据加载
# =============================================================================

def load_trento_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 Trento 数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        hsi: 高光谱数据, shape = (H, W, Bands)
        lidar: LiDAR 数据, shape = (H, W)
        gt: 地面真值标签, shape = (H, W)
    """
    # 加载 HSI
    hsi_path = os.path.join(data_dir, "HSI_Trento.mat")
    hsi_data = sio.loadmat(hsi_path)
    hsi_key = list(hsi_data.keys())[-1]
    hsi = hsi_data[hsi_key]
    
    # 加载 LiDAR
    lidar_path = os.path.join(data_dir, "Lidar_Trento.mat")
    lidar_data = sio.loadmat(lidar_path)
    lidar_key = list(lidar_data.keys())[-1]
    lidar = lidar_data[lidar_key]
    
    # 加载 GT
    gt_path = os.path.join(data_dir, "GT_Trento.mat")
    gt_data = sio.loadmat(gt_path)
    gt_key = list(gt_data.keys())[-1]
    gt = gt_data[gt_key]
    
    return hsi, lidar, gt


def load_houston_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 Houston2013 数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        hsi: 高光谱数据, shape = (Bands, H, W)
        lidar: LiDAR 数据, shape = (1, H, W)
        train_gt: 训练标签, shape = (H, W)
        test_gt: 测试标签, shape = (H, W)
    """
    # 加载 HSI
    hsi_path = os.path.join(data_dir, "HSI.mat")
    hsi_data = sio.loadmat(hsi_path)
    hsi_key = list(hsi_data.keys())[-1]
    hsi = hsi_data[hsi_key]
    
    # 加载 LiDAR
    lidar_path = os.path.join(data_dir, "LiDAR_1.mat")
    lidar_data = sio.loadmat(lidar_path)
    lidar_key = list(lidar_data.keys())[-1]
    lidar = lidar_data[lidar_key]
    
    # 加载训练和测试标签
    train_gt_path = os.path.join(data_dir, "TRLabel.mat")
    train_gt_data = sio.loadmat(train_gt_path)
    train_gt_key = list(train_gt_data.keys())[-1]
    train_gt = train_gt_data[train_gt_key]
    # 处理稀疏矩阵格式
    if hasattr(train_gt, 'todense'):
        train_gt = np.array(train_gt.todense())
    
    test_gt_path = os.path.join(data_dir, "TSLabel.mat")
    test_gt_data = sio.loadmat(test_gt_path)
    test_gt_key = list(test_gt_data.keys())[-1]
    test_gt = test_gt_data[test_gt_key]
    # 处理稀疏矩阵格式
    if hasattr(test_gt, 'todense'):
        test_gt = np.array(test_gt.todense())
    
    return hsi, lidar, train_gt, test_gt


def load_muufl_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 MUUFL Gulfport 数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        hsi: 高光谱数据, shape = (H, W, Bands)
        lidar: LiDAR 数据 (DSM), shape = (H, W)
        gt: 地面真值标签, shape = (H, W)
    """
    # 加载主数据文件
    hsi_path = os.path.join(data_dir, "muufl_gulfport_campus_1_hsi_220_label.mat")
    hsi_data = sio.loadmat(hsi_path)
    
    # 获取结构体
    hsi_struct = hsi_data['hsi'][0, 0]
    
    # 提取 HSI 数据
    hsi = hsi_struct['Data']  # shape: (325, 220, 64)
    
    # 提取标签
    scene_labels = hsi_struct['sceneLabels'][0, 0]
    gt = scene_labels['labels']  # shape: (325, 220)
    
    # 将 -1 (无标签) 转换为 0
    gt = gt.copy()
    gt[gt == -1] = 0
    
    # 提取 LiDAR 数据
    # 尝试从带 LiDAR 的文件加载
    lidar_path = os.path.join(data_dir, "muufl_gulfport_campus_w_lidar_1.mat")
    if os.path.exists(lidar_path):
        lidar_data = sio.loadmat(lidar_path)
        lidar_struct = lidar_data['hsi'][0, 0]
        
        # LiDAR 存储为结构化点云，需要插值到栅格
        lidar_info = lidar_struct['Lidar']
        lidar = _interpolate_lidar_to_grid(lidar_info, hsi.shape[:2])
    else:
        # 如果没有 LiDAR 文件，创建虚拟数据
        print("警告: 未找到 LiDAR 数据，创建虚拟 LiDAR 数据")
        lidar = np.zeros(hsi.shape[:2], dtype=np.float32)
    
    return hsi, lidar, gt


def _interpolate_lidar_to_grid(lidar_info: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    """
    将 LiDAR 点云插值到图像栅格
    
    Args:
        lidar_info: LiDAR 结构化数据
        grid_shape: 目标栅格形状 (H, W)
        
    Returns:
        lidar_grid: 插值后的 LiDAR DSM, shape = (H, W)
    """
    from scipy.interpolate import griddata
    
    H, W = grid_shape
    
    # 尝试解析 LiDAR 数据结构
    try:
        # LiDAR 数据结构: lidar_info[0, 0] 包含 x, y, z 字段
        lidar_struct = lidar_info[0, 0]
        
        # 获取 x, y, z 坐标
        x_field = lidar_struct['x']
        y_field = lidar_struct['y']
        z_field = lidar_struct['z']
        
        # 提取实际数据
        if hasattr(x_field, 'flatten'):
            x = x_field.flatten()
            y = y_field.flatten()
            z = z_field.flatten()
        else:
            # 可能是嵌套结构
            x = np.array(x_field).flatten()
            y = np.array(y_field).flatten()
            z = np.array(z_field).flatten()
        
        # 创建目标网格
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        
        # 使用最近邻插值
        if len(x) > 0 and len(y) > 0 and len(z) > 0:
            # 归一化坐标到图像空间
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * (W - 1)
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8) * (H - 1)
            
            # 插值
            points = np.column_stack([x_norm, y_norm])
            lidar_grid = griddata(points, z, (grid_x, grid_y), method='nearest')
            
            # 填充 NaN 值
            if np.any(np.isnan(lidar_grid)):
                lidar_grid = np.nan_to_num(lidar_grid, nan=0.0)
        else:
            lidar_grid = np.zeros((H, W), dtype=np.float32)
            
    except Exception as e:
        print(f"LiDAR 插值失败: {e}")
        lidar_grid = np.zeros((H, W), dtype=np.float32)
    
    return lidar_grid.astype(np.float32)


def load_houston2018_data(data_dir: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 Houston 2018 数据集
    
    优先从本地目录加载，否则使用 rs-fusion-datasets 库自动下载。
    
    Args:
        data_dir: 数据目录路径 (可选)
        
    Returns:
        hsi: 高光谱数据, shape = (H, W, 50)
        lidar: LiDAR DSM 数据, shape = (H, W)
        gt: 地面真值标签, shape = (H, W)
    """
    # 尝试从本地目录加载
    if data_dir and os.path.exists(data_dir):
        print(f"从本地目录加载 Houston 2018 数据集: {data_dir}")
        
        # 加载 HSI
        hsi_path = os.path.join(data_dir, "houston_hsi.mat")
        hsi_data = sio.loadmat(hsi_path)
        hsi_key = [k for k in hsi_data.keys() if not k.startswith('_')][-1]
        hsi = hsi_data[hsi_key]
        
        # 加载 LiDAR
        lidar_path = os.path.join(data_dir, "houston_lidar.mat")
        lidar_data = sio.loadmat(lidar_path)
        lidar_key = [k for k in lidar_data.keys() if not k.startswith('_')][-1]
        lidar = lidar_data[lidar_key]
        
        # 加载 GT
        gt_path = os.path.join(data_dir, "houston_gt.mat")
        gt_data = sio.loadmat(gt_path)
        gt_key = [k for k in gt_data.keys() if not k.startswith('_')][-1]
        gt = gt_data[gt_key]
        
        # 处理稀疏矩阵
        if hasattr(gt, 'todense'):
            gt = np.array(gt.todense())
        if hasattr(lidar, 'todense'):
            lidar = np.array(lidar.todense())
        
        # 处理 HSI 格式
        if hsi.ndim == 3 and hsi.shape[0] < hsi.shape[2]:
            # (Bands, H, W) -> (H, W, Bands)
            hsi = np.transpose(hsi, (1, 2, 0))
        
        # 处理 LiDAR 格式
        if lidar.ndim == 3:
            if lidar.shape[0] == 1:
                lidar = lidar[0]  # (1, H, W) -> (H, W)
            elif lidar.shape[2] == 1:
                lidar = lidar[:, :, 0]  # (H, W, 1) -> (H, W)
        
        print(f"  HSI shape: {hsi.shape}")
        print(f"  LiDAR shape: {lidar.shape}")
        print(f"  GT shape: {gt.shape}")
        
        return hsi, lidar, gt
    
    # 使用 rs-fusion-datasets 加载
    elif RS_FUSION_AVAILABLE:
        print("使用 rs-fusion-datasets 加载 Houston 2018 数据集...")
        
        hsi, dsm, train_label, test_label, all_label, info = fetch_houston2018_ouc()
        
        # 合并训练和测试标签
        if hasattr(train_label, 'todense'):
            train_label = np.array(train_label.todense())
        if hasattr(test_label, 'todense'):
            test_label = np.array(test_label.todense())
        if hasattr(all_label, 'todense'):
            gt = np.array(all_label.todense())
        else:
            gt = train_label + test_label
        
        # 处理 HSI 格式
        if hsi.ndim == 3 and hsi.shape[0] < hsi.shape[2]:
            hsi = np.transpose(hsi, (1, 2, 0))
        
        # 处理 LiDAR 格式
        if dsm.ndim == 3:
            if dsm.shape[0] == 1:
                lidar = dsm[0]
            elif dsm.shape[0] <= 3:
                lidar = np.transpose(dsm, (1, 2, 0))
            else:
                lidar = dsm.squeeze()
        else:
            lidar = dsm
        
        print(f"  HSI shape: {hsi.shape}")
        print(f"  LiDAR shape: {lidar.shape}")
        print(f"  GT shape: {gt.shape}")
        
        return hsi, lidar, gt
    else:
        raise ImportError(
            "需要安装 rs-fusion-datasets 库或提供本地数据路径:\n"
            "pip install rs-fusion-datasets\n"
            "或将数据放到 Dataset/Houston2018 目录"
        )

# =============================================================================
# Patch 提取
# =============================================================================

def extract_patches(hsi: np.ndarray, lidar: np.ndarray, gt: np.ndarray,
                    class_dict: Dict, patch_size: int = 1,
                    use_all_samples: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取 HSI 和 LiDAR patches
    
    Args:
        hsi: 高光谱数据
        lidar: LiDAR 数据
        gt: 地面真值
        class_dict: 类别字典
        patch_size: Patch 大小
        use_all_samples: 是否使用全部样本
        
    Returns:
        hsi_patches: HSI patches, shape = (N, patch_size, patch_size, Bands)
        lidar_patches: LiDAR patches, shape = (N, patch_size, patch_size, 1)
        labels: 标签, shape = (N,)
    """
    half_size = patch_size // 2
    h, w = gt.shape
    
    # 获取类别列表
    class_ids = list(class_dict.keys())
    
    # 存储样本
    hsi_samples = []
    lidar_samples = []
    labels = []
    
    # 类别计数
    class_count = {cid: 0 for cid in class_ids}
    
    # 目标样本数
    target_samples = {cid: class_dict[cid]["samples"] for cid in class_ids}
    
    def all_classes_completed():
        return all(class_count[cid] >= target_samples[cid] for cid in class_ids)
    
    # 遍历所有像素
    np.random.seed(42)
    
    while not all_classes_completed():
        for label in class_ids:
            if class_count[label] >= target_samples[label]:
                continue
            
            # 获取该类别的所有像素坐标
            coords = np.argwhere(gt == label)
            if len(coords) == 0:
                continue
            
            np.random.shuffle(coords)
            
            for coord in coords:
                i, j = coord
                
                # 计算 patch 边界
                i_start = max(0, i - half_size)
                i_end = min(h, i + half_size + 1)
                j_start = max(0, j - half_size)
                j_end = min(w, j + half_size + 1)
                
                # 检查边界
                if i_end - i_start < patch_size or j_end - j_start < patch_size:
                    continue
                
                # 提取 HSI patch
                if hsi.ndim == 3 and hsi.shape[2] < hsi.shape[0]:
                    # H, W, Bands 格式
                    hsi_patch = hsi[i_start:i_end, j_start:j_end, :]
                else:
                    # Bands, H, W 格式
                    hsi_patch = hsi[:, i_start:i_end, j_start:j_end]
                    hsi_patch = np.transpose(hsi_patch, (1, 2, 0))  # 转为 H, W, Bands
                
                # 提取 LiDAR patch
                if lidar.ndim == 2:
                    lidar_patch = lidar[i_start:i_end, j_start:j_end]
                elif lidar.ndim == 3:
                    if lidar.shape[0] == 1:
                        # (1, H, W) 格式
                        lidar_patch = lidar[0, i_start:i_end, j_start:j_end]
                    else:
                        # (H, W, 1) 格式
                        lidar_patch = lidar[i_start:i_end, j_start:j_end, 0]
                else:
                    lidar_patch = lidar[i_start:i_end, j_start:j_end]
                
                lidar_patch = lidar_patch[..., np.newaxis]  # 添加通道维度
                
                hsi_samples.append(hsi_patch)
                lidar_samples.append(lidar_patch)
                labels.append(label)
                class_count[label] += 1
                
                if class_count[label] >= target_samples[label]:
                    break
        
        # 防止无限循环
        if all_classes_completed():
            break
    
    return np.array(hsi_samples), np.array(lidar_samples), np.array(labels)


# =============================================================================
# 数据增强
# =============================================================================

def augment_data(hsi_data: np.ndarray, labels: np.ndarray,
                 rotations: List[int] = [45, 90, 135],
                 flip_up_down: bool = True,
                 flip_left_right: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据增强
    
    Args:
        hsi_data: HSI 数据, shape = (N, H, W, Bands)
        labels: 标签, shape = (N,)
        rotations: 旋转角度列表
        flip_up_down: 是否上下翻转
        flip_left_right: 是否左右翻转
        
    Returns:
        augmented_data: 增强后的数据
        augmented_labels: 增强后的标签
    """
    augmented_hsi = []
    augmented_labels = []
    
    for hsi, label in zip(hsi_data, labels):
        # 原始数据
        augmented_hsi.append(hsi)
        augmented_labels.append(label)
        
        # 旋转
        for angle in rotations:
            hsi_rotated = rotate(hsi, angle, axes=(0, 1), reshape=False, mode='nearest')
            augmented_hsi.append(hsi_rotated)
            augmented_labels.append(label)
        
        # 上下翻转
        if flip_up_down:
            hsi_flipped_ud = np.flipud(hsi)
            augmented_hsi.append(hsi_flipped_ud)
            augmented_labels.append(label)
        
        # 左右翻转
        if flip_left_right:
            hsi_flipped_lr = np.fliplr(hsi)
            augmented_hsi.append(hsi_flipped_lr)
            augmented_labels.append(label)
    
    return np.array(augmented_hsi), np.array(augmented_labels)


def augment_paired_data(hsi_data: np.ndarray,
                        lidar_data: np.ndarray,
                        labels: np.ndarray,
                        rotations: List[int] = [45, 90, 135],
                        flip_up_down: bool = True,
                        flip_left_right: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 HSI 和 LiDAR/DSM 应用完全相同的几何增强。

    多模态像素级融合依赖严格配准。旧流程只变换 HSI、随后重复原始
    LiDAR，会人为制造模态错位。该函数把每一种旋转和翻转同步应用到
    两个模态，并保持样本顺序一致。
    """
    if len(hsi_data) != len(lidar_data) or len(hsi_data) != len(labels):
        raise ValueError("HSI, LiDAR and labels must contain the same number of samples")

    augmented_hsi = []
    augmented_lidar = []
    augmented_labels = []

    def append_pair(hsi_patch, lidar_patch, label):
        # np.flip* may create negative strides, which torch.from_numpy cannot use.
        augmented_hsi.append(np.ascontiguousarray(hsi_patch))
        augmented_lidar.append(np.ascontiguousarray(lidar_patch))
        augmented_labels.append(label)

    for hsi, lidar, label in zip(hsi_data, lidar_data, labels):
        append_pair(hsi, lidar, label)

        for angle in rotations:
            hsi_rotated = rotate(
                hsi, angle, axes=(0, 1), reshape=False, mode="nearest"
            )
            lidar_rotated = rotate(
                lidar, angle, axes=(0, 1), reshape=False, mode="nearest"
            )
            append_pair(hsi_rotated, lidar_rotated, label)

        if flip_up_down:
            append_pair(np.flipud(hsi), np.flipud(lidar), label)

        if flip_left_right:
            append_pair(np.fliplr(hsi), np.fliplr(lidar), label)

    return (
        np.asarray(augmented_hsi),
        np.asarray(augmented_lidar),
        np.asarray(augmented_labels),
    )


# =============================================================================
# 数据集类
# =============================================================================

class HSILidarDataset(Dataset):
    """
    高光谱图像与 LiDAR 融合数据集
    
    返回分离的 HSI 和 LiDAR 数据，支持 Cross-Attention 融合
    """
    
    def __init__(self, hsi_data: np.ndarray, lidar_data: np.ndarray, 
                 labels: np.ndarray, transform=None, use_cross_attention: bool = True):
        """
        Args:
            hsi_data: HSI 数据, shape = (N, H, W, Bands)
            lidar_data: LiDAR 数据, shape = (N, H, W, Channels)
            labels: 标签, shape = (N,)
            transform: 可选的数据变换
            use_cross_attention: 是否使用交叉注意力融合（True 则返回分离数据）
        """
        self.hsi_data = torch.from_numpy(hsi_data.astype(np.float32))
        self.lidar_data = torch.from_numpy(lidar_data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transform = transform
        self.use_cross_attention = use_cross_attention
        
        # 如果不使用交叉注意力，则拼接 HSI 和 LiDAR（兼容旧模式）
        if not use_cross_attention:
            self.data = torch.cat([self.hsi_data, self.lidar_data], dim=-1)
        else:
            self.data = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx] - 1  # 标签从 0 开始
        
        if self.use_cross_attention:
            # 返回分离的 HSI 和 LiDAR
            hsi = self.hsi_data[idx]  # [H, W, Bands]
            lidar = self.lidar_data[idx]  # [H, W, Channels]
            
            # 转换 LiDAR 为 [Channels, H, W] 格式
            if lidar.dim() == 3:
                lidar = lidar.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            if self.transform:
                hsi = self.transform(hsi)
            
            return hsi, lidar, label
        else:
            # 兼容旧模式：返回拼接数据
            data = self.data[idx]
            
            if self.transform:
                data = self.transform(data)
            
            return data, label


# =============================================================================
# Patch 提取（基于坐标）
# =============================================================================

def extract_patches_from_coords(
    hsi: np.ndarray, 
    lidar: np.ndarray,
    coords: np.ndarray,
    patch_size: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据坐标提取 HSI 和 LiDAR patches
    
    Args:
        hsi: 高光谱数据
        lidar: LiDAR 数据
        coords: 坐标数组, shape = (N, 3), 每行为 (row, col, class_id)
        patch_size: Patch 大小
        
    Returns:
        hsi_patches: HSI patches, shape = (N, patch_size, patch_size, Bands)
        lidar_patches: LiDAR patches, shape = (N, patch_size, patch_size, 1)
        labels: 标签, shape = (N,)
    """
    half_size = patch_size // 2
    
    # 获取数据格式（先判断格式，再正确获取 H, W）
    hsi_format = "H,W,Bands" if (hsi.ndim == 3 and hsi.shape[2] < hsi.shape[0]) else "Bands,H,W"
    
    # 根据 HSI 格式正确获取空间维度 H, W
    if hsi_format == "H,W,Bands":
        H, W = hsi.shape[0], hsi.shape[1]
    else:
        # Bands, H, W 格式
        H, W = hsi.shape[1], hsi.shape[2]
    
    # 或者从 LiDAR 数据获取（更可靠）
    if lidar.ndim == 2:
        H, W = lidar.shape
    elif lidar.ndim == 3:
        if lidar.shape[0] == 1:
            H, W = lidar.shape[1], lidar.shape[2]
        else:
            H, W = lidar.shape[0], lidar.shape[1]
    
    hsi_patches = []
    lidar_patches = []
    labels = []
    
    for coord in coords:
        row, col, class_id = int(coord[0]), int(coord[1]), int(coord[2])
        
        # 计算 patch 边界
        r_start = max(0, row - half_size)
        r_end = min(H, row + half_size + 1)
        c_start = max(0, col - half_size)
        c_end = min(W, col + half_size + 1)
        
        # 提取 HSI patch
        if hsi_format == "H,W,Bands":
            hsi_patch = hsi[r_start:r_end, c_start:c_end, :]
        else:
            # Bands, H, W 格式
            hsi_patch = hsi[:, r_start:r_end, c_start:c_end]
            hsi_patch = np.transpose(hsi_patch, (1, 2, 0))  # 转为 H, W, Bands
        
        # 提取 LiDAR patch
        if lidar.ndim == 2:
            lidar_patch = lidar[r_start:r_end, c_start:c_end]
        elif lidar.ndim == 3:
            if lidar.shape[0] == 1:
                lidar_patch = lidar[0, r_start:r_end, c_start:c_end]
            else:
                lidar_patch = lidar[r_start:r_end, c_start:c_end, 0]
        else:
            lidar_patch = lidar[r_start:r_end, c_start:c_end]
        
        lidar_patch = lidar_patch[..., np.newaxis]
        
        hsi_patches.append(hsi_patch)
        lidar_patches.append(lidar_patch)
        labels.append(class_id)
    
    return np.array(hsi_patches), np.array(lidar_patches), np.array(labels)


# =============================================================================
# 数据准备函数
# =============================================================================

def prepare_dataloaders(dataset_type: str,
                        data_dir: str,
                        class_dict: Dict,
                        patch_size: int = 1,
                        batch_size: int = 32,
                        val_ratio: float = 0.1,
                        augment: bool = True,
                        augment_rotations: List[int] = [45, 90, 135],
                        num_workers: int = 0,
                        spatial_split_method: str = "disjoint",
                        spatial_split_seed: int = 42,
                        min_test_ratio: float = 0.2,
                        min_test_samples: int = 50,
                        use_cross_attention: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    准备数据加载器（使用空间划分，避免 patch 重叠泄露）
    
    Args:
        dataset_type: 数据集类型 ('trento' 或 'houston')
        data_dir: 数据目录
        class_dict: 类别字典
        patch_size: Patch 大小
        batch_size: 批量大小
        val_ratio: 验证集比例
        augment: 是否进行数据增强
        augment_rotations: 旋转角度
        num_workers: DataLoader 工作进程数
        spatial_split_method: 空间划分方法 ('disjoint', 'random', 'grid', 'block')
        spatial_split_seed: 空间划分随机种子
        min_test_ratio: 测试集最小比例
        min_test_samples: 测试集最小样本数
        use_cross_attention: 是否使用交叉注意力融合（True 则返回分离的 HSI 和 LiDAR）
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    print(f"\n{'='*60}")
    print(f"加载数据集: {dataset_type}")
    print(f"{'='*60}")
    
    # 加载数据
    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower == "trento":
        hsi, lidar, gt = load_trento_data(data_dir)
    elif dataset_type_lower == "houston":
        hsi, lidar, train_gt, test_gt = load_houston_data(data_dir)
        gt = train_gt + test_gt  # 合并训练和测试标签
    elif dataset_type_lower == "muufl":
        hsi, lidar, gt = load_muufl_data(data_dir)
    elif dataset_type_lower == "houston2018":
        hsi, lidar, gt = load_houston2018_data(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")
    
    print(f"  HSI shape: {hsi.shape}")
    print(f"  LiDAR shape: {lidar.shape}")
    print(f"  GT shape: {gt.shape}")
    
    # 归一化
    hsi_normalized = normalize_data(hsi, method="minmax")
    lidar_normalized = normalize_data(lidar, method="minmax")
    
    print(f"  HSI 归一化范围: [{hsi_normalized.min():.4f}, {hsi_normalized.max():.4f}]")
    print(f"  LiDAR 归一化范围: [{lidar_normalized.min():.4f}, {lidar_normalized.max():.4f}]")
    
    # 空间划分
    print(f"\n{'='*60}")
    print(f"空间划分 (method={spatial_split_method}, patch_size={patch_size})")
    print(f"{'='*60}")
    
    train_coords, test_coords = spatial_train_test_split(
        gt=gt,
        class_dict=class_dict,
        method=spatial_split_method,
        patch_size=patch_size,
        seed=spatial_split_seed,
        min_test_ratio=min_test_ratio,
        min_test_samples=min_test_samples
    )
    
    # 打印划分统计
    print_split_statistics(train_coords, test_coords, class_dict, patch_size, min_test_samples)
    
    # 检查重叠
    if len(train_coords) > 0 and len(test_coords) > 0:
        n_overlaps, overlap_ratio = check_overlap(train_coords, test_coords, patch_size)
        if overlap_ratio > 0:
            print(f"\n⚠️  警告: 检测到 {overlap_ratio:.2%} 的测试样本与训练集 patch 重叠")
        else:
            print(f"\n✓ 无 patch 重叠，空间划分有效")
    
    # 根据坐标提取 patches
    print(f"\n{'='*60}")
    print("提取 patches...")
    print(f"{'='*60}")
    
    hsi_train, lidar_train, labels_train = extract_patches_from_coords(
        hsi_normalized, lidar_normalized, train_coords, patch_size
    )
    
    hsi_test, lidar_test, labels_test = extract_patches_from_coords(
        hsi_normalized, lidar_normalized, test_coords, patch_size
    )
    
    print(f"  训练集 HSI patches: {hsi_train.shape}")
    print(f"  训练集 LiDAR patches: {lidar_train.shape}")
    print(f"  训练集 Labels: {labels_train.shape}")
    print(f"  测试集 HSI patches: {hsi_test.shape}")
    print(f"  测试集 LiDAR patches: {lidar_test.shape}")
    print(f"  测试集 Labels: {labels_test.shape}")
    
    # 数据增强
    if augment:
        print(f"\n数据增强中...")
        hsi_train, lidar_train, labels_train = augment_paired_data(
            hsi_train, lidar_train, labels_train,
            rotations=augment_rotations,
            flip_up_down=True,
            flip_left_right=True
        )
        print(f"增强后训练集大小: {len(labels_train)}")
    
    # 标准化
    scaler = StandardScaler()
    n_samples, h, w, n_channels = hsi_train.shape
    
    # 分别标准化 HSI 和 LiDAR
    hsi_train_flat = hsi_train.reshape(n_samples, -1)
    hsi_train_normalized = scaler.fit_transform(hsi_train_flat).reshape(hsi_train.shape)
    
    hsi_test_flat = hsi_test.reshape(len(hsi_test), -1)
    hsi_test_normalized = scaler.transform(hsi_test_flat).reshape(hsi_test.shape)
    
    # LiDAR 单独标准化（因为尺度不同）
    lidar_scaler = StandardScaler()
    lidar_train_flat = lidar_train.reshape(n_samples, -1)
    lidar_train_normalized = lidar_scaler.fit_transform(lidar_train_flat).reshape(lidar_train.shape)
    
    lidar_test_flat = lidar_test.reshape(len(lidar_test), -1)
    lidar_test_normalized = lidar_scaler.transform(lidar_test_flat).reshape(lidar_test.shape)
    
    # 划分验证集
    n_train = len(hsi_train_normalized)
    indices = np.arange(n_train)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_val = int(n_train * val_ratio)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # 创建数据集（根据 use_cross_attention 选择模式）
    if use_cross_attention:
        # 交叉注意力模式：分离 HSI 和 LiDAR
        train_dataset = HSILidarDataset(
            hsi_train_normalized[train_indices],
            lidar_train_normalized[train_indices],
            labels_train[train_indices],
            use_cross_attention=True
        )
        
        val_dataset = HSILidarDataset(
            hsi_train_normalized[val_indices],
            lidar_train_normalized[val_indices],
            labels_train[val_indices],
            use_cross_attention=True
        )
        
        test_dataset = HSILidarDataset(
            hsi_test_normalized,
            lidar_test_normalized,
            labels_test,
            use_cross_attention=True
        )
    else:
        # Early Fusion 模式：拼接 HSI 和 LiDAR
        train_data = np.concatenate([hsi_train_normalized[train_indices], 
                                     lidar_train_normalized[train_indices]], axis=-1)
        val_data = np.concatenate([hsi_train_normalized[val_indices], 
                                   lidar_train_normalized[val_indices]], axis=-1)
        test_data = np.concatenate([hsi_test_normalized, lidar_test_normalized], axis=-1)
        
        train_dataset = TensorDataset(
            torch.from_numpy(train_data.astype(np.float32)),
            torch.from_numpy(labels_train[train_indices].astype(np.int64)) - 1
        )
        
        val_dataset = TensorDataset(
            torch.from_numpy(val_data.astype(np.float32)),
            torch.from_numpy(labels_train[val_indices].astype(np.int64)) - 1
        )
        
        test_dataset = TensorDataset(
            torch.from_numpy(test_data.astype(np.float32)),
            torch.from_numpy(labels_test.astype(np.int64)) - 1
        )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"\n{'='*60}")
    print("DataLoader 创建完成:")
    print(f"{'='*60}")
    print(f"  训练 batches: {len(train_loader)}")
    print(f"  验证 batches: {len(val_loader)}")
    print(f"  测试 batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    from config import get_dataset_config, get_class_dict
    
    print("=" * 60)
    print("测试数据集模块（含空间划分）")
    print("=" * 60)
    
    # 测试 Trento 数据加载
    print("\n[测试 Trento 数据集]")
    data_dir = r"E:\项目\HSLiNets\Dataset\TrentoDataset"
    dataset_config = get_dataset_config("trento")
    class_dict = get_class_dict(dataset_config["class_info"])
    
    try:
        hsi, lidar, gt = load_trento_data(data_dir)
        print(f"  HSI shape: {hsi.shape}")
        print(f"  LiDAR shape: {lidar.shape}")
        print(f"  GT shape: {gt.shape}")
        print("  Trento 数据加载成功!")
        
        # 测试空间划分
        print("\n  测试空间划分...")
        train_coords, test_coords = spatial_train_test_split(
            gt=gt,
            class_dict=class_dict,
            method="disjoint",
            patch_size=9,
            seed=42
        )
        print_split_statistics(train_coords, test_coords, class_dict, patch_size=9)
        
        # 测试完整数据加载流程
        print("\n  测试完整数据加载流程...")
        train_loader, val_loader, test_loader = prepare_dataloaders(
            dataset_type="trento",
            data_dir=data_dir,
            class_dict=class_dict,
            patch_size=9,
            batch_size=32,
            spatial_split_method="disjoint"
        )
        
        # 测试一个 batch
        for batch_data, batch_labels in train_loader:
            print(f"\n  测试 batch:")
            print(f"    data shape: {batch_data.shape}")
            print(f"    labels shape: {batch_labels.shape}")
            break
        
        print("\n  ✓ Trento 数据加载流程测试成功!")
        
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

