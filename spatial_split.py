"""
空间划分模块
实现训练/测试区域不重叠的划分策略，解决 patch 空间重叠泄露问题
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from collections import defaultdict


def get_pixel_coordinates(gt: np.ndarray, class_id: int) -> np.ndarray:
    """
    获取指定类别的所有像素坐标
    
    Args:
        gt: 地面真值, shape = (H, W)
        class_id: 类别 ID
        
    Returns:
        坐标数组, shape = (N, 2), 每行为
    """
    coords = np.argwhere(gt == class_id)
    return coords


def compute_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """
    计算两个坐标之间的欧氏距离
    
    Args:
        coord1: 坐标1, shape = (2,)
        coord2: 坐标2, shape = (2,)
        
    Returns:
        欧氏距离
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def compute_min_distance_to_set(coord: np.ndarray, coord_set: np.ndarray) -> float:
    """
    计算一个坐标到坐标集合的最小距离
    
    Args:
        coord: 单个坐标, shape = (2,)
        coord_set: 坐标集合, shape = (N, 2)
        
    Returns:
        最小距离
    """
    if len(coord_set) == 0:
        return float('inf')
    distances = np.sqrt(np.sum((coord_set - coord) ** 2, axis=1))
    return np.min(distances)


# =============================================================================
# 划分策略
# =============================================================================

def spatial_split_random(
    gt: np.ndarray,
    class_dict: Dict,
    patch_size: int = 9,
    min_distance: Optional[float] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    空间划分策略1：随机划分 + 最小距离约束
    
    确保训练集和测试集的 patch 中心距离大于 min_distance
    
    Args:
        gt: 地面真值, shape = (H, W)
        class_dict: 类别字典
        patch_size: Patch 大小
        min_distance: 最小距离阈值（默认为 patch_size）
        seed: 随机种子
        
    Returns:
        train_coords: 训练集坐标, shape = (N_train, 3), 每行为 (row, col, class_id)
        test_coords: 测试集坐标, shape = (N_test, 3)
    """
    np.random.seed(seed)
    
    if min_distance is None:
        min_distance = patch_size  # 确保 patch 不重叠
    
    train_coords_list = []
    test_coords_list = []
    
    for class_id in class_dict.keys():
        # 获取该类别的所有坐标
        all_coords = get_pixel_coordinates(gt, class_id)
        
        if len(all_coords) == 0:
            continue
        
        # 打乱顺序
        np.random.shuffle(all_coords)
        
        n_train = class_dict[class_id]["training"]
        n_total = len(all_coords)
        
        # 选择训练集坐标（带距离约束）
        train_selected = []
        remaining_coords = list(all_coords)
        
        for coord in remaining_coords:
            if len(train_selected) >= n_train:
                break
            
            # 检查与已选训练坐标的距离
            if len(train_selected) == 0:
                train_selected.append(coord)
            else:
                min_dist = compute_min_distance_to_set(
                    coord, np.array(train_selected)
                )
                if min_dist >= min_distance:
                    train_selected.append(coord)
        
        train_selected = np.array(train_selected) if train_selected else np.array([]).reshape(0, 2)
        
        # 从剩余坐标中选择测试集（与训练集保持距离）
        test_selected = []
        for coord in all_coords:
            if len(test_selected) >= n_total - n_train:
                break
            
            # 不在训练集中的坐标
            if len(train_selected) > 0:
                in_train = any(np.array_equal(coord, tc) for tc in train_selected)
                if in_train:
                    continue
                
                # 检查与训练集的距离
                min_dist = compute_min_distance_to_set(coord, train_selected)
                if min_dist >= min_distance:
                    test_selected.append(coord)
            else:
                test_selected.append(coord)
        
        test_selected = np.array(test_selected) if test_selected else np.array([]).reshape(0, 2)
        
        # 添加类别标签
        if len(train_selected) > 0:
            train_with_class = np.column_stack([
                train_selected, 
                np.full(len(train_selected), class_id)
            ])
            train_coords_list.append(train_with_class)
        
        if len(test_selected) > 0:
            test_with_class = np.column_stack([
                test_selected,
                np.full(len(test_selected), class_id)
            ])
            test_coords_list.append(test_with_class)
    
    train_coords = np.vstack(train_coords_list) if train_coords_list else np.array([]).reshape(0, 3)
    test_coords = np.vstack(test_coords_list) if test_coords_list else np.array([]).reshape(0, 3)
    
    return train_coords, test_coords


def spatial_split_grid(
    gt: np.ndarray,
    class_dict: Dict,
    patch_size: int = 9,
    grid_size: int = 2,
    test_ratio: float = 0.5,
    seed: int = 42,
    min_test_samples: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    空间划分策略2：网格划分（改进版，支持 min_test_samples）
    
    将图像划分为网格，交替选择网格作为训练/测试区域
    对于测试样本不足的类别，从训练网格边界附近的样本补充
    
    Args:
        gt: 地面真值, shape = (H, W)
        class_dict: 类别字典
        patch_size: Patch 大小
        grid_size: 网格大小（将图像划分为 grid_size x grid_size 的网格）
        test_ratio: 测试网格比例
        seed: 随机种子
        min_test_samples: 每个类别最小测试样本数
        
    Returns:
        train_coords: 训练集坐标
        test_coords: 测试集坐标
    """
    np.random.seed(seed)
    
    H, W = gt.shape
    half = patch_size // 2
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    # 生成网格
    grids = []
    for i in range(grid_size):
        for j in range(grid_size):
            grids.append((i, j))
    
    # 随机选择测试网格
    n_test_grids = int(len(grids) * test_ratio)
    np.random.shuffle(grids)
    test_grids = set(grids[:n_test_grids])
    train_grids = set(grids[n_test_grids:])
    
    # 按类别收集坐标
    train_coords_by_class = {cid: [] for cid in class_dict.keys()}
    test_coords_by_class = {cid: [] for cid in class_dict.keys()}
    
    for class_id in class_dict.keys():
        all_coords = get_pixel_coordinates(gt, class_id)
        
        for coord in all_coords:
            row, col = coord
            
            # 检查边界
            if row < half or row >= H - half or col < half or col >= W - half:
                continue
            
            # 判断属于哪个网格
            grid_i = min(row // cell_h, grid_size - 1)
            grid_j = min(col // cell_w, grid_size - 1)
            
            coord_with_class = np.array([row, col, class_id])
            
            if (grid_i, grid_j) in test_grids:
                test_coords_by_class[class_id].append((coord_with_class, grid_i, grid_j))
            else:
                train_coords_by_class[class_id].append((coord_with_class, grid_i, grid_j))
    
    # 首先确保每个类别有足够的训练样本
    # 如果训练样本不足，从测试网格中补充
    for class_id in class_dict.keys():
        train_samples = train_coords_by_class[class_id]
        test_samples = test_coords_by_class[class_id]
        
        n_train_target = class_dict[class_id].get("training", 0)
        n_train = len(train_samples)
        
        if n_train < n_train_target and len(test_samples) > 0:
            # 计算需要补充的训练样本数
            n_train_needed = n_train_target - n_train
            
            # 计算每个测试样本到最近训练网格的距离
            # 距离越小，越靠近边界，补充后重叠风险越小
            samples_with_distance = []
            for sample, grid_i, grid_j in test_samples:
                min_dist = float('inf')
                for ti, tj in train_grids:
                    dist = abs(grid_i - ti) + abs(grid_j - tj)  # 曼哈顿距离
                    min_dist = min(min_dist, dist)
                samples_with_distance.append((sample, min_dist, grid_i, grid_j))
            
            # 按距离排序，优先选择离训练网格最近的
            samples_with_distance.sort(key=lambda x: x[1])
            
            # 补充训练样本
            n_to_add = min(n_train_needed, len(samples_with_distance))
            for i in range(n_to_add):
                sample, dist, grid_i, grid_j = samples_with_distance[i]
                train_samples.append((sample, grid_i, grid_j))
            
            # 更新测试集
            added_samples = set(
                tuple(samples_with_distance[i][0]) for i in range(n_to_add)
            )
            test_coords_by_class[class_id] = [
                (s, gi, gj) for s, gi, gj in test_samples
                if tuple(s) not in added_samples
            ]
    
    # 对于测试样本不足的类别，从训练样本中补充
    # 优先选择靠近测试网格边界的样本，以减少 patch 重叠风险
    # 同时确保训练样本数不少于训练目标
    for class_id in class_dict.keys():
        test_samples = test_coords_by_class[class_id]
        train_samples = train_coords_by_class[class_id]
        
        n_test = len(test_samples)
        n_train = len(train_samples)
        n_needed = min_test_samples - n_test
        
        # 获取训练目标
        n_train_target = class_dict[class_id].get("training", 0)
        
        if n_needed > 0 and len(train_samples) > 0:
            # 计算可以补充的最大样本数（确保训练样本不少于训练目标）
            max_to_add = n_train - n_train_target
            if max_to_add <= 0:
                # 训练样本已不足，跳过补充
                continue
            
            # 计算每个训练样本到最近测试网格的距离
            # 距离越小，越靠近边界，补充后重叠风险越小
            samples_with_distance = []
            for sample, grid_i, grid_j in train_samples:
                min_dist = float('inf')
                for ti, tj in test_grids:
                    dist = abs(grid_i - ti) + abs(grid_j - tj)  # 曼哈顿距离
                    min_dist = min(min_dist, dist)
                samples_with_distance.append((sample, min_dist, grid_i, grid_j))
            
            # 按距离排序，优先选择离测试网格最近的（距离=1表示相邻网格）
            samples_with_distance.sort(key=lambda x: x[1])
            
            # 补充样本（不超过可补充数量）
            n_to_add = min(n_needed, len(samples_with_distance), max_to_add)
            for i in range(n_to_add):
                sample, dist, grid_i, grid_j = samples_with_distance[i]
                # 从训练集移除，添加到测试集
                test_samples.append((sample, grid_i, grid_j))
            
            # 更新训练集
            added_samples = set(
                tuple(samples_with_distance[i][0]) for i in range(n_to_add)
            )
            train_coords_by_class[class_id] = [
                (s, gi, gj) for s, gi, gj in train_samples 
                if tuple(s) not in added_samples
            ]
    
    # 合并所有坐标
    train_coords_list = [
        item[0] for samples in train_coords_by_class.values() 
        for item in samples
    ]
    test_coords_list = [
        item[0] for samples in test_coords_by_class.values() 
        for item in samples
    ]
    
    train_coords = np.array(train_coords_list) if train_coords_list else np.array([]).reshape(0, 3)
    test_coords = np.array(test_coords_list) if test_coords_list else np.array([]).reshape(0, 3)
    
    return train_coords, test_coords


def spatial_split_disjoint(
    gt: np.ndarray,
    class_dict: Dict,
    patch_size: int = 9,
    buffer_size: Optional[int] = None,
    seed: int = 42,
    min_test_ratio: float = 0.2,
    min_test_samples: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    空间划分策略3：严格不相交划分（改进版）
    
    优先保证训练集有足够样本，然后在剩余空间分配测试集
    
    Args:
        gt: 地面真值, shape = (H, W)
        class_dict: 类别字典
        patch_size: Patch 大小
        buffer_size: 缓冲区大小（默认为 patch_size // 2）
        seed: 随机种子
        min_test_ratio: 测试集最小比例（相对于总样本数）
        min_test_samples: 测试集最小样本数
        
    Returns:
        train_coords: 训练集坐标
        test_coords: 测试集坐标
    """
    np.random.seed(seed)
    
    if buffer_size is None:
        buffer_size = patch_size // 2
    
    H, W = gt.shape
    half = patch_size // 2
    
    # 全局覆盖掩码（记录已使用的 patch 区域）
    used_mask = np.zeros((H, W), dtype=bool)
    
    train_coords_list = []
    test_coords_list = []
    
    # 收集所有类别的坐标信息，按训练目标排序（大类优先）
    all_class_info = []
    for class_id in class_dict.keys():
        all_coords = get_pixel_coordinates(gt, class_id)
        
        if len(all_coords) == 0:
            continue
        
        # 过滤边界坐标
        valid_coords = []
        for coord in all_coords:
            row, col = coord
            if row >= half and row < H - half and col >= half and col < W - half:
                valid_coords.append(coord)
        
        if len(valid_coords) == 0:
            continue
        
        valid_coords = np.array(valid_coords)
        np.random.shuffle(valid_coords)
        
        n_total = len(valid_coords)
        n_train_target = class_dict[class_id]["training"]
        
        # 计算测试集目标样本数
        n_test_target = max(
            int(n_total * min_test_ratio),
            min_test_samples
        )
        
        # 确保目标合理（训练集优先）
        n_train_target = min(n_train_target, n_total)
        remaining = n_total - n_train_target
        n_test_target = min(n_test_target, remaining)
        
        all_class_info.append({
            'class_id': class_id,
            'coords': valid_coords,
            'n_total': n_total,
            'n_train_target': n_train_target,
            'n_test_target': n_test_target
        })
    
    # 按训练目标排序（大类优先，确保大类有足够训练样本）
    all_class_info.sort(key=lambda x: x['n_train_target'], reverse=True)
    
    # 第一轮：分配训练集
    for info in all_class_info:
        class_id = info['class_id']
        coords = info['coords']
        n_train_target = info['n_train_target']
        
        train_selected = []
        
        for coord in coords:
            if len(train_selected) >= n_train_target:
                break
            
            row, col = coord
            
            # 检查是否在已使用的 patch 区域内
            if used_mask[row, col]:
                continue
            
            # 标记覆盖区域
            r_start = max(0, row - half)
            r_end = min(H, row + half + 1)
            c_start = max(0, col - half)
            c_end = min(W, col + half + 1)
            
            train_selected.append([row, col, class_id])
            used_mask[r_start:r_end, c_start:c_end] = True
        
        info['train_selected'] = train_selected
        train_coords_list.extend(train_selected)
    
    # 第二轮：分配测试集
    for info in all_class_info:
        class_id = info['class_id']
        coords = info['coords']
        n_test_target = info['n_test_target']
        train_selected = info.get('train_selected', [])
        train_coords_set = set((c[0], c[1]) for c in train_selected)
        
        test_selected = []
        
        for coord in coords:
            if len(test_selected) >= n_test_target:
                break
            
            row, col = coord
            
            # 跳过已选为训练集的坐标
            if (row, col) in train_coords_set:
                continue
            
            # 检查是否在训练集 patch 区域内（不重叠要求）
            if used_mask[row, col]:
                continue
            
            # 标记覆盖区域
            r_start = max(0, row - half)
            r_end = min(H, row + half + 1)
            c_start = max(0, col - half)
            c_end = min(W, col + half + 1)
            
            test_selected.append([row, col, class_id])
            used_mask[r_start:r_end, c_start:c_end] = True
        
        test_coords_list.extend(test_selected)
    
    train_coords = np.array(train_coords_list) if train_coords_list else np.array([]).reshape(0, 3)
    test_coords = np.array(test_coords_list) if test_coords_list else np.array([]).reshape(0, 3)
    
    return train_coords, test_coords


def spatial_split_block(
    gt: np.ndarray,
    class_dict: Dict,
    patch_size: int = 9,
    test_block_ratio: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    空间划分策略4：块状划分
    
    随机选择若干连续块作为测试区域，其余为训练区域
    
    Args:
        gt: 地面真值, shape = (H, W)
        class_dict: 类别字典
        patch_size: Patch 大小
        test_block_ratio: 测试块占总面积的比例
        seed: 随机种子
        
    Returns:
        train_coords: 训练集坐标
        test_coords: 测试集坐标
    """
    np.random.seed(seed)
    
    H, W = gt.shape
    half = patch_size // 2
    
    # 计算测试块大小
    total_area = H * W
    test_area = int(total_area * test_block_ratio)
    block_size = int(np.sqrt(test_area))
    
    # 随机选择测试块的左上角
    block_row = np.random.randint(half, max(half + 1, H - block_size - half))
    block_col = np.random.randint(half, max(half + 1, W - block_size - half))
    
    # 创建测试掩码
    test_mask = np.zeros((H, W), dtype=bool)
    test_mask[block_row:block_row + block_size, 
              block_col:block_col + block_size] = True
    
    train_coords_list = []
    test_coords_list = []
    
    for class_id in class_dict.keys():
        all_coords = get_pixel_coordinates(gt, class_id)
        
        for coord in all_coords:
            row, col = coord
            
            # 边界检查
            if row < half or row >= H - half or col < half or col >= W - half:
                continue
            
            coord_with_class = [row, col, class_id]
            
            if test_mask[row, col]:
                test_coords_list.append(coord_with_class)
            else:
                train_coords_list.append(coord_with_class)
    
    train_coords = np.array(train_coords_list) if train_coords_list else np.array([]).reshape(0, 3)
    test_coords = np.array(test_coords_list) if test_coords_list else np.array([]).reshape(0, 3)
    
    return train_coords, test_coords


# =============================================================================
# 主划分函数
# =============================================================================

def spatial_train_test_split(
    gt: np.ndarray,
    class_dict: Dict,
    method: str = "disjoint",
    patch_size: int = 9,
    seed: int = 42,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    空间训练/测试划分主函数
    
    Args:
        gt: 地面真值, shape = (H, W)
        class_dict: 类别字典
        method: 划分方法
            - 'random': 随机划分 + 最小距离约束
            - 'grid': 网格划分
            - 'disjoint': 严格不相交划分（推荐）
            - 'block': 块状划分
        patch_size: Patch 大小
        seed: 随机种子
        **kwargs: 各方法的额外参数
        
    Returns:
        train_coords: 训练集坐标, shape = (N_train, 3), 每行为 (row, col, class_id)
        test_coords: 测试集坐标, shape = (N_test, 3)
    """
    methods = {
        'random': spatial_split_random,
        'grid': spatial_split_grid,
        'disjoint': spatial_split_disjoint,
        'block': spatial_split_block,
    }
    
    # 每个方法支持的参数列表
    method_params = {
        'random': {'min_distance'},
        'grid': {'grid_size', 'test_ratio', 'min_test_samples'},
        'disjoint': {'buffer_size', 'min_test_ratio', 'min_test_samples'},
        'block': {'test_block_ratio'},
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    # 过滤只传递该方法支持的参数
    allowed_params = method_params[method]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
    
    return methods[method](gt, class_dict, patch_size, seed=seed, **filtered_kwargs)


# =============================================================================
# 辅助函数
# =============================================================================

def check_overlap(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    patch_size: int = 9
) -> Tuple[int, float]:
    """
    检查训练集和测试集 patch 的重叠情况
    
    Args:
        train_coords: 训练集坐标
        test_coords: 测试集坐标
        patch_size: Patch 大小
        
    Returns:
        n_overlaps: 重叠的测试样本数
        overlap_ratio: 重叠比例
    """
    if len(train_coords) == 0 or len(test_coords) == 0:
        return 0, 0.0
    
    half = patch_size // 2
    
    # 使用向量化操作优化性能
    train_rows = train_coords[:, 0]
    train_cols = train_coords[:, 1]
    test_rows = test_coords[:, 0]
    test_cols = test_coords[:, 1]
    
    n_overlaps = 0
    
    # 分批处理以避免内存溢出（每批 10000 个测试样本）
    batch_size = 10000
    for i in range(0, len(test_coords), batch_size):
        test_rows_batch = test_rows[i:i+batch_size]
        test_cols_batch = test_cols[i:i+batch_size]
        
        # 使用广播计算距离矩阵
        # shape: (batch_size, n_train)
        row_diff = np.abs(test_rows_batch[:, np.newaxis] - train_rows[np.newaxis, :])
        col_diff = np.abs(test_cols_batch[:, np.newaxis] - train_cols[np.newaxis, :])
        
        # 检查是否有任何一个训练样本的 patch 覆盖该测试样本
        overlap_matrix = (row_diff <= half) & (col_diff <= half)
        n_overlaps += np.any(overlap_matrix, axis=1).sum()
    
    overlap_ratio = n_overlaps / len(test_coords)
    
    return n_overlaps, overlap_ratio


def print_split_statistics(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    class_dict: Dict,
    patch_size: int = 9,
    min_test_samples: int = 50
):
    """
    打印划分统计信息
    
    Args:
        train_coords: 训练集坐标
        test_coords: 测试集坐标
        class_dict: 类别字典
        patch_size: Patch 大小
        min_test_samples: 测试集最小样本数警告阈值
    """
    print("\n" + "=" * 60)
    print("空间划分统计")
    print("=" * 60)
    
    print(f"\nPatch Size: {patch_size}")
    
    # 按类别统计
    print("\n按类别统计:")
    print(f"{'类别':<10} {'训练集':<10} {'测试集':<10} {'训练目标':<10} {'状态':<10}")
    
    total_train = 0
    total_test = 0
    warning_classes = []
    
    for class_id in sorted(class_dict.keys()):
        n_train = np.sum(train_coords[:, 2] == class_id) if len(train_coords) > 0 else 0
        n_test = np.sum(test_coords[:, 2] == class_id) if len(test_coords) > 0 else 0
        target = class_dict[class_id]["training"]
        
        total_train += n_train
        total_test += n_test
        
        # 检查测试集是否过少
        if n_test < min_test_samples:
            status = "⚠️ 少"
            warning_classes.append(class_id)
        else:
            status = "✓"
        
        print(f"{class_id:<10} {n_train:<10} {n_test:<10} {target:<10} {status:<10}")
    
    print("-" * 50)
    print(f"{'总计':<10} {total_train:<10} {total_test:<10}")
    
    # 警告测试集过少的类别
    if warning_classes:
        print(f"\n⚠️  警告: 以下类别测试集样本少于 {min_test_samples}: {warning_classes}")
        print("   建议: 增加 min_test_samples 参数或调整数据集配置")
    
    # 检查重叠
    if len(train_coords) > 0 and len(test_coords) > 0:
        n_overlaps, overlap_ratio = check_overlap(train_coords, test_coords, patch_size)
        
        print(f"\n重叠检查:")
        print(f"  重叠测试样本数: {n_overlaps}")
        print(f"  重叠比例: {overlap_ratio:.2%}")
    else:
        print(f"\n⚠️  警告: 训练集或测试集为空!")
    
    print("=" * 60)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试空间划分模块")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    H, W = 100, 100
    gt = np.zeros((H, W), dtype=int)
    
    # 创建几个类别区域
    gt[10:30, 10:30] = 1
    gt[40:60, 10:30] = 2
    gt[10:30, 40:60] = 3
    gt[40:60, 40:60] = 4
    
    class_dict = {
        1: {"training": 50, "samples": 400},
        2: {"training": 50, "samples": 400},
        3: {"training": 50, "samples": 400},
        4: {"training": 50, "samples": 400},
    }
    
    patch_size = 9
    
    # 测试不同方法
    methods = ['random', 'grid', 'disjoint', 'block']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print("=" * 60)
        
        train_coords, test_coords = spatial_train_test_split(
            gt, class_dict, method=method, patch_size=patch_size
        )
        
        print_split_statistics(train_coords, test_coords, class_dict, patch_size)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
