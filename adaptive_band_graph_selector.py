"""
AdaptiveBandGraphSelector: 自适应波段选择模块
基于图注意力网络（GAT）的高光谱图像波段选择，融合 LiDAR 辅助信息

Author: HSLiNets Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# 1. 节点特征提取模块
# =============================================================================

class BandFeatureExtractor(nn.Module):
    """
    为每个波段提取节点特征
    
    节点特征包含:
    - 当前 band 的 mean
    - 当前 band 的 std
    - 当前 band 与 LiDAR 的相关性
    
    输出节点特征维度: d = 3
    """
    
    def __init__(self):
        super(BandFeatureExtractor, self).__init__()
    
    def forward(self, hsi_patch: torch.Tensor, lidar_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hsi_patch: HSI 数据, shape = [B, H, W], B 为波段数
            lidar_patch: LiDAR 数据, shape = [1, H, W]
            
        Returns:
            node_features: 节点特征矩阵, shape = [B, 3]
        """
        num_bands = hsi_patch.shape[0]
        H, W = hsi_patch.shape[1], hsi_patch.shape[2]
        
        # 存储所有波段特征
        features_list = []
        
        # 展开 LiDAR 用于相关性计算
        lidar_flat = lidar_patch.reshape(-1)  # [H*W]
        lidar_mean = lidar_flat.mean()
        lidar_std = lidar_flat.std() + 1e-8
        
        for i in range(num_bands):
            band = hsi_patch[i]  # [H, W]
            band_flat = band.reshape(-1)  # [H*W]
            
            # 特征1: 当前 band 的均值
            band_mean = band_flat.mean().unsqueeze(0)
            
            # 特征2: 当前 band 的标准差
            band_std = band_flat.std().unsqueeze(0)
            
            # 特征3: 当前 band 与 LiDAR 的 Pearson 相关系数
            band_mean_val = band_flat.mean()
            band_std_val = band_flat.std() + 1e-8
            
            # Pearson correlation: cov(X,Y) / (std(X) * std(Y))
            covariance = ((band_flat - band_mean_val) * (lidar_flat - lidar_mean)).mean()
            correlation = (covariance / (band_std_val * lidar_std)).unsqueeze(0)
            
            # 拼接三个特征
            band_features = torch.cat([band_mean, band_std, correlation], dim=0)  # [3]
            features_list.append(band_features)
        
        # 堆叠所有波段特征
        node_features = torch.stack(features_list, dim=0)  # [B, 3]
        
        return node_features


# =============================================================================
# 2. 图构建模块
# =============================================================================

class ChainGraphBuilder(nn.Module):
    """
    构建链式图结构
    
    链式图: band i 与 band i+1 相连
    即: (0)-(1)-(2)-...-(B-1)
    
    输出边索引，用于 PyTorch Geometric 风格的图表示
    """
    
    def __init__(self):
        super(ChainGraphBuilder, self).__init__()
    
    def forward(self, num_bands: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_bands: 波段数量 B
            device: 计算设备
            
        Returns:
            edge_index: 边索引, shape = [2, 2*(B-1)]
                        第一行为源节点，第二行为目标节点
            edge_index_bidirectional: 双向边索引, shape = [2, 4*(B-1)]
        """
        if num_bands < 2:
            # 单波段情况，返回自环
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            return edge_index, edge_index
        
        # 构建链式边: (i, i+1)
        source_nodes = torch.arange(0, num_bands - 1, dtype=torch.long, device=device)
        target_nodes = torch.arange(1, num_bands, dtype=torch.long, device=device)
        
        # 单向边索引
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)  # [2, B-1]
        
        # 双向边索引（用于 GAT 的无向图）
        edge_index_bidirectional = torch.cat([
            edge_index,  # 正向边
            torch.stack([target_nodes, source_nodes], dim=0)  # 反向边
        ], dim=1)  # [2, 2*(B-1)]
        
        return edge_index, edge_index_bidirectional


# =============================================================================
# 3. 图注意力网络（GAT）模块
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    单层图注意力网络
    
    实现: h_i' = σ(Σ_{j∈N(i)} α_{ij} W h_j)
    其中 α_{ij} 通过注意力机制计算
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度（每个 head）
            n_heads: 注意力头数
            dropout: dropout 概率
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads if out_features >= n_heads else 1
        
        # 确保输出维度能被头数整除
        self.out_features_per_head = out_features // n_heads
        self.actual_out_features = self.out_features_per_head * n_heads
        
        # 线性变换矩阵 W
        self.W = nn.Parameter(torch.zeros(size=(in_features, self.actual_out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力参数 a
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features_per_head, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: 节点特征, shape = [num_nodes, in_features]
            edge_index: 边索引, shape = [2, num_edges]
            
        Returns:
            out: 更新后的节点特征, shape = [num_nodes, out_features]
        """
        num_nodes = h.shape[0]
        
        # 线性变换: [num_nodes, in_features] -> [num_nodes, n_heads, out_features_per_head]
        Wh = torch.mm(h, self.W).view(num_nodes, self.n_heads, self.out_features_per_head)
        
        # 提取边的源节点和目标节点
        source = edge_index[0]  # [num_edges]
        target = edge_index[1]  # [num_edges]
        
        # 获取源节点和目标节点的特征
        Wh_source = Wh[source]  # [num_edges, n_heads, out_features_per_head]
        Wh_target = Wh[target]  # [num_edges, n_heads, out_features_per_head]
        
        # 拼接计算注意力分数
        # [num_edges, n_heads, 2 * out_features_per_head]
        concat_features = torch.cat([Wh_source, Wh_target], dim=-1)
        
        # 计算注意力系数 e_ij
        # [num_edges, n_heads, 1] -> [num_edges, n_heads]
        e = self.leakyrelu(torch.matmul(concat_features, self.a).squeeze(-1))
        
        # 对每个节点的入边进行 softmax 归一化
        # 创建稀疏注意力矩阵
        attention = torch.zeros(num_nodes, num_nodes, self.n_heads, device=h.device)
        for head in range(self.n_heads):
            attention[target, source, head] = e[:, head]
        
        # Softmax 归一化
        attention = F.softmax(attention, dim=1)
        
        # 应用注意力权重聚合邻居特征
        # [num_nodes, n_heads, out_features_per_head]
        out = torch.zeros(num_nodes, self.n_heads, self.out_features_per_head, device=h.device)
        for head in range(self.n_heads):
            for i in range(num_nodes):
                # 获取节点 i 的邻居
                neighbors = edge_index[1][edge_index[0] == i]
                if len(neighbors) > 0:
                    weights = attention[i, neighbors, head]
                    out[i, head] = (Wh[neighbors, head] * weights.unsqueeze(-1)).sum(dim=0)
        
        # 多头拼接
        out = out.view(num_nodes, self.actual_out_features)
        
        return out


class TwoLayerGAT(nn.Module):
    """
    两层图注意力网络
    
    结构: Input -> GAT Layer 1 -> ELU -> GAT Layer 2 -> Output
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, 
                 n_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
            out_features: 输出特征维度
            n_heads: 注意力头数
            dropout: dropout 概率
        """
        super(TwoLayerGAT, self).__init__()
        
        # 第一层 GAT
        self.gat1 = GraphAttentionLayer(
            in_features, hidden_features, n_heads=n_heads, dropout=dropout
        )
        
        # 第二层 GAT
        self.gat2 = GraphAttentionLayer(
            hidden_features, out_features, n_heads=n_heads, dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: 节点特征, shape = [num_nodes, in_features]
            edge_index: 边索引, shape = [2, num_edges]
            
        Returns:
            out: 输出特征, shape = [num_nodes, out_features]
        """
        # 第一层
        h = self.gat1(h, edge_index)
        h = self.elu(h)
        h = self.dropout(h)
        
        # 第二层
        h = self.gat2(h, edge_index)
        
        return h


# =============================================================================
# 4. Score Head 模块
# =============================================================================

class BandScoreHead(nn.Module):
    """
    波段评分头
    
    使用小型 MLP 输出每个波段的分数，并通过 sigmoid 映射到 [0, 1]
    """
    
    def __init__(self, in_features: int, hidden_features: int = 64):
        """
        Args:
            in_features: 输入特征维度（GAT 输出维度）
            hidden_features: 隐藏层维度
        """
        super(BandScoreHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: 节点特征, shape = [num_bands, in_features]
            
        Returns:
            scores: 波段权重, shape = [num_bands], 范围 [0, 1]
        """
        scores = self.mlp(h).squeeze(-1)  # [num_bands]
        weights = self.sigmoid(scores)     # [num_bands]
        
        return weights


# =============================================================================
# 5. 主模块: AdaptiveBandGraphSelector
# =============================================================================

class AdaptiveBandGraphSelector(nn.Module):
    """
    自适应波段选择模块
    
    基于图注意力网络的高光谱图像波段选择，融合 LiDAR 辅助信息
    
    工作流程:
    1. 提取每个波段的节点特征（mean, std, 与 LiDAR 相关性）
    2. 构建链式图结构
    3. 使用两层 GAT 建模波段间关系
    4. 通过 MLP 输出每个波段的权重
    
    支持两种输出模式:
    - mode='weights': 仅返回波段权重
    - mode='weighted': 返回加权后的 HSI
    """
    
    def __init__(self, 
                 node_feature_dim: int = 3,
                 gat_hidden_dim: int = 64,
                 gat_out_dim: int = 32,
                 score_hidden_dim: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            node_feature_dim: 节点特征维度（默认3: mean, std, correlation）
            gat_hidden_dim: GAT 隐藏层维度
            gat_out_dim: GAT 输出维度
            score_hidden_dim: Score Head 隐藏层维度
            n_heads: GAT 注意力头数
            dropout: dropout 概率
        """
        super(AdaptiveBandGraphSelector, self).__init__()
        
        # 节点特征提取器
        self.feature_extractor = BandFeatureExtractor()
        
        # 图构建器
        self.graph_builder = ChainGraphBuilder()
        
        # 两层 GAT
        self.gat = TwoLayerGAT(
            in_features=node_feature_dim,
            hidden_features=gat_hidden_dim,
            out_features=gat_out_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # 波段评分头
        self.score_head = BandScoreHead(
            in_features=gat_out_dim,
            hidden_features=score_hidden_dim
        )
    
    def forward(self, 
                hsi_patch: torch.Tensor, 
                lidar_patch: torch.Tensor,
                mode: str = 'weights') -> torch.Tensor:
        """
        Args:
            hsi_patch: HSI 数据, shape = [B, H, W], B 为波段数
            lidar_patch: LiDAR 数据, shape = [1, H, W]
            mode: 输出模式
                  - 'weights': 仅返回波段权重, shape = [B]
                  - 'weighted': 返回加权后的 HSI, shape = [B, H, W]
                  
        Returns:
            output: 根据 mode 返回波段权重或加权 HSI
        """
        # Step 1: 提取节点特征
        node_features = self.feature_extractor(hsi_patch, lidar_patch)  # [B, 3]
        
        # Step 2: 构建链式图
        num_bands = hsi_patch.shape[0]
        _, edge_index = self.graph_builder(num_bands, hsi_patch.device)
        
        # Step 3: GAT 前向传播
        graph_features = self.gat(node_features, edge_index)  # [B, gat_out_dim]
        
        # Step 4: 计算波段权重
        band_weights = self.score_head(graph_features)  # [B]
        
        # Step 5: 根据模式返回结果
        if mode == 'weights':
            return band_weights
        elif mode == 'weighted':
            # 对 HSI 加权: [B, H, W] * [B, 1, 1]
            weighted_hsi = hsi_patch * band_weights.view(-1, 1, 1)
            return weighted_hsi
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'weights' or 'weighted'.")


# =============================================================================
# 6. 最小可运行示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AdaptiveBandGraphSelector 最小可运行示例")
    print("=" * 60)
    
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    
    # 创建随机输入数据
    num_bands = 63      # 波段数
    height = 11         # 空间高度
    width = 11          # 空间宽度
    
    # 随机 HSI patch: [B, H, W]
    hsi_patch = torch.randn(num_bands, height, width)
    
    # 随机 LiDAR patch: [1, H, W]
    lidar_patch = torch.randn(1, height, width)
    
    print(f"\n输入数据:")
    print(f"  HSI patch shape: {hsi_patch.shape}")
    print(f"  LiDAR patch shape: {lidar_patch.shape}")
    
    # 实例化模块
    model = AdaptiveBandGraphSelector(
        node_feature_dim=3,
        gat_hidden_dim=64,
        gat_out_dim=32,
        score_hidden_dim=64,
        n_heads=4,
        dropout=0.1
    )
    
    print(f"\n模型结构:")
    print(model)
    
    # 计算波段权重
    print(f"\n--- 模式 1: 仅返回波段权重 ---")
    band_weights = model(hsi_patch, lidar_patch, mode='weights')
    print(f"Band weights shape: {band_weights.shape}")
    print(f"Band weights 范围: [{band_weights.min().item():.4f}, {band_weights.max().item():.4f}]")
    print(f"Band weights 前5个值: {band_weights[:5].tolist()}")
    
    # 计算加权 HSI
    print(f"\n--- 模式 2: 返回加权后的 HSI ---")
    weighted_hsi = model(hsi_patch, lidar_patch, mode='weighted')
    print(f"Weighted HSI shape: {weighted_hsi.shape}")
    print(f"Weighted HSI 范围: [{weighted_hsi.min().item():.4f}, {weighted_hsi.max().item():.4f}]")
    
    # 验证权重应用正确性
    print(f"\n--- 验证 ---")
    manual_weighted = hsi_patch * band_weights.view(-1, 1, 1)
    diff = torch.abs(weighted_hsi - manual_weighted).max()
    print(f"手动加权与模型输出最大差异: {diff.item():.10f}")
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)
