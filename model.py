"""
HSLiNets 模型定义
包含 SpectralSpatialMamba、CrossAttentionFusion、Classifier、HSIClassificationMambaModel

改进版本：
- SpectralSpatialMamba: 光谱-空间联合状态空间模型，支持选择性扫描
- CrossAttentionFusion: HSI-LiDAR 交叉注意力融合
- 移除无效的 SpatialFeatureProcessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# 尝试导入 mamba_ssm（可选）
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("Warning: mamba_ssm not available, using pure PyTorch implementation")

# 尝试导入波段选择模块（可选）
try:
    from band_graph_attention_selector import BandGraphAttentionSelector
    BAND_SELECTOR_AVAILABLE = True
except ImportError:
    BAND_SELECTOR_AVAILABLE = False


# =============================================================================
# 选择性扫描模块（纯 PyTorch 实现）
# =============================================================================

class SelectiveScan(nn.Module):
    """
    选择性扫描模块（Selective Scan）
    
    纯 PyTorch 实现，用于替代 mamba_ssm 的选择性扫描
    
    核心思想：
    - 输入经过线性变换生成 B, C, D 参数
    - 使用这些参数进行状态空间更新
    - 支持前向和后向扫描
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        """
        Args:
            d_model: 输入维度
            d_state: 状态空间维度
            d_conv: 卷积核大小
            expand: 扩展因子
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM 参数投影
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        
        # dt 投影
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # 状态空间参数
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.A_log = nn.Parameter(torch.log(torch.abs(self.A) + 1e-6))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape = [Batch, Length, d_model]
            
        Returns:
            输出张量, shape = [Batch, Length, d_model]
        """
        Batch, Length, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # [B, L, d_inner * 2]
        x_proj, z = xz.chunk(2, dim=-1)  # 各 [B, L, d_inner]
        
        # 卷积
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :Length].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM 参数投影
        x_dbl = self.x_proj(x_conv)  # [B, L, d_state * 2 + d_inner]
        dt, B, C = torch.split(x_dbl, [self.d_state, self.d_state, self.d_inner], dim=-1)
        
        # dt 投影
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)
        
        # 状态空间更新（向量化实现）
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # 初始化状态 [B, d_inner, d_state]
        h = torch.zeros(Batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        # 逐时间步处理
        outputs = []
        for t in range(Length):
            # 当前输入 [B, d_inner]
            x_t = x_conv[:, t, :]
            
            # 选择性参数
            B_t = B[:, t, :]  # [B, d_state]
            C_t = C[:, t, :]  # [B, d_inner]
            dt_t = dt[:, t, :]  # [B, d_inner]
            
            # 离散化状态转移矩阵
            # A: [d_inner, d_state], dt_t: [B, d_inner] -> [B, d_inner, d_state]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]
            
            # 状态更新: h_t = A_bar * h_{t-1} + B_t * x_t
            # h: [B, d_inner, d_state]
            # B_t: [B, d_state] -> unsqueeze(1) -> [B, 1, d_state]
            # x_t: [B, d_inner] -> unsqueeze(-1) -> [B, d_inner, 1]
            # B_t * x_t -> [B, d_inner, d_state] (广播)
            h = A_bar * h + B_t.unsqueeze(1) * x_t.unsqueeze(-1)  # [B, d_inner, d_state]
            
            # 输出: y_t = (C_t.unsqueeze(-1) * h).sum(dim=-1) + D * x_t
            # C_t: [B, d_inner] -> unsqueeze(-1) -> [B, d_inner, 1]
            # h: [B, d_inner, d_state]
            # C_t * h -> [B, d_inner, d_state], sum(dim=-1) -> [B, d_inner]
            y_t = (C_t.unsqueeze(-1) * h).sum(dim=-1) + self.D.unsqueeze(0) * x_t  # [B, d_inner]
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)  # [B, L, d_model]
        
        return output


class SelectiveScanMamba(nn.Module):
    """
    使用 mamba_ssm 库的选择性扫描（如果可用）
    否则回退到纯 PyTorch 实现
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        
        self.use_mamba_ssm = MAMBA_SSM_AVAILABLE
        
        if self.use_mamba_ssm:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.mamba = SelectiveScan(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


# =============================================================================
# 2D Cross-Scan 模块
# =============================================================================

class CrossScan2D(nn.Module):
    """
    2D Cross-Scan 模块
    
    将 2D 特征图展平为 4 个方向的 1D 序列，分别进行选择性扫描后合并
    
    方向：
    - 从左到右 (H 方向)
    - 从右到左 (H 方向反向)
    - 从上到下 (W 方向)
    - 从下到上 (W 方向反向)
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        
        self.ssm = SelectiveScanMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape = [Batch, H, W, Channels]
            
        Returns:
            输出张量, shape = [Batch, H, W, Channels]
        """
        B, H, W, C = x.shape
        
        # 方向 1: 从左到右（按行展平）
        x_h = x.reshape(B, H * W, C)  # [B, H*W, C]
        
        # 方向 2: 从右到左
        x_h_reverse = torch.flip(x_h, dims=[1])
        
        # 方向 3: 从上到下（按列展平）
        x_w = x.transpose(1, 2).reshape(B, H * W, C)  # [B, H*W, C]
        
        # 方向 4: 从下到上
        x_w_reverse = torch.flip(x_w, dims=[1])
        
        # 拼接所有方向
        x_all = torch.cat([x_h, x_h_reverse, x_w, x_w_reverse], dim=0)  # [4B, H*W, C]
        
        # 选择性扫描
        y_all = self.ssm(x_all)  # [4B, H*W, C]
        
        # 分离各方向
        y_h, y_h_reverse, y_w, y_w_reverse = torch.chunk(y_all, 4, dim=0)
        
        # 恢复原始形状
        y_h = y_h.reshape(B, H, W, C)
        y_h_reverse = torch.flip(y_h_reverse.reshape(B, H, W, C), dims=[1])
        y_w = y_w.reshape(B, W, H, C).transpose(1, 2)
        y_w_reverse = torch.flip(y_w_reverse.reshape(B, W, H, C), dims=[1]).transpose(1, 2)
        
        # 合并
        y = y_h + y_h_reverse + y_w + y_w_reverse
        
        return self.norm(y)


# =============================================================================
# SpectralSpatialMamba - 光谱-空间联合 Mamba 模块
# =============================================================================

class SpectralSpatialMamba(nn.Module):
    """
    光谱-空间联合 Mamba 模块
    
    架构：
    1. 光谱分支：沿波段维度进行选择性扫描
    2. 空间分支：使用 2D Cross-Scan 捕获空间依赖
    3. 特征融合 + 残差连接
    
    输入: [Batch, H, W, Bands]
    输出: [Batch, H, W, output_dim]
    """
    
    def __init__(self, num_bands: int, hidden_dim: int, output_dim: int,
                 d_state: int = 16, d_conv: int = 3, expand: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            num_bands: 输入波段数
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            d_state: SSM 状态维度
            d_conv: SSM 卷积核大小
            expand: SSM 扩展因子
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(num_bands)
        
        # ============ 光谱分支 ============
        # 沿波段维度扫描
        self.spectral_proj = nn.Linear(num_bands, hidden_dim)
        self.spectral_norm = nn.LayerNorm(hidden_dim)
        self.spectral_ssm = SelectiveScanMamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # ============ 空间分支 ============
        # 使用 2D Cross-Scan
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_ssm = CrossScan2D(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # ============ 融合层 ============
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)  # 拼接后的维度
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # ============ 残差连接 ============
        self.residual_proj = nn.Linear(num_bands, output_dim) if num_bands != output_dim else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape = [Batch, H, W, Bands]
            
        Returns:
            输出张量, shape = [Batch, H, W, output_dim]
        """
        B, H, W, _ = x.shape
        
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # ============ 光谱分支 ============
        # 将每个像素的光谱作为序列
        x_spectral = x_norm.reshape(B, H * W, self.num_bands)  # [B, H*W, Bands]
        x_spectral = self.spectral_proj(x_spectral)  # [B, H*W, hidden]
        x_spectral = self.spectral_norm(x_spectral)
        x_spectral = self.spectral_ssm(x_spectral)  # [B, H*W, hidden]
        x_spectral = x_spectral.reshape(B, H, W, self.hidden_dim)  # [B, H, W, hidden]
        
        # ============ 空间分支 ============
        x_spatial = x_spectral.reshape(B, H, W, self.hidden_dim)
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial = self.spatial_ssm(x_spatial)  # [B, H, W, hidden]
        
        # ============ 特征融合 ============
        # 拼接光谱和空间特征
        x_fused = torch.cat([x_spectral, x_spatial], dim=-1)  # [B, H, W, hidden*2]
        x_fused = self.fusion_norm(x_fused)
        x_fused = self.fusion_proj(x_fused)  # [B, H, W, output_dim]
        
        # ============ 残差连接 ============
        x_residual = self.residual_proj(x)  # [B, H, W, output_dim]
        output = self.dropout(x_fused + x_residual)
        
        return output


# =============================================================================
# CrossAttentionFusion - HSI-LiDAR 交叉注意力融合模块
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """
    HSI-LiDAR 交叉注意力融合模块
    
    架构：
    1. HSI 和 LiDAR 分别进行特征提取
    2. 使用交叉注意力进行特征交互
    3. 残差连接 + 门控融合
    
    HSI 作为 Query，LiDAR 作为 Key/Value
    """
    
    def __init__(self, hsi_dim: int, lidar_dim: int, hidden_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            hsi_dim: HSI 特征维度
            lidar_dim: LiDAR 特征维度
            hidden_dim: 融合后的隐藏维度
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # HSI 投影
        self.hsi_proj = nn.Sequential(
            nn.Linear(hsi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # LiDAR 投影
        self.lidar_proj = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 交叉注意力 Q, K, V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, hsi_feat: torch.Tensor, lidar_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hsi_feat: HSI 特征, shape = [Batch, H, W, hsi_dim]
            lidar_feat: LiDAR 特征, shape = [Batch, H, W, lidar_dim]
            
        Returns:
            融合特征, shape = [Batch, H, W, hidden_dim]
        """
        B, H, W, _ = hsi_feat.shape
        
        # 投影到统一维度
        hsi = self.hsi_proj(hsi_feat)      # [B, H, W, hidden]
        lidar = self.lidar_proj(lidar_feat) # [B, H, W, hidden]
        
        # 展平空间维度
        hsi_flat = hsi.view(B, H * W, self.hidden_dim)     # [B, H*W, hidden]
        lidar_flat = lidar.view(B, H * W, self.hidden_dim) # [B, H*W, hidden]
        
        # ============ 交叉注意力 ============
        # HSI 作为 Query，LiDAR 作为 Key/Value
        Q = self.q_proj(hsi_flat)    # [B, H*W, hidden]
        K = self.k_proj(lidar_flat)  # [B, H*W, hidden]
        V = self.v_proj(lidar_flat)  # [B, H*W, hidden]
        
        # 重塑为多头
        Q = Q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, H*W, head_dim]
        K = K.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 注意力输出
        attn_out = torch.matmul(attn, V)  # [B, heads, H*W, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, self.hidden_dim)  # [B, H*W, hidden]
        attn_out = self.out_proj(attn_out)
        
        # 残差连接
        hsi_flat = self.norm1(hsi_flat + self.dropout(attn_out))
        
        # FFN
        hsi_flat = self.norm2(hsi_flat + self.ffn(hsi_flat))
        
        # ============ 门控融合 ============
        # 恢复空间维度
        hsi_out = hsi_flat.view(B, H, W, self.hidden_dim)
        
        # 门控融合 HSI 和 LiDAR 特征
        concat_feat = torch.cat([hsi_out, lidar], dim=-1)  # [B, H, W, hidden*2]
        gate_weight = self.gate(concat_feat)  # [B, H, W, hidden]
        
        # 最终融合
        fused = gate_weight * hsi_out + (1 - gate_weight) * lidar
        
        return fused


# =============================================================================
# LiDARFeatureExtractor - LiDAR 特征提取器
# =============================================================================

class LiDARFeatureExtractor(nn.Module):
    """
    LiDAR 特征提取器
    
    使用卷积网络从 LiDAR 数据中提取空间结构特征
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim: 输入通道数（通常为 1 或 3，取决于 LiDAR 类型）
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            
            # 第二层
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # 第三层
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: LiDAR 数据, shape = [Batch, Channels, H, W]
            
        Returns:
            特征, shape = [Batch, H, W, output_dim]
        """
        x = self.conv_layers(x)  # [B, output_dim, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, output_dim]
        return x


# =============================================================================
# Classifier - 分类器
# =============================================================================

class Classifier(nn.Module):
    """
    分类器模块
    
    使用全局平均池化 + MLP
    
    输入: [Batch, H, W, in_features]
    输出: [Batch, num_classes]
    """
    
    def __init__(self, in_features: int, num_classes: int,
                 hidden_features: int = 512, dropout: float = 0.3):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 分类类别数
            hidden_features: 隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape = [Batch, H, W, in_features]
            
        Returns:
            输出张量, shape = [Batch, num_classes]
        """
        # 全局平均池化
        x = x.mean(dim=(1, 2))  # [Batch, in_features]
        x = self.fc_layers(x)
        return x


# =============================================================================
# HSIClassificationMambaModel - 主模型
# =============================================================================

class HSIClassificationMambaModel(nn.Module):
    """
    高光谱图像分类 Mamba 模型（改进版）
    
    架构：
    1. HSI -> SpectralSpatialMamba -> HSI Features
    2. LiDAR -> LiDARFeatureExtractor -> LiDAR Features
    3. HSI Features + LiDAR Features -> CrossAttentionFusion -> Fused Features
    4. Fused Features -> Classifier -> Output
    
    输入: 
        - HSI: [Batch, H, W, Bands]
        - LiDAR: [Batch, Channels, H, W]
    输出: [Batch, num_classes]
    """
    
    def __init__(self, spatial_dim: int, num_bands: int, hidden_dim: int,
                 output_dim: int, delta_param_init: float, num_classes: int,
                 classifier_hidden: int = 512, dropout: float = 0.3,
                 use_band_selector: bool = False,
                 band_selector_hidden: int = 64,
                 band_selector_out: int = 32,
                 lidar_channels: int = 1,
                 d_state: int = 16,
                 d_conv: int = 3,
                 expand: int = 2,
                 fusion_heads: int = 4,
                 use_cross_attention: bool = True):
        """
        Args:
            spatial_dim: 空间维度 (patch size)
            num_bands: HSI 波段数（不含 LiDAR）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            delta_param_init: SSM delta 参数初始值（保留兼容性）
            num_classes: 分类类别数
            classifier_hidden: 分类器隐藏层维度
            dropout: Dropout 概率
            use_band_selector: 是否使用波段选择模块
            band_selector_hidden: 波段选择模块隐藏维度
            band_selector_out: 波段选择模块输出维度
            lidar_channels: LiDAR 通道数
            d_state: SSM 状态维度
            d_conv: SSM 卷积核大小
            expand: SSM 扩展因子
            fusion_heads: 融合注意力头数
            use_cross_attention: 是否使用交叉注意力融合（False 则使用 Early Fusion）
        """
        super().__init__()
        
        self.use_band_selector = use_band_selector
        self.use_cross_attention = use_cross_attention
        self.num_bands = num_bands
        self.output_dim = output_dim
        self.lidar_channels = lidar_channels
        
        # 波段选择模块 (可选)
        if use_band_selector:
            if not BAND_SELECTOR_AVAILABLE:
                raise ImportError(
                    "BandGraphAttentionSelector not available. "
                    "Please ensure band_graph_attention_selector.py is in the same directory."
                )
            self.band_selector = BandGraphAttentionSelector(
                node_feature_dim=3,
                gat_hidden_dim=band_selector_hidden,
                gat_out_dim=band_selector_out,
                score_hidden_dim=band_selector_hidden,
                n_heads=4,
                dropout=0.1
            )
        
        # ============ HSI 特征提取 ============
        self.hsi_mamba = SpectralSpatialMamba(
            num_bands=num_bands,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # ============ LiDAR 特征提取 ============
        if use_cross_attention:
            # 独立的 LiDAR 特征提取器
            self.lidar_extractor = LiDARFeatureExtractor(
                input_dim=lidar_channels,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            
            # 交叉注意力融合
            self.fusion = CrossAttentionFusion(
                hsi_dim=output_dim,
                lidar_dim=output_dim,
                hidden_dim=output_dim,
                num_heads=fusion_heads,
                dropout=dropout
            )
        
        # ============ 分类器 ============
        self.classifier = Classifier(
            in_features=output_dim,
            num_classes=num_classes,
            hidden_features=classifier_hidden,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, lidar: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: HSI 数据, shape = [Batch, H, W, Bands]
               如果 use_band_selector=True，则需要单独传入 hsi 和 lidar
            lidar: LiDAR 数据, shape = [Batch, Channels, H, W]
            
        Returns:
            输出张量, shape = [Batch, num_classes]
        """
        B, H, W, _ = x.shape
        
        # 波段选择 (可选)
        if self.use_band_selector and lidar is not None:
            # x: [Batch, H, W, Bands] 需要拆分 HSI 和 LiDAR
            hsi_only = x[..., :-self.lidar_channels]  # [Batch, H, W, Bands-1]
            
            # 转换为 [Bands, H, W] 格式用于波段选择
            batch_size = x.shape[0]
            weighted_hsi_list = []
            
            for i in range(batch_size):
                hsi_patch = hsi_only[i].permute(2, 0, 1)  # [Bands, H, W]
                lidar_patch = lidar[i:i+1]  # [1, Channels, H, W]
                
                # 波段选择加权
                weighted_hsi = self.band_selector(hsi_patch, lidar_patch, mode='weighted')
                weighted_hsi_list.append(weighted_hsi.permute(1, 2, 0))  # [H, W, Bands]
            
            weighted_hsi_batch = torch.stack(weighted_hsi_list, dim=0)  # [Batch, H, W, Bands]
            x = weighted_hsi_batch
        
        # ============ HSI 特征提取 ============
        hsi_feat = self.hsi_mamba(x)  # [B, H, W, output_dim]
        
        # ============ 多模态融合 ============
        if self.use_cross_attention and lidar is not None:
            # LiDAR 特征提取
            lidar_feat = self.lidar_extractor(lidar)  # [B, H, W, output_dim]
            
            # 交叉注意力融合
            fused_feat = self.fusion(hsi_feat, lidar_feat)  # [B, H, W, output_dim]
        else:
            # Early Fusion 回退
            fused_feat = hsi_feat
        
        # ============ 分类 ============
        output = self.classifier(fused_feat)  # [B, num_classes]
        
        return output


# =============================================================================
# 模型工厂函数
# =============================================================================

def create_model(config) -> HSIClassificationMambaModel:
    """
    根据配置创建模型
    
    Args:
        config: ModelConfig 实例
        
    Returns:
        HSIClassificationMambaModel 实例
    """
    return HSIClassificationMambaModel(
        spatial_dim=config.spatial_dim,
        num_bands=config.num_bands,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        delta_param_init=config.delta_param_init,
        num_classes=config.num_classes,
        classifier_hidden=config.classifier_hidden,
        dropout=config.dropout,
        use_band_selector=config.use_band_selector,
        band_selector_hidden=config.band_selector_hidden,
        band_selector_out=config.band_selector_out,
        lidar_channels=getattr(config, 'lidar_channels', 1),
        d_state=getattr(config, 'd_state', 16),
        d_conv=getattr(config, 'd_conv', 3),
        expand=getattr(config, 'expand', 2),
        fusion_heads=getattr(config, 'fusion_heads', 4),
        use_cross_attention=getattr(config, 'use_cross_attention', True)
    )


# =============================================================================
# 兼容旧接口
# =============================================================================

# 保留旧的 HSIVimBlock 作为别名（向后兼容）
HSIVimBlock = SpectralSpatialMamba

# 保留旧的 SpatialFeatureProcessing（向后兼容，但标记为废弃）
class SpatialFeatureProcessing(nn.Module):
    """已废弃：在新架构中不再使用"""
    def __init__(self, input_channels: int):
        super().__init__()
        self.identity = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试模型定义（改进版）")
    print("=" * 70)
    
    # 测试 SpectralSpatialMamba
    print("\n[1] 测试 SpectralSpatialMamba")
    mamba = SpectralSpatialMamba(
        num_bands=64,
        hidden_dim=128,
        output_dim=64,
        d_state=16,
        d_conv=3,
        expand=2
    )
    
    x = torch.randn(2, 9, 9, 64)  # [Batch, H, W, Bands]
    y = mamba(x)
    print(f"  输入 shape: {x.shape}")
    print(f"  输出 shape: {y.shape}")
    
    # 测试 CrossAttentionFusion
    print("\n[2] 测试 CrossAttentionFusion")
    fusion = CrossAttentionFusion(
        hsi_dim=64,
        lidar_dim=64,
        hidden_dim=64,
        num_heads=4
    )
    
    hsi_feat = torch.randn(2, 9, 9, 64)
    lidar_feat = torch.randn(2, 9, 9, 64)
    fused = fusion(hsi_feat, lidar_feat)
    print(f"  HSI 特征 shape: {hsi_feat.shape}")
    print(f"  LiDAR 特征 shape: {lidar_feat.shape}")
    print(f"  融合特征 shape: {fused.shape}")
    
    # 测试完整模型
    print("\n[3] 测试 HSIClassificationMambaModel")
    model = HSIClassificationMambaModel(
        spatial_dim=9,
        num_bands=64,
        hidden_dim=128,
        output_dim=64,
        delta_param_init=0.01,
        num_classes=11,
        classifier_hidden=256,
        dropout=0.2,
        lidar_channels=1,
        use_cross_attention=True
    )
    
    hsi = torch.randn(2, 9, 9, 64)  # [B, H, W, Bands]
    lidar = torch.randn(2, 1, 9, 9)  # [B, Channels, H, W]
    
    output = model(hsi, lidar)
    print(f"  HSI 输入 shape: {hsi.shape}")
    print(f"  LiDAR 输入 shape: {lidar.shape}")
    print(f"  输出 shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 测试 Early Fusion 模式
    print("\n[4] 测试 Early Fusion 模式（use_cross_attention=False）")
    model_early = HSIClassificationMambaModel(
        spatial_dim=9,
        num_bands=64,
        hidden_dim=128,
        output_dim=64,
        delta_param_init=0.01,
        num_classes=11,
        use_cross_attention=False
    )
    
    output_early = model_early(hsi)
    print(f"  输出 shape: {output_early.shape}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)