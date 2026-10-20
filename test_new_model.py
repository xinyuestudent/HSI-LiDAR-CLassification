"""
测试新模型架构
验证 SpectralSpatialMamba 和 CrossAttentionFusion 模块
"""

import sys
import torch

# 添加模块路径
sys.path.insert(0, '.')

from model import (
    SpectralSpatialMamba,
    CrossAttentionFusion,
    LiDARFeatureExtractor,
    HSIClassificationMambaModel,
    create_model
)
from config import ModelConfig


def test_spectral_spatial_mamba():
    """测试 SpectralSpatialMamba 模块"""
    print("\n" + "=" * 60)
    print("测试 SpectralSpatialMamba 模块")
    print("=" * 60)
    
    batch_size = 4
    h, w = 9, 9
    num_bands = 64
    hidden_dim = 96
    output_dim = 96
    
    model = SpectralSpatialMamba(
        num_bands=num_bands,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, h, w, num_bands)
    
    print(f"输入 shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输出 shape: {output.shape}")
    print(f"预期 shape: [({batch_size}, {h}, {w}, {output_dim})]")
    
    assert output.shape == (batch_size, h, w, output_dim), f"Shape mismatch: {output.shape}"
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
    
    print("✓ SpectralSpatialMamba 测试通过")
    return True


def test_cross_attention_fusion():
    """测试 CrossAttentionFusion 模块"""
    print("\n" + "=" * 60)
    print("测试 CrossAttentionFusion 模块")
    print("=" * 60)
    
    batch_size = 4
    h, w = 9, 9
    hsi_dim = 96
    lidar_dim = 32
    hidden_dim = 64
    num_heads = 4
    
    model = CrossAttentionFusion(
        hsi_dim=hsi_dim,
        lidar_dim=lidar_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1
    )
    
    hsi_feat = torch.randn(batch_size, h, w, hsi_dim)
    lidar_feat = torch.randn(batch_size, h, w, lidar_dim)
    
    print(f"HSI 特征 shape: {hsi_feat.shape}")
    print(f"LiDAR 特征 shape: {lidar_feat.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(hsi_feat, lidar_feat)
    
    print(f"融合输出 shape: {output.shape}")
    print(f"预期 shape: [({batch_size}, {h}, {w}, {hidden_dim})]")
    
    assert output.shape == (batch_size, h, w, hidden_dim), f"Shape mismatch: {output.shape}"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
    
    print("✓ CrossAttentionFusion 测试通过")
    return True


def test_lidar_feature_extractor():
    """测试 LiDARFeatureExtractor 模块"""
    print("\n" + "=" * 60)
    print("测试 LiDARFeatureExtractor 模块")
    print("=" * 60)
    
    batch_size = 4
    h, w = 9, 9
    lidar_channels = 1
    hidden_dim = 16
    output_dim = 32
    
    model = LiDARFeatureExtractor(
        input_dim=lidar_channels,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # LiDARFeatureExtractor 期望 [B, C, H, W] 格式
    lidar = torch.randn(batch_size, lidar_channels, h, w)
    
    print(f"LiDAR 输入 shape: {lidar.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(lidar)
    
    print(f"输出 shape: {output.shape}")
    print(f"预期 shape: [({batch_size}, {h}, {w}, {output_dim})]")
    
    assert output.shape == (batch_size, h, w, output_dim), f"Shape mismatch: {output.shape}"
    
    print("✓ LiDARFeatureExtractor 测试通过")
    return True


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 60)
    print("测试完整 HSIClassificationMambaModel")
    print("=" * 60)
    
    batch_size = 4
    h, w = 9, 9
    num_bands = 64  # HSI bands only
    lidar_channels = 1
    num_classes = 11
    
    # 创建模型配置
    config = ModelConfig(
        spatial_dim=h,
        num_bands=num_bands,
        lidar_channels=lidar_channels,
        hidden_dim=96,
        output_dim=96,
        num_classes=num_classes,
        fusion_heads=4,
        use_cross_attention=True
    )
    
    model = create_model(config)
    
    # 创建测试输入
    hsi = torch.randn(batch_size, h, w, num_bands)
    lidar = torch.randn(batch_size, lidar_channels, h, w)
    
    print(f"HSI 输入 shape: {hsi.shape}")
    print(f"LiDAR 输入 shape: {lidar.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(hsi, lidar)
    
    print(f"模型输出 shape: {output.shape}")
    print(f"预期 shape: [({batch_size}, {num_classes})]")
    
    assert output.shape == (batch_size, num_classes), f"Shape mismatch: {output.shape}"
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("✓ 完整模型测试通过")
    return True


def test_early_fusion_mode():
    """测试 Early Fusion 模式"""
    print("\n" + "=" * 60)
    print("测试 Early Fusion 模式")
    print("=" * 60)
    
    batch_size = 4
    h, w = 9, 9
    num_bands = 64
    lidar_channels = 1
    num_classes = 11
    
    # 创建模型配置 (关闭交叉注意力)
    config = ModelConfig(
        spatial_dim=h,
        num_bands=num_bands + lidar_channels,  # 包含 LiDAR
        lidar_channels=lidar_channels,
        hidden_dim=96,
        output_dim=96,
        num_classes=num_classes,
        use_cross_attention=False
    )
    
    model = create_model(config)
    
    # 创建测试输入 (HSI + LiDAR 拼接)
    x = torch.randn(batch_size, h, w, num_bands + lidar_channels)
    
    print(f"输入 shape (HSI+LiDAR): {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"模型输出 shape: {output.shape}")
    print(f"预期 shape: [({batch_size}, {num_classes})]")
    
    assert output.shape == (batch_size, num_classes), f"Shape mismatch: {output.shape}"
    
    print("✓ Early Fusion 模式测试通过")
    return True


def main():
    print("=" * 70)
    print("HSLiNets 新模型架构测试")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed &= test_spectral_spatial_mamba()
    except Exception as e:
        print(f"✗ SpectralSpatialMamba 测试失败: {e}")
        all_passed = False
    
    try:
        all_passed &= test_cross_attention_fusion()
    except Exception as e:
        print(f"✗ CrossAttentionFusion 测试失败: {e}")
        all_passed = False
    
    try:
        all_passed &= test_lidar_feature_extractor()
    except Exception as e:
        print(f"✗ LiDARFeatureExtractor 测试失败: {e}")
        all_passed = False
    
    try:
        all_passed &= test_full_model()
    except Exception as e:
        print(f"✗ 完整模型测试失败: {e}")
        all_passed = False
    
    try:
        all_passed &= test_early_fusion_mode()
    except Exception as e:
        print(f"✗ Early Fusion 模式测试失败: {e}")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)