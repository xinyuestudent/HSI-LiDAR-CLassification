"""
简单测试 SelectiveScan 模块
"""

import sys
import torch

sys.path.insert(0, '.')

from model import SelectiveScan, SelectiveScanMamba

def test_selective_scan():
    print("=" * 60)
    print("测试 SelectiveScan 模块")
    print("=" * 60)
    
    batch_size = 4
    length = 81
    d_model = 96
    d_state = 16
    expand = 2
    
    model = SelectiveScan(
        d_model=d_model,
        d_state=d_state,
        d_conv=3,
        expand=expand
    )
    
    x = torch.randn(batch_size, length, d_model)
    print(f"输入 shape: {x.shape}")
    print(f"d_model: {d_model}")
    print(f"d_inner: {model.d_inner}")
    
    # 测试前向传播
    try:
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"输出 shape: {output.shape}")
        print(f"期望 shape: [{batch_size}, {length}, {d_model}]")
        
        if output.shape == (batch_size, length, d_model):
            print("✓ SelectiveScan 测试通过")
        else:
            print("✗ SelectiveScan 维度不匹配")
    except Exception as e:
        print(f"✗ SelectiveScan 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

def test_spectral_spatial_mamba_debug():
    print("\n" + "=" * 60)
    print("测试 SpectralSpatialMamba 模块 (调试)")
    print("=" * 60)
    
    from model import SpectralSpatialMamba
    
    batch_size = 4
    h, w = 9, 9
    num_bands = 64
    hidden_dim = 96
    
    model = SpectralSpatialMamba(
        num_bands=num_bands,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, h, w, num_bands)
    print(f"输入 shape: {x.shape}")
    
    # 逐步测试
    try:
        model.eval()
        with torch.no_grad():
            # 输入归一化
            x_norm = model.input_norm(x)
            print(f"input_norm output: {x_norm.shape}")
            
            # 光谱分支
            x_spectral = x_norm.reshape(batch_size, h * w, num_bands)
            print(f"reshape: {x_spectral.shape}")
            
            x_spectral = model.spectral_proj(x_spectral)
            print(f"spectral_proj output: {x_spectral.shape}")
            
            x_spectral = model.spectral_norm(x_spectral)
            print(f"spectral_norm output: {x_spectral.shape}")
            
            x_spectral = model.spectral_ssm(x_spectral)
            print(f"spectral_ssm output: {x_spectral.shape}")
            
            x_spectral = x_spectral.reshape(batch_size, h, w, hidden_dim)
            print(f"spectral reshape output: {x_spectral.shape}")
            
            # 空间分支
            x_spatial = x_spectral
            print(f"x_spatial: {x_spatial.shape}")
            
            x_spatial = model.spatial_norm(x_spatial)
            print(f"spatial_norm output: {x_spatial.shape}")
            
            x_spatial = model.spatial_ssm(x_spatial)
            print(f"spatial_ssm output: {x_spatial.shape}")
            
            # 融合
            x_fused = torch.cat([x_spectral, x_spatial], dim=-1)
            print(f"concat output: {x_fused.shape}")
            
            x_fused = model.fusion_norm(x_fused)
            print(f"fusion_norm output: {x_fused.shape}")
            
            x_fused = model.fusion_proj(x_fused)
            print(f"fusion_proj output: {x_fused.shape}")
            
            # 残差
            x_residual = model.residual_proj(x)
            print(f"residual_proj output: {x_residual.shape}")
            
            output = model.dropout(x_fused + x_residual)
            print(f"final output: {output.shape}")
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_selective_scan()
    test_spectral_spatial_mamba_debug()
