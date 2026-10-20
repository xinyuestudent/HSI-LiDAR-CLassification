"""
测试 Houston 2018 数据集加载
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 Houston 2018 配置")
print("=" * 60)

try:
    from config import HOUSTON2018_CONFIG, get_dataset_config, DATASET_CONFIGS
    
    print("\n1. HOUSTON2018_CONFIG 配置:")
    print(f"   名称: {HOUSTON2018_CONFIG['name']}")
    print(f"   HSI 波段: {HOUSTON2018_CONFIG['hsi_bands']}")
    print(f"   LiDAR 波段: {HOUSTON2018_CONFIG['lidar_bands']}")
    print(f"   类别数: {HOUSTON2018_CONFIG['num_classes']}")
    print(f"   数据来源: {HOUSTON2018_CONFIG['data_source']}")
    
    print("\n2. get_dataset_config('houston2018'):")
    config = get_dataset_config('houston2018')
    print(f"   名称: {config['name']}")
    
    print("\n3. DATASET_CONFIGS:")
    print(f"   可用数据集: {list(DATASET_CONFIGS.keys())}")
    
    print("\n✓ 配置加载成功!")
    
except Exception as e:
    print(f"\n✗ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试数据加载函数")
print("=" * 60)

try:
    from dataset import load_houston2018_data, RS_FUSION_AVAILABLE
    
    print(f"\nrs-fusion-datasets 可用: {RS_FUSION_AVAILABLE}")
    
    if RS_FUSION_AVAILABLE:
        print("\n尝试加载 Houston 2018 数据集...")
        hsi, lidar, gt = load_houston2018_data()
        print(f"\n数据加载成功!")
        print(f"   HSI shape: {hsi.shape}")
        print(f"   LiDAR shape: {lidar.shape}")
        print(f"   GT shape: {gt.shape}")
    else:
        print("\n⚠ rs-fusion-datasets 未安装")
        print("请运行: pip install rs-fusion-datasets")
        
except ImportError as e:
    print(f"\n✗ 导入失败: {e}")
    print("\n请安装 rs-fusion-datasets:")
    print("  pip install rs-fusion-datasets")
except Exception as e:
    print(f"\n✗ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
