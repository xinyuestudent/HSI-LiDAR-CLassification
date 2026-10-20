"""测试 Houston 2018 数据集完整加载流程"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 Houston 2018 数据集")
print("=" * 60)

# 1. 测试数据下载
print("\n[1] 下载数据集...")
from rs_fusion_datasets import fetch_houston2018_ouc
try:
    hsi, dsm, train_label, test_label, all_label, info = fetch_houston2018_ouc()
    print(f"  HSI shape: {hsi.shape}")
    print(f"  DSM shape: {dsm.shape}")
    print(f"  Train label shape: {train_label.shape}")
    print(f"  Test label shape: {test_label.shape}")
    print(f"  类别数: {info['n_class']}")
except Exception as e:
    print(f"  下载失败: {e}")
    sys.exit(1)

# 2. 测试配置
print("\n[2] 测试配置...")
from config import get_dataset_config, get_class_dict
config = get_dataset_config("houston2018")
class_dict = get_class_dict(config["class_info"])
print(f"  数据集名称: {config['name']}")
print(f"  类别数: {config['num_classes']}")

# 3. 测试数据加载函数
print("\n[3] 测试数据加载函数...")
from dataset import load_houston2018_data
try:
    hsi2, lidar2, gt2 = load_houston2018_data("dummy_path")
    print(f"  HSI shape: {hsi2.shape}")
    print(f"  LiDAR shape: {lidar2.shape}")
    print(f"  GT shape: {gt2.shape}")
    print(f"  ✓ 数据加载成功!")
except Exception as e:
    print(f"  加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ Houston 2018 测试完成!")
print("=" * 60)