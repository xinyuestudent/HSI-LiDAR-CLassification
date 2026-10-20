"""快速测试 Houston 2018 配置"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("快速测试 Houston 2018")
print("=" * 60)

# 1. 测试配置
from config import HOUSTON2018_CONFIG, get_dataset_config, DATASET_CONFIGS
print("\n✓ 配置加载成功")
print(f"  可用数据集: {list(DATASET_CONFIGS.keys())}")

# 2. 测试 rs-fusion-datasets
from dataset import RS_FUSION_AVAILABLE
print(f"\n✓ rs-fusion-datasets 可用: {RS_FUSION_AVAILABLE}")

# 3. 测试导入
if RS_FUSION_AVAILABLE:
    from rs_fusion_datasets import fetch_houston2018_ouc
    print("\n✓ fetch_houston2018_ouc 导入成功")

print("\n" + "=" * 60)
print("配置测试通过！数据加载需要下载约 500MB 数据。")
print("=" * 60)
