"""测试 rs_fusion_datasets 导入"""
import sys
print(f"Python: {sys.executable}")

try:
    from rs_fusion_datasets import fetch_houston2018_ouc, Houston2018Ouc
    print("fetch_houston2018_ouc import: OK")
except ImportError as e:
    print(f"fetch_houston2018_ouc import FAILED: {e}")