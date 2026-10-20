"""
HSLiNets 训练模块
包含训练器类、训练循环、验证和测试
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np

from config import ModelConfig, TrainingConfig
from utils import (
    TrainingLogger, compute_metrics, print_metrics, 
    get_scheduler, save_checkpoint, load_checkpoint,
    Timer, get_memory_usage, plot_training_curves
)


# =============================================================================
# 训练器类
# =============================================================================

class Trainer:
    """
    模型训练器
    
    封装训练循环、验证、测试、模型保存等功能
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 output_dir: str = "checkpoints",
                 class_names: Optional[list] = None):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            model_config: 模型配置
            training_config: 训练配置
            output_dir: 输出目录
            class_names: 类别名称列表
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.class_names = class_names or [f"Class_{i}" for i in range(model_config.num_classes)]
        
        # 设备
        self.device = torch.device(
            training_config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = get_scheduler(self.optimizer, training_config)
        
        # 日志记录器
        os.makedirs(output_dir, exist_ok=True)
        self.logger = TrainingLogger(
            save_path=os.path.join(output_dir, "training_log.json")
        )
        
        # 最佳模型
        self.best_model_wts = None
        self.best_val_loss = float('inf')
        
        # 早停计数器
        self.early_stop_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch
        
        Returns:
            avg_loss: 平均损失
            accuracy: 精度
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 支持两种数据格式：(hsi, lidar, labels) 或 (data, labels)
            if len(batch_data) == 3:
                hsi, lidar, labels = batch_data
                hsi = hsi.to(self.device)
                lidar = lidar.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播（使用交叉注意力）
                self.optimizer.zero_grad()
                outputs = self.model(hsi, lidar)
            else:
                data, labels = batch_data
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播（Early Fusion 模式）
                self.optimizer.zero_grad()
                outputs = self.model(data)
            
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        验证
        
        Args:
            loader: 数据加载器 (默认使用验证集)
            
        Returns:
            avg_loss: 平均损失
            accuracy: 精度
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in loader:
                # 支持两种数据格式
                if len(batch_data) == 3:
                    hsi, lidar, labels = batch_data
                    hsi = hsi.to(self.device)
                    lidar = lidar.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(hsi, lidar)
                else:
                    data, labels = batch_data
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(data)
                
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def test(self) -> Dict:
        """
        测试模型
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                # 支持两种数据格式
                if len(batch_data) == 3:
                    hsi, lidar, labels = batch_data
                    hsi = hsi.to(self.device)
                    lidar = lidar.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(hsi, lidar)
                else:
                    data, labels = batch_data
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        metrics = compute_metrics(
            np.array(all_labels), 
            np.array(all_preds),
            num_classes=self.model_config.num_classes
        )
        
        return metrics
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数 (默认使用配置中的值)
            
        Returns:
            最佳模型的测试指标
        """
        if epochs is None:
            epochs = self.training_config.epochs
        
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"初始学习率: {self.training_config.learning_rate}")
        print(f"批量大小: {self.training_config.batch_size}")
        print(get_memory_usage())
        print("=" * 60)
        
        timer = Timer()
        timer.start()
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录日志
            self.logger.log(epoch, train_loss, val_loss, train_acc, val_acc)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
                
                # 保存检查点
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    {"val_loss": val_loss, "val_acc": val_acc},
                    os.path.join(self.output_dir, "best_model.pth")
                )
                print(f"  Validation Loss Decreased -> Saving Model")
            else:
                self.early_stop_counter += 1
            
            # 早停
            if self.early_stop_counter >= self.training_config.early_stop_patience:
                print(f"\n早停: 验证损失在 {self.training_config.early_stop_patience} 个 epoch 内未改善")
                break
        
        # 训练结束
        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"总训练时间: {timer}")
        print(f"最佳验证损失: {self.logger.best_val_loss:.4f}")
        print(f"最佳验证精度: {self.logger.best_val_acc:.2f}%")
        print(f"最佳 Epoch: {self.logger.best_epoch}")
        
        # 保存训练日志
        self.logger.save()
        
        # 绘制训练曲线
        plot_training_curves(
            self.logger,
            save_path=os.path.join(self.output_dir, "training_curves.png")
        )
        
        # 加载最佳模型进行测试
        print("\n加载最佳模型进行测试...")
        self.model.load_state_dict(self.best_model_wts)
        
        # 测试
        test_metrics = self.test()
        print_metrics(test_metrics, self.class_names)
        
        return test_metrics


# =============================================================================
# 便捷训练函数
# =============================================================================

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                test_loader: DataLoader,
                model_config: ModelConfig,
                training_config: TrainingConfig,
                output_dir: str = "checkpoints",
                class_names: Optional[list] = None) -> Dict:
    """
    训练模型的便捷函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        model_config: 模型配置
        training_config: 训练配置
        output_dir: 输出目录
        class_names: 类别名称列表
        
    Returns:
        测试指标字典
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
        class_names=class_names
    )
    
    return trainer.train()


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试训练模块")
    print("=" * 60)
    
    # 创建模拟数据
    print("\n创建模拟数据...")
    
    batch_size = 32
    num_samples = 100
    num_bands = 64
    num_classes = 6
    
    # 模拟数据
    train_data = torch.randn(num_samples, 1, 1, num_bands)
    train_labels = torch.randint(0, num_classes, (num_samples,))
    
    val_data = torch.randn(20, 1, 1, num_bands)
    val_labels = torch.randint(0, num_classes, (20,))
    
    test_data = torch.randn(30, 1, 1, num_bands)
    test_labels = torch.randint(0, num_classes, (30,))
    
    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data, val_labels),
        batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels),
        batch_size=batch_size
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    print("\n测试完成")
    print("=" * 60)
