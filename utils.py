"""
HSLiNets 工具函数模块
包含评估指标、可视化等工具
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


# =============================================================================
# 评估指标
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    num_classes: int) -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
        
    Returns:
        指标字典 {oa, aa, kappa, class_accuracy}
    """
    # Overall Accuracy (OA)
    oa = accuracy_score(y_true, y_pred)
    
    # Average Accuracy (AA) - 各类别精度的平均
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    aa = np.mean(class_accuracy)
    
    # Kappa 系数
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return {
        "OA": oa * 100,
        "AA": aa * 100,
        "Kappa": kappa * 100,
        "class_accuracy": class_accuracy * 100,
        "confusion_matrix": cm
    }


def print_metrics(metrics: Dict, class_names: Optional[List[str]] = None):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        class_names: 类别名称列表
    """
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"Overall Accuracy (OA): {metrics['OA']:.2f}%")
    print(f"Average Accuracy (AA): {metrics['AA']:.2f}%")
    print(f"Kappa Coefficient: {metrics['Kappa']:.2f}%")
    
    if class_names and "class_accuracy" in metrics:
        print("\n各类别精度:")
        for i, (name, acc) in enumerate(zip(class_names, metrics["class_accuracy"])):
            print(f"  {i+1}. {name}: {acc:.2f}%")
    
    print("=" * 50)


# =============================================================================
# 混淆矩阵可视化
# =============================================================================

def plot_confusion_matrix(cm: np.ndarray, 
                          class_names: List[str],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10)):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 归一化
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标注
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.close()


# =============================================================================
# 训练日志
# =============================================================================

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Args:
            save_path: 日志保存路径
        """
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.save_path = save_path
    
    def log(self, epoch: int, train_loss: float, val_loss: float,
            train_acc: float, val_acc: float):
        """
        记录一个 epoch 的训练结果
        
        Args:
            epoch: 当前 epoch
            train_loss: 训练损失
            val_loss: 验证损失
            train_acc: 训练精度
            val_acc: 验证精度
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # 更新最佳结果
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch
        
        # 打印日志
        print(f"Epoch [{epoch}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    def save(self):
        """保存日志"""
        if self.save_path:
            import json
            log_data = {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accs": self.train_accs,
                "val_accs": self.val_accs,
                "best_val_loss": self.best_val_loss,
                "best_val_acc": self.best_val_acc,
                "best_epoch": self.best_epoch
            }
            with open(self.save_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"训练日志已保存至: {self.save_path}")


# =============================================================================
# 学习率调度
# =============================================================================

def get_scheduler(optimizer, config):
    """
    根据配置获取学习率调度器
    
    Args:
        optimizer: 优化器
        config: TrainingConfig 实例
        
    Returns:
        学习率调度器
    """
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    elif config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.gamma, patience=10
        )
    else:
        return None


# =============================================================================
# 模型保存与加载
# =============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, save_path: str):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        metrics: 评估指标字典
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"模型检查点已保存至: {save_path}")


def load_checkpoint(model, optimizer, load_path: str, device: str = 'cuda'):
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        load_path: 检查点路径
        device: 设备
        
    Returns:
        加载的 epoch 和 metrics
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('metrics', {})


# =============================================================================
# 内存和时间统计
# =============================================================================

def get_memory_usage():
    """获取当前 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        return f"GPU Memory: Allocated={allocated:.2f}MB, Cached={cached:.2f}MB"
    return "CUDA not available"


class Timer:
    """计时器"""
    
    def __init__(self):
        self.start_time = None
    
    def start(self):
        import time
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        import time
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def __str__(self):
        elapsed = self.elapsed()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =============================================================================
# 绘制训练曲线
# =============================================================================

def plot_training_curves(logger: TrainingLogger, save_path: Optional[str] = None):
    """
    绘制训练曲线
    
    Args:
        logger: TrainingLogger 实例
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(logger.train_losses, label='Train Loss')
    axes[0].plot(logger.val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 精度曲线
    axes[1].plot(logger.train_accs, label='Train Acc')
    axes[1].plot(logger.val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")
    
    plt.close()


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试工具函数")
    print("=" * 60)
    
    # 测试评估指标
    print("\n[测试评估指标]")
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    print(f"OA: {metrics['OA']:.2f}%")
    print(f"AA: {metrics['AA']:.2f}%")
    print(f"Kappa: {metrics['Kappa']:.2f}%")
    
    # 测试计时器
    print("\n[测试计时器]")
    timer = Timer()
    timer.start()
    import time
    time.sleep(1)
    print(f"经过时间: {timer}")
    
    # 测试内存
    print("\n[测试内存]")
    print(get_memory_usage())
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
