"""
配置参数
"""
import torch

class Config:
    # 数据配置
    image_size = 28
    num_channels = 1
    num_classes = 10

    # 扩散配置
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # 训练配置
    batch_size = 64
    epochs = 100
    learning_rate = 2e-4

    # 模型配置
    base_channels = 32

    # 路径配置
    data_dir = "./data/mnist"
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return f"设备: {self.device}\n时间步: {self.timesteps}\n批次大小: {self.batch_size}\n训练轮数: {self.epochs}"