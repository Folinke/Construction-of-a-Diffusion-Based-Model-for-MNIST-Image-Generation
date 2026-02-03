"""
数据加载器
"""
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class MNISTLoader:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_loaders(self):
        """获取训练和测试数据加载器"""
        train_dataset = torchvision.datasets.MNIST(
            root=self.config.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=self.config.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print(f"训练集: {len(train_dataset)} 张图片")
        print(f"测试集: {len(test_dataset)} 张图片")

        return train_loader, test_loader