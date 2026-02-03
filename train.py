"""
训练脚本
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入自定义模块
from config import Config
from data_loader import MNISTLoader
from model import SimpleUNet
from diffusion import DiffusionProcess


def train():
    """训练函数"""
    print("=" * 60)
    print("开始训练扩散模型")
    print("=" * 60)

    # 创建配置
    config = Config()
    print(config)

    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    loader = MNISTLoader(config)
    train_loader, _ = loader.get_loaders()

    # 创建模型和扩散过程
    print("创建模型和扩散过程...")
    model = SimpleUNet(config)
    model.to(config.device)

    diffusion = DiffusionProcess(config)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 训练
    print(f"\n开始训练 {config.epochs} 个epoch...")
    losses = []
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')

        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(config.device)

            # 随机采样时间步
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=config.device)

            # 计算损失
            loss = diffusion.compute_loss(model, images, t)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()
            losses.append(loss.item())

            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 每100个batch打印一次
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}')

        # 每10个epoch保存一次模型并生成样本
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            # 保存模型
            checkpoint_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'模型已保存到: {checkpoint_path}')

            # 生成样本
            generate_samples(model, diffusion, config, epoch+1)

    # 训练时间
    training_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {training_time:.2f}秒")

    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存到: {final_path}")

    # 绘制损失曲线
    plot_loss_curve(losses, config)

    return model, diffusion


def generate_samples(model, diffusion, config, epoch):
    """生成样本"""
    model.eval()
    with torch.no_grad():
        # 生成16个样本
        shape = (16, config.num_channels, config.image_size, config.image_size)
        samples = diffusion.sample(model, shape, config.device)

        # 转换为0-1范围
        samples = (samples.clamp(-1, 1) + 1) / 2

        # 保存图像
        grid = torchvision.utils.make_grid(samples.cpu(), nrow=4, pad_value=1.0)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f'Generated MNIST Digits (Epoch {epoch})')
        plt.axis('off')

        save_path = os.path.join(config.results_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'  样本已保存到: {save_path}')

    model.train()


def plot_loss_curve(losses, config):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))

    # 原始损失
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # 平滑损失
    plt.subplot(1, 2, 2)
    if len(losses) > 100:
        window = 100
        smooth_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(smooth_losses)
        plt.title(f'Smoothed Loss (window={window})')
    else:
        plt.plot(losses)
        plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(config.results_dir, 'training_loss.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'损失曲线已保存到: {save_path}')


def test_model():
    """测试模型是否能正常工作"""
    print("测试模型...")

    config = Config()
    config.timesteps = 10  # 测试时减少时间步

    # 创建模型
    model = SimpleUNet(config)
    model.to(config.device)

    # 测试前向传播
    x = torch.randn(2, 1, 28, 28).to(config.device)
    t = torch.randint(0, config.timesteps, (2,)).to(config.device)

    output = model(x, t)
    print(f"✓ 模型前向传播测试通过")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")

    # 测试扩散过程
    diffusion = DiffusionProcess(config)
    loss = diffusion.compute_loss(model, x, t)
    print(f"✓ 损失计算测试通过")
    print(f"  损失值: {loss.item():.4f}")

    return True


if __name__ == "__main__":
    # 先测试模型
    if test_model():
        print("\n模型测试通过，开始正式训练...")
        # 开始训练
        model, diffusion = train()

        print("\n" + "=" * 60)
        print("训练完成! 检查结果:")
        print(f"  模型检查点: {Config().checkpoint_dir}/")
        print(f"  生成结果: {Config().results_dir}/")
        print("=" * 60)