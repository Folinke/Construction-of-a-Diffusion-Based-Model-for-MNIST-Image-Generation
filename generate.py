"""
生成脚本
"""
import os
#避免OMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torchvision
import matplotlib.pyplot as plt
from config import Config
from model import SimpleUNet
from diffusion import DiffusionProcess


def load_model(config, checkpoint_path):
    """加载模型"""
    model = SimpleUNet(config)
    model.to(config.device)

    if os.path.exists(checkpoint_path):
        # 使用 weights_only=True 避免安全警告
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            # 尝试直接加载模型状态字典
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(model_state_dict, strict=False)
        else:
            # 如果检查点本身就是模型状态字典
            model.load_state_dict(checkpoint)
        print(f"加载模型: {checkpoint_path}")
        model.eval()  # 设置为评估模式
    else:
        # 如果没有找到最终模型，尝试加载最新的检查点
        checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
            checkpoint_path = os.path.join(config.checkpoint_dir, latest)
            print(f"未找到最终模型，尝试加载: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            raise FileNotFoundError(f"未找到任何检查点文件，请先训练模型")

    return model


def generate():
    """生成数字"""
    print("生成MNIST数字...")
    print(f"设备: {Config().device}")

    # 配置
    config = Config()

    # 确保结果目录存在
    os.makedirs(config.results_dir, exist_ok=True)

    # 加载模型
    checkpoint_path = os.path.join(config.checkpoint_dir, "model_final.pth")
    model = load_model(config, checkpoint_path)

    # 创建扩散过程
    diffusion = DiffusionProcess(config)

    # 生成数字 - 减少数量以确保内存足够
    num_images = 9  # 改为9个，更容易显示
    shape = (num_images, config.num_channels, config.image_size, config.image_size)

    print(f"正在生成 {num_images} 个图像...")

    with torch.no_grad():
        #samples = diffusion.sample(model, shape, config.device)
        samples = diffusion.ddim_sample(model, shape, config.device,
                                        sampling_timesteps=50, eta=0.0)
        samples = (samples.clamp(-1, 1) + 1) / 2

    # 显示和保存
    grid = torchvision.utils.make_grid(samples.cpu(), nrow=3, pad_value=1.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.title("Generated MNIST Digits")
    plt.axis('off')

    save_path = os.path.join(config.results_dir, "final_generated.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"生成结果已保存到: {save_path}")

    # 显示图像
    plt.show()

    return samples


def generate_specific_digit(digit=None, num_images=9):
    """生成特定数字（如果模型支持条件生成）"""
    print(f"生成数字: {digit if digit is not None else '随机'}")

    # 注意：这个简单版本不支持条件生成
    # 但我们可以多次生成并选择想要的数字
    config = Config()
    checkpoint_path = os.path.join(config.checkpoint_dir, "model_final.pth")
    model = load_model(config, checkpoint_path)
    diffusion = DiffusionProcess(config)

    if digit is None:
        # 生成随机数字
        shape = (num_images, config.num_channels, config.image_size, config.image_size)
        with torch.no_grad():
            samples = diffusion.sample(model, shape, config.device)
            samples = (samples.clamp(-1, 1) + 1) / 2

        grid = torchvision.utils.make_grid(samples.cpu(), nrow=3, pad_value=1.0)

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f"Generated MNIST Digits (Random)")
        plt.axis('off')

        save_path = os.path.join(config.results_dir, f"random_digits.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("注意：这个简单模型不支持条件生成。")
        print("要生成特定数字，您需要训练一个条件扩散模型。")


if __name__ == "__main__":
    # 检查检查点目录是否存在
    if not os.path.exists(Config().checkpoint_dir):
        print(f"错误: 检查点目录 {Config().checkpoint_dir} 不存在")
        print("请先运行 train.py 训练模型")
    elif not os.listdir(Config().checkpoint_dir):
        print(f"错误: 检查点目录 {Config().checkpoint_dir} 为空")
        print("请先运行 train.py 训练模型")
    else:
        generate()