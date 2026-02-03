"""
主程序入口
"""
import os
import sys
import argparse
import torch
from config import Config
from data_loader import MNISTDataLoader
from diffusion_process import DiffusionProcess
from unet_model import ConditionalUNet, BasicUNet
from trainer import DiffusionTrainer
from generator import ImageGenerator
from utils import set_seed, save_config, print_model_summary
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于扩散的MNIST图片生成模型")

    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'generate', 'test', 'visualize'],
                        help='运行模式: train, generate, test, visualize')

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--image_size', type=int, default=28, help='图像大小')

    # 模型参数
    parser.add_argument('--model', type=str, default='conditional',
                        choices=['basic', 'conditional'],
                        help='模型类型: basic, conditional')
    parser.add_argument('--base_channels', type=int, default=32, help='基础通道数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散时间步')

    # 生成参数
    parser.add_argument('--num_images', type=int, default=16, help='生成图像数量')
    parser.add_argument('--digit', type=int, default=None, help='要生成的数字')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='引导尺度')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--use_wandb', action='store_true', help='使用wandb记录')

    return parser.parse_args()


def train_mode(config, args):
    """训练模式"""
    print("=" * 60)
    print("开始训练模式")
    print("=" * 60)

    # 设置随机种子
    set_seed(args.seed)

    # 创建组件
    dataloader = MNISTDataLoader(config)
    train_loader, val_loader = dataloader.get_dataloaders()
    diffusion = DiffusionProcess(config)

    # 创建模型
    if args.model == 'basic':
        model = BasicUNet()
    else:
        model = ConditionalUNet(config)

    # 打印模型摘要
    print_model_summary(model)

    # 创建训练器
    trainer = DiffusionTrainer(model, diffusion, config)

    # 初始化wandb
    if args.use_wandb:
        trainer.init_wandb()

    # 加载检查点（如果提供）
    if args.checkpoint and os.path.exists(args.checkpoint):
        trainer.load_checkpoint(args.checkpoint)

    # 开始训练
    trainer.train(train_loader, val_loader)

    print("训练完成!")


def generate_mode(config, args):
    """生成模式"""
    print("=" * 60)
    print("开始生成模式")
    print("=" * 60)

    # 设置随机种子
    set_seed(args.seed)

    # 创建组件
    diffusion = DiffusionProcess(config)

    # 创建模型
    if args.model == 'basic':
        model = BasicUNet()
    else:
        model = ConditionalUNet(config)

    # 加载检查点
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点: {args.checkpoint}")
    else:
        # 尝试加载最佳模型
        best_checkpoint = os.path.join(config.checkpoint_dir, "model_best.pth")
        if os.path.exists(best_checkpoint):
            checkpoint = torch.load(best_checkpoint, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载最佳模型: {best_checkpoint}")
        else:
            print("警告: 未找到检查点，使用随机初始化的模型")

    model.to(config.device)
    model.eval()

    # 创建生成器
    generator = ImageGenerator(model, diffusion, config)

    # 生成图像
    if args.digit is not None:
        # 生成特定数字
        print(f"生成数字 {args.digit} 的图像...")
        generator.generate_digit_grid(
            digit=args.digit,
            num_rows=4,
            num_cols=4,
            guidance_scale=args.guidance_scale
        )

        # 可视化采样过程
        generator.visualize_sampling_process(digit=args.digit)
    else:
        # 生成所有数字
        print("生成所有数字的图像...")
        generator.generate_digit_grid(
            digit=None,
            num_rows=2,
            num_cols=5,
            guidance_scale=args.guidance_scale
        )

        # 生成随机图像
        generator.generate_images(
            num_images=args.num_images,
            guidance_scale=args.guidance_scale
        )

    print("生成完成!")


def test_mode(config, args):
    """测试模式"""
    print("=" * 60)
    print("开始测试模式")
    print("=" * 60)

    # 设置随机种子
    set_seed(args.seed)

    # 创建组件
    dataloader = MNISTDataLoader(config)
    train_loader, test_loader = dataloader.get_dataloaders()
    diffusion = DiffusionProcess(config)

    # 创建模型
    if args.model == 'basic':
        model = BasicUNet()
    else:
        model = ConditionalUNet(config)

    # 加载检查点
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点: {args.checkpoint}")
    else:
        print("错误: 需要提供检查点路径")
        return

    model.to(config.device)
    model.eval()

    # 测试数据可视化
    print("可视化测试数据...")
    dataloader.visualize_dataset(save_path=os.path.join(config.result_dir, "test_samples.png"))

    # 可视化扩散过程
    print("可视化扩散过程...")
    x, _ = next(iter(test_loader))
    x = x[:8].to(config.device)
    diffusion.visualize_diffusion_process(
        x,
        save_path=os.path.join(config.result_dir, "diffusion_process.png")
    )

    # 测试生成器
    print("测试生成器...")
    generator = ImageGenerator(model, diffusion, config)
    generator.generate_images(num_images=9)

    print("测试完成!")


def visualize_mode(config, args):
    """可视化模式"""
    print("=" * 60)
    print("开始可视化模式")
    print("=" * 60)

    # 设置随机种子
    set_seed(args.seed)

    # 可视化数据集
    dataloader = MNISTDataLoader(config)
    dataloader.visualize_dataset(save_path=os.path.join(config.result_dir, "mnist_dataset.png"))

    # 可视化扩散过程参数
    diffusion = DiffusionProcess(config)

    from utils import visualize_noise_schedule
    visualize_noise_schedule(
        diffusion,
        save_path=os.path.join(config.result_dir, "noise_schedule.png")
    )

    # 可视化模型结构
    if args.model == 'basic':
        model = BasicUNet()
    else:
        model = ConditionalUNet(config)

    print_model_summary(model)

    print("可视化完成!")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 创建配置
    config = Config()

    # 更新配置
    config.batch_size = args.batch_size
    config.image_size = args.image_size
    config.base_channels = args.base_channels
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.timesteps = args.timesteps

    # 打印配置
    print("配置参数:")
    print(config)
    print()

    # 保存配置
    save_config(config, config.checkpoint_dir)

    # 根据模式运行
    if args.mode == 'train':
        train_mode(config, args)
    elif args.mode == 'generate':
        generate_mode(config, args)
    elif args.mode == 'test':
        test_mode(config, args)
    elif args.mode == 'visualize':
        visualize_mode(config, args)
    else:
        print(f"未知模式: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()