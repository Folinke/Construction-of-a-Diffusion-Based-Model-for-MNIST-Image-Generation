"""
图片生成器
"""
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import imageio
from tqdm import tqdm


class ImageGenerator:
    def __init__(self, model, diffusion, config, device=None):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.device = device or config.device
        self.model.to(self.device)
        self.model.eval()

        # 创建结果目录
        os.makedirs(os.path.join(config.result_dir, "generated_images"), exist_ok=True)

    @torch.no_grad()
    def generate_images(self, num_images=16, labels=None, guidance_scale=3.0,
                        sampling_method="ddim", save=True, show=True):
        """
        生成图像

        参数:
        - num_images: 生成图像数量
        - labels: 标签列表，如果为None则随机生成
        - guidance_scale: 分类器自由引导尺度
        - sampling_method: 采样方法 ("ddim" 或 "standard")
        - save: 是否保存图像
        - show: 是否显示图像
        """
        # 准备标签
        if labels is None:
            labels = torch.randint(0, self.config.num_classes, (num_images,))
        elif isinstance(labels, list):
            labels = torch.tensor(labels)

        labels = labels.to(self.device)

        print(f"生成 {num_images} 张图像")
        print(f"标签: {labels.cpu().numpy()}")
        print(f"引导尺度: {guidance_scale}")
        print(f"采样方法: {sampling_method}")

        # 生成图像
        shape = (num_images, self.config.num_channels, self.config.image_size, self.config.image_size)

        if sampling_method == "ddim":
            samples, intermediate_samples = self.diffusion.ddim_sample(
                self.model,
                shape,
                sampling_timesteps=self.config.sampling_timesteps,
                labels=labels,
                guidance_scale=guidance_scale,
                eta=0.0
            )
        else:
            samples, intermediate_samples = self.diffusion.p_sample_loop(
                self.model,
                shape,
                labels=labels,
                guidance_scale=guidance_scale
            )

        # 转换为0-1范围
        samples = (samples.clamp(-1, 1) + 1) / 2

        # 创建网格
        grid = torchvision.utils.make_grid(samples.cpu(), nrow=int(np.sqrt(num_images)), pad_value=1.0)

        # 显示图像
        if show:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(f"生成的MNIST数字 (标签: {labels.cpu().numpy()})")
            ax.axis('off')
            plt.show()

        # 保存图像
        if save:
            # 保存网格图像
            timestamp = torch.randint(0, 10000, (1,)).item()
            grid_path = os.path.join(
                self.config.result_dir,
                "generated_images",
                f"generated_grid_{timestamp}.png"
            )

            torchvision.utils.save_image(
                grid,
                grid_path,
                normalize=True,
                value_range=(0, 1)
            )

            print(f"网格图像已保存到: {grid_path}")

            # 保存单个图像
            for i in range(num_images):
                img = samples[i].cpu()
                label = labels[i].item()

                img_path = os.path.join(
                    self.config.result_dir,
                    "generated_images",
                    f"digit_{label}_{timestamp}_{i}.png"
                )

                torchvision.utils.save_image(
                    img,
                    img_path,
                    normalize=True,
                    value_range=(0, 1)
                )

            print(f"单个图像已保存到: {self.config.result_dir}/generated_images/")

        return samples, labels

    @torch.no_grad()
    def generate_digit_grid(self, digit=None, num_rows=4, num_cols=4, guidance_scale=3.0):
        """
        生成特定数字的网格

        参数:
        - digit: 要生成的数字 (0-9)，如果为None则生成所有数字
        - num_rows: 行数
        - num_cols: 列数
        - guidance_scale: 引导尺度
        """
        if digit is not None:
            # 生成特定数字
            num_images = num_rows * num_cols
            labels = torch.full((num_images,), digit, dtype=torch.long)
        else:
            # 生成所有数字
            num_images = self.config.num_classes
            labels = torch.arange(0, self.config.num_classes)
            num_rows = 2
            num_cols = 5

        images, _ = self.generate_images(
            num_images=num_images,
            labels=labels,
            guidance_scale=guidance_scale,
            save=False,
            show=False
        )

        # 创建网格
        grid = torchvision.utils.make_grid(
            images.cpu(),
            nrow=num_cols,
            pad_value=1.0
        )

        # 显示
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(grid.permute(1, 2, 0))

        if digit is not None:
            title = f"生成的数字 {digit} ({num_rows}x{num_cols})"
        else:
            title = f"生成的数字 0-9"

        ax.set_title(title)
        ax.axis('off')

        # 保存
        timestamp = torch.randint(0, 10000, (1,)).item()
        save_path = os.path.join(
            self.config.result_dir,
            "generated_images",
            f"digit_{digit if digit is not None else 'all'}_{timestamp}.png"
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"数字网格已保存到: {save_path}")

        return images

    @torch.no_grad()
    def create_interpolation_video(self, digit1=0, digit2=9, num_frames=60, fps=10):
        """
        创建数字插值视频

        参数:
        - digit1: 起始数字
        - digit2: 结束数字
        - num_frames: 帧数
        - fps: 帧率
        """
        print(f"创建数字 {digit1} 到 {digit2} 的插值视频...")

        # 生成潜在向量
        z1 = torch.randn(1, self.config.num_channels, self.config.image_size, self.config.image_size).to(self.device)
        z2 = torch.randn(1, self.config.num_channels, self.config.image_size, self.config.image_size).to(self.device)

        # 标签
        label1 = torch.tensor([digit1], device=self.device)
        label2 = torch.tensor([digit2], device=self.device)

        # 插值
        frames = []
        for alpha in tqdm(np.linspace(0, 1, num_frames)):
            # 潜在向量插值
            z = (1 - alpha) * z1 + alpha * z2

            # 标签插值 (独热编码)
            label_onehot = torch.zeros(1, self.config.num_classes).to(self.device)
            label_onehot[0, digit1] = 1 - alpha
            label_onehot[0, digit2] = alpha

            # 生成图像 (简化版，实际需要修改模型支持独热标签)
            # 这里我们简单生成两个图像然后混合
            with torch.no_grad():
                img1, _ = self.generate_images(
                    num_images=1,
                    labels=label1,
                    guidance_scale=3.0,
                    save=False,
                    show=False
                )

                img2, _ = self.generate_images(
                    num_images=1,
                    labels=label2,
                    guidance_scale=3.0,
                    save=False,
                    show=False
                )

                # 图像混合
                img = (1 - alpha) * img1 + alpha * img2
                img = (img.clamp(-1, 1) + 1) / 2

                # 转换为PIL图像
                img_pil = transforms.ToPILImage()(img.squeeze().cpu())
                frames.append(img_pil)

        # 保存为GIF
        gif_path = os.path.join(
            self.config.result_dir,
            "generated_images",
            f"interpolation_{digit1}_to_{digit2}.gif"
        )

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )

        print(f"插值视频已保存到: {gif_path}")

        return gif_path

    @torch.no_grad()
    def visualize_sampling_process(self, digit=5, num_steps=10):
        """
        可视化采样过程

        参数:
        - digit: 要生成的数字
        - num_steps: 显示的步骤数
        """
        print(f"可视化数字 {digit} 的采样过程...")

        # 生成图像并获取中间结果
        shape = (1, self.config.num_channels, self.config.image_size, self.config.image_size)
        labels = torch.tensor([digit], device=self.device)

        _, intermediate_samples = self.diffusion.ddim_sample(
            self.model,
            shape,
            sampling_timesteps=50,
            labels=labels,
            guidance_scale=3.0
        )

        # 选择要显示的步骤
        step_indices = np.linspace(0, len(intermediate_samples) - 1, num_steps, dtype=int)

        # 创建图像网格
        fig, axes = plt.subplots(2, num_steps // 2, figsize=(15, 6))
        axes = axes.flatten()

        for idx, step_idx in enumerate(step_indices):
            img = intermediate_samples[step_idx]
            img = (img.clamp(-1, 1) + 1) / 2

            # 显示图像
            ax = axes[idx]
            ax.imshow(img.squeeze().cpu(), cmap='gray')
            ax.set_title(f"Step {step_idx}")
            ax.axis('off')

        plt.suptitle(f"数字 {digit} 的采样过程 (从噪声到清晰)")
        plt.tight_layout()

        # 保存
        save_path = os.path.join(
            self.config.result_dir,
            "generated_images",
            f"sampling_process_digit_{digit}.png"
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"采样过程可视化已保存到: {save_path}")

        return intermediate_samples


# 测试生成器
if __name__ == "__main__":
    from config import Config
    from diffusion_process import DiffusionProcess
    from unet_model import BasicUNet

    config = Config()
    diffusion = DiffusionProcess(config)
    model = BasicUNet()

    # 加载预训练模型 (如果有的话)
    checkpoint_path = "./checkpoints/model_best.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载预训练模型: {checkpoint_path}")

    # 创建生成器
    generator = ImageGenerator(model, diffusion, config)

    # 测试生成图像
    print("测试生成图像...")
    generator.generate_images(num_images=9, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # 测试生成数字网格
    print("\n测试生成数字网格...")
    generator.generate_digit_grid(digit=5, num_rows=2, num_cols=4)