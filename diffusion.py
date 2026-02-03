"""
扩散过程（余弦调度版）
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from schedulers import cosine_beta_schedule   # 新增这一行


class DiffusionProcess:
    def __init__(self, config):
        self.config = config
        self.timesteps = config.timesteps

        #使用余弦调度
        self.betas = cosine_beta_schedule(config.timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 移动到设备
        device = config.device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def compute_loss(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy, true_noise = self.add_noise(x_start, t, noise)
        pred_noise = model(x_noisy, t)
        return F.mse_loss(pred_noise, true_noise)

    @torch.no_grad()
    def sample_step(self, model, x, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)
        model_output = model(x, t)
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * model_output) / \
                  self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        if t[0].item() > 0:
            noise = torch.randn_like(x)
            posterior_variance = self.betas[t].view(-1, 1, 1, 1)
            return model_mean + torch.sqrt(posterior_variance) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, model, shape, device):
        model.eval()
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.sample_step(model, x, t)
        model.train()
        return x

    def visualize_noise_process(self, x_start, num_steps=5, save_path=None):
        fig, axes = plt.subplots(2, num_steps, figsize=(15, 6))
        timesteps = torch.linspace(0, self.timesteps - 1, num_steps, dtype=torch.long)
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x_start.shape[0],), t.item(), device=x_start.device)
            x_noisy, noise = self.add_noise(x_start, t_batch)
            axes[0, i].imshow(x_noisy[0].cpu().squeeze(), cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Step {t.item()}')
            axes[0, i].axis('off')
            axes[1, i].imshow(noise[0].cpu().squeeze(), cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title('Noise')
            axes[1, i].axis('off')
        plt.suptitle('Forward Diffusion Process')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        plt.show()
    @torch.no_grad()
    def ddim_sample(self, model, shape, device, sampling_timesteps=50, eta=0.0):
        """
        DDIM 快速采样（确定性/随机性可控）
        sampling_timesteps: 实际跳步数，<< 1000 时加速明显
        eta: 0→完全确定，1→等价 DDPM
        """
        model.eval()
        # 均匀跳步
        step = self.timesteps // sampling_timesteps
        ts = torch.arange(self.timesteps - 1, -1, -step,
                          device=device).long()  # [T/step]
        # 预计算 alpha 序列
        alphas_cumprod_t = self.alphas_cumprod[ts]
        alphas_cumprod_t_prev = torch.cat(
            [self.alphas_cumprod[ts[1:]], self.alphas_cumprod[0:1]])

        # 初始纯噪声
        x = torch.randn(shape, device=device)
        for i, t in enumerate(ts):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            # 模型预测噪声
            pred_noise = model(x, t_batch)
            # 计算 x0 预测
            pred_x0 = (x - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) / \
                      self.sqrt_alphas_cumprod[t]
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            # DDIM 更新
            alpha_t = alphas_cumprod_t[i]
            alpha_t_prev = alphas_cumprod_t_prev[i]
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            # 方向项
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * pred_noise
            # 随机项
            if i == len(ts) - 1:
                noise = 0
            else:
                noise = torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
        model.train()
        return x