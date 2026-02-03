import torch
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power=1):
    """
    余弦噪声调度，DDIM
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)
    betas = torch.from_numpy(betas).float()
    return betas[1:timesteps+1]      # 去掉首尾辅助值