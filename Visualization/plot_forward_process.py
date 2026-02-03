import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch, torchvision, matplotlib.pyplot as plt
from config import Config
from diffusion import DiffusionProcess
from data_loader import MNISTLoader

cfg = Config()
diff = DiffusionProcess(cfg)
loader = MNISTLoader(cfg)
train, _ = loader.get_loaders()
x, _ = next(iter(train))
x = x[:1].to(cfg.device)

steps = torch.linspace(0, cfg.timesteps-1, 9, dtype=torch.long)
plt.figure(figsize=(5, 5), dpi=200)
for i, t in enumerate(steps):
    t_b = torch.tensor([t], device=cfg.device)
    x_noisy, _ = diff.add_noise(x, t_b)
    img = (x_noisy.squeeze().cpu().clamp(-1,1)+1)/2
    plt.subplot(3, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f't={t.item()}'); plt.axis('off')
plt.suptitle('Forward Diffusion Process')
plt.tight_layout()
plt.savefig('results/plots/forward_process.png')
plt.close()
print('saved → results/plots/forward_process.png')