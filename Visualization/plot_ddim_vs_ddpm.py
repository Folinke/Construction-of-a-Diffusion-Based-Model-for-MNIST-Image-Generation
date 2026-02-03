import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch, time, matplotlib.pyplot as plt
from config import Config
from model import SimpleUNet
from diffusion import DiffusionProcess

cfg = Config()
diff = DiffusionProcess(cfg)
model = SimpleUNet(cfg).to(cfg.device)
model.eval()

shape = (16, 1, 28, 28)
steps = [10, 20, 50, 100, 200, 1000]
times, labels = [], []

with torch.no_grad():
    for n in steps:
        t0 = time.time()
        if n == 1000:
            _ = diff.sample(model, shape, cfg.device)      # DDPM
        else:
            _ = diff.ddim_sample(model, shape, cfg.device, sampling_timesteps=n, eta=0.)
        elap = time.time() - t0
        times.append(elap)
        labels.append(f'{n}' if n < 1000 else 'DDPM')

plt.figure(figsize=(4, 3), dpi=200)
plt.bar(labels, times, color='#74b9ff')
plt.ylabel('Sampling Time (s)'); plt.grid(axis='y', alpha=0.3)
plt.title('DDIM Step vs Speed (16 images)')
plt.tight_layout()
plt.savefig('results/plots/ddim_speed.png')
plt.close()
print('saved → results/plots/ddim_speed.png')