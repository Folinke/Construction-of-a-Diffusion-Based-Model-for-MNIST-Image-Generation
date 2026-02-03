import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, matplotlib.pyplot as plt, torch
from schedulers import cosine_beta_schedule   # 你已有的文件

T = 1000
lin = torch.linspace(1e-4, 0.02, T)
cos = cosine_beta_schedule(T)

plt.figure(figsize=(4.5, 3), dpi=200)
plt.plot(lin, label='Linear', lw=1.5)
plt.plot(cos, label='Cosine', lw=1.5)
plt.xlabel('Timestep'); plt.ylabel('β'); plt.grid(alpha=0.3)
plt.legend(); plt.title('Noise Schedule Comparison')
plt.tight_layout()
plt.savefig('results/plots/noise_schedule.png')
plt.close()
print('saved → results/plots/noise_schedule.png')