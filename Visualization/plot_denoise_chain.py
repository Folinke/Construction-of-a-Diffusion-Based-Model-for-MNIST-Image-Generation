# 必须在所有 import 之前解决 OpenMP 冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch, matplotlib.pyplot as plt
from pathlib import Path
import sys

# 定位项目根目录
repo_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_dir))

from config import Config
from model import SimpleUNet
from diffusion import DiffusionProcess

# 权重绝对路径
CKPT_PATH = repo_dir / 'checkpoints' / 'model_final.pth'

cfg = Config()
diff = DiffusionProcess(cfg)
model = SimpleUNet(cfg).to(cfg.device)

# 加载 checkpoint（兼容两种格式）
ckpt = torch.load(CKPT_PATH, map_location=cfg.device, weights_only=True)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

# 记录中间去噪步骤
steps = 50
intermediates = []
shape = (1, 1, 28, 28)
with torch.no_grad():
    x = torch.randn(shape, device=cfg.device)
    # 均匀跳步
    ts = torch.arange(steps - 1, -1, -1, device=cfg.device)
    for i, t in enumerate(ts):
        t_b = torch.tensor([t], device=cfg.device)
        x = diff.sample_step(model, x, t_b)
        if i % 5 == 0 or i == steps - 1:
            intermediates.append((x.squeeze().cpu().clamp(-1, 1) + 1) / 2)

# 绘图
rows = len(intermediates)
plt.figure(figsize=(2, rows * 0.6), dpi=200)
for idx, img in enumerate(intermediates):
    plt.subplot(rows, 1, idx + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle('Reverse Denoising Chain (DDPM)', y=0.92)
plt.tight_layout()

# 保存
OUT_DIR = repo_dir / 'results' / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / 'denoise_chain.png')
plt.close()
print(f'saved → {OUT_DIR / "denoise_chain.png"}')