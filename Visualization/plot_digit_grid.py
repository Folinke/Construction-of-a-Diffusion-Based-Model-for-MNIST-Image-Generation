# 头两行照旧：解决 OpenMP 冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch, matplotlib.pyplot as plt, torchvision
from pathlib import Path
import sys

# 把项目根目录加入 Python 路径
repo_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_dir))

from config import Config
from model import SimpleUNet
from diffusion import DiffusionProcess

# 绝对路径定位权重
CKPT_PATH = repo_dir / 'checkpoints' / 'model_final.pth'

cfg = Config()
diff = DiffusionProcess(cfg)
model = SimpleUNet(cfg).to(cfg.device)

# 加载 checkpoint —— 兼容两种格式
ckpt = torch.load(CKPT_PATH, map_location=cfg.device, weights_only=True)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    # 纯状态字典
    model.load_state_dict(ckpt)
model.eval()

# 出图逻辑与之前完全一致
plt.figure(figsize=(8, 5), dpi=200)
with torch.no_grad():
    for digit in range(10):
        label = torch.tensor([digit] * 8)
        shape = (8, 1, 28, 28)
        sam = diff.ddim_sample(model, shape, cfg.device, sampling_timesteps=50, eta=0.)
        sam = (sam.clamp(-1, 1) + 1) / 2
        grid = torchvision.utils.make_grid(sam.cpu(), nrow=8, pad_value=1)
        plt.subplot(2, 5, digit + 1)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f'Digit {digit}')
        plt.axis('off')

plt.suptitle('Generated Digit Grid (DDIM 50 steps)')
plt.tight_layout()

OUT_DIR = repo_dir / 'results' / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / 'digit_grid.png')
plt.close()
print(f'saved → {OUT_DIR / "digit_grid.png"}')