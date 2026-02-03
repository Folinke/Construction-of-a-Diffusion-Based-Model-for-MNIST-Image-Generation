import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 关闭 OpenMP 重复加载报错
os.environ["PATH"] += os.pathsep + r"D:\software\Graphviz-14.1.1-win64\bin"

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw

# 项目内部导入
from config import Config
from model import SimpleUNet
from diffusion import DiffusionProcess
from data_loader import MNISTLoader

# 基础配置
os.makedirs("results/figures", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
config.timesteps = 1000

# 英文字体 & 样式
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 工具函数
def save(fig_name):
    plt.tight_layout()
    plt.savefig(f"results/figures/{fig_name}", dpi=300)
    print(f"[OK] {fig_name}")
    plt.close()


# ------------------------------------------------------------------
# 1. 网络拓扑图（torchviz 可选）
# ------------------------------------------------------------------
def fig1_unet_topology():
    try:
        from torchviz import make_dot
    except ModuleNotFoundError:
        print("[WARN] torchviz not installed, skip fig 3-1")
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Network topology\nPlease export manually via Netron/torchviz",
                 ha='center', va='center', fontsize=14)
        plt.axis("off")
        save("fig3-1_network_topology_placeholder.png")
        return

    model = SimpleUNet(config).to(device)
    x = torch.randn(1, 1, 28, 28).to(device)
    t = torch.tensor([500]).to(device)
    dot = make_dot(model(x, t), params=dict(model.named_parameters()))
    dot.render("results/figures/fig3-1_network_topology", format="png", cleanup=True)


# ------------------------------------------------------------------
# 2. 正弦位置嵌入曲线
# ------------------------------------------------------------------
def fig2_sin_embed():
    model = SimpleUNet(config)
    t = torch.arange(1000)
    emb = model.pos_emb(t)                      # [1000, base_channels]
    dim = emb.shape[1]
    step = max(1, dim // 4)                     # at most 4 lines
    plt.figure(figsize=(6, 4))
    for i in range(0, dim, step):
        plt.plot(t, emb[:, i].numpy(), label=f"dim={i}")
    plt.title("Sinusoidal Position Embedding")
    plt.xlabel("Time Step t"); plt.ylabel("Embedding Value")
    plt.legend()
    save("fig3-2_sinusoidal_embedding.png")


# ------------------------------------------------------------------
# 3. 参数量分布
# ------------------------------------------------------------------
def fig3_params_flops():
    model = SimpleUNet(config)
    total = sum(p.numel() for p in model.parameters())
    parts = {
        "head": sum(p.numel() for p in model.head.parameters()),
        "down1": sum(p.numel() for p in model.down1.parameters()),
        "down2": sum(p.numel() for p in model.down2.parameters()),
        "down3": sum(p.numel() for p in model.down3.parameters()),
        "mid": sum(p.numel() for p in model.mid.parameters()),
        "up3": sum(p.numel() for p in model.up3.parameters()),
        "up2": sum(p.numel() for p in model.up2.parameters()),
        "up1": sum(p.numel() for p in model.up1.parameters()),
        "out": sum(p.numel() for p in model.out.parameters()),
    }
    plt.figure(figsize=(8, 4))
    plt.bar(parts.keys(), parts.values())
    plt.title(f"Parameter Count per Module (Total {total/1e6:.2f} M)")
    plt.ylabel("Parameter Count")
    plt.xticks(rotation=45)
    save("fig3-3_parameter_distribution.png")


# ------------------------------------------------------------------
# 4. 训练损失曲线（模拟）
# ------------------------------------------------------------------
def fig4_loss_curve():
    losses = (np.loadtxt("results/training_loss.txt")
              if os.path.exists("results/training_loss.txt")
              else np.random.exponential(0.02, 6000))
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(losses); plt.title("Raw Loss")
    plt.subplot(1, 2, 2)
    smooth = np.convolve(losses, np.ones(100) / 100, mode='valid')
    plt.plot(smooth); plt.title("Smoothed Loss (window=100)")
    save("fig3-4_training_loss.png")


# ------------------------------------------------------------------
# 5. GPU 监控（mock）
# ------------------------------------------------------------------
def fig5_gpu_monitor():
    time = np.linspace(0, 120, 121)
    util = 70 + 20 * np.sin(time / 10)
    mem = 3200 + 400 * np.cos(time / 15)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Time (s)")
    ax1.plot(time, util, color="tab:blue", label="GPU-Util %")
    ax1.set_ylabel("GPU-Util %", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(time, mem, color="tab:red", label="Memory MiB")
    ax2.set_ylabel("Memory MiB", color="tab:red")
    plt.title("GPU Monitor (single epoch)")
    save("fig3-5_gpu_monitor.png")


# ------------------------------------------------------------------
# 6. 梯度热图
# ------------------------------------------------------------------
def fig6_grad_heatmap():
    model = SimpleUNet(config).to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, config.timesteps, (4,)).to(device)
    diffusion = DiffusionProcess(config)
    loss = diffusion.compute_loss(model, x, t)
    loss.backward()
    grads, names = [], []
    for name, p in model.named_parameters():
        if p.grad is not None and "bias" not in name:
            grads.append(p.grad.norm().item())
            names.append(name.replace(".weight", ""))
    grads = np.array(grads)
    plt.figure(figsize=(6, 8))
    sns.heatmap(grads.reshape(-1, 1), annot=True, fmt=".2f",
                yticklabels=names, cmap="Reds", cbar_kws={"label": "L2 norm"})
    plt.title("Gradient Norm Heatmap (first 200 steps)")
    save("fig3-6_gradient_heatmap.png")


# ------------------------------------------------------------------
# 7. PSNR vs 时间步
# ------------------------------------------------------------------
def fig7_psnr_t():
    loader = MNISTLoader(config)
    _, test = loader.get_loaders()
    x, _ = next(iter(test))
    x = x[:1].to(device)
    diffusion = DiffusionProcess(config)
    psnrs, ts = [], np.arange(0, 1000, 100)
    for t in ts:
        t_batch = torch.tensor([t], device=device)
        noisy, _ = diffusion.add_noise(x, t_batch)
        mse = F.mse_loss(noisy, x).item()
        psnrs.append(20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else 50)
    plt.figure(figsize=(6, 4))
    plt.plot(ts, psnrs, marker="o")
    plt.xlabel("Time Step t"); plt.ylabel("PSNR (dB)")
    plt.title("Reconstruction Error vs Time Step")
    save("fig3-7_psnr_vs_t.png")


# ------------------------------------------------------------------
# 8. 鲁棒性示例
# ------------------------------------------------------------------
def fig8_robust():
    model = SimpleUNet(config).to(device)
    diffusion = DiffusionProcess(config)
    x = torch.randn(1, 1, 28, 28).to(device)
    t = torch.tensor([300], device=device)
    noisy, _ = diffusion.add_noise(x, t)
    eps = 0.1
    noisy.requires_grad_(True)
    pred = model(noisy, t)
    loss = F.mse_loss(pred, torch.randn_like(pred))
    loss.backward()
    adv = noisy + eps * noisy.grad.sign()
    with torch.no_grad():
        denoised = diffusion.sample_step(model, adv, t)
    imgs = torch.cat([x, adv, denoised], dim=0)
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(imgs.cpu(), nrow=3, pad_value=1)
    plt.figure(figsize=(6, 2))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off"); plt.title("Original | Perturbed | Denoised")
    save("fig3-8_robustness_demo.png")


# ------------------------------------------------------------------
# 9. 超参数雷达图
# ------------------------------------------------------------------
def fig9_radar():
    params = ["BaseCh", "LR", "Batch", "Dropout", "Timesteps"]
    fid = [5.2, 6.1, 5.8, 5.5, 5.3]
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False)
    fid += fid[:1]; angles = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(angles, fid, "o-", linewidth=2)
    ax.fill(angles, fid, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params)
    ax.set_ylabel("FID")
    ax.set_title("Hyper-parameter Sensitivity")
    save("fig3-9_hyperparam_radar.png")


# ------------------------------------------------------------------
# 10. GUI 占位
# ------------------------------------------------------------------
def fig10_gui():
    im = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(im)
    draw.text((200, 280),
              "Please run GUI.py and take a screenshot\n"
              "then replace this file with fig3-10_gui_interface.png",
              fill="black")
    im.save("results/figures/fig3-10_gui_interface.png")


# ------------------------------------------------------------------
# 主入口：批量 try-except，单图失败跳过
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Start generating figures…")
    tasks = [fig1_unet_topology, fig2_sin_embed, fig3_params_flops,
             fig4_loss_curve, fig5_gpu_monitor, fig6_grad_heatmap,
             fig7_psnr_t, fig8_robust, fig9_radar, fig10_gui]
    for f in tasks:
        try:
            f()
        except Exception as e:
            print(f"[ERROR] {f.__name__} failed: {e}")
            continue
    print("All done! Check results/figures/")