"""
升级 U-Net：GroupNorm + GeLU + 正弦位置嵌入 + 抗锯齿池化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    """正弦位置嵌入（与扩散时间步 t 对应）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    """(GroupNorm → GeLU → Conv) *2 + 残差 + 时间投影"""
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h += self.time_mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class Down(nn.Module):
    """抗锯齿平均池化下采样"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.res = ResBlock(in_ch, out_ch, time_emb_dim)

    def forward(self, x, t):
        x = self.pool(x)
        return self.res(x, t)


class Up(nn.Module):
    """最近邻上采样 + 拼接 + ResBlock"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.res = ResBlock(in_ch + out_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t)


class SimpleUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.base_channels
        time_dim = c * 4

        self.pos_emb = SinusoidalPosEmb(c)
        self.time_mlp = nn.Sequential(
            nn.Linear(c, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.head = nn.Conv2d(config.num_channels, c, 3, padding=1)

        # 下采样
        self.down1 = ResBlock(c, c, time_dim)
        self.down2 = Down(c, c * 2, time_dim)
        self.down3 = Down(c * 2, c * 4, time_dim)

        # 中间
        self.mid = ResBlock(c * 4, c * 4, time_dim)

        # 上采样
        self.up3 = Up(c * 4, c * 2, time_dim)
        self.up2 = Up(c * 2, c, time_dim)
        self.up1 = ResBlock(c, c, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, config.num_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # 时间嵌入
        t = self.pos_emb(t)
        t = self.time_mlp(t)

        # 主干
        x = self.head(x)
        d1 = self.down1(x, t)
        d2 = self.down2(d1, t)
        d3 = self.down3(d2, t)

        m = self.mid(d3, t)

        u3 = self.up3(m, d2, t)
        u2 = self.up2(u3, d1, t)
        u1 = self.up1(u2, t)

        return self.out(u1)