"""
扩散模型GUI可视化界面
"""
import os

# 避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import threading
import queue
import json

# 导入项目模块
from config import Config
from model import SimpleUNet
from diffusion import DiffusionProcess
from data_loader import MNISTLoader


class DiffusionGUI:
    def __init__(self, master):
        """初始化GUI"""
        self.master = master
        master.title("MNIST扩散模型可视化系统")
        master.geometry("1200x800")

        # 设置样式
        self.setup_styles()

        # 初始化变量
        self.config = Config()
        self.model = None
        self.diffusion = None
        self.loader = MNISTLoader(self.config)
        self.train_loader, self.test_loader = None, None
        self.current_image = None
        self.current_noisy_image = None
        self.current_reconstructed_image = None
        self.current_timestep = 500
        self.timestep_var = tk.IntVar(value=self.current_timestep)
        self.is_processing = False
        self.result_queue = queue.Queue()

        # 创建界面
        self.create_widgets()

        # 加载数据
        self.load_data()

        # 检查模型
        self.check_model()

        # 启动结果处理线程
        self.start_result_thread()

        # 绑定关闭事件
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()

        # 设置主题
        style.theme_use('clam')

        # 自定义颜色
        self.bg_color = "#f0f0f0"
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#6b8cbc"
        self.accent_color = "#ff6b6b"
        self.text_color = "#333333"

        # 配置样式
        style.configure("Title.TLabel",
                        font=("Helvetica", 16, "bold"),
                        foreground=self.primary_color)
        style.configure("Subtitle.TLabel",
                        font=("Helvetica", 12, "bold"),
                        foreground=self.secondary_color)
        style.configure("Info.TLabel",
                        font=("Helvetica", 10),
                        foreground=self.text_color)
        style.configure("Primary.TButton",
                        font=("Helvetica", 10, "bold"),
                        background=self.primary_color,
                        foreground="white")
        style.configure("Accent.TButton",
                        font=("Helvetica", 10, "bold"),
                        background=self.accent_color,
                        foreground="white")
        style.configure("Status.TLabel",
                        font=("Helvetica", 9),
                        foreground="#666666")

        # 配置按钮样式
        style.map("Primary.TButton",
                  background=[('active', self.secondary_color)])
        style.map("Accent.TButton",
                  background=[('active', "#ff8585")])

    def create_widgets(self):
        """创建界面控件"""
        # 主容器
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame,
                                text="MNIST扩散模型可视化系统",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # 模型控制
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="模型状态:", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.model_status_label = ttk.Label(model_frame, text="未加载", style="Info.TLabel")
        self.model_status_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Button(model_frame, text="加载模型",
                   command=self.load_model,
                   style="Primary.TButton").pack(fill=tk.X, pady=(0, 5))

        ttk.Button(model_frame, text="重新训练模型",
                   command=self.retrain_model,
                   style="Primary.TButton").pack(fill=tk.X)

        # 图像控制
        image_frame = ttk.Frame(control_frame)
        image_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(image_frame, text="图像控制:", style="Subtitle.TLabel").pack(anchor=tk.W)

        ttk.Button(image_frame, text="随机选择图像",
                   command=self.random_image,
                   style="Primary.TButton").pack(fill=tk.X, pady=(0, 5))

        ttk.Button(image_frame, text="从测试集选择",
                   command=self.select_from_testset,
                   style="Primary.TButton").pack(fill=tk.X)

        # 扩散控制
        diffusion_frame = ttk.Frame(control_frame)
        diffusion_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(diffusion_frame, text="扩散参数:", style="Subtitle.TLabel").pack(anchor=tk.W)

        # 时间步控制
        timestep_frame = ttk.Frame(diffusion_frame)
        timestep_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(timestep_frame, text="时间步:", style="Info.TLabel").pack(side=tk.LEFT)
        self.timestep_label = ttk.Label(timestep_frame, text=f"{self.current_timestep}/{self.config.timesteps}",
                                        style="Info.TLabel")
        self.timestep_label.pack(side=tk.RIGHT)

        self.timestep_scale = ttk.Scale(timestep_frame,
                                        from_=0,
                                        to=self.config.timesteps - 1,
                                        variable=self.timestep_var,
                                        command=self.on_timestep_change)
        self.timestep_scale.pack(fill=tk.X, pady=(5, 0))

        # 操作按钮
        operation_frame = ttk.Frame(control_frame)
        operation_frame.pack(fill=tk.X)

        ttk.Button(operation_frame, text="添加噪声",
                   command=self.add_noise,
                   style="Accent.TButton").pack(fill=tk.X, pady=(0, 5))

        ttk.Button(operation_frame, text="复原图像",
                   command=self.reconstruct,
                   style="Accent.TButton").pack(fill=tk.X, pady=(0, 5))

        ttk.Button(operation_frame, text="完整扩散过程",
                   command=self.full_diffusion_process,
                   style="Accent.TButton").pack(fill=tk.X, pady=(0, 5))

        ttk.Button(operation_frame, text="生成新数字",
                   command=self.generate_new_digit,
                   style="Accent.TButton").pack(fill=tk.X)

        # 信息面板
        info_frame = ttk.LabelFrame(control_frame, text="系统信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # 图像显示区域
        image_display_frame = ttk.LabelFrame(main_frame, text="图像对比", padding="10")
        image_display_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 使用网格布局三张图片
        ttk.Label(image_display_frame, text="原始图像", style="Subtitle.TLabel").grid(row=0, column=0, padx=20)
        ttk.Label(image_display_frame, text="加噪后图像", style="Subtitle.TLabel").grid(row=0, column=1, padx=20)
        ttk.Label(image_display_frame, text="复原后图像", style="Subtitle.TLabel").grid(row=0, column=2, padx=20)

        # 创建三个画布用于显示图像
        self.original_canvas = tk.Canvas(image_display_frame, width=200, height=200, bg="white")
        self.original_canvas.grid(row=1, column=0, padx=20, pady=10)

        self.noisy_canvas = tk.Canvas(image_display_frame, width=200, height=200, bg="white")
        self.noisy_canvas.grid(row=1, column=1, padx=20, pady=10)

        self.reconstructed_canvas = tk.Canvas(image_display_frame, width=200, height=200, bg="white")
        self.reconstructed_canvas.grid(row=1, column=2, padx=20, pady=10)

        # 图像信息标签
        self.original_info = ttk.Label(image_display_frame, text="未选择图像", style="Info.TLabel")
        self.original_info.grid(row=2, column=0, padx=20)

        self.noisy_info = ttk.Label(image_display_frame, text="未添加噪声", style="Info.TLabel")
        self.noisy_info.grid(row=2, column=1, padx=20)

        self.reconstructed_info = ttk.Label(image_display_frame, text="未复原", style="Info.TLabel")
        self.reconstructed_info.grid(row=2, column=2, padx=20)

        # 状态栏
        self.status_label = ttk.Label(main_frame, text="就绪", style="Status.TLabel")
        self.status_label.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # 配置网格权重
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        image_display_frame.columnconfigure(0, weight=1)
        image_display_frame.columnconfigure(1, weight=1)
        image_display_frame.columnconfigure(2, weight=1)

    def load_data(self):
        """加载数据"""
        try:
            self.train_loader, self.test_loader = self.loader.get_loaders()
            self.log_info("数据加载成功")
        except Exception as e:
            self.log_error(f"数据加载失败: {str(e)}")

    def check_model(self):
        """检查模型"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "model_final.pth")
        if os.path.exists(checkpoint_path):
            self.log_info(f"找到模型检查点: {checkpoint_path}")
            self.load_model()
        else:
            self.log_warning("未找到训练好的模型，请先训练模型")

    def load_model(self):
        """加载模型"""
        if self.is_processing:
            messagebox.showwarning("警告", "当前有任务正在运行，请稍后再试")
            return

        self.is_processing = True
        self.set_status("正在加载模型...")

        def task():
            try:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, "model_final.pth")

                if not os.path.exists(checkpoint_path):
                    # 尝试加载最新的检查点
                    checkpoints = [f for f in os.listdir(self.config.checkpoint_dir)
                                   if f.endswith('.pth') and 'epoch' in f]
                    if checkpoints:
                        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        checkpoint_path = os.path.join(self.config.checkpoint_dir, latest)

                self.model = SimpleUNet(self.config)
                self.model.to(self.config.device)

                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path,
                                            map_location=self.config.device,
                                            weights_only=False)  # 使用False以兼容旧版本

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif isinstance(checkpoint, dict):
                        # 尝试直接加载状态字典
                        model_state_dict = {}
                        for k, v in checkpoint.items():
                            if 'module.' in k:
                                model_state_dict[k.replace('module.', '')] = v
                            else:
                                model_state_dict[k] = v
                        self.model.load_state_dict(model_state_dict, strict=False)
                    else:
                        self.model.load_state_dict(checkpoint)

                    self.model.eval()

                    self.result_queue.put(("model_loaded", {
                        "message": f"模型加载成功: {os.path.basename(checkpoint_path)}",
                        "checkpoint": checkpoint_path
                    }))
                else:
                    self.result_queue.put(("model_error", {
                        "message": "未找到模型检查点，请先训练模型"
                    }))

            except Exception as e:
                self.result_queue.put(("model_error", {
                    "message": f"模型加载失败: {str(e)}"
                }))
            finally:
                self.result_queue.put(("processing_done", None))

        threading.Thread(target=task, daemon=True).start()

    def random_image(self):
        """随机选择一张图像"""
        if not self.test_loader:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            # 从测试集随机选择一个batch
            data_iter = iter(self.test_loader)
            images, labels = next(data_iter)

            # 随机选择一张图像
            idx = np.random.randint(0, len(images))
            self.current_image = images[idx].unsqueeze(0).to(self.config.device)
            self.current_label = labels[idx].item()

            # 显示原始图像
            self.display_image(self.current_image.cpu(), self.original_canvas)
            self.original_info.config(text=f"标签: {self.current_label}")

            # 重置其他图像
            self.clear_canvas(self.noisy_canvas)
            self.clear_canvas(self.reconstructed_canvas)
            self.noisy_info.config(text="未添加噪声")
            self.reconstructed_info.config(text="未复原")

            self.log_info(f"选择图像 - 标签: {self.current_label}")
            self.set_status("图像选择完成")

        except Exception as e:
            self.log_error(f"选择图像失败: {str(e)}")

    def select_from_testset(self):
        """从测试集选择特定图像"""
        if not self.test_loader:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 创建选择对话框
        dialog = tk.Toplevel(self.master)
        dialog.title("选择测试图像")
        dialog.geometry("300x400")

        # 显示所有测试集图像的小图
        test_iter = iter(self.test_loader)
        images, labels = next(test_iter)

        # 创建画布用于显示
        canvas = tk.Canvas(dialog, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)

        # 显示图像
        photo_images = []
        for i, (img, label) in enumerate(zip(images[:20], labels[:20])):  # 显示前20个
            # 转换为PIL图像
            img_pil = transforms.ToPILImage()(img)
            img_pil = img_pil.resize((40, 40), Image.Resampling.LANCZOS)

            # 转换为Tkinter图像
            photo = ImageTk.PhotoImage(img_pil)
            photo_images.append(photo)

            # 计算位置
            row = i // 5
            col = i % 5

            # 创建按钮
            btn = tk.Button(dialog, image=photo,
                            command=lambda idx=i, lab=label.item():
                            self.select_specific_image(idx, lab, dialog),
                            relief=tk.FLAT)
            btn.place(x=col * 50 + 10, y=row * 50 + 10, width=40, height=40)

            # 添加标签
            label_text = tk.Label(dialog, text=str(label.item()))
            label_text.place(x=col * 50 + 25, y=row * 50 + 60, anchor=tk.CENTER)

        # 保存引用
        dialog.photo_images = photo_images

    def select_specific_image(self, idx, label, dialog):
        """选择特定图像"""
        test_iter = iter(self.test_loader)
        images, labels = next(test_iter)

        self.current_image = images[idx].unsqueeze(0).to(self.config.device)
        self.current_label = label

        # 显示原始图像
        self.display_image(self.current_image.cpu(), self.original_canvas)
        self.original_info.config(text=f"标签: {self.current_label}")

        # 重置其他图像
        self.clear_canvas(self.noisy_canvas)
        self.clear_canvas(self.reconstructed_canvas)
        self.noisy_info.config(text="未添加噪声")
        self.reconstructed_info.config(text="未复原")

        self.log_info(f"选择图像 - 标签: {self.current_label}")
        self.set_status("图像选择完成")

        dialog.destroy()

    def on_timestep_change(self, value):
        """时间步变化回调"""
        self.current_timestep = int(float(value))
        self.timestep_label.config(text=f"{self.current_timestep}/{self.config.timesteps}")

    def add_noise(self):
        """添加噪声"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择一张图像")
            return

        if self.diffusion is None:
            self.diffusion = DiffusionProcess(self.config)

        self.is_processing = True
        self.set_status("正在添加噪声...")

        def task():
            try:
                # 创建时间步张量
                t = torch.tensor([self.current_timestep], device=self.config.device)

                # 添加噪声
                noisy_image, noise = self.diffusion.add_noise(self.current_image, t)
                self.current_noisy_image = noisy_image

                # 计算噪声强度
                noise_intensity = torch.mean(torch.abs(noise)).item()

                self.result_queue.put(("noise_added", {
                    "noisy_image": noisy_image.cpu(),
                    "timestep": self.current_timestep,
                    "noise_intensity": noise_intensity
                }))

            except Exception as e:
                self.result_queue.put(("error", {
                    "message": f"添加噪声失败: {str(e)}"
                }))
            finally:
                self.result_queue.put(("processing_done", None))

        threading.Thread(target=task, daemon=True).start()

    def reconstruct(self):
        """复原图像"""
        if self.current_noisy_image is None:
            messagebox.showwarning("警告", "请先添加噪声")
            return

        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return

        self.is_processing = True
        self.set_status("正在复原图像...")

        def task():
            try:
                # 创建时间步张量
                t = torch.tensor([self.current_timestep], device=self.config.device)

                # 使用模型预测噪声
                with torch.no_grad():
                    predicted_noise = self.model(self.current_noisy_image, t)

                # 计算去噪后的图像
                sqrt_alphas_cumprod_t = self.diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

                reconstructed = (
                                            self.current_noisy_image - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
                reconstructed = torch.clamp(reconstructed, -1.0, 1.0)

                self.current_reconstructed_image = reconstructed

                # 计算复原质量
                if self.current_image is not None:
                    mse = torch.mean((reconstructed - self.current_image) ** 2).item()
                    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                else:
                    mse = psnr = None

                self.result_queue.put(("reconstructed", {
                    "reconstructed_image": reconstructed.cpu(),
                    "mse": mse,
                    "psnr": psnr
                }))

            except Exception as e:
                self.result_queue.put(("error", {
                    "message": f"复原图像失败: {str(e)}"
                }))
            finally:
                self.result_queue.put(("processing_done", None))

        threading.Thread(target=task, daemon=True).start()

    def full_diffusion_process(self):
        """完整扩散过程可视化"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择一张图像")
            return

        self.is_processing = True
        self.set_status("正在生成完整扩散过程...")

        def task():
            try:
                if self.diffusion is None:
                    self.diffusion = DiffusionProcess(self.config)

                # 选择几个关键时间步
                timesteps = torch.linspace(0, self.config.timesteps - 1, 10, dtype=torch.long)

                images = []
                noise_levels = []

                for ts in timesteps:
                    t_batch = torch.full((self.current_image.shape[0],), ts.item(),
                                         device=self.config.device)
                    noisy_img, noise = self.diffusion.add_noise(self.current_image, t_batch)

                    # 转换为PIL图像
                    img_np = noisy_img[0].cpu().numpy().squeeze()
                    img_np = (img_np + 1) / 2  # 归一化到0-1
                    img_np = (img_np * 255).astype(np.uint8)

                    images.append(Image.fromarray(img_np, mode='L'))
                    noise_levels.append(torch.mean(torch.abs(noise)).item())

                self.result_queue.put(("full_diffusion", {
                    "images": images,
                    "timesteps": timesteps.tolist(),
                    "noise_levels": noise_levels
                }))

            except Exception as e:
                self.result_queue.put(("error", {
                    "message": f"生成扩散过程失败: {str(e)}"
                }))
            finally:
                self.result_queue.put(("processing_done", None))

        threading.Thread(target=task, daemon=True).start()

    def generate_new_digit(self):
        """生成新数字"""
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return

        self.is_processing = True
        self.set_status("正在生成新数字...")

        def task():
            try:
                if self.diffusion is None:
                    self.diffusion = DiffusionProcess(self.config)

                # 生成形状
                shape = (1, self.config.num_channels, self.config.image_size, self.config.image_size)

                # 完整采样
                with torch.no_grad():
                    generated = self.diffusion.sample(self.model, shape, self.config.device)

                # 归一化到0-1
                generated = (generated.clamp(-1, 1) + 1) / 2

                self.result_queue.put(("generated", {
                    "generated_image": generated.cpu(),
                    "message": "新数字生成完成"
                }))

            except Exception as e:
                self.result_queue.put(("error", {
                    "message": f"生成数字失败: {str(e)}"
                }))
            finally:
                self.result_queue.put(("processing_done", None))

        threading.Thread(target=task, daemon=True).start()

    def retrain_model(self):
        """重新训练模型"""
        response = messagebox.askyesno("确认",
                                       "重新训练模型可能需要较长时间，是否继续？")
        if response:
            self.log_info("开始重新训练模型...")
            self.set_status("正在重新训练模型...")

            # 这里可以调用训练脚本，为了简化，我们只显示信息
            messagebox.showinfo("提示",
                                "模型重新训练功能需要在命令行中运行 train.py\n"
                                "请打开终端并执行: python train.py")

            self.set_status("就绪")

    def display_image(self, image_tensor, canvas):
        """在画布上显示图像"""
        try:
            # 转换为numpy数组
            if image_tensor.dim() == 4:
                img = image_tensor[0].numpy().squeeze()
            else:
                img = image_tensor.numpy().squeeze()

            # 归一化到0-255
            img = (img + 1) / 2  # 从[-1,1]到[0,1]
            img = (img * 255).astype(np.uint8)

            # 转换为PIL图像
            pil_img = Image.fromarray(img, mode='L')

            # 调整大小以适应画布
            canvas_width = canvas.winfo_width() or 200
            canvas_height = canvas.winfo_height() or 200
            pil_img = pil_img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            # 转换为Tkinter图像
            tk_img = ImageTk.PhotoImage(pil_img)

            # 清除画布并显示图像
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_img)

            # 保存引用
            canvas.image = tk_img

            return True

        except Exception as e:
            self.log_error(f"显示图像失败: {str(e)}")
            return False

    def clear_canvas(self, canvas):
        """清除画布"""
        canvas.delete("all")
        if hasattr(canvas, 'image'):
            canvas.image = None

    def start_result_thread(self):
        """启动结果处理线程"""

        def process_results():
            while True:
                try:
                    result_type, data = self.result_queue.get(timeout=0.1)

                    if result_type == "model_loaded":
                        self.model_status_label.config(text="已加载", foreground="green")
                        self.log_info(data["message"])
                        self.set_status("模型加载完成")

                    elif result_type == "model_error":
                        self.model_status_label.config(text="加载失败", foreground="red")
                        self.log_error(data["message"])
                        self.set_status("模型加载失败")

                    elif result_type == "noise_added":
                        self.display_image(data["noisy_image"], self.noisy_canvas)
                        self.noisy_info.config(
                            text=f"时间步: {data['timestep']}\n"
                                 f"噪声强度: {data['noise_intensity']:.4f}"
                        )
                        self.log_info(f"添加噪声完成 - 时间步: {data['timestep']}")
                        self.set_status("噪声添加完成")

                    elif result_type == "reconstructed":
                        self.display_image(data["reconstructed_image"], self.reconstructed_canvas)

                        info_text = "复原完成"
                        if data["mse"] is not None:
                            info_text += f"\nMSE: {data['mse']:.6f}"
                            info_text += f"\nPSNR: {data['psnr']:.2f} dB"

                        self.reconstructed_info.config(text=info_text)
                        self.log_info(f"图像复原完成 - MSE: {data['mse']:.6f}")
                        self.set_status("图像复原完成")

                    elif result_type == "full_diffusion":
                        self.show_full_diffusion_dialog(data)

                    elif result_type == "generated":
                        # 显示生成的图像
                        self.display_image(data["generated_image"], self.original_canvas)
                        self.original_info.config(text="生成的新数字")
                        self.log_info(data["message"])
                        self.set_status("新数字生成完成")

                        # 清除其他图像
                        self.clear_canvas(self.noisy_canvas)
                        self.clear_canvas(self.reconstructed_canvas)
                        self.noisy_info.config(text="未添加噪声")
                        self.reconstructed_info.config(text="未复原")

                    elif result_type == "error":
                        messagebox.showerror("错误", data["message"])
                        self.log_error(data["message"])
                        self.set_status("操作失败")

                    elif result_type == "processing_done":
                        self.is_processing = False

                except queue.Empty:
                    continue
                except Exception as e:
                    self.log_error(f"结果处理错误: {str(e)}")

        threading.Thread(target=process_results, daemon=True).start()

    def show_full_diffusion_dialog(self, data):
        """显示完整扩散过程对话框"""
        dialog = tk.Toplevel(self.master)
        dialog.title("完整扩散过程")
        dialog.geometry("800x600")

        # 创建画布
        canvas = tk.Canvas(dialog, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)

        # 创建滚动条
        scrollbar = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        # 内部框架
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

        # 显示图像
        photo_images = []
        for i, (img, ts, noise_level) in enumerate(zip(data["images"],
                                                       data["timesteps"],
                                                       data["noise_levels"])):
            # 调整图像大小
            img = img.resize((100, 100), Image.Resampling.LANCZOS)

            # 转换为Tkinter图像
            photo = ImageTk.PhotoImage(img)
            photo_images.append(photo)

            # 创建标签框架
            frame = ttk.Frame(inner_frame, relief=tk.RAISED, borderwidth=1)
            frame.grid(row=i // 5, column=i % 5, padx=5, pady=5)

            # 显示图像
            label = ttk.Label(frame, image=photo)
            label.pack(padx=5, pady=5)

            # 显示信息
            info = ttk.Label(frame,
                             text=f"t={ts}\n噪声={noise_level:.4f}",
                             font=("Helvetica", 8))
            info.pack(padx=5, pady=(0, 5))

        # 更新滚动区域
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # 保存引用
        dialog.photo_images = photo_images

        self.log_info("完整扩散过程已显示")
        self.set_status("就绪")

    def set_status(self, message):
        """设置状态栏消息"""
        self.status_label.config(text=f"状态: {message}")
        self.master.update_idletasks()

    def log_info(self, message):
        """记录信息日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] INFO: {message}\n")
        self.info_text.see(tk.END)
        self.info_text.update_idletasks()

    def log_warning(self, message):
        """记录警告日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] WARNING: {message}\n", "warning")
        self.info_text.see(tk.END)
        self.info_text.update_idletasks()

    def log_error(self, message):
        """记录错误日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] ERROR: {message}\n", "error")
        self.info_text.see(tk.END)
        self.info_text.update_idletasks()

    def on_closing(self):
        """关闭窗口时的处理"""
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.master.destroy()


def main():
    """主函数"""
    # 创建主窗口
    root = tk.Tk()

    # 配置文本标签样式
    root.option_add('*TCombobox*Listbox.font', ('Helvetica', 10))

    # 创建GUI
    app = DiffusionGUI(root)

    # 配置日志标签样式
    app.info_text.tag_config("warning", foreground="orange")
    app.info_text.tag_config("error", foreground="red")

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()