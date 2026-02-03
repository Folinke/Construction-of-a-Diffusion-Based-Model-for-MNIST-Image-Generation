# Construction-of-a-Diffusion-Based-Model-for-MNIST-Image-Generation
本项目基于扩散模型，搭建了一套 MNIST 手写数字的生成与可视化系统。 它采用 U-Net 作为核心网络架构，借助余弦调度的前向扩散过程，逐步给图像添 加噪声。之后系统通过逆向去噪过程，从随机噪声里生成高质量的手写数字图像。 项目具备完整的训练、生成和可视化功能，支持 DDIM 算法快速采样、交 互式 GUI 操作，还包含多种生成模式。
