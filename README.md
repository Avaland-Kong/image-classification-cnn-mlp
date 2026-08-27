# MNIST‑MLP‑CNN 手写数字分类对比实验
> 项目：对比多层感知机(MLP)与卷积神经网络(SimpleCNN)在MNIST手写数字数据集上的分类性能。
> 
> 环境：Python 3.11 + PyTorch 2.5.1+cu121，RTX4060 Laptop GPU

## 项目简介
本项目基于PyTorch实现**MLP全连接网络**与**简易CNN卷积网络**，在MNIST手写数字数据集上完成训练与测试，对比两种网络架构的识别精度，理解卷积操作对图像任务的增益。

- MLP：纯全连接网络，将28×28图像展平为784维向量输入网络；
- CNN：利用卷积+池化提取图像局部空间特征，更适配图像任务。

## 项目目录结构
```
mnist‑mlp‑cnn/
├─ main.py                 # 入口文件：训练、评估、绘图
├─ dataset_config.py       # 数据集与DataLoader配置
├─ models/
│   ├─ __init__.py
│   ├─ mlp.py              # MLP网络定义
│   └─ cnn.py              # SimpleCNN网络定义
├─ compare_result.png      # 实验结果精度对比图
├─ .gitignore              # git忽略配置
└─ README.md
```

## 环境依赖
```txt
torch==2.5.1+cu121
torchvision
matplotlib
```

### 环境安装
```bash
# 创建conda环境
conda create -n mnist python=3.11
conda activate mnist
# 安装GPU版本PyTorch，也可使用whl离线包安装
pip install torch torchvision matplotlib --index-url https://download.pytorch.org/whl/cu121
```

## 快速复现
1. 克隆仓库
```bash
git clone https://github.com/Avaland-Kong/MNIST.git
cd MNIST
conda activate mnist
```

2. 运行主程序
```bash
python main.py
```
- 首次运行自动下载MNIST数据集至`./data`目录；
- 训练完成后根目录生成`compare_result.png`精度对比曲线图。