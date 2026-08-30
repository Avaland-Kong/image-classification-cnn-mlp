# MNIST 手写数字分类

基于 PyTorch 的 MNIST 分类项目，支持 **MLP** 与 **CNN** 两种模型，可通过命令行选择模型结构、损失函数、优化器，并保存 / 加载训练权重。

## 项目简介

- **MLP**：全连接网络，将图像展平后输入，结构由 `--dims` 指定隐藏层大小。
- **CNN**：卷积 + BatchNorm + 池化提取空间特征，结构由 `--dims` 指定各层通道数。

数据集与模型结构解耦：`image_shape`、`num_classes` 从 Dataset 自动读取，便于后续更换数据集。

## 目录结构

```
MNIST/
├── main.py              # 入口：参数解析、建模型、训练 / 测试、保存 / 加载
├── train.py             # train_model 训练循环
├── test.py              # test_model 评估逻辑
├── dataset_config.py    # 数据集、DataLoader、image_shape、num_classes
├── models/
│   ├── __init__.py
│   ├── mlp.py           # MLP 定义
│   └── cnn.py           # CNN 定义
├── checkpoints/         # 模型权重（训练后生成，已被 .gitignore 忽略）
├── data/                # 各数据集目录 data/<name>/（首次运行自动下载，已被 .gitignore 忽略）
├── .gitignore
└── README.md
```

## 环境依赖

- Python 3.10+
- PyTorch
- torchvision

```bash
# 示例：conda 环境
conda create -n mnist python=3.11
conda activate mnist

# CPU
pip install torch torchvision

# GPU（按本机 CUDA 版本选择，见 https://pytorch.org）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 快速开始

```bash
git clone https://github.com/Avaland-Kong/MNIST.git
cd MNIST
conda activate mnist

# 训练 + 测试（默认 MLP）
python main.py -mode train_test

# 训练 MLP 并保存为 model_0
python main.py -mode train -m mlp -mn model_0 -d 256 128

# 训练 CNN 并保存为 model_1
python main.py -mode train_test -m cnn -mn model_1 -d 32 64

# 仅测试（只需指定 checkpoint 名称，结构从文件恢复）
python main.py -mode test -mn model_1
```

首次运行会自动下载 MNIST 到 `./data`。权重保存在 `checkpoints/{model_name}.pth`。

## 命令行参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--mode` | `-mode` | `train` | `train` / `test` / `train_test` |
| `--model_name` | `-mn` | `model_0` | 保存 / 加载的 checkpoint 名称 |
| `--model` | `-m` | `mlp` | 模型类型：`mlp` / `cnn` |
| `--dims` | `-d` | `256 256` | MLP 隐藏层维度；CNN 各卷积层通道数 |
| `--loss_function` | `-lf` | `cross_entropy` | 损失函数 |
| `--optimizer` | `-opt` | `adam` | 优化器：`adam` / `sgd` |
| `--lr` | `-lr` | `0.001` | 学习率 |
| `--epochs` | `-e` | `10` | 训练轮数 |

查看全部参数：

```bash
python main.py -h
```

### 参数说明

- **`--dims`**：对 MLP 表示隐藏层神经元数量（如 `256 128`）；对 CNN 表示各卷积层输出通道数（如 `32 64`）。CNN 每层后有一次 `MaxPool2d(2)`，层数受输入图像尺寸限制。
- **`--model_name`**：仅用于区分不同实验的保存文件名，与 `--model`（网络类型）无关。同名会覆盖旧权重。
- **测试模式**：只需 `-mode test -mn <名称>`，模型类型与结构从 checkpoint 中恢复。

## Checkpoint 内容

每个 `.pth` 文件包含：

| 字段 | 说明 |
|------|------|
| `model_state_dict` | 模型权重 |
| `model_type` | `mlp` 或 `cnn` |
| `model_dims` | 训练时使用的 `--dims` |
| `image_shape` | 输入图像形状 `(C, H, W)` |
| `num_classes` | 分类类别数 |

## 参考结果

在 MNIST 上、默认训练 10 epoch 的典型测试准确率：

| 模型 | 配置示例 | Test Accuracy |
|------|----------|---------------|
| MLP | `-d 256 128` | ~97.9% |
| CNN | `-d 32 64` | ~99.1% |

实际结果因随机初始化与硬件略有差异。

## 许可证

本项目仅供学习使用。
