# 阶段 B：主对比实验

| 项目 | 内容 |
|------|------|
| 阶段 | 主对比（阶段 B） |
| 日期 | 2026-08-30 |
| 模式 | `-mode train_test`（训练后在官方测试集评估） |
| 冻结配置 | 第 6.1 / 6.2 节（A3 已通过后冻结） |
| 结构 | MLP `-d 256 128`；CNN `-d 32 64` |
| 超参 | Adam，`lr=0.001`，`-vr 0.1` |
| 轮次 | MNIST / Fashion-MNIST：`10`；CIFAR-10：`20` |
| 种子 | `17`，`32`，`61`（每个「数据集 × 模型」各一次） |
| 运行次数 | 18（完整重跑，覆盖此前未完成的阶段 B 日志） |
| 环境 | `image-cls`；PyTorch 2.5.1+cu121；GPU RTX 4060 Laptop |
| 总耗时 | 约 28 分钟（18 次合计） |

评价依据：官方测试集 **`final_test_acc`**（`metrics.json`）。mean 与 sample std 按三个种子计算（设计书第 10 节）。

命令模板：

```bash
python main.py -mode train_test \
  -m mlp -ds <dataset> -d 256 128 \
  -opt adam -lr 0.001 -e <10|20> -s <seed> -vr 0.1 \
  -mn exp1_<dataset>_mlp_d256-128_s<seed>

python main.py -mode train_test \
  -m cnn -ds <dataset> -d 32 64 \
  -opt adam -lr 0.001 -e <10|20> -s <seed> -vr 0.1 \
  -mn exp1_<dataset>_cnn_d32-64_s<seed>
```

CIFAR-10 使用 `-e 20`，其余 `-e 10`。

---

## 1. 逐次运行（测试集）

### 1.1 MNIST（划分 54000 / 6000 / 10000）

| 运行名 | 模型 | seed | test_loss | **test_acc** | val_acc |
|--------|------|------|-----------|--------------|---------|
| `exp1_mnist_mlp_d256-128_s17` | MLP | 17 | 0.1065 | **0.9731** | 0.9732 |
| `exp1_mnist_mlp_d256-128_s32` | MLP | 32 | 0.0834 | **0.9790** | 0.9797 |
| `exp1_mnist_mlp_d256-128_s61` | MLP | 61 | 0.0818 | **0.9790** | 0.9792 |
| `exp1_mnist_cnn_d32-64_s17` | CNN | 17 | 0.0500 | **0.9877** | 0.9867 |
| `exp1_mnist_cnn_d32-64_s32` | CNN | 32 | 0.0344 | **0.9896** | 0.9927 |
| `exp1_mnist_cnn_d32-64_s61` | CNN | 61 | 0.0402 | **0.9897** | 0.9902 |

### 1.2 Fashion-MNIST（划分 54000 / 6000 / 10000）

| 运行名 | 模型 | seed | test_loss | **test_acc** | val_acc |
|--------|------|------|-----------|--------------|---------|
| `exp1_fashion_mnist_mlp_d256-128_s17` | MLP | 17 | 0.3469 | **0.8858** | 0.8880 |
| `exp1_fashion_mnist_mlp_d256-128_s32` | MLP | 32 | 0.3525 | **0.8847** | 0.8915 |
| `exp1_fashion_mnist_mlp_d256-128_s61` | MLP | 61 | 0.3372 | **0.8879** | 0.8987 |
| `exp1_fashion_mnist_cnn_d32-64_s17` | CNN | 17 | 0.2695 | **0.9150** | 0.9217 |
| `exp1_fashion_mnist_cnn_d32-64_s32` | CNN | 32 | 0.2616 | **0.9190** | 0.9250 |
| `exp1_fashion_mnist_cnn_d32-64_s61` | CNN | 61 | 0.2497 | **0.9195** | 0.9237 |

### 1.3 CIFAR-10（划分 45000 / 5000 / 10000）

| 运行名 | 模型 | seed | test_loss | **test_acc** | val_acc |
|--------|------|------|-----------|--------------|---------|
| `exp1_cifar10_mlp_d256-128_s17` | MLP | 17 | 2.1344 | **0.5104** | 0.5040 |
| `exp1_cifar10_mlp_d256-128_s32` | MLP | 32 | 2.1245 | **0.5057** | 0.5222 |
| `exp1_cifar10_mlp_d256-128_s61` | MLP | 61 | 2.1229 | **0.5110** | 0.5042 |
| `exp1_cifar10_cnn_d32-64_s17` | CNN | 17 | 1.1723 | **0.7126** | 0.7230 |
| `exp1_cifar10_cnn_d32-64_s32` | CNN | 32 | 1.1707 | **0.7279** | 0.7304 |
| `exp1_cifar10_cnn_d32-64_s61` | CNN | 61 | 1.3332 | **0.7197** | 0.7250 |

日志目录：`logs/exp1_*/`（含 `history.csv`、`config.json`、`metrics.json`）；权重：`checkpoints/exp1_*.pth`。

---

## 2. 主结果表（测试准确率 mean±std）

| 数据集 | MLP Acc (mean±std) | CNN Acc (mean±std) | 差值 (CNN−MLP) |
|--------|--------------------|--------------------|----------------|
| MNIST | 0.9770 ± 0.0034 | 0.9890 ± 0.0011 | 0.0120 |
| Fashion-MNIST | 0.8861 ± 0.0016 | 0.9178 ± 0.0025 | 0.0317 |
| CIFAR-10 | 0.5090 ± 0.0029 | 0.7201 ± 0.0077 | 0.2110 |

测试损失 mean±std：

| 数据集 | MLP Loss (mean±std) | CNN Loss (mean±std) |
|--------|---------------------|---------------------|
| MNIST | 0.0906 ± 0.0138 | 0.0415 ± 0.0079 |
| Fashion-MNIST | 0.3455 ± 0.0078 | 0.2603 ± 0.0100 |
| CIFAR-10 | 2.1272 ± 0.0063 | 1.2254 ± 0.0933 |

对比图：`docs/results/exp1_test_acc.png`。

已同步写入 `docs/experiment_design.md` 第 7.5 节。

---

## 3. 简要观察

- 三个数据集上 **CNN 测试准确率均高于 MLP**；多种子波动较小（Fashion 与 MNIST 的 std 约 0.001～0.003；CIFAR-10 上 CNN std 约 0.008）。
- 随任务变难，两模型准确率都下降，**差值扩大**：MNIST 1.20 个百分点 → Fashion-MNIST 3.17 个百分点 → CIFAR-10 21.10 个百分点。
- CIFAR-10 绝对准确率明显低于前两个数据集（MLP 约 50.9%，CNN 约 72.0%），符合浅网络、无数据增强、超参由 MNIST 选定的设定；训练过程收敛，未出现贴近随机猜测（10%）的崩溃。
- CIFAR-10 上两模型均出现 train 明显高于 val/test 的过拟合（MLP 最终 train_acc 约 0.83、test_acc 约 0.51；CNN 最终 train_acc 约 0.92～0.95、test_acc 约 0.72）。这是该协议下的现象，不回改第 6 节配置。
