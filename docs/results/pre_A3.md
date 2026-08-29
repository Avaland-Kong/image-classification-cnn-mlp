# 预实验 A3：CNN 接通验证

| 项目 | 内容 |
|------|------|
| 阶段 | 预实验 A3 |
| 日期 | 2026-08-30 |
| 数据集 | `mnist` |
| 划分 | train/val/test = 54000/6000/10000（`val_ratio=0.1`，`seed=42`） |
| 模式 | `-mode train`（不评估测试集） |
| 结构 | CNN `-d 32 64`（A1 入选） |
| 超参 | Adam，`lr=0.001`，`epochs=10`（A2 入选 T2） |
| 运行名 | `pre_A3_cnn_d32-64_s42` |
| 环境 | `image-cls`；PyTorch 2.5.1+cu121；GPU RTX 4060 Laptop |

目的：确认由 MLP 选定的训练协议可迁移到 CNN。本步不重新选参，不与 MLP 比分。

命令：

```bash
python main.py -mode train -m cnn -ds mnist -d 32 64 \
  -opt adam -lr 0.001 -e 10 -s 42 -vr 0.1 \
  -mn pre_A3_cnn_d32-64_s42
```

---

## 1. 结果

| 指标 | 最终 epoch |
|------|------------|
| train_loss | 0.0117 |
| train_acc | 0.9958 |
| val_loss | 0.0399 |
| **val_acc** | **0.9905** |

训练过程稳定，损失总体下降；`val_acc ≥ 0.90`。

**判定：通过。**

日志：`logs/pre_A3_cnn_d32-64_s42/`。

---

## 2. 冻结生效

A3 通过后，第 6.1 / 6.2 节配置正式冻结，可进入阶段 B 主对比实验。
