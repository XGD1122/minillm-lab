# MiniLLM-Lab 实验报告

生成时间: (运行 `python -m src.evaluation.report` 自动填充)

---

## 实验 A: Mini Pretrain

| 项目 | 值 |
|------|----|
| 模型 | distilgpt2 (82M) |
| 数据 | WikiText-2 |
| 配置 | configs/cpu_pretrain_tiny.yaml |
| 状态 | 未运行 |

## 实验 B: SFT

| 项目 | 值 |
|------|----|
| 模型 | distilgpt2 (82M) |
| 配置 | configs/cpu_sft_tiny.yaml |
| 状态 | 未运行 |

## 实验 C: DPO

| 项目 | 值 |
|------|----|
| 模型 | distilgpt2 (82M) |
| 配置 | configs/cpu_dpo_tiny.yaml |
| 状态 | 未运行 |

---

## 总结

三个实验对应 LLM 训练的三个关键阶段:

1. **Mini Pretrain** — 因果语言建模,让模型学会预测下一个 token
2. **SFT** — 指令微调,让模型学会理解并遵循人类指令
3. **DPO** — 偏好优化,让模型的输出更符合人类偏好

> 运行训练后,查看 `runs/` 下的 TensorBoard 日志获取 loss 曲线。
