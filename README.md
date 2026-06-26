# MiniLLM-Lab

> 用 distilgpt2 (82M 参数) 在 GPU/CPU 上跑通 LLM 训练全流程：预训练 → SFT → DPO。

## 项目定位

算法工程师能力展示项目 — 在小模型上验证 LLM 训练的完整流水线，证明你理解 causal language modeling、instruction tuning 和 preference optimization 的原理和实现。

## 三个实验

| 实验 | 目的 | 数据 | 核心输出 |
|------|------|------|---------|
| A: Mini Pretrain | 因果语言模型继续训练 | 本地文本语料 | loss 曲线 + 训练前后生成对比 |
| B: SFT | 指令微调 | 自制 49 条 instruction 数据 | 微调前后同一 prompt 回答对比 |
| C: DPO | 偏好优化 | 自制 37 条偏好对 | chosen/rejected reward 变化 + 偏好案例 |

## 技术栈

| 层 | 选型 |
|----|------|
| 框架 | HuggingFace Transformers + Datasets + PEFT |
| 模型 | distilgpt2 (82M params) |
| 日志 | TensorBoard（不可用时自动降级为 CSV） |
| 配置 | YAML + CLI argparse |

## 快速开始

```bash
git clone <repo-url>
cd minillm-lab
pip install -r requirements.txt

# 安装 tensorboard (可选，缺失时自动降级为 CSV)
pip install tensorboard

# 冒烟测试 (50 steps, ~5 min)
python -m src.training.mini_pretrain --config configs/cpu_pretrain_tiny.yaml --smoke
python -m src.training.sft --config configs/cpu_sft_tiny.yaml --smoke
python -m src.training.dpo --config configs/cpu_dpo_tiny.yaml --smoke

# 完整运行 (500 steps)
python -m src.training.mini_pretrain --config configs/cpu_pretrain_tiny.yaml

# 查看结果
tensorboard --logdir runs/
python -m src.evaluation.report  # 生成实验报告
```

## 项目结构

```
minillm-lab/
├── src/
│   ├── config.py                   # YAML 配置 + CLI 参数解析
│   ├── data/
│   │   ├── causal_lm_dataset.py    # 文本 tokenize + 分块
│   │   ├── sft_dataset.py          # instruction 数据格式化
│   │   └── dpo_dataset.py          # chosen/rejected 偏好对
│   ├── training/
│   │   ├── mini_pretrain.py        # 实验 A: causal LM 训练
│   │   ├── sft.py                  # 实验 B: SFT 训练
│   │   └── dpo.py                  # 实验 C: 手写 DPO Loss
│   ├── evaluation/
│   │   ├── perplexity.py           # 困惑度计算
│   │   ├── generation_compare.py   # 训练前后生成对比
│   │   └── report.py               # Markdown 实验报告生成
│   └── utils/
│       ├── model.py                # 模型加载 (CPU/CUDA 自动切换)
│       └── logging.py              # TensorBoard / CSV 日志
├── configs/                        # 三实验 YAML 配置文件
├── data/samples/                   # 自制 SFT + DPO 样例数据
├── notebooks/                      # Jupyter 全流程 walkthrough
└── tests/                          # 基础测试
```

## 配置

```yaml
# configs/cpu_pretrain_tiny.yaml
model: "distilgpt2"
max_steps: 500
smoke_steps: 50         # --smoke 时使用
batch_size: 2
gradient_accumulation_steps: 4
block_size: 256
learning_rate: 5.0e-5
device: "cuda"          # cuda / cpu / mps，或删除此行自动检测
```

CLI 覆盖规则：`--config` 加载 YAML，`--smoke` 用 `smoke_steps` 替代 `max_steps`。

## 耗时预估

| 实验 | smoke (50 steps) | 完整 (500 steps) |
|------|-----------------|------------------|
| Mini Pretrain | ~2 min (GPU) / ~5 min (CPU) | ~15 min / ~40 min |
| SFT | ~2 min / ~5 min | ~10 min / ~30 min |
| DPO | ~3 min / ~5 min | ~20 min / ~50 min |
