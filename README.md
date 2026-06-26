# MiniLLM-Lab

用 distilgpt2 (82M 参数) 跑通 LLM 训练的完整流水线：因果语言模型预训练 → SFT 指令微调 → DPO 偏好优化。

## 三个实验

| 实验 | 目的 | 核心输出 |
|------|------|---------|
| A: Mini Pretrain | 因果语言模型继续训练 | loss 曲线 + 训练前后生成对比 |
| B: SFT | 指令微调 | 微调前后同一 prompt 回答对比 |
| C: DPO | 偏好优化 | chosen/rejected reward 变化 + 偏好案例 |

## 技术栈

PyTorch / HuggingFace Transformers / Datasets / PEFT / TensorBoard / YAML

## 快速开始

```bash
git clone https://github.com/XGD1122/minillm-lab.git
cd minillm-lab
pip install -r requirements.txt

# 冒烟测试 (50 steps)
python -m src.training.mini_pretrain --config configs/cpu_pretrain_tiny.yaml --smoke
python -m src.training.sft --config configs/cpu_sft_tiny.yaml --smoke
python -m src.training.dpo --config configs/cpu_dpo_tiny.yaml --smoke

# 完整运行 (500 steps)
python -m src.training.mini_pretrain --config configs/cpu_pretrain_tiny.yaml

# 查看结果
tensorboard --logdir runs/
python -m src.evaluation.report
```

## 项目结构

```
minillm-lab/
├── src/
│   ├── config.py                   # YAML 配置 + CLI 参数解析
│   ├── data/
│   │   ├── causal_lm_dataset.py    # 文本 tokenize + 分块
│   │   ├── sft_dataset.py          # instruction 数据格式化
│   │   └── dpo_dataset.py          # preference pair 数据格式化
│   ├── training/
│   │   ├── mini_pretrain.py        # 实验 A: causal LM 训练
│   │   ├── sft.py                  # 实验 B: SFT 训练
│   │   └── dpo.py                  # 实验 C: DPO 训练 (手写 loss)
│   ├── evaluation/
│   │   ├── perplexity.py           # 困惑度计算
│   │   ├── generation_compare.py   # 训练前后生成对比
│   │   └── report.py               # 实验报告生成
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
model: "distilgpt2"
max_steps: 500
smoke_steps: 50
batch_size: 2
gradient_accumulation_steps: 4
block_size: 256
learning_rate: 5.0e-5
device: "cuda"
```

`--config` 加载 YAML，`--smoke` 将 `smoke_steps` 应用到 `max_steps`。

## 耗时预估

| 实验 | smoke (50 steps) | 完整 (500 steps) |
|------|-----------------|------------------|
| Mini Pretrain | ~2 min (GPU) | ~15 min |
| SFT | ~2 min | ~10 min |
| DPO | ~3 min | ~20 min |
