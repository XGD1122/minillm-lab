"""Generate experiment summary report in Markdown."""

import os
import json
from datetime import datetime


def generate_report():
    """Generate a summary report by scanning outputs/ and runs/ directories."""
    report_path = "reports/experiment_report.md"
    os.makedirs("reports", exist_ok=True)

    lines = []
    lines.append(f"# MiniLLM-Lab 实验报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    # Experiment A: Pretrain
    lines.append("## 实验 A: Mini Pretrain\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|----|")
    lines.append("| 模型 | distilgpt2 (82M) |")
    lines.append("| 数据 | WikiText-2 |")
    lines.append("| 配置 | configs/cpu_pretrain_tiny.yaml |")

    pretrain_out = "outputs/pretrain"
    if os.path.exists(pretrain_out):
        lines.append(f"| 输出目录 | `{pretrain_out}/` |")
        checkpoints = [d for d in os.listdir(pretrain_out) if d.startswith("checkpoint-")]
        lines.append(f"| Checkpoints | {len(checkpoints)} 个 |")
    else:
        lines.append("| 状态 | 未运行 |")
    lines.append("")

    # Experiment B: SFT
    lines.append("## 实验 B: SFT\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|----|")
    lines.append("| 模型 | distilgpt2 (82M) |")
    sft_data = "data/samples/sft_samples.json"
    if os.path.exists(sft_data):
        with open(sft_data, "r", encoding="utf-8") as f:
            n = len(json.load(f))
        lines.append(f"| 数据量 | {n} 条 instruction |")
    lines.append("| 配置 | configs/cpu_sft_tiny.yaml |")

    sft_out = "outputs/sft"
    if os.path.exists(sft_out):
        lines.append(f"| 输出目录 | `{sft_out}/` |")
    else:
        lines.append("| 状态 | 未运行 |")
    lines.append("")

    # Experiment C: DPO
    lines.append("## 实验 C: DPO\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|----|")
    lines.append("| 模型 | distilgpt2 (82M) |")
    dpo_data = "data/samples/dpo_samples.json"
    if os.path.exists(dpo_data):
        with open(dpo_data, "r", encoding="utf-8") as f:
            n = len(json.load(f))
        lines.append(f"| 偏好对数量 | {n} 条 |")
    lines.append("| 配置 | configs/cpu_dpo_tiny.yaml |")

    dpo_out = "outputs/dpo"
    if os.path.exists(dpo_out):
        lines.append(f"| 输出目录 | `{dpo_out}/` |")
    else:
        lines.append("| 状态 | 未运行 |")
    lines.append("")

    # Summary
    lines.append("---\n")
    lines.append("## 总结\n")
    lines.append("三个实验对应 LLM 训练的三个关键阶段:\n")
    lines.append("1. **Mini Pretrain** — 因果语言建模,让模型学会预测下一个 token")
    lines.append("2. **SFT** — 指令微调,让模型学会理解并遵循人类指令")
    lines.append("3. **DPO** — 偏好优化,让模型的输出更符合人类偏好\n")
    lines.append("> 运行训练后,查看 `runs/` 下的 TensorBoard 日志获取 loss 曲线,运行 `python -m src.evaluation.report` 重新生成此报告。")

    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Report generated: {report_path}")
    print(content)


if __name__ == "__main__":
    generate_report()
