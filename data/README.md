# 数据说明

## SFT 数据格式

文件: `data/samples/sft_samples.json`

```json
[
  {
    "instruction": "把下面这句话翻译成英文",
    "input": "今天天气很好。",
    "output": "The weather is nice today."
  }
]
```

- `instruction`: 任务指令
- `input`: (可选) 任务输入
- `output`: 期望的模型输出

共约 50 条,覆盖翻译、总结、广告语、情感分析、JSON格式转换、解释概念、问答、续写、写信、列优缺点、代码生成、观点表态、建议、编故事、信息提取、改写、成语解释、菜谱、客服回复等任务类型。

## DPO 数据格式

文件: `data/samples/dpo_samples.json`

```json
[
  {
    "prompt": "如何学习编程?",
    "chosen": "建议从Python开始,多做项目练习...",
    "rejected": "随便学学就会了,不用太认真..."
  }
]
```

- `prompt`: 用户问题
- `chosen`: 高质量、有帮助、负责任的回答
- `rejected`: 低质量、敷衍或不负责任的回答

共约 37 条偏好对。chosen 回答体现:具体建议、多维度分析、积极价值观。rejected 回答体现:敷衍、极端、消极或不负责的态度。

## 扩充数据

可以用 LLM (如 ChatGPT/Claude) 批量生成更多 SFT/DPO 数据,只需保持相同格式。
