"""Causal LM dataset: local text tokenization and chunking.

Uses text from local SFT/DPO samples + built-in corpus to eliminate
network dependency at training time.
"""

import json
import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Built-in English text corpus (Wikipedia-style passages) as fallback
_BUILTIN_TEXT = [
    "Machine learning is a field of inquiry devoted to understanding and building methods that learn from data.",
    "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence.",
    "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
    "The Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
    "Computers are electronic devices that process data according to instructions stored in their memory.",
    "The scientific method is a systematic way of learning about the world through observation and experimentation.",
    "Language is a structured system of communication used by humans, based on speech and gesture.",
    "Mathematics is the study of numbers, quantities, shapes, and patterns. It is essential in many fields.",
    "The Internet is a global network of interconnected computers that communicate via standardized protocols.",
    "Climate change refers to long-term shifts in temperatures and weather patterns on Earth.",
    "Artificial neural networks are computing systems inspired by biological neural networks in animal brains.",
    "Deep learning is a class of machine learning algorithms that uses multiple layers to extract higher-level features.",
    "Natural language processing is a subfield of linguistics and artificial intelligence focused on human language.",
    "Computer vision enables machines to derive meaningful information from digital images and videos.",
    "Reinforcement learning is an area of machine learning where an agent learns by interacting with its environment.",
    "Transfer learning is a technique where a model trained on one task is repurposed for a second related task.",
    "Data science combines statistics, computer science, and domain knowledge to extract insights from data.",
    "Cloud computing delivers computing services over the internet, including storage and processing power.",
    "Cybersecurity protects computer systems from theft, damage, or disruption of their hardware and software.",
    "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy to fuel their growth.",
    "Gravity is a fundamental force of nature that attracts objects with mass toward each other.",
    "The Industrial Revolution transformed economies from agriculture-based to industry-based production.",
    "Quantum mechanics is a fundamental theory in physics describing nature at the smallest scales.",
    "Biology is the scientific study of life, from molecular processes to complex ecosystems.",
    "Economics studies the production, distribution, and consumption of goods and services.",
    "Philosophy explores fundamental questions about existence, knowledge, values, reason, and language.",
    "Music is the art of arranging sounds in time through melody, harmony, rhythm, and timbre.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Democracy is a form of government in which the people have the authority to make laws.",
    "The printing press was invented by Johannes Gutenberg in the 15th century, revolutionizing communication.",
    "Albert Einstein developed the theory of relativity, which transformed our understanding of space and time.",
    "DNA is a molecule that carries the genetic instructions used in the growth and development of organisms.",
    "The solar system consists of the Sun and the objects that orbit it, including eight planets.",
    "Electricity is the set of physical phenomena associated with the presence and motion of electric charge.",
    "Literature encompasses written works, especially those considered of superior or lasting artistic merit.",
    "Architecture is both the process and product of planning, designing, and constructing buildings.",
    "The cell is the basic structural and functional unit of all known living organisms.",
    "Evolution is the process by which species of organisms change over generations through natural selection.",
    "Photography is the art and practice of creating images by recording light on a photosensitive surface.",
    "Robotics integrates computer science and engineering to design and operate robots.",
    "The atmosphere of Earth is a layer of gases that surrounds the planet and is retained by gravity.",
    "Energy cannot be created or destroyed but can be transformed from one form to another.",
    "The Constitution is the supreme law of the United States of America.",
    "Water covers about 71 percent of the Earth's surface, mostly in seas and oceans.",
    "The periodic table organizes chemical elements by atomic number and electron configuration.",
    "Happiness is often considered to be a state of mind or a feeling of contentment and satisfaction.",
    "Education is the process of facilitating learning and acquiring knowledge and skills.",
    "The telephone was invented by Alexander Graham Bell in 1876, changing how people communicate.",
    "Volcanoes are openings in the Earth's crust through which molten rock and gases erupt.",
    "The Internet has transformed how people work, shop, learn, and connect with each other globally.",
    "Stars are massive luminous spheres of plasma held together by their own gravity.",
    "The Great Wall of China is one of the most impressive architectural feats in history.",
    "Biodiversity refers to the variety of life on Earth at all levels, from genes to ecosystems.",
    "Genetics is the study of genes, genetic variation, and heredity in living organisms.",
    "The speed of light in a vacuum is approximately 300,000 kilometers per second.",
    "Human rights are moral principles that describe certain standards of human behavior.",
    "The ocean contains about 97 percent of Earth's water and covers vast undersea landscapes.",
    "Innovation drives progress in technology, medicine, and other fields of human endeavor.",
    "Memory is the faculty by which the brain encodes, stores, and retrieves information.",
]


def _load_local_text() -> list[str]:
    """Collect text from local JSON files and built-in corpus."""
    passages = list(_BUILTIN_TEXT)

    # Extract text from SFT samples
    sft_path = "data/samples/sft_samples.json"
    if os.path.exists(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                for key in ("instruction", "input", "output"):
                    val = item.get(key, "")
                    if val and isinstance(val, str) and len(val) > 10:
                        passages.append(val)

    # Extract text from DPO samples
    dpo_path = "data/samples/dpo_samples.json"
    if os.path.exists(dpo_path):
        with open(dpo_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                for key in ("prompt", "chosen", "rejected"):
                    val = item.get(key, "")
                    if val and isinstance(val, str) and len(val) > 10:
                        passages.append(val)

    return passages


def get_wikitext_data(tokenizer: PreTrainedTokenizer, block_size: int = 256):
    """Build causal LM dataset from local text corpus.

    Returns train and eval PyTorch Datasets ready for causal LM training.
    """
    passages = _load_local_text()
    print(f"  Using local text corpus: {len(passages)} passages")

    # Tokenize each passage individually, then concatenate and chunk
    all_ids = []
    for passage in passages:
        encoded = tokenizer.encode(passage, truncation=True, max_length=block_size * 2)
        if len(encoded) > 1:  # skip empty
            all_ids.extend(encoded)
    all_ids = torch.tensor(all_ids, dtype=torch.long)

    # Chunk into block_size windows
    chunks = []
    stride = block_size // 2  # overlap for more data
    for i in range(0, len(all_ids) - block_size, stride):
        chunk = all_ids[i : i + block_size]
        chunks.append(chunk)
        if len(chunks) >= 500:  # enough for demo
            break

    if not chunks:
        # Fallback: random token IDs
        chunks = [torch.randint(0, 10000, (block_size,), dtype=torch.long)]

    ids_tensor = torch.stack(chunks)
    print(f"  Created {len(ids_tensor)} chunks of size {block_size}")

    # Split: 80% train, 20% eval
    split = int(len(ids_tensor) * 0.8)
    train_ids = ids_tensor[:split]
    eval_ids = ids_tensor[split:]

    return CausalLMDataset(train_ids), CausalLMDataset(eval_ids)


class CausalLMDataset(Dataset):
    """Dataset for causal language modeling. input_ids == labels."""

    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}
