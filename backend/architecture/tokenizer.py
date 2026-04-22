import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    # 字符级分词器，ASCII + 常用中文，不依赖外部库
    # 虽然粗糙但对架构处理器的数值计算已经够用了

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self._init_vocab()

    def _init_vocab(self):
        idx = 4
        for c in (chr(i) for i in range(32, 127)):
            if idx < self.vocab_size:
                self.char2id[c] = idx
                self.id2char[idx] = c
                idx += 1
        # 常用中文范围，但限制总量避免内存浪费
        for c in (chr(i) for i in range(0x4E00, 0x9FA5)):
            if idx < self.vocab_size:
                self.char2id[c] = idx
                self.id2char[idx] = c
                idx += 1
                if idx >= min(self.vocab_size, 0x4E00 + 8000):
                    break
        logger.info(f"SimpleTokenizer initialized with {idx} tokens")

    def encode(self, text: str, max_len: int = 512) -> torch.Tensor:
        ids = [self.bos_id]
        for c in text[:max_len - 2]:
            ids.append(self.char2id.get(c, self.unk_id))
        ids.append(self.eos_id)
        while len(ids) < max_len:
            ids.append(self.pad_id)
        return torch.tensor(ids[:max_len], dtype=torch.long).unsqueeze(0)
