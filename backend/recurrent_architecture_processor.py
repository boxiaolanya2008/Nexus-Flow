# 循环架构处理器：RecurrentModel 的数值计算核心
# 处理后的 logits 和 halting 信息编码为结构化信号注入 LLM 请求

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import logging
import os

from .architecture.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class RecurrentArchitectureProcessor:
    # tokenize -> RecurrentModel forward -> logits/halting 分析 -> 结构化信号

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        max_loops: int = 4,
        use_act: bool = False,
        device: str = "cpu"
    ):
        from .architecture.recurrent_architecture import RecurrentModel

        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        self.device = torch.device(device)
        self.model = RecurrentModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_loops=max_loops,
            use_act=use_act
        ).to(self.device)
        self.model.eval()

        logger.info(
            f"RecurrentArchitectureProcessor initialized: "
            f"{sum(p.numel() for p in self.model.parameters()):,} params, "
            f"device={device}"
        )

    def process(self, text: str, num_loops: int = None) -> Dict[str, Any]:
        input_ids = self.tokenizer.encode(text, max_len=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, num_loops=num_loops)
            logits = outputs['logits']
            halting_steps = outputs.get('halting_steps')

        architecture_analysis = self._analyze_architecture(logits, halting_steps, text)
        encoded_signal = self._encode_architecture_to_signal(architecture_analysis, text)

        return {
            "architecture_signal": encoded_signal,
            "logits_stats": {
                "mean": logits.mean().item(),
                "std": logits.std().item(),
                "min": logits.min().item(),
                "max": logits.max().item()
            },
            "halting_steps": halting_steps.cpu().numpy().tolist() if halting_steps is not None else None,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "encoder_status": "recurrent"
        }

    def _analyze_architecture(
        self,
        logits: torch.Tensor,
        halting_steps: torch.Tensor,
        text: str
    ) -> Dict[str, Any]:
        batch_size, seq_len, vocab_size = logits.shape
        top_k_logits, top_k_indices = torch.topk(logits, k=5, dim=-1)

        analysis = {
            "architecture_type": "RecurrentModel",
            "input_length": len(text),
            "sequence_length": seq_len,
            "vocabulary_size": vocab_size,
            "logits_statistics": {
                "mean": logits.mean().item(),
                "std": logits.std().item(),
                "min": logits.min().item(),
                "max": logits.max().item()
            },
            "top_predictions": top_k_indices[0, -1, :].cpu().numpy().tolist(),
            "halting_info": None
        }

        if halting_steps is not None:
            analysis["halting_info"] = {
                "mean_steps": halting_steps.float().mean().item(),
                "max_steps": halting_steps.max().item(),
                "min_steps": halting_steps.min().item(),
                "steps_distribution": halting_steps.cpu().numpy().tolist()
            }

        return analysis

    def _encode_architecture_to_signal(
        self,
        architecture_analysis: Dict,
        original_text: str
    ) -> str:
        import json

        signal_parts = [
            "[循环架构信号]",
            "以下是由 RecurrentModel 提取的架构分析：",
            "",
            "## 架构类型",
            architecture_analysis["architecture_type"],
            "",
            "## 输入信息",
            f"输入长度: {architecture_analysis['input_length']} 字符",
            f"序列长度: {architecture_analysis['sequence_length']}",
            f"词汇表大小: {architecture_analysis['vocabulary_size']}",
            "",
            "## Logits 统计",
            json.dumps(architecture_analysis["logits_statistics"], ensure_ascii=False, indent=2),
            "",
            "## Top 预测",
            f"Top-5 token IDs: {architecture_analysis['top_predictions']}",
        ]

        if architecture_analysis["halting_info"]:
            signal_parts.extend([
                "",
                "## ACT Halting 信息",
                json.dumps(architecture_analysis["halting_info"], ensure_ascii=False, indent=2),
            ])

        signal_parts.extend([
            "",
            "## 分析摘要",
            f"循环架构已处理输入文本，输出 logits 分布：均值={architecture_analysis['logits_statistics']['mean']:.3f}, "
            f"标准差={architecture_analysis['logits_statistics']['std']:.3f}",
            "",
            "[循环架构信号结束]"
        ])

        return "\n".join(signal_parts)

    def get_architecture_status(self) -> Dict[str, Any]:
        return {
            "architecture_name": "RecurrentModel",
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "status": "active",
            "mode": "recurrent",
            "semantic_encoder": {
                "status": "not_applicable",
                "note": "RecurrentModel uses different architecture than CustomModel"
            }
        }
