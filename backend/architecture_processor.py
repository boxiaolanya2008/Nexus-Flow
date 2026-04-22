# CustomModel 架构处理器
# forward 传播后从 hidden states 提取统计特征，编码为结构化信号注入 LLM 请求

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import os

from .architecture.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class ArchitectureProcessor:
    # tokenize -> CustomModel forward -> hidden states 统计分析 -> 结构化信号

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        window_size: int = 64,
        memory_size: int = 128,
        device: str = "cpu",
        semantic_encoder_path: str = None  # 保留参数签名兼容，不再使用
    ):
        from .architecture import CustomModel

        self.tokenizer = SimpleTokenizer(vocab_size=50000)
        self.device = torch.device(device)
        self.d_model = d_model
        self.model = CustomModel(
            vocab_size=50000,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
            memory_size=memory_size,
            max_seq_len=512,
            dropout=0.0
        ).to(self.device)
        self.model.eval()

        self.extract_layers = [0, n_layers // 2, n_layers - 1]

        logger.info(
            f"ArchitectureProcessor initialized: "
            f"{sum(p.numel() for p in self.model.parameters()):,} params, "
            f"device={device}"
        )

    def process(self, text: str) -> Dict[str, Any]:
        input_ids = self.tokenizer.encode(text, max_len=512).to(self.device)

        with torch.no_grad():
            batch_size, seq_len = input_ids.shape
            token_emb = self.model.token_embedding(input_ids)
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.model.position_embedding(positions)
            x = self.model.embedding_norm(token_emb + pos_emb)
            x = self.model.embedding_dropout(x)

            # 收集中间层 hidden states
            layer_outputs = []
            for block in self.model.blocks:
                x = block(x)
                layer_outputs.append(x)

            final_hidden = self.model.final_norm(x)

        # 直接从 hidden states 提取统计特征，不依赖语义编码器
        feature_analysis = self._extract_features(final_hidden, layer_outputs, text)
        encoded_signal = self._encode_features_to_signal(feature_analysis, text)

        return {
            "architecture_signal": encoded_signal,
            "feature_analysis": feature_analysis,
            "final_hidden": final_hidden.cpu().numpy(),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "encoder_status": "direct_extraction"
        }

    def _extract_features(
        self,
        final_hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor],
        text: str
    ) -> Dict[str, Any]:
        # 从 hidden states 提取多维度统计特征
        # 不依赖外部编码器，直接用张量运算

        # 全局统计
        hidden_stats = {
            "mean": final_hidden.mean().item(),
            "std": final_hidden.std().item(),
            "min": final_hidden.min().item(),
            "max": final_hidden.max().item(),
            "norm_mean": final_hidden.norm(dim=-1).mean().item(),
            "norm_std": final_hidden.norm(dim=-1).std().item(),
        }

        # 逐层特征（首、中、尾三层）
        layer_features = {}
        for layer_idx in self.extract_layers:
            h = layer_outputs[layer_idx]
            layer_features[f"layer_{layer_idx}"] = {
                "mean": h.mean().item(),
                "std": h.std().item(),
                "activation_sparsity": (h.abs() < 0.1).float().mean().item(),
                "norm_mean": h.norm(dim=-1).mean().item(),
            }

        # token 级别的激活分布
        token_norms = final_hidden.norm(dim=-1).squeeze(0)  # (seq_len,)
        top_active_tokens = torch.topk(token_norms, min(5, token_norms.shape[0]))

        # 层间变化率（捕捉信息流动）
        layer_deltas = {}
        for i in range(len(layer_outputs) - 1):
            delta = (layer_outputs[i + 1] - layer_outputs[i]).norm().item()
            layer_deltas[f"layer_{i}_to_{i+1}"] = delta

        # 输入特征
        input_features = {
            "text_length": len(text),
            "sequence_length": final_hidden.shape[1],
            "has_cjk": any('\u4e00' <= c <= '\u9fff' for c in text),
            "has_code_chars": any(c in text for c in '{}[]()=;'),
        }

        return {
            "hidden_stats": hidden_stats,
            "layer_features": layer_features,
            "top_active_positions": top_active_tokens.indices.tolist(),
            "top_active_values": [round(v, 3) for v in top_active_tokens.values.tolist()],
            "layer_deltas": layer_deltas,
            "input_features": input_features,
        }

    def _encode_features_to_signal(
        self,
        feature_analysis: Dict,
        original_text: str
    ) -> str:
        import json

        signal_parts = [
            "[架构信号]",
            "以下是由 CustomNeuralArchitecture 提取的数值特征分析：",
            "",
            "## Hidden States 统计",
            json.dumps(feature_analysis["hidden_stats"], ensure_ascii=False, indent=2),
            "",
            "## 逐层特征",
            json.dumps(feature_analysis["layer_features"], ensure_ascii=False, indent=2),
            "",
            "## 激活分布",
            f"Top 活跃位置: {feature_analysis['top_active_positions']}",
            f"Top 活跃值: {feature_analysis['top_active_values']}",
            "",
            "## 层间信息流动",
            json.dumps(feature_analysis["layer_deltas"], ensure_ascii=False, indent=2),
            "",
            "## 输入特征",
            json.dumps(feature_analysis["input_features"], ensure_ascii=False, indent=2),
            "",
            "[架构信号结束]"
        ]

        return "\n".join(signal_parts)

    def get_architecture_status(self) -> Dict[str, Any]:
        return {
            "architecture_name": "CustomNeuralArchitecture",
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "layers": len(self.model.blocks),
            "d_model": self.model.d_model,
            "extract_layers": self.extract_layers,
            "status": "active",
            "mode": "direct_feature_extraction"
        }
