# 架构处理器：CustomModel 的真实数值计算 + 语义编码
# hidden states 经过 SemanticEncoder 映射到可解释的语义空间后注入 API 请求

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import os

from .architecture.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class ArchitectureProcessor:
    # 工作流：tokenize -> CustomModel forward -> SemanticEncoder -> 结构化信号 -> 注入 LLM 提示词
    # 这样外部 LLM 收到的是经过数值化预处理的输入，而非原始文本

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        window_size: int = 64,
        memory_size: int = 128,
        device: str = "cpu",
        semantic_encoder_path: str = None
    ):
        from .architecture import CustomModel
        from .architecture.semantic_encoder import SemanticEncoder, SemanticCodeAnalyzer

        self.tokenizer = SimpleTokenizer(vocab_size=50000)
        self.device = torch.device(device)
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

        self.semantic_encoder = SemanticEncoder(
            d_model=d_model,
            semantic_dim=64
        ).to(self.device)

        if semantic_encoder_path is None:
            semantic_encoder_path = os.path.join(
                os.path.dirname(__file__),
                "architecture",
                "semantic_encoder.pt"
            )

        if os.path.exists(semantic_encoder_path):
            try:
                checkpoint = torch.load(semantic_encoder_path, map_location=self.device)
                self.semantic_encoder.load_state_dict(checkpoint["encoder_state_dict"])
                self.semantic_encoder.eval()
                logger.info(f"SemanticEncoder loaded from {semantic_encoder_path}")
                self.using_trained_encoder = True
            except Exception as e:
                logger.warning(f"Failed to load SemanticEncoder: {e}")
                self.using_trained_encoder = False
        else:
            logger.warning(f"SemanticEncoder checkpoint not found at {semantic_encoder_path}")
            logger.warning("Using untrained SemanticEncoder - semantic signals will be random")
            self.using_trained_encoder = False

        self.semantic_analyzer = SemanticCodeAnalyzer(self.semantic_encoder)
        self.extract_layers = [0, n_layers // 2, n_layers - 1]

        logger.info(
            f"ArchitectureProcessor initialized: "
            f"{sum(p.numel() for p in self.model.parameters()):,} params, "
            f"device={device}, "
            f"semantic_encoder={'trained' if self.using_trained_encoder else 'untrained'}"
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

            for block in self.model.blocks:
                x = block(x)

            final_hidden = self.model.final_norm(x)

        semantic_vector = self.semantic_encoder(final_hidden)
        semantic_analysis = self.semantic_analyzer.analyze(final_hidden, text)
        encoded_signal = self._encode_semantic_to_signal(semantic_analysis, text)

        return {
            "architecture_signal": encoded_signal,
            "semantic_vector": semantic_vector.cpu().numpy().tolist(),
            "semantic_analysis": semantic_analysis,
            "final_hidden": final_hidden.cpu().numpy(),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "encoder_status": "trained" if self.using_trained_encoder else "untrained"
        }

    def _encode_semantic_to_signal(
        self,
        semantic_analysis: Dict,
        original_text: str
    ) -> str:
        import json

        signal_parts = [
            "[架构语义信号]",
            "以下是由 CustomNeuralArchitecture 提取的代码语义分析：",
            "",
            "## 语义向量表示",
            json.dumps(semantic_analysis["semantic_vector"], ensure_ascii=False, indent=2),
            "",
            "## 关键特征",
        ]

        for feature_name, value in semantic_analysis["top_features"]:
            signal_parts.append(f"  - {feature_name}: {value:.3f}")

        signal_parts.extend([
            "",
            "## 结构分析",
            json.dumps(semantic_analysis["structure_analysis"], ensure_ascii=False, indent=2),
            "",
            "## 质量评估",
            json.dumps(semantic_analysis["quality_assessment"], ensure_ascii=False, indent=2),
            "",
            "## 代码类型推断",
            json.dumps(semantic_analysis["inferred_type"], ensure_ascii=False, indent=2),
            "",
            "## 复杂度评估",
            json.dumps(semantic_analysis["complexity_assessment"], ensure_ascii=False, indent=2),
            "",
            "## 分析摘要",
            semantic_analysis["summary"],
            "",
            f"[语义编码器状态: {'已训练' if self.using_trained_encoder else '未训练'}]",
            "[架构语义信号结束]"
        ])

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
            "mode": "semantic_encoding",
            "semantic_encoder": {
                "status": "trained" if self.using_trained_encoder else "untrained",
                "semantic_dim": self.semantic_encoder.semantic_dim,
                "dimensions": self.semantic_encoder.SEMANTIC_DIMENSIONS[:self.semantic_encoder.semantic_dim]
            }
        }
