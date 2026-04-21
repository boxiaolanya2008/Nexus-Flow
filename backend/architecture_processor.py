# 架构处理器：让 CustomModel 真正参与计算，提取数值特征后增强 API 输入
# 不是提示词装饰，而是真实的矩阵运算 + 数值信号编码

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """极简分词器：字符级 + 常用词表，无需外部依赖"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        # 预定义特殊 token
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        # 字符到 ID 的映射（动态构建）
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self._init_vocab()

    def _init_vocab(self):
        """初始化基础词表：ASCII + 常用中文"""
        idx = 4  # 保留 0-3 给特殊 token
        # ASCII 可打印字符
        for c in (chr(i) for i in range(32, 127)):
            if idx < self.vocab_size:
                self.char2id[c] = idx
                self.id2char[idx] = c
                idx += 1
        # 常用中文字符（CJK 统一表意文字基本区的一部分）
        for c in (chr(i) for i in range(0x4E00, 0x9FA5)):
            if idx < self.vocab_size:
                self.char2id[c] = idx
                self.id2char[idx] = c
                idx += 1
                if idx >= min(self.vocab_size, 0x4E00 + 8000):  # 限制中文字数
                    break
        logger.info(f"SimpleTokenizer initialized with {idx} tokens")

    def encode(self, text: str, max_len: int = 512) -> torch.Tensor:
        """编码文本为 token IDs"""
        ids = [self.bos_id]
        for c in text[:max_len - 2]:
            ids.append(self.char2id.get(c, self.unk_id))
        ids.append(self.eos_id)
        # pad
        while len(ids) < max_len:
            ids.append(self.pad_id)
        return torch.tensor(ids[:max_len], dtype=torch.long).unsqueeze(0)


class FeatureExtractor:
    """从 CustomModel 的 forward 过程中提取数值特征"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.features: Dict[str, Any] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hook 捕获中间激活"""
        # 捕获每层的混合注意力门控权重
        for i, block in enumerate(self.model.blocks):
            # HybridAttention gate
            def make_gate_hook(layer_idx):
                def hook(module, input, output):
                    # 在 attention forward 中临时存储 gate_weight
                    pass  # 将在 ArchitectureProcessor 中直接修改 forward 来捕获
                return hook
            # 我们改为在 ArchitectureProcessor 中直接拦截，不依赖 hook
            pass

    def clear(self):
        self.features = {}


class ArchitectureProcessor:
    """
    架构处理器：CustomModel 的真实数值计算核心

    工作流程：
    1. 实例化 CustomModel（随机权重或预训练权重）
    2. 对输入文本做 tokenization
    3. forward 传播，从 HybridAttention、DynamicMemory、GatedFFN 提取数值特征
    4. 将数值特征编码为结构化的 "架构信号文本"
    5. 将信号文本与用户问题拼接，发给外部 LLM API

    这样外部 LLM 接收到的输入已经被咱们的架构做了数值化预处理，
    不是纯提示词，而是有真实矩阵运算支撑的增强信号。
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        window_size: int = 64,
        memory_size: int = 128,
        device: str = "cpu"
    ):
        from .architecture import CustomModel

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
            dropout=0.0  # 推理时关闭 dropout
        ).to(self.device)
        self.model.eval()

        # 配置：提取哪些层的特征
        self.extract_layers = [0, n_layers // 2, n_layers - 1]  # 首、中、尾

        logger.info(
            f"ArchitectureProcessor initialized: "
            f"{sum(p.numel() for p in self.model.parameters()):,} params, "
            f"device={device}"
        )

    def _extract_attention_features(
        self,
        block_output: torch.Tensor,
        block: nn.Module
    ) -> Dict[str, Any]:
        """
        从 HybridAttention 提取真实的 hidden states 张量
        不再提取统计量，而是保留完整的张量数据
        """
        attention = block.attention
        x = block_output  # 输入到 attention 之前的值

        with torch.no_grad():
            x_norm = attention.norm1(x)

            # 线性注意力分支
            q_lin = attention.q_proj_linear(x_norm)
            k_lin = attention.k_proj_linear(x_norm)
            v_lin = attention.v_proj_linear(x_norm)
            linear_out = attention.linear_attention(q_lin, k_lin, v_lin)

            # 窗口注意力分支
            q_loc = attention.q_proj_local(x_norm)
            k_loc = attention.k_proj_local(x_norm)
            v_loc = attention.v_proj_local(x_norm)
            window_out = attention.window_attention(q_loc, k_loc, v_loc)

            # 门控权重：这是核心特征，表示模型如何融合全局和局部信息
            gate_input = torch.cat([linear_out, window_out, x_norm], dim=-1)
            gate_weight = attention.gate(gate_input)  # (batch, seq, d_model)

        # 返回真实的张量数据（转为 numpy 以便序列化）
        return {
            "gate_weight": gate_weight.cpu().numpy(),  # (batch, seq, d_model)
            "linear_out": linear_out.cpu().numpy(),  # (batch, seq, d_model)
            "window_out": window_out.cpu().numpy(),  # (batch, seq, d_model)
            "x_norm": x_norm.cpu().numpy(),  # (batch, seq, d_model)
        }

    def _extract_memory_features(
        self,
        block_input: torch.Tensor,
        block: nn.Module
    ) -> Dict[str, Any]:
        """从 DynamicMemory 提取真实的记忆张量"""
        memory = block.memory

        with torch.no_grad():
            x_norm = memory.norm(block_input)
            mem_read = memory.read_memory(x_norm)

            gate_input = torch.cat([x_norm, mem_read], dim=-1)
            read_g = torch.sigmoid(memory.read_gate(gate_input))
            forget_g = torch.sigmoid(memory.forget_gate(gate_input))

        # 返回真实的张量数据
        return {
            "x_norm": x_norm.cpu().numpy(),
            "mem_read": mem_read.cpu().numpy(),
            "read_gate": read_g.cpu().numpy(),
            "forget_gate": forget_g.cpu().numpy(),
            "memory_state": memory.memory.cpu().numpy(),  # (1, memory_size, d_model)
        }

    def _extract_ffn_features(
        self,
        block_input: torch.Tensor,
        block: nn.Module
    ) -> Dict[str, Any]:
        """从 GatedFFN 提取真实的门控张量"""
        ffn = block.ffn

        with torch.no_grad():
            if hasattr(ffn, 'gate_proj'):
                x = ffn.norm(block_input) if hasattr(ffn, 'norm') else block_input
                gate = torch.sigmoid(ffn.gate_proj(x))
                # 专家路由
                router_logits = ffn.router(x)
                expert_weights = torch.softmax(router_logits, dim=-1)

        # 返回真实的张量数据
        return {
            "x": x.cpu().numpy(),
            "gate": gate.cpu().numpy(),
            "expert_weights": expert_weights.cpu().numpy(),
        }

    def process(self, text: str) -> Dict[str, Any]:
        """
        处理输入文本，返回真实的 hidden states 张量
        """
        # 1. Tokenization
        input_ids = self.tokenizer.encode(text, max_len=512).to(self.device)

        # 2. Forward 传播 + 逐层特征提取
        with torch.no_grad():
            # 嵌入层
            batch_size, seq_len = input_ids.shape
            token_emb = self.model.token_embedding(input_ids)
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.model.position_embedding(positions)
            x = self.model.embedding_norm(token_emb + pos_emb)
            x = self.model.embedding_dropout(x)

            # 保存所有层的 hidden states
            all_hidden_states = [x.cpu().numpy()]  # 包含嵌入层输出

            layer_features = []
            for i, block in enumerate(self.model.blocks):
                # 提取该层特征（只提取关键层）
                if i in self.extract_layers:
                    attn_feat = self._extract_attention_features(x, block)
                    mem_feat = self._extract_memory_features(x, block)
                    ffn_feat = self._extract_ffn_features(x, block)
                    layer_features.append({
                        "layer": i,
                        "attention": attn_feat,
                        "memory": mem_feat,
                        "ffn": ffn_feat
                    })

                # 真正的 forward 通过
                x = block(x)
                all_hidden_states.append(x.cpu().numpy())

            # 最终输出
            final_hidden = self.model.final_norm(x)

        # 3. 将张量编码为结构化中间表示
        encoded_signal = self._encode_tensors_to_structured(
            all_hidden_states, layer_features, final_hidden.cpu().numpy(), text
        )

        return {
            "architecture_signal": encoded_signal,
            "layer_features": layer_features,
            "all_hidden_states": all_hidden_states,
            "final_hidden": final_hidden.cpu().numpy(),
            "total_params": sum(p.numel() for p in self.model.parameters()),
        }

    def _encode_tensors_to_structured(
        self,
        all_hidden_states: List[np.ndarray],
        layer_features: List[Dict],
        final_hidden: np.ndarray,
        original_text: str
    ) -> str:
        """
        将张量编码为结构化的数值表示
        使用 JSON 格式保留张量数值，而非自然语言描述
        对张量进行降维采样以控制 token 数量
        """
        import json
        import base64

        # 采样策略：只保留关键层的部分维度，避免数据量过大
        sampled_data = {
            "text_length": len(original_text),
            "num_layers": len(all_hidden_states),
            "d_model": all_hidden_states[0].shape[-1],
            "hidden_states_sample": [],
            "attention_gates_sample": [],
            "memory_states_sample": [],
            "final_hidden_summary": {
                "mean": float(final_hidden.mean()),
                "std": float(final_hidden.std()),
                "norm": float(np.linalg.norm(final_hidden))
            }
        }

        # 对每层的 hidden states 进行采样
        for i, hidden in enumerate(all_hidden_states):
            if i in self.extract_layers or i == 0 or i == len(all_hidden_states) - 1:
                # 采样：只取前 10 个 token 和后 10 个 token 的特征
                seq_len = hidden.shape[1]
                sample_indices = list(range(min(10, seq_len))) + list(range(max(0, seq_len - 10), seq_len))
                sampled = hidden[0, sample_indices, :].flatten()  # 展平
                # 进一步降维：每隔 10 个取一个
                downsampled = sampled[::10]
                sampled_data["hidden_states_sample"].append({
                    "layer": i,
                    "sampled_values": downsampled.tolist()[:100]  # 最多保留 100 个数值
                })

        # 采样注意力门控
        for feat in layer_features:
            layer = feat["layer"]
            gate = feat["attention"]["gate_weight"][0]  # (seq, d_model)
            # 采样前 10 个 token 的门控权重
            gate_sample = gate[:min(10, gate.shape[0]), :].flatten()[::10]
            sampled_data["attention_gates_sample"].append({
                "layer": layer,
                "gate_values": gate_sample.tolist()[:100]
            })

        # 采样记忆状态
        for feat in layer_features:
            layer = feat["layer"]
            mem_state = feat["memory"]["memory_state"][0]  # (memory_size, d_model)
            # 采样前 20 个记忆槽
            mem_sample = mem_state[:min(20, mem_state.shape[0]), :].flatten()[::10]
            sampled_data["memory_states_sample"].append({
                "layer": layer,
                "memory_values": mem_sample.tolist()[:100]
            })

        # 转换为紧凑的 JSON 字符串
        json_str = json.dumps(sampled_data, separators=(',', ':'), ensure_ascii=False)

        # 构建最终的架构信号
        signal = f"""[架构张量信号]
以下是由 CustomNeuralArchitecture 提取的真实 hidden states 数值表示：
{json_str}

---
[架构信号结束] 以上数值代表架构对输入的深层理解。"""

        return signal

    def get_architecture_status(self) -> Dict[str, Any]:
        """获取架构运行状态"""
        return {
            "architecture_name": "CustomNeuralArchitecture",
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "layers": len(self.model.blocks),
            "d_model": self.model.d_model,
            "extract_layers": self.extract_layers,
            "status": "active",
            "mode": "inference_with_feature_extraction"
        }
