# 自定义架构基础块：组合所有核心组件
import torch
import torch.nn as nn
from .hybrid_attention import HybridAttention
from .dynamic_memory import DynamicMemory
from .gated_ffn import GatedFFN

class CustomBlock(nn.Module):
    """
    自定义架构基础块
    组合混合注意力、动态记忆和门控前馈网络
    形成一个完整的处理单元
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 64,
        memory_size: int = 128,
        d_ff: int = None,
        dropout: float = 0.1,
        num_experts: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        
        # 混合注意力层
        self.attention = HybridAttention(d_model, n_heads, window_size)
        
        # 动态记忆层
        self.memory = DynamicMemory(d_model, memory_size)
        
        # 门控前馈网络（使用改进的多专家版本）
        self.ffn = GatedFFN(d_model, d_ff, dropout, num_experts)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 混合注意力
        x = self.attention(x)
        
        # 动态记忆
        x = self.memory(x)
        
        # 门控前馈网络
        x = self.ffn(x)
        
        return x
