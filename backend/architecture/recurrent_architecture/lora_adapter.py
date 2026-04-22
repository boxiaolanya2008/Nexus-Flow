# LoRA adapter (depth-wise) 实现
import torch
import torch.nn as nn
from typing import List


class LoRAAdapter(nn.Module):
    """
    LoRA (Low-Rank Adaptation) adapter
    深度-wise LoRA 适配器，用于参数高效微调
    """
    
    def __init__(
        self,
        d_model: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化 B 为零，这样初始时 LoRA 不影响原始输出
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
        
        Returns:
            LoRA 增量 (batch, seq_len, d_model)
        """
        # LoRA: B * A * x
        delta = torch.matmul(x, self.lora_A)  # (batch, seq_len, rank)
        delta = self.dropout(delta)
        delta = torch.matmul(delta, self.lora_B)  # (batch, seq_len, d_model)
        delta = delta * self.scaling
        
        return delta


class DepthWiseLoRA(nn.Module):
    """
    深度-wise LoRA 适配器
    为每个层单独的 LoRA 适配器
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 为每层创建 LoRA 适配器
        self.adapters = nn.ModuleList([
            LoRAAdapter(d_model, rank, alpha, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            layer_idx: 层索引
        
        Returns:
            LoRA 增量 (batch, seq_len, d_model)
        """
        return self.adapters[layer_idx](x)
