# Loop-index sinusoidal embedding 实现
import torch
import torch.nn as nn
import math


class LoopIndexEmbedding(nn.Module):
    """
    Loop-index sinusoidal embedding
    为循环层提供基于循环索引的正弦位置编码
    """
    
    def __init__(self, d_model: int, max_loops: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_loops = max_loops
        
        # 预计算循环索引编码
        self._build_loop_encoding(max_loops)
    
    def _build_loop_encoding(self, max_loops: int):
        """构建循环索引编码"""
        loop_idx = torch.arange(max_loops).unsqueeze(1)  # (max_loops, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        pe = torch.zeros(max_loops, self.d_model)
        pe[:, 0::2] = torch.sin(loop_idx * div_term)
        pe[:, 1::2] = torch.cos(loop_idx * div_term)
        
        self.register_buffer('loop_encoding', pe)
    
    def forward(self, loop_idx: int) -> torch.Tensor:
        """
        获取指定循环索引的编码
        
        Args:
            loop_idx: 循环索引 (0 到 max_loops-1)
        
        Returns:
            编码向量 (d_model,)
        """
        if loop_idx >= self.max_loops:
            # 如果超出范围，扩展编码
            self._build_loop_encoding(loop_idx + 1)
            self.max_loops = loop_idx + 1
        
        return self.loop_encoding[loop_idx]
    
    def forward_batch(self, loop_indices: torch.Tensor) -> torch.Tensor:
        """
        批量获取循环索引编码
        
        Args:
            loop_indices: 循环索引张量 (batch,)
        
        Returns:
            编码张量 (batch, d_model)
        """
        max_idx = loop_indices.max().item()
        if max_idx >= self.max_loops:
            self._build_loop_encoding(max_idx + 1)
            self.max_loops = max_idx + 1
        
        return self.loop_encoding[loop_indices]
