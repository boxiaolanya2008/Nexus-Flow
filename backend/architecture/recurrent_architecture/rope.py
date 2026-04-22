# RoPE (Rotary Position Embedding) 实现
import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    通过旋转位置编码将位置信息注入到查询和键向量中
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算位置编码
        self._build_positional_encoding(max_seq_len)
    
    def _build_positional_encoding(self, seq_len: int):
        """构建位置编码"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int = None) -> tuple:
        """
        应用 RoPE 到查询和键

        Args:
            q: 查询张量 (batch, n_heads, seq_len, head_dim)
            k: 键张量 (batch, n_heads, seq_len, head_dim)
            seq_len: 序列长度（如果与张量长度不同）

        Returns:
            旋转后的 q, k
        """
        if seq_len is None:
            seq_len = q.shape[-2]

        # 如果需要，扩展缓存
        if seq_len > self.max_seq_len:
            self._build_positional_encoding(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # 应用旋转
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        return q_rot, k_rot
    
    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """应用旋转操作"""
        # 将 x 分为两半
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]

        # 将 cos/sin 也分成两半
        cos1, cos2 = cos[..., :cos.shape[-1]//2], cos[..., cos.shape[-1]//2:]
        sin1, sin2 = sin[..., :sin.shape[-1]//2], sin[..., sin.shape[-1]//2:]

        # 确保 cos/sin 的形状与 x 匹配
        # cos/sin: (1, 1, seq_len, head_dim/2)
        # x: (batch, n_heads, seq_len, head_dim/2)
        # 广播到 (batch, n_heads, seq_len, head_dim/2)
        cos1 = cos1[:, :, :x.shape[-2], :]
        cos2 = cos2[:, :, :x.shape[-2], :]
        sin1 = sin1[:, :, :x.shape[-2], :]
        sin2 = sin2[:, :, :x.shape[-2], :]

        # 旋转
        x_rot = torch.cat([
            x1 * cos1 - x2 * sin2,
            x1 * sin1 + x2 * cos2
        ], dim=-1)

        return x_rot
