# GQA (Grouped Query Attention) 实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .rope import RoPE


class GQAAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    通过分组查询头来减少 KV cache 的内存占用
    每组查询头共享相同的键值头
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        max_seq_len: int = 8192,
        rope_base: int = 10000,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads
        
        # 确保 head_dim 是整数
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        # 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RoPE(self.head_dim, max_seq_len, rope_base)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        kv_cache: Optional[tuple] = None
    ) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            mask: 注意力掩码 (batch, 1, seq_len, seq_len)
            use_kv_cache: 是否使用 KV cache
            kv_cache: 缓存的 k, v
        
        Returns:
            output: 输出张量 (batch, seq_len, d_model)
            new_kv_cache: 新的 KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dim)
        
        # 应用 RoPE
        q, k = self.rope(q, k, seq_len)
        
        # KV cache 处理
        if use_kv_cache and kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_kv_cache = (k, v) if use_kv_cache else None
        
        # 扩展 KV 以匹配查询头数
        # (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # 应用掩码
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)
        
        # 合并头
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        output = output.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        # 输出投影
        output = self.o_proj(output)
        
        return output, new_kv_cache
