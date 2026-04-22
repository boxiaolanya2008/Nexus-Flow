# Transformer block 实现
import torch
import torch.nn as nn
from .gqa_attention import GQAAttention
from .swiglu_ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    """
    基础 Transformer block
    包含 GQA Attention 和 SwiGLU FFN
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 8192,
        rope_base: int = 10000,
        dropout: float = 0.0,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.use_moe = use_moe
        
        # Attention
        self.attention = GQAAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout
        )
        
        # FFN
        if use_moe:
            from .moe_ffn import MoEFFN
            self.ffn = MoEFFN(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
        else:
            self.ffn = SwiGLUFFN(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout
            )
        
        # Layer norms
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        use_kv_cache: bool = False,
        kv_cache: tuple = None
    ) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            mask: 注意力掩码
            use_kv_cache: 是否使用 KV cache
            kv_cache: KV cache
        
        Returns:
            output: 输出张量 (batch, seq_len, d_model)
            new_kv_cache: 新的 KV cache
        """
        residual = x
        
        # Attention
        x_norm = self.norm1(x)
        attn_out, new_kv_cache = self.attention(
            x_norm,
            mask=mask,
            use_kv_cache=use_kv_cache,
            kv_cache=kv_cache
        )
        x = residual + self.dropout(attn_out)
        
        # FFN
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out)
        
        return x, new_kv_cache
