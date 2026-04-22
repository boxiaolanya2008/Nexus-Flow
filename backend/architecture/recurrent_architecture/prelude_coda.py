# Prelude 和 Coda 共享同一个结构：2 层 dense Transformer blocks
# 只是语义上分为"前处理"和"后处理"，实现完全一致

import torch
import torch.nn as nn
from .transformer_block import TransformerBlock


class TransformerStack(nn.Module):
    # 可复用的 Transformer 块堆叠，Prelude 和 Coda 都是它的实例
    # 以后如果要改层数或者加残差，改这里一处就行

    def __init__(
        self,
        n_blocks: int,
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
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                dropout=dropout,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k=top_k
            )
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        for block in self.blocks:
            x, _ = block(x, mask=mask)
        return x


class Prelude(TransformerStack):
    # 循环之前的前处理，2 层 dense FFN
    # 语义上叫 Prelude 方便理解模型架构，实现就是 TransformerStack

    def __init__(self, d_model, n_heads=32, n_kv_heads=8, d_ff=None,
                 max_seq_len=8192, rope_base=10000, dropout=0.0):
        super().__init__(
            n_blocks=2, d_model=d_model, n_heads=n_heads,
            n_kv_heads=n_kv_heads, d_ff=d_ff, max_seq_len=max_seq_len,
            rope_base=rope_base, dropout=dropout, use_moe=False
        )


class Coda(TransformerStack):
    # 循环之后的后处理，2 层 dense FFN
    # 和 Prelude 结构完全一样，只是语义不同

    def __init__(self, d_model, n_heads=32, n_kv_heads=8, d_ff=None,
                 max_seq_len=8192, rope_base=10000, dropout=0.0):
        super().__init__(
            n_blocks=2, d_model=d_model, n_heads=n_heads,
            n_kv_heads=n_kv_heads, d_ff=d_ff, max_seq_len=max_seq_len,
            rope_base=rope_base, dropout=dropout, use_moe=False
        )
