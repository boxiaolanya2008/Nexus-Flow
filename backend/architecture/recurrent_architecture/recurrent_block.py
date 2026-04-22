# Recurrent block 实现
import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .loop_embedding import LoopIndexEmbedding
from .lora_adapter import DepthWiseLoRA
from .lti_injection import LTIInjection
from .act_halting import ACTRecurrentBlock


class RecurrentBlock(nn.Module):
    """
    Recurrent block with T loops
    共享权重，在推理时循环执行
    包含：Attention, MoE FFN, LoRA adapter, LTI injection, ACT halting
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 8192,
        rope_base: int = 10000,
        max_loops: int = 32,
        num_experts: int = 8,
        top_k: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        use_act: bool = True,
        act_max_steps: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_loops = max_loops
        self.use_act = use_act
        
        # Loop-index embedding
        self.loop_embedding = LoopIndexEmbedding(d_model, max_loops)
        
        # Transformer block (shared weights)
        self.transformer_block = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout,
            use_moe=True,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # LoRA adapter (depth-wise)
        self.lora_adapter = DepthWiseLoRA(
            d_model=d_model,
            num_layers=max_loops,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=dropout
        )
        
        # LTI injection
        self.lti_injection = LTIInjection(d_model)
        
        # ACT halting
        if use_act:
            self.act_recurrent = ACTRecurrentBlock(
                d_model=d_model,
                max_steps=act_max_steps
            )
        
        self.norm = nn.RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        num_loops: int = None,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            e: 冻结的输入嵌入 (batch, seq_len, d_model)
            num_loops: 循环次数（如果为 None，使用 max_loops）
            mask: 注意力掩码
        
        Returns:
            output: 输出张量 (batch, seq_len, d_model)
            halting_steps: ACT 停止步数（如果使用 ACT）
        """
        if num_loops is None:
            num_loops = self.max_loops
        
        batch_size, seq_len, d_model = x.shape
        h = x
        
        # 定义循环内的块函数
        def loop_block(h_current, loop_idx):
            # 添加 loop embedding
            loop_emb = self.loop_embedding(loop_idx).unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            h_current = h_current + loop_emb
            
            # Transformer block
            h_current, _ = self.transformer_block(h_current, mask=mask)
            
            # LoRA adapter
            lora_delta = self.lora_adapter(h_current, loop_idx)
            h_current = h_current + lora_delta
            
            # LTI injection
            h_current = self.lti_injection(h_current, e, h_current)
            
            return h_current
        
        # 执行循环
        if self.use_act:
            # 使用 ACT halting
            output, halting_steps = self.act_recurrent(
                h,
                loop_block
            )
        else:
            # 固定循环次数
            output = h
            halting_steps = None
            for loop_idx in range(num_loops):
                output = loop_block(output, loop_idx)
        
        output = self.norm(output)
        
        return output, halting_steps
