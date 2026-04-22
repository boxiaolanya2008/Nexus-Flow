# 完整的 RecurrentModel 实现
import torch
import torch.nn as nn
from typing import Optional
from .prelude_coda import Prelude, Coda
from .recurrent_block import RecurrentBlock


class RecurrentModel(nn.Module):
    """
    完整的循环模型
    
    架构流程:
    1. Input tokens -> Token embedding + RoPE frequencies
    2. Prelude (2 transformer blocks with dense FFN)
    3. Frozen input injection (e)
    4. Recurrent block × T loops (shared weights)
       - Loop-index sinusoidal embedding
       - Attention (GQA or MLA)
       - MoE FFN (top-K routed)
       - LoRA adapter (depth-wise)
       - LTI injection (h = Ah + Be + out)
       - ACT halting (early exit per token)
    5. Coda (2 transformer blocks with dense FFN)
    6. RMSNorm + LM head (weight-tied to embedding)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
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
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # RoPE frequencies (integrated into attention)
        # RoPE is handled inside GQAAttention
        
        # Prelude
        self.prelude = Prelude(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout
        )
        
        # Recurrent block
        self.recurrent_block = RecurrentBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            max_loops=max_loops,
            num_experts=num_experts,
            top_k=top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_act=use_act,
            act_max_steps=act_max_steps,
            dropout=dropout
        )
        
        # Coda
        self.coda = Coda(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout
        )
        
        # Final RMSNorm
        self.norm = nn.RMSNorm(d_model)
        
        # LM head (weight-tied to embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        num_loops: int = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            num_loops: 循环次数（如果为 None，使用 max_loops）
            mask: 注意力掩码 (batch, 1, seq_len, seq_len)
        
        Returns:
            dict containing:
                logits: 输出 logits (batch, seq_len, vocab_size)
                halting_steps: ACT 停止步数（如果使用 ACT）
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # 保存冻结的输入嵌入用于 LTI injection
        e = x.detach().clone()
        
        # Prelude
        x = self.prelude(x, mask=mask)
        
        # Recurrent block with frozen input injection
        x, halting_steps = self.recurrent_block(
            x,
            e=e,
            num_loops=num_loops,
            mask=mask
        )
        
        # Coda
        x = self.coda(x, mask=mask)
        
        # Final norm
        x = self.norm(x)
        
        # LM head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return {
            'logits': logits,
            'halting_steps': halting_steps
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        num_loops: int = None,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            num_loops: 循环次数
            temperature: 采样温度
            top_k: top-k 采样
            eos_token_id: EOS token ID
        
        Returns:
            生成的 token IDs (batch, seq_len + new_tokens)
        """
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            current_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # 前向传播
                outputs = self.forward(current_ids, num_loops=num_loops)
                logits = outputs['logits'][:, -1, :]  # (batch, vocab_size)
                
                # 应用温度
                logits = logits / temperature
                
                # Top-k 采样
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, indices, values)
                
                # 采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 拼接
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # 检查 EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
            
            return current_ids
