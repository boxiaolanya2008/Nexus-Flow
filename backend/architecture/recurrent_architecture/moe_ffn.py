# MoE FFN (top-K routed) 实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MoEExpert(nn.Module):
    """MoE 专家网络"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN with top-K routing
    使用 top-K 路由选择专家
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            MoEExpert(d_model, d_ff) for _ in range(num_experts)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
        
        Returns:
            输出张量 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 计算门控分数
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Top-K 选择
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(gate_logits, dim=-1),
            self.top_k,
            dim=-1
        )  # (batch, seq_len, top_k)
        
        # 归一化权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 对每个专家进行处理
        for expert_idx in range(self.num_experts):
            # 找到使用该专家的位置
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (batch, seq_len)
            
            if not expert_mask.any():
                continue
            
            # 获取对应的输入和权重
            expert_input = x[expert_mask]  # (n_selected, d_model)
            expert_weights = top_k_weights[expert_mask]  # (n_selected, top_k)
            
            # 找到该专家在 top_k 中的位置
            expert_weight_indices = (top_k_indices[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]
            expert_weights = expert_weights[torch.arange(len(expert_weights)), expert_weight_indices].unsqueeze(-1)
            
            # 专家计算
            expert_output = self.experts[expert_idx](expert_input)
            
            # 加权累加
            weighted_output = expert_output * expert_weights
            output[expert_mask] += weighted_output.squeeze(-1)
        
        output = self.dropout(output)
        return output
