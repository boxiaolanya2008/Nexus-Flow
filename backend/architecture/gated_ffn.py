# 门控前馈网络：替代传统前馈网络
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFFN(nn.Module):
    """
    门控前馈网络
    使用门控机制替代传统的前馈网络，提升非线性表达能力和推理效率
    采用 GLU (Gated Linear Unit) 变体
    新增：更多专家、改进路由机制
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1, num_experts: int = 4):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 默认隐藏层维度为 4 倍模型维度
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        
        # 门控线性单元
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        
        # 专家混合机制（扩展到多个专家）
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_ff) for _ in range(num_experts)
        ])
        self.expert_gate = nn.Linear(d_model, num_experts)
        
        # 专家路由网络，用于更智能的专家选择
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用改进的专家路由机制
        """
        residual = x
        x = self.norm(x)
        
        # 使用路由网络计算专家权重
        router_logits = self.router(x)
        expert_weights = F.softmax(router_logits, dim=-1)
        
        # 计算所有专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))
        
        # 加权混合专家输出
        mixed_expert = sum(
            expert_weights[:, :, i:i+1] * expert_outputs[i]
            for i in range(self.num_experts)
        )
        
        # 门控线性单元
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        
        # 门控激活
        gated_output = gate * up
        
        # 混合专家输出
        output = gated_output + mixed_expert
        
        # 下投影
        output = self.down_proj(output)
        
        # Dropout 和残差连接
        output = self.dropout(output) + residual
        
        return output
