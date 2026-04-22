# SwiGLU FFN 实现
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) FFN
    使用 Swish 激活函数的门控线性单元
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # SwiGLU 使用三个投影
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
        
        Returns:
            输出张量 (batch, seq_len, d_model)
        """
        # SwiGLU: Swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        
        # 下投影
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output
