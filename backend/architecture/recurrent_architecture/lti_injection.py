# LTI (Linear Transformer Injection) 实现
import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    """
    Linear Transformer Injection (LTI)
    注入公式: h = Ah + Be + out
    其中 h 是隐藏状态，e 是冻结的输入嵌入，A 和 B 是可学习参数
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 可学习的注入参数
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
    
    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        out: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 隐藏状态 (batch, seq_len, d_model)
            e: 冻结的输入嵌入 (batch, seq_len, d_model)
            out: 当前层的输出 (batch, seq_len, d_model)
        
        Returns:
            注入后的隐藏状态 (batch, seq_len, d_model)
        """
        # h = Ah + Be + out
        Ah = torch.matmul(h, self.A)
        Be = torch.matmul(e, self.B)
        
        h_new = Ah + Be + out
        return h_new
