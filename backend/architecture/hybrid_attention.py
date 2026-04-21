# 混合注意力机制：结合线性注意力和局部窗口注意力
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class HybridAttention(nn.Module):
    """
    混合注意力机制
    结合线性注意力的全局上下文和局部窗口注意力的细节捕捉
    优化长序列处理，降低计算复杂度从 O(n²) 到 O(n)
    新增：相对位置编码、自适应门控权重
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        # 线性注意力参数
        self.q_proj_linear = nn.Linear(d_model, d_model)
        self.k_proj_linear = nn.Linear(d_model, d_model)
        self.v_proj_linear = nn.Linear(d_model, d_model)
        
        # 局部窗口注意力参数
        self.q_proj_local = nn.Linear(d_model, d_model)
        self.k_proj_local = nn.Linear(d_model, d_model)
        self.v_proj_local = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 自适应门控机制，根据输入动态调整线性注意力和局部注意力的权重
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 相对位置编码
        self.relative_pos_bias = nn.Parameter(torch.randn(2 * window_size + 1, n_heads))
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        线性注意力，计算复杂度 O(n)
        使用特征映射避免显式计算注意力矩阵
        """
        # 使用 ELU + 1 作为特征映射
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 计算 QK^T 和 KV
        batch_size, seq_len, d_model = q.shape
        q = q.transpose(0, 1)  # (seq_len, batch, d_model)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # KV 聚合
        kv = torch.einsum('sbd,sbc->bdc', k, v)
        
        # Q * KV
        output = torch.einsum('sbd,bdc->sbc', q, kv)
        
        # 归一化
        k_sum = torch.sum(k, dim=0)
        output = output / (k_sum.unsqueeze(-1) + 1e-6)
        
        output = output.transpose(0, 1)  # (batch, seq_len, d_model)
        return output
    
    def window_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        局部窗口注意力，捕捉局部细节
        使用滑动窗口机制和相对位置编码
        """
        batch_size, seq_len, d_model = q.shape
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算窗口内的注意力
        outputs = []
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            q_i = q[:, :, i:i+1, :]  # (batch, n_heads, 1, head_dim)
            k_win = k[:, :, start:end, :]  # (batch, n_heads, window, head_dim)
            v_win = v[:, :, start:end, :]
            
            # 计算注意力分数
            scores = torch.matmul(q_i, k_win.transpose(-2, -1)) / sqrt(self.head_dim)
            
            # 添加相对位置编码
            rel_pos = torch.arange(start, end, device=q.device) - i
            rel_pos_idx = rel_pos + self.window_size
            rel_pos_bias = self.relative_pos_bias[rel_pos_idx].unsqueeze(0).unsqueeze(0)
            scores = scores + rel_pos_bias.transpose(-2, -1)
            
            attn_weights = F.softmax(scores, dim=-1)
            
            # 加权求和
            output = torch.matmul(attn_weights, v_win)
            outputs.append(output)
        
        # 拼接输出
        output = torch.cat(outputs, dim=2)  # (batch, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        output = output.view(batch_size, seq_len, d_model)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用改进的自适应门控机制
        """
        residual = x
        x = self.norm1(x)
        
        # 线性注意力分支
        q_linear = self.q_proj_linear(x)
        k_linear = self.k_proj_linear(x)
        v_linear = self.v_proj_linear(x)
        linear_out = self.linear_attention(q_linear, k_linear, v_linear)
        
        # 局部窗口注意力分支
        q_local = self.q_proj_local(x)
        k_local = self.k_proj_local(x)
        v_local = self.v_proj_local(x)
        window_out = self.window_attention(q_local, k_local, v_local)
        
        # 自适应门控融合，使用原始输入作为额外信息
        gate_input = torch.cat([linear_out, window_out, x], dim=-1)
        gate_weight = self.gate(gate_input)
        output = gate_weight * linear_out + (1 - gate_weight) * window_out
        
        # 残差连接
        output = self.out_proj(output) + residual
        
        return output
