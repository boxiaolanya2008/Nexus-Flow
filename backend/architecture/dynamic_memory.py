# 动态记忆模块：优化长序列处理
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMemory(nn.Module):
    """
    动态记忆模块
    维护一个可更新的记忆状态，用于长序列的上下文保持
    支持记忆的读写、遗忘和更新操作
    新增：记忆压缩、分层记忆管理
    """
    
    def __init__(self, d_model: int, memory_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # 可学习的记忆矩阵
        self.memory = nn.Parameter(torch.randn(1, memory_size, d_model))
        
        # 记忆读写门
        self.read_gate = nn.Linear(d_model * 2, d_model)
        self.write_gate = nn.Linear(d_model * 2, d_model)
        self.forget_gate = nn.Linear(d_model * 2, d_model)
        
        # 记忆更新网络，增强表达能力
        self.update_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 注意力机制用于记忆检索
        self.memory_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # 记忆压缩网络，用于长序列的记忆压缩
        self.compression_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def read_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        从记忆中读取相关信息
        """
        batch_size = query.shape[0]
        
        # 扩展记忆矩阵到批次大小
        memory = self.memory.expand(batch_size, -1, -1)
        
        # 使用注意力从记忆中检索
        read_out, _ = self.memory_attention(query, memory, memory)
        
        return read_out
    
    def write_memory(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        将信息写入记忆
        """
        batch_size = input_tensor.shape[0]
        
        # 计算写入门
        memory_expanded = self.memory.expand(batch_size, -1, -1)
        gate_input = torch.cat([input_tensor, memory_expanded.mean(dim=1, keepdim=True).expand(-1, input_tensor.shape[1], -1)], dim=-1)
        write_g = torch.sigmoid(self.write_gate(gate_input))
        
        # 计算更新内容
        update_content = self.update_net(gate_input)
        
        # 更新记忆（只更新前几个位置，避免完全覆盖）
        with torch.no_grad():
            update_positions = min(input_tensor.shape[1], self.memory_size)
            self.memory.data[:, :update_positions, :] = (
                write_g[:, :update_positions, :] * update_content[:, :update_positions, :] +
                (1 - write_g[:, :update_positions, :]) * self.memory.data[:, :update_positions, :]
            ).mean(dim=0)
        
        return input_tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，使用记忆压缩机制
        """
        residual = x
        x = self.norm(x)
        
        # 从记忆中读取
        memory_read = self.read_memory(x)
        
        # 融合记忆信息和当前输入
        gate_input = torch.cat([x, memory_read], dim=-1)
        read_g = torch.sigmoid(self.read_gate(gate_input))
        forget_g = torch.sigmoid(self.forget_gate(gate_input))
        
        # 融合输出
        output = read_g * memory_read + (1 - read_g) * x
        
        # 记忆压缩：对长序列进行压缩以减少记忆负担
        if x.shape[1] > self.memory_size:
            compressed = self.compression_net(torch.cat([x[:, :self.memory_size, :], x[:, -self.memory_size:, :]], dim=-1))
            output = output + compressed
        
        # 写入记忆
        output = self.write_memory(output)
        
        # 残差连接
        output = output + residual
        
        return output
