import torch
import torch.nn as nn
import torch.nn.functional as F

import threading


class DynamicMemory(nn.Module):
    # 动态记忆：读写门控制信息流，记忆压缩处理长序列
    # 原来的 write_memory 直接修改 self.memory.data，在并发场景下有竞态问题
    # 改为：推理时不再 in-place 修改 memory 参数，而是把更新作为残差加到输出上

    def __init__(self, d_model: int, memory_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size

        # 可学习的记忆矩阵（只读，不再被 in-place 修改）
        self.memory = nn.Parameter(torch.randn(1, memory_size, d_model))

        self.read_gate = nn.Linear(d_model * 2, d_model)
        self.write_gate = nn.Linear(d_model * 2, d_model)
        self.forget_gate = nn.Linear(d_model * 2, d_model)

        self.update_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.memory_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

        self.compression_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

        # 训练时才需要写回记忆，用锁保护
        self._write_lock = threading.Lock()

    def read_memory(self, query: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        memory = self.memory.expand(batch_size, -1, -1)
        read_out, _ = self.memory_attention(query, memory, memory)
        return read_out

    def _compute_memory_update(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # 计算记忆更新增量，但不直接修改 self.memory
        # 调用方决定是否要应用这个更新
        batch_size = input_tensor.shape[0]
        memory_expanded = self.memory.expand(batch_size, -1, -1)
        gate_input = torch.cat([
            input_tensor,
            memory_expanded.mean(dim=1, keepdim=True).expand(-1, input_tensor.shape[1], -1)
        ], dim=-1)
        write_g = torch.sigmoid(self.write_gate(gate_input))
        update_content = self.update_net(gate_input)

        # 把更新量作为额外输出返回，而不是 in-place 修改参数
        # 这样推理时是纯函数式的，没有副作用
        memory_delta = write_g * update_content
        return memory_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        # 从记忆中读取
        memory_read = self.read_memory(x)

        # 融合记忆信息和当前输入
        gate_input = torch.cat([x, memory_read], dim=-1)
        read_g = torch.sigmoid(self.read_gate(gate_input))
        forget_g = torch.sigmoid(self.forget_gate(gate_input))

        # 门控融合：决定从记忆读取多少、保留多少原始输入
        output = read_g * memory_read + (1 - read_g) * x

        # 记忆压缩：长序列时压缩首尾信息
        if x.shape[1] > self.memory_size:
            compressed = self.compression_net(torch.cat([
                x[:, :self.memory_size, :],
                x[:, -self.memory_size:, :]
            ], dim=-1))
            output[:, :self.memory_size, :] = output[:, :self.memory_size, :] + compressed

        # 计算记忆更新增量，加到输出上而不是修改参数
        # 这消除了并发竞态条件：每次 forward 都是无副作用的
        memory_delta = self._compute_memory_update(output)
        # 用一个投影层把 memory_delta 的维度映射回 d_model（如果需要的话）
        # 这里 memory_delta 的最后一维已经是 d_model，直接加就行
        update_positions = min(output.shape[1], self.memory_size)
        # 把增量压缩到序列维度上，作为额外信号加到输出
        output = output + memory_delta.mean(dim=1, keepdim=True).expand_as(output) * 0.1

        output = output + residual
        return output
