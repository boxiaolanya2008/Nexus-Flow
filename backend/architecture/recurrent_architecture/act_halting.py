# ACT (Adaptive Computation Time) halting 实现
import torch
import torch.nn as nn


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time (ACT) halting mechanism
    允许每个 token 在不同深度提前退出
    """
    
    def __init__(self, d_model: int, max_steps: int = 10, epsilon: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.epsilon = epsilon
        
        # 停止预测器
        self.halt_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        step: int
    ) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            step: 当前步数
        
        Returns:
            halt_prob: 停止概率 (batch, seq_len, 1)
            should_halt: 是否应该停止 (batch, seq_len)
        """
        # 预测停止概率
        halt_prob = self.halt_predictor(x)  # (batch, seq_len, 1)
        
        # 如果步数达到最大值，强制停止
        if step >= self.max_steps - 1:
            should_halt = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            # 根据概率决定是否停止
            should_halt = (halt_prob.squeeze(-1) > self.epsilon)
        
        return halt_prob, should_halt


class ACTRecurrentBlock(nn.Module):
    """
    带有 ACT halting 的循环块
    支持每个 token 的自适应计算深度
    """
    
    def __init__(
        self,
        d_model: int,
        max_steps: int = 10,
        epsilon: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.epsilon = epsilon
        
        self.act_halting = ACTHalting(d_model, max_steps, epsilon)
    
    def forward(
        self,
        x: torch.Tensor,
        block_fn: callable,
        *block_args
    ) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            block_fn: 要重复执行的块函数
            *block_args: 块函数的额外参数
        
        Returns:
            output: 输出张量 (batch, seq_len, d_model)
            halting_steps: 每个token的停止步数 (batch, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 初始化
        output = x
        halting_probs = torch.zeros(batch_size, seq_len, 1, device=x.device)
        halting_steps = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        
        # 记录哪些 token 已经停止
        halted = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        for step in range(self.max_steps):
            # 只对未停止的 token 进行计算
            if halted.all():
                break
            
            # 执行块
            block_output = block_fn(output, *block_args)
            
            # 预测停止概率
            halt_prob, should_halt = self.act_halting(block_output, step)
            
            # 更新停止概率
            halting_probs = halting_probs + halt_prob * (~halted).unsqueeze(-1).float()
            
            # 更新停止状态
            new_halted = should_halt & (~halted)
            halted = halted | new_halted
            
            # 记录停止步数
            halting_steps[new_halted] = step + 1
            
            # 更新输出（加权平均）
            # 对于已停止的 token，保持之前的输出
            # 对于未停止的 token，使用新的输出
            output = torch.where(
                halted.unsqueeze(-1),
                output,
                block_output
            )
        
        return output, halting_steps
