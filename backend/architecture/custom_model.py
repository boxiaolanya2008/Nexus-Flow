# 完整模型架构：自定义神经网络模型
import torch
import torch.nn as nn
from .custom_block import CustomBlock

class CustomModel(nn.Module):
    """
    完整的自定义神经网络模型
    替代传统 Transformer 架构
    优化编码能力和推理能力
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        window_size: int = 64,
        memory_size: int = 128,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        num_experts: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 嵌入层归一化
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 自定义架构块堆叠（使用改进的多专家版本）
        self.blocks = nn.ModuleList([
            CustomBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                memory_size=memory_size,
                d_ff=d_ff,
                dropout=dropout,
                num_experts=num_experts
            )
            for _ in range(n_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（词嵌入和输出投影共享权重）
        self.output_proj.weight = self.token_embedding.weight
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        token_emb = self.token_embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # 组合嵌入
        x = self.embedding_norm(token_emb + pos_emb)
        x = self.embedding_dropout(x)
        
        # 通过自定义架构块
        for block in self.blocks:
            x = block(x)
        
        # 最终层归一化
        x = self.final_norm(x)
        
        # 输出投影
        logits = self.output_proj(x)
        
        return logits
    
    def get_architecture_info(self) -> dict:
        """获取架构信息"""
        return {
            'architecture_name': 'Custom Neural Architecture',
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.blocks[0].attention.n_heads,
            'window_size': self.blocks[0].attention.window_size,
            'memory_size': self.blocks[0].memory.memory_size,
            'total_params': sum(p.numel() for p in self.parameters()),
            'components': [
                'Hybrid Attention (Linear + Window)',
                'Dynamic Memory Module',
                'Gated Feed-Forward Network'
            ]
        }
