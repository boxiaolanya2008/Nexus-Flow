# 测试循环架构
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from backend.architecture.recurrent_architecture import RecurrentModel


def test_recurrent_model():
    """测试循环模型的基本功能"""
    print("Testing RecurrentModel...")
    
    # 模型参数
    vocab_size = 1000
    d_model = 128
    n_heads = 8
    n_kv_heads = 4
    max_loops = 4
    batch_size = 2
    seq_len = 16
    
    # 创建模型
    model = RecurrentModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_loops=max_loops,
        use_act=False  # 先禁用 ACT 简化测试
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 测试前向传播
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # 前向传播
    outputs = model(input_ids, num_loops=2)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    assert outputs['logits'].shape == (batch_size, seq_len, vocab_size), "Output shape mismatch!"
    
    print("Forward pass test passed!")
    
    # 测试生成
    print("\nTesting generation...")
    generated = model.generate(
        input_ids=input_ids[:, :8],
        max_new_tokens=5,
        num_loops=2,
        temperature=1.0,
        top_k=10
    )
    
    print(f"Generated shape: {generated.shape}")
    print(f"Expected shape: ({batch_size}, {8 + 5})")
    
    assert generated.shape == (batch_size, 13), "Generation shape mismatch!"
    
    print("Generation test passed!")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_recurrent_model()
