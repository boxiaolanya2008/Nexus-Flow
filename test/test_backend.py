import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.architecture import CustomModel


def test_architecture():
    """测试自定义架构"""
    print("测试自定义架构...")
    try:
        model = CustomModel(
            vocab_size=50000,
            d_model=512,
            n_layers=6,
            n_heads=8
        )
        info = model.get_architecture_info()
        print(f"架构信息: {info}")
        assert info['total_params'] > 0
        assert info['d_model'] == 512
        assert info['n_layers'] == 6
        print("自定义架构测试成功")
        return True
    except Exception as e:
        print(f"自定义架构测试失败: {e}")
        return False


def test_forward_pass():
    # 前向传播是模型能跑通的基本验证，不能跳过
    print("测试前向传播...")
    try:
        model = CustomModel(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 32, 1000)
        print("前向传播测试成功")
        return True
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始后端测试...")

    arch_result = test_architecture()
    forward_result = test_forward_pass()

    if arch_result and forward_result:
        print("\n所有测试通过")
    else:
        print("\n部分测试失败")
