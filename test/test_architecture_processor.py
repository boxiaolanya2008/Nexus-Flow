import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.architecture_processor import ArchitectureProcessor


def test_processor():
    print("初始化 ArchitectureProcessor...")
    try:
        processor = ArchitectureProcessor(device="cpu")
        print("初始化成功")
    except Exception as e:
        print(f"初始化失败: {e}")
        return False

    print("\n处理测试输入...")
    try:
        result = processor.process("这是一个测试输入，用于验证架构特征提取功能。")
        print("处理成功")
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n架构信号长度: {len(result['architecture_signal'])}")
    print(f"前500字符:\n{result['architecture_signal'][:500]}")

    # 验证返回键
    expected_keys = [
        "architecture_signal",
        "feature_analysis",
        "final_hidden",
        "total_params",
        "encoder_status",
    ]
    print(f"\n返回的键: {list(result.keys())}")
    for key in expected_keys:
        assert key in result, f"缺少返回键: {key}"
    print("所有返回键验证通过")

    # feature_analysis 应该包含统计特征
    analysis = result["feature_analysis"]
    assert "hidden_stats" in analysis
    assert "layer_features" in analysis
    assert "layer_deltas" in analysis
    assert "input_features" in analysis
    print("特征分析结构验证通过")

    # final_hidden 应该是 numpy 数组
    assert result["final_hidden"].shape[0] == 1
    assert result["final_hidden"].shape[2] == 512
    print(f"Hidden states shape: {result['final_hidden'].shape}")

    # total_params 应该是正整数
    assert result["total_params"] > 0
    print(f"模型参数量: {result['total_params']:,}")

    # encoder_status 应该是 direct_extraction
    assert result["encoder_status"] == "direct_extraction"
    print(f"编码器状态: {result['encoder_status']}")

    return True


if __name__ == "__main__":
    success = test_processor()
    print("\n测试结果:", "成功" if success else "失败")
