import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.architecture_processor import ArchitectureProcessor


def test_processor():
    """测试 ArchitectureProcessor 是否正常工作"""
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

    # 当前 API 返回的键：architecture_signal, semantic_vector, semantic_analysis, final_hidden, total_params, encoder_status
    expected_keys = [
        "architecture_signal",
        "semantic_vector",
        "semantic_analysis",
        "final_hidden",
        "total_params",
        "encoder_status",
    ]
    print(f"\n返回的键: {list(result.keys())}")
    for key in expected_keys:
        assert key in result, f"缺少返回键: {key}"
    print("所有返回键验证通过")

    # semantic_vector 应该是列表，长度等于语义维度
    assert isinstance(result["semantic_vector"], list)
    assert len(result["semantic_vector"]) > 0
    print(f"语义向量维度: {len(result['semantic_vector'])}")

    # semantic_analysis 应该包含必要的子字段
    analysis = result["semantic_analysis"]
    analysis_keys = ["semantic_vector", "top_features", "structure_analysis", "quality_assessment", "inferred_type", "complexity_assessment", "summary"]
    for key in analysis_keys:
        assert key in analysis, f"语义分析缺少键: {key}"
    print("语义分析结构验证通过")

    # final_hidden 应该是 numpy 数组
    assert result["final_hidden"].shape[0] == 1
    assert result["final_hidden"].shape[2] == 512
    print(f"Hidden states shape: {result['final_hidden'].shape}")

    # total_params 应该是正整数
    assert result["total_params"] > 0
    print(f"模型参数量: {result['total_params']:,}")

    # encoder_status 应该是 trained 或 untrained
    assert result["encoder_status"] in ("trained", "untrained")
    print(f"编码器状态: {result['encoder_status']}")

    return True


if __name__ == "__main__":
    success = test_processor()
    print("\n测试结果:", "成功" if success else "失败")
