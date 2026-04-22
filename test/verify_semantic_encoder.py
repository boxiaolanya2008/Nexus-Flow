#!/usr/bin/env python3
"""
语义编码器验证脚本
验证修复后的架构信号是否真正具备语义价值
"""

import torch
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.architecture_processor import ArchitectureProcessor
from backend.architecture.semantic_encoder import (
    SemanticEncoder, 
    SemanticCodeAnalyzer,
    create_semantic_signal
)


def test_semantic_consistency():
    """测试语义一致性：相似代码应该产生相似的语义向量"""
    print("=" * 60)
    print("测试1: 语义一致性")
    print("=" * 60)
    
    processor = ArchitectureProcessor(device="cpu")
    
    # 相似代码对
    similar_codes = [
        ("def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
         "def bubble_sort(data):\n    n = len(data)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if data[j] > data[j+1]:\n                data[j], data[j+1] = data[j+1], data[j]\n    return data"),
        
        ("class DataProcessor:\n    def process(self, data):\n        return [x*2 for x in data]",
         "class DataHandler:\n    def handle(self, items):\n        return [item*2 for item in items]")
    ]
    
    for i, (code1, code2) in enumerate(similar_codes, 1):
        print(f"\n相似代码对 {i}:")
        
        result1 = processor.process(code1)
        result2 = processor.process(code2)
        
        vec1 = torch.tensor(result1["semantic_vector"])
        vec2 = torch.tensor(result2["semantic_vector"])
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)
        
        print(f"  代码1长度: {len(code1)}, 代码2长度: {len(code2)}")
        print(f"  语义向量余弦相似度: {cos_sim.item():.4f}")
        
        if cos_sim.item() > 0.8:
            print(f"  ✓ 相似代码产生了相似的语义向量")
        elif cos_sim.item() > 0.5:
            print(f"  ~ 语义向量有一定相似性（需要更多训练）")
        else:
            print(f"  ✗ 相似代码语义向量差异较大（编码器未训练）")


def test_semantic_discrimination():
    """测试语义区分性：不同代码应该产生不同的语义向量"""
    print("\n" + "=" * 60)
    print("测试2: 语义区分性")
    print("=" * 60)
    
    processor = ArchitectureProcessor(device="cpu")
    
    # 不同类型的代码
    codes = {
        "排序算法": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
        "文件IO": "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()",
        "类定义": "class User:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    def greet(self):\n        return f'Hello, {self.name}'",
        "网络请求": "import requests\ndef fetch(url):\n    response = requests.get(url)\n    return response.json()"
    }
    
    print("\n不同类型代码的语义向量差异:")
    
    results = {}
    for name, code in codes.items():
        result = processor.process(code)
        results[name] = torch.tensor(result["semantic_vector"])
    
    # 计算两两相似度
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            vec1, vec2 = results[name1], results[name2]
            cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)
            print(f"  {name1} vs {name2}: {cos_sim.item():.4f}")


def test_semantic_interpretability():
    """测试语义可解释性：检查语义向量是否对应可解释的特征"""
    print("\n" + "=" * 60)
    print("测试3: 语义可解释性")
    print("=" * 60)
    
    processor = ArchitectureProcessor(device="cpu")
    
    # 复杂代码
    complex_code = """
def process_data(data, config):
    results = []
    for item in data:
        if item['type'] == 'A':
            processed = transform_a(item, config)
        elif item['type'] == 'B':
            processed = transform_b(item, config)
        else:
            processed = default_transform(item)
        
        if validate(processed):
            results.append(processed)
    
    return sorted(results, key=lambda x: x['score'])
"""
    
    result = processor.process(complex_code)
    analysis = result["semantic_analysis"]
    
    print("\n语义分析结果:")
    print(f"  编码器状态: {result['encoder_status']}")
    print(f"\n  Top 5 语义特征:")
    for feature, value in analysis["top_features"]:
        print(f"    - {feature}: {value:.3f}")
    
    print(f"\n  结构分析:")
    for key, value in analysis["structure_analysis"].items():
        print(f"    - {key}: {value}")
    
    print(f"\n  复杂度评估:")
    complexity = analysis["complexity_assessment"]
    print(f"    - 分数: {complexity['score']:.3f}")
    print(f"    - 等级: {complexity['level']}")
    print(f"    - 描述: {complexity['description']}")
    
    print(f"\n  分析摘要:")
    print(f"    {analysis['summary']}")


def test_architecture_signal_format():
    """测试架构信号格式"""
    print("\n" + "=" * 60)
    print("测试4: 架构信号格式")
    print("=" * 60)
    
    processor = ArchitectureProcessor(device="cpu")
    
    code = "def test():\n    return 42"
    result = processor.process(code)
    signal = result["architecture_signal"]
    
    print("\n架构信号预览（前1000字符）:")
    print(signal[:1000])
    print("\n...")
    
    # 验证信号包含关键部分
    required_parts = [
        "[架构语义信号]",
        "语义向量表示",
        "关键特征",
        "结构分析",
        "质量评估",
        "代码类型推断",
        "复杂度评估",
        "分析摘要",
        "[架构语义信号结束]"
    ]
    
    print("\n信号完整性检查:")
    for part in required_parts:
        if part in signal:
            print(f"  ✓ 包含: {part}")
        else:
            print(f"  ✗ 缺失: {part}")


def test_encoder_status():
    """测试编码器状态报告"""
    print("\n" + "=" * 60)
    print("测试5: 编码器状态")
    print("=" * 60)
    
    processor = ArchitectureProcessor(device="cpu")
    status = processor.get_architecture_status()
    
    print("\n架构状态:")
    print(f"  架构名称: {status['architecture_name']}")
    print(f"  总参数量: {status['total_parameters']:,}")
    print(f"  设备: {status['device']}")
    print(f"  层数: {status['layers']}")
    print(f"  模型维度: {status['d_model']}")
    print(f"  模式: {status['mode']}")
    
    print("\n语义编码器状态:")
    semantic_status = status['semantic_encoder']
    print(f"  状态: {semantic_status['status']}")
    print(f"  语义维度: {semantic_status['semantic_dim']}")
    print(f"  语义维度定义（前10个）:")
    for dim in semantic_status['dimensions'][:10]:
        print(f"    - {dim}")


def main():
    """主函数：运行所有测试"""
    print("\n" + "=" * 60)
    print("语义编码器验证")
    print("=" * 60)
    print("\n注意：如果编码器未训练，语义向量将是随机的。")
    print("运行以下命令训练编码器：")
    print("  python backend/architecture/train_semantic_encoder.py")
    print()
    
    try:
        test_semantic_consistency()
        test_semantic_discrimination()
        test_semantic_interpretability()
        test_architecture_signal_format()
        test_encoder_status()
        
        print("\n" + "=" * 60)
        print("所有测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
