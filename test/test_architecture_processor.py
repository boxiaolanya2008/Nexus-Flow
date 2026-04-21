# 测试重构后的 ArchitectureProcessor
import sys
import os

# 添加 backend 目录到 Python 路径
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# 修改为绝对导入
import importlib.util
spec = importlib.util.spec_from_file_location("architecture_processor", os.path.join(backend_path, "architecture_processor.py"))
architecture_processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(architecture_processor_module)
ArchitectureProcessor = architecture_processor_module.ArchitectureProcessor

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
    
    # 检查返回的数据结构
    print(f"\n返回的键: {result.keys()}")
    print(f"all_hidden_states 层数: {len(result['all_hidden_states'])}")
    print(f"layer_features 数量: {len(result['layer_features'])}")
    
    # 检查张量数据
    if result['layer_features']:
        first_layer = result['layer_features'][0]
        print(f"\n第一层特征键: {first_layer.keys()}")
        print(f"attention 键: {first_layer['attention'].keys()}")
        print(f"gate_weight shape: {first_layer['attention']['gate_weight'].shape}")
    
    return True

if __name__ == "__main__":
    success = test_processor()
    print("\n测试结果:", "成功" if success else "失败")
