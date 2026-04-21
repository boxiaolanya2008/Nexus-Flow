# 后端测试文件
import sys
import os

# 添加 backend 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database.database import init_db
from database.models import Base
from architecture import CustomModel

async def test_database():
    """测试数据库初始化"""
    print("测试数据库初始化...")
    try:
        await init_db()
        print("数据库初始化成功")
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False

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
        print("自定义架构测试成功")
        return True
    except Exception as e:
        print(f"自定义架构测试失败: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    
    print("开始后端测试...")
    
    # 测试架构
    arch_result = test_architecture()
    
    # 测试数据库
    db_result = asyncio.run(test_database())
    
    if arch_result and db_result:
        print("\n所有测试通过")
    else:
        print("\n部分测试失败")
