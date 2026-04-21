# 多 API 管理器：从 .env 读取 API 配置
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class MultiAPIManager:
    """多 API 管理器，从 .env 读取 API 配置"""
    
    def __init__(self):
        self.api_url = os.getenv('api')
        self.model = os.getenv('model')
        self.api_key = os.getenv('key')
        
        if not all([self.api_url, self.model, self.api_key]):
            raise ValueError("缺少必要的 API 配置：api、model、key")
    
    def get_config(self) -> Dict[str, str]:
        """获取当前 API 配置"""
        return {
            'api_url': self.api_url,
            'model': self.model,
            'api_key': self.api_key
        }
    
    def reload_config(self):
        """重新从 .env 加载配置"""
        load_dotenv(override=True)
        self.api_url = os.getenv('api')
        self.model = os.getenv('model')
        self.api_key = os.getenv('key')
