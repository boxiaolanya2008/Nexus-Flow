# 前端设计器：直接使用架构信号，不使用提示词模板
from typing import Dict, Any

class FrontendDesigner:
    """前端设计器"""
    
    def __init__(self):
        self.framework = 'react'
        self.styling = 'tailwindcss'
    
    def analyze_design_request(
        self,
        description: str,
        architecture_signal: str = ''
    ) -> Dict[str, Any]:
        """分析设计请求并返回架构信号"""
        return {
            'architecture_signal': architecture_signal,
            'framework': self.framework,
            'styling': self.styling
        }
