# 代码增强器：直接使用架构信号，不使用提示词模板
from typing import Dict, Any
from .code_analyzer import CodeAnalyzer

class CodeEnhancer:
    """代码增强器"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    def analyze_and_enhance(
        self,
        code: str,
        architecture_signal: str = ''
    ) -> Dict[str, Any]:
        """分析代码并返回架构信号"""
        analysis = self.analyzer.analyze(code)
        
        return {
            'analysis': {
                'language': analysis.language,
                'functions': analysis.functions,
                'classes': analysis.classes,
                'complexity': analysis.complexity,
                'issues': analysis.issues
            },
            'architecture_signal': architecture_signal
        }
