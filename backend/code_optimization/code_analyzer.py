# 代码分析器：AST 解析、依赖分析、质量评估
import ast
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class CodeAnalysisResult:
    """代码分析结果"""
    language: str
    functions: List[str]
    classes: List[str]
    variables: List[str]
    dependencies: List[str]
    complexity: int
    issues: List[str]

class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self):
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c'
        }
    
    def detect_language(self, code: str, filename: str = '') -> str:
        """检测代码语言"""
        if filename:
            for ext, lang in self.language_map.items():
                if filename.endswith(ext):
                    return lang
        
        # 基于内容检测
        if 'def ' in code and 'import ' in code:
            return 'python'
        elif 'function ' in code and 'const ' in code:
            return 'javascript'
        elif 'class ' in code and 'public ' in code:
            return 'java'
        elif 'func ' in code and 'package ' in code:
            return 'go'
        
        return 'unknown'
    
    def analyze_python(self, code: str) -> CodeAnalysisResult:
        """分析 Python 代码"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return CodeAnalysisResult(
                language='python',
                functions=[],
                classes=[],
                variables=[],
                dependencies=[],
                complexity=0,
                issues=['语法错误']
            )
        
        functions = []
        classes = []
        variables = []
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Name):
                if node.id not in variables:
                    variables.append(node.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                dependencies.append(node.module or '')
        
        # 计算复杂度（简化版）
        complexity = len(functions) + len(classes) * 2
        
        # 检测问题
        issues = []
        if complexity > 20:
            issues.append('代码复杂度较高')
        if len(functions) > 10:
            issues.append('函数数量较多，建议拆分')
        
        return CodeAnalysisResult(
            language='python',
            functions=functions,
            classes=classes,
            variables=variables[:50],  # 限制数量
            dependencies=list(set(dependencies)),
            complexity=complexity,
            issues=issues
        )
    
    def analyze_javascript(self, code: str) -> CodeAnalysisResult:
        """分析 JavaScript 代码"""
        functions = re.findall(r'function\s+(\w+)', code)
        classes = re.findall(r'class\s+(\w+)', code)
        variables = re.findall(r'(?:const|let|var)\s+(\w+)', code)
        dependencies = re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', code)
        
        complexity = len(functions) + len(classes) * 2
        
        issues = []
        if complexity > 20:
            issues.append('代码复杂度较高')
        
        return CodeAnalysisResult(
            language='javascript',
            functions=functions,
            classes=classes,
            variables=variables[:50],
            dependencies=dependencies,
            complexity=complexity,
            issues=issues
        )
    
    def analyze(self, code: str, filename: str = '') -> CodeAnalysisResult:
        """分析代码"""
        language = self.detect_language(code, filename)
        
        if language == 'python':
            return self.analyze_python(code)
        elif language in ['javascript', 'typescript']:
            return self.analyze_javascript(code)
        else:
            # 通用分析
            functions = re.findall(r'(?:def|function)\s+(\w+)', code)
            classes = re.findall(r'class\s+(\w+)', code)
            variables = re.findall(r'(?:const|let|var|\w+)\s*=\s*', code)
            
            return CodeAnalysisResult(
                language=language,
                functions=functions,
                classes=classes,
                variables=variables[:50],
                dependencies=[],
                complexity=len(functions) + len(classes) * 2,
                issues=['语言支持有限']
            )
    
    def extract_context(self, code: str, max_length: int = 2000) -> str:
        """提取代码上下文摘要"""
        result = self.analyze(code)
        
        context_parts = [
            f"语言: {result.language}",
            f"函数: {', '.join(result.functions[:10])}",
            f"类: {', '.join(result.classes[:5])}",
            f"复杂度: {result.complexity}"
        ]
        
        if result.issues:
            context_parts.append(f"问题: {', '.join(result.issues)}")
        
        return '\n'.join(context_parts)
