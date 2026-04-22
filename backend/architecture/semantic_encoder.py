# 语义编码器：将 hidden states 映射到可解释的低维语义空间
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json


class SemanticEncoder(nn.Module):
    """
    语义编码器：将 CustomModel 的 hidden states 编码为可解释的语义向量
    
    设计目标：
    1. 将高维 hidden states (d_model=512) 压缩到低维语义空间 (semantic_dim=64)
    2. 每个维度对应一个可解释的代码特征（如：循环复杂度、函数嵌套深度等）
    3. 通过对比学习训练，使语义相似的代码产生相近的语义向量
    """
    
    # 预定义的语义维度定义
    SEMANTIC_DIMENSIONS = [
        # 代码结构特征
        "loop_complexity",      # 循环复杂度
        "nesting_depth",        # 嵌套深度
        "function_count",       # 函数数量
        "class_count",          # 类数量
        "branch_complexity",    # 分支复杂度
        
        # 代码质量特征
        "code_cohesion",        # 内聚性
        "coupling_degree",      # 耦合度
        "readability_score",    # 可读性
        "maintainability",      # 可维护性
        
        # 语义特征
        "algorithm_type",       # 算法类型（0=顺序，1=递归，2=迭代等）
        "data_structure",       # 数据结构使用
        "io_intensity",         # I/O密集度
        "compute_intensity",    # 计算密集度
        "memory_intensity",     # 内存密集度
        
        # 语言特征
        "language_pattern",     # 语言特定模式
        "framework_usage",      # 框架使用程度
        "api_complexity",       # API复杂度
        "async_pattern",        # 异步模式
        
        # 任务类型特征
        "is_algorithm",         # 是否是算法实现
        "is_data_processing",   # 是否是数据处理
        "is_ui_related",        # 是否UI相关
        "is_network_io",        # 是否网络I/O
        "is_file_io",           # 是否文件I/O
        
        # 更多维度填充到64
        "reserved_21", "reserved_22", "reserved_23", "reserved_24",
        "reserved_25", "reserved_26", "reserved_27", "reserved_28",
        "reserved_29", "reserved_30", "reserved_31", "reserved_32",
        "reserved_33", "reserved_34", "reserved_35", "reserved_36",
        "reserved_37", "reserved_38", "reserved_39", "reserved_40",
        "reserved_41", "reserved_42", "reserved_43", "reserved_44",
        "reserved_45", "reserved_46", "reserved_47", "reserved_48",
        "reserved_49", "reserved_50", "reserved_51", "reserved_52",
        "reserved_53", "reserved_54", "reserved_55", "reserved_56",
        "reserved_57", "reserved_58", "reserved_59", "reserved_60",
        "reserved_61", "reserved_62", "reserved_63", "reserved_64",
    ]
    
    def __init__(self, d_model: int = 512, semantic_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.semantic_dim = semantic_dim
        
        # 多层投影网络
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, semantic_dim),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """使用 Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        将 hidden states 编码为语义向量
        
        Args:
            hidden_states: (batch, seq_len, d_model)
        
        Returns:
            semantic_vector: (batch, semantic_dim) 归一化到 [0, 1]
        """
        # 对序列维度取平均，得到句子级别的表示
        pooled = hidden_states.mean(dim=1)  # (batch, d_model)
        
        # 投影到语义空间
        semantic = self.projector(pooled)  # (batch, semantic_dim)
        
        return semantic
    
    def encode_to_dict(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        将 hidden states 编码为语义字典
        
        Returns:
            每个语义维度的具体数值
        """
        with torch.no_grad():
            semantic_vector = self.forward(hidden_states)  # (1, semantic_dim)
            values = semantic_vector.squeeze(0).cpu().numpy()
        
        return {
            dim_name: float(values[i])
            for i, dim_name in enumerate(self.SEMANTIC_DIMENSIONS[:self.semantic_dim])
        }
    
    def get_top_features(self, hidden_states: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        获取最显著的语义特征
        
        Returns:
            [(feature_name, value), ...] 按 value 降序排列
        """
        semantic_dict = self.encode_to_dict(hidden_states)
        sorted_features = sorted(semantic_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]


class ContrastiveTrainer:
    """
    对比学习训练器：训练 SemanticEncoder 使相似代码产生相似语义向量
    """
    
    def __init__(self, encoder: SemanticEncoder, temperature: float = 0.07):
        self.encoder = encoder
        self.temperature = temperature
    
    def contrastive_loss(
        self,
        anchor: torch.Tensor,      # (batch, semantic_dim)
        positive: torch.Tensor,    # (batch, semantic_dim) - 相似代码
        negative: torch.Tensor     # (batch, semantic_dim) - 不相似代码
    ) -> torch.Tensor:
        """
        InfoNCE 对比损失
        """
        # 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # 正样本相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        
        # 负样本相似度
        neg_sim = torch.sum(anchor * negative, dim=-1) / self.temperature
        
        # InfoNCE 损失
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        
        return loss.mean()
    
    def code_similarity_loss(
        self,
        code1_hidden: torch.Tensor,
        code2_hidden: torch.Tensor,
        is_similar: bool  # True 表示代码相似，False 表示不相似
    ) -> torch.Tensor:
        """
        基于代码对相似性的损失
        """
        vec1 = self.encoder(code1_hidden)
        vec2 = self.encoder(code2_hidden)
        
        # 余弦相似度
        cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
        
        if is_similar:
            # 相似代码：最大化相似度
            loss = (1 - cos_sim).mean()
        else:
            # 不相似代码：最小化相似度（但保留一定 margin）
            margin = 0.5
            loss = torch.clamp(cos_sim - margin, min=0).mean()
        
        return loss


class SemanticCodeAnalyzer:
    """
    基于语义向量的代码分析器
    将语义向量转换为人类可读的代码分析报告
    """
    
    # 语义阈值定义
    THRESHOLDS = {
        "loop_complexity": {"low": 0.3, "medium": 0.6, "high": 0.8},
        "nesting_depth": {"low": 0.3, "medium": 0.5, "high": 0.7},
        "readability_score": {"poor": 0.3, "fair": 0.5, "good": 0.7, "excellent": 0.9},
        "maintainability": {"poor": 0.3, "fair": 0.5, "good": 0.7, "excellent": 0.9},
    }
    
    def __init__(self, encoder: SemanticEncoder):
        self.encoder = encoder
    
    def analyze(self, hidden_states: torch.Tensor, code_text: str = "") -> Dict:
        """
        生成完整的代码语义分析报告
        """
        semantic_dict = self.encoder.encode_to_dict(hidden_states)
        top_features = self.encoder.get_top_features(hidden_states, top_k=8)
        
        # 结构分析
        structure_analysis = self._analyze_structure(semantic_dict)
        
        # 质量评估
        quality_assessment = self._assess_quality(semantic_dict)
        
        # 类型推断
        code_type = self._infer_code_type(semantic_dict)
        
        # 复杂度评估
        complexity = self._assess_complexity(semantic_dict)
        
        return {
            "semantic_vector": semantic_dict,
            "top_features": top_features,
            "structure_analysis": structure_analysis,
            "quality_assessment": quality_assessment,
            "inferred_type": code_type,
            "complexity_assessment": complexity,
            "summary": self._generate_summary(semantic_dict, code_type, complexity)
        }
    
    def _analyze_structure(self, semantic: Dict[str, float]) -> Dict:
        """分析代码结构"""
        loop_level = self._categorize_value(semantic.get("loop_complexity", 0.5), "loop_complexity")
        nesting_level = self._categorize_value(semantic.get("nesting_depth", 0.5), "nesting_depth")
        
        return {
            "loop_complexity_level": loop_level,
            "nesting_depth_level": nesting_level,
            "function_count_indicator": "high" if semantic.get("function_count", 0) > 0.7 else "medium" if semantic.get("function_count", 0) > 0.4 else "low",
            "class_count_indicator": "high" if semantic.get("class_count", 0) > 0.7 else "low",
            "branch_complexity": "complex" if semantic.get("branch_complexity", 0) > 0.6 else "simple"
        }
    
    def _assess_quality(self, semantic: Dict[str, float]) -> Dict:
        """评估代码质量"""
        readability = self._categorize_value(semantic.get("readability_score", 0.5), "readability_score")
        maintainability = self._categorize_value(semantic.get("maintainability", 0.5), "maintainability")
        
        return {
            "readability": readability,
            "maintainability": maintainability,
            "cohesion": "high" if semantic.get("code_cohesion", 0) > 0.7 else "medium" if semantic.get("code_cohesion", 0) > 0.4 else "low",
            "coupling": "low" if semantic.get("coupling_degree", 0) < 0.3 else "medium" if semantic.get("coupling_degree", 0) < 0.6 else "high"
        }
    
    def _infer_code_type(self, semantic: Dict[str, float]) -> Dict:
        """推断代码类型"""
        types = []
        
        if semantic.get("is_algorithm", 0) > 0.6:
            types.append("algorithm_implementation")
        if semantic.get("is_data_processing", 0) > 0.6:
            types.append("data_processing")
        if semantic.get("is_ui_related", 0) > 0.6:
            types.append("ui_related")
        if semantic.get("is_network_io", 0) > 0.6:
            types.append("network_io")
        if semantic.get("is_file_io", 0) > 0.6:
            types.append("file_io")
        
        # 算法模式
        algo_pattern = "unknown"
        algo_type_val = semantic.get("algorithm_type", 0.5)
        if algo_type_val < 0.33:
            algo_pattern = "sequential"
        elif algo_type_val < 0.66:
            algo_pattern = "recursive"
        else:
            algo_pattern = "iterative"
        
        return {
            "primary_types": types if types else ["general"],
            "algorithm_pattern": algo_pattern,
            "compute_intensity": "high" if semantic.get("compute_intensity", 0) > 0.7 else "medium" if semantic.get("compute_intensity", 0) > 0.4 else "low",
            "io_intensity": "high" if semantic.get("io_intensity", 0) > 0.7 else "medium" if semantic.get("io_intensity", 0) > 0.4 else "low"
        }
    
    def _assess_complexity(self, semantic: Dict[str, float]) -> Dict:
        """评估整体复杂度"""
        # 综合多个维度计算复杂度分数
        complexity_score = (
            semantic.get("loop_complexity", 0.5) * 0.3 +
            semantic.get("nesting_depth", 0.5) * 0.25 +
            semantic.get("branch_complexity", 0.5) * 0.25 +
            semantic.get("function_count", 0.5) * 0.2
        )
        
        if complexity_score < 0.3:
            level = "low"
            description = "代码结构简单，易于理解"
        elif complexity_score < 0.5:
            level = "medium"
            description = "代码结构适中，有一定复杂性"
        elif complexity_score < 0.7:
            level = "high"
            description = "代码结构复杂，需要仔细分析"
        else:
            level = "very_high"
            description = "代码结构非常复杂，建议重构"
        
        return {
            "score": complexity_score,
            "level": level,
            "description": description
        }
    
    def _categorize_value(self, value: float, dimension: str) -> str:
        """根据阈值对数值进行分类"""
        if dimension not in self.THRESHOLDS:
            return "medium"
        
        thresholds = self.THRESHOLDS[dimension]
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
        
        for category, threshold in sorted_thresholds:
            if value < threshold:
                return category
        
        return sorted_thresholds[-1][0]
    
    def _generate_summary(self, semantic: Dict[str, float], code_type: Dict, complexity: Dict) -> str:
        """生成自然语言摘要"""
        parts = []
        
        # 类型描述
        primary_type = code_type["primary_types"][0] if code_type["primary_types"] else "general"
        type_names = {
            "algorithm_implementation": "算法实现",
            "data_processing": "数据处理",
            "ui_related": "UI相关",
            "network_io": "网络I/O",
            "file_io": "文件I/O",
            "general": "通用代码"
        }
        parts.append(f"这是一个{type_names.get(primary_type, primary_type)}类型的代码")
        
        # 复杂度描述
        parts.append(f"，整体复杂度为{complexity['level']}（{complexity['description']}）")
        
        # 算法模式
        if code_type["algorithm_pattern"] != "unknown":
            pattern_names = {"sequential": "顺序执行", "recursive": "递归", "iterative": "迭代"}
            parts.append(f"，主要采用{pattern_names.get(code_type['algorithm_pattern'], code_type['algorithm_pattern'])}模式")
        
        # 质量提示
        readability = semantic.get("readability_score", 0.5)
        if readability < 0.4:
            parts.append("，可读性较差，建议添加注释")
        elif readability > 0.8:
            parts.append("，可读性良好")
        
        return "".join(parts)


def create_semantic_signal(
    hidden_states: torch.Tensor,
    encoder: SemanticEncoder,
    code_text: str = ""
) -> str:
    """
    创建结构化的语义信号字符串，用于注入到 LLM 提示词中
    """
    analyzer = SemanticCodeAnalyzer(encoder)
    analysis = analyzer.analyze(hidden_states, code_text)
    
    # 构建语义信号
    signal_parts = [
        "[架构语义信号]",
        "以下是由 CustomNeuralArchitecture 提取的代码语义分析：",
        "",
        "## 语义向量表示",
        json.dumps(analysis["semantic_vector"], ensure_ascii=False, indent=2),
        "",
        "## 关键特征",
    ]
    
    for feature_name, value in analysis["top_features"]:
        signal_parts.append(f"  - {feature_name}: {value:.3f}")
    
    signal_parts.extend([
        "",
        "## 结构分析",
        json.dumps(analysis["structure_analysis"], ensure_ascii=False, indent=2),
        "",
        "## 质量评估",
        json.dumps(analysis["quality_assessment"], ensure_ascii=False, indent=2),
        "",
        "## 代码类型推断",
        json.dumps(analysis["inferred_type"], ensure_ascii=False, indent=2),
        "",
        "## 复杂度评估",
        json.dumps(analysis["complexity_assessment"], ensure_ascii=False, indent=2),
        "",
        "## 分析摘要",
        analysis["summary"],
        "",
        "[架构语义信号结束]"
    ])
    
    return "\n".join(signal_parts)
