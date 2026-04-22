# 语义编码器测试
import unittest
import torch
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.architecture.semantic_encoder import (
    SemanticEncoder, 
    SemanticCodeAnalyzer,
    ContrastiveTrainer,
    create_semantic_signal
)
from backend.architecture import CustomModel


class TestSemanticEncoder(unittest.TestCase):
    """测试语义编码器"""

    def setUp(self):
        """测试前准备"""
        self.d_model = 512
        self.semantic_dim = 64
        self.encoder = SemanticEncoder(d_model=self.d_model, semantic_dim=self.semantic_dim)
        self.encoder.eval()

    def test_encoder_output_shape(self):
        """测试编码器输出形状"""
        batch_size = 2
        seq_len = 100
        hidden_states = torch.randn(batch_size, seq_len, self.d_model)
        
        semantic = self.encoder(hidden_states)
        
        self.assertEqual(semantic.shape, (batch_size, self.semantic_dim))
        self.assertTrue(torch.all(semantic >= 0) and torch.all(semantic <= 1))

    def test_encode_to_dict(self):
        """测试编码为字典"""
        hidden_states = torch.randn(1, 50, self.d_model)
        
        semantic_dict = self.encoder.encode_to_dict(hidden_states)
        
        self.assertEqual(len(semantic_dict), self.semantic_dim)
        self.assertIn("loop_complexity", semantic_dict)
        self.assertIn("readability_score", semantic_dict)
        self.assertTrue(all(0 <= v <= 1 for v in semantic_dict.values()))

    def test_get_top_features(self):
        """测试获取 top 特征"""
        hidden_states = torch.randn(1, 50, self.d_model)
        
        top_features = self.encoder.get_top_features(hidden_states, top_k=5)
        
        self.assertEqual(len(top_features), 5)
        self.assertTrue(all(isinstance(name, str) and isinstance(value, float) 
                           for name, value in top_features))
        # 验证按值降序排列
        values = [v for _, v in top_features]
        self.assertEqual(values, sorted(values, reverse=True))


class TestSemanticCodeAnalyzer(unittest.TestCase):
    """测试语义代码分析器"""

    def setUp(self):
        self.encoder = SemanticEncoder(d_model=512, semantic_dim=64)
        self.analyzer = SemanticCodeAnalyzer(self.encoder)

    def test_analyze_structure(self):
        """测试结构分析"""
        hidden_states = torch.randn(1, 50, 512)
        
        analysis = self.analyzer.analyze(hidden_states, "test code")
        
        self.assertIn("semantic_vector", analysis)
        self.assertIn("structure_analysis", analysis)
        self.assertIn("quality_assessment", analysis)
        self.assertIn("inferred_type", analysis)
        self.assertIn("complexity_assessment", analysis)
        self.assertIn("summary", analysis)

    def test_complexity_assessment(self):
        """测试复杂度评估"""
        hidden_states = torch.randn(1, 50, 512)
        
        analysis = self.analyzer.analyze(hidden_states, "test")
        complexity = analysis["complexity_assessment"]
        
        self.assertIn("score", complexity)
        self.assertIn("level", complexity)
        self.assertIn("description", complexity)
        self.assertTrue(0 <= complexity["score"] <= 1)
        self.assertIn(complexity["level"], ["low", "medium", "high", "very_high"])

    def test_code_type_inference(self):
        """测试代码类型推断"""
        hidden_states = torch.randn(1, 50, 512)
        
        analysis = self.analyzer.analyze(hidden_states, "test")
        code_type = analysis["inferred_type"]
        
        self.assertIn("primary_types", code_type)
        self.assertIn("algorithm_pattern", code_type)
        self.assertIn("compute_intensity", code_type)
        self.assertIn("io_intensity", code_type)


class TestContrastiveTrainer(unittest.TestCase):
    """测试对比学习训练器"""

    def setUp(self):
        self.encoder = SemanticEncoder(d_model=512, semantic_dim=64)
        self.trainer = ContrastiveTrainer(self.encoder)

    def test_contrastive_loss(self):
        """测试对比损失计算"""
        batch_size = 4
        semantic_dim = 64
        
        anchor = torch.randn(batch_size, semantic_dim)
        positive = torch.randn(batch_size, semantic_dim)
        negative = torch.randn(batch_size, semantic_dim)
        
        loss = self.trainer.contrastive_loss(anchor, positive, negative)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)

    def test_code_similarity_loss(self):
        """测试代码相似性损失"""
        hidden1 = torch.randn(1, 50, 512)
        hidden2 = torch.randn(1, 50, 512)
        
        # 相似代码
        loss_similar = self.trainer.code_similarity_loss(hidden1, hidden2, is_similar=True)
        self.assertIsInstance(loss_similar, torch.Tensor)
        
        # 不相似代码
        loss_dissimilar = self.trainer.code_similarity_loss(hidden1, hidden2, is_similar=False)
        self.assertIsInstance(loss_dissimilar, torch.Tensor)


class TestSemanticSignalCreation(unittest.TestCase):
    """测试语义信号生成"""

    def setUp(self):
        self.encoder = SemanticEncoder(d_model=512, semantic_dim=64)
        self.encoder.eval()

    def test_create_semantic_signal(self):
        """测试创建语义信号"""
        hidden_states = torch.randn(1, 50, 512)
        code_text = "def test(): pass"
        
        signal = create_semantic_signal(hidden_states, self.encoder, code_text)
        
        self.assertIn("[架构语义信号]", signal)
        self.assertIn("语义向量表示", signal)
        self.assertIn("关键特征", signal)
        self.assertIn("结构分析", signal)
        self.assertIn("质量评估", signal)
        self.assertIn("[架构语义信号结束]", signal)


class TestIntegration(unittest.TestCase):
    """集成测试：与 CustomModel 配合使用"""

    def test_end_to_end(self):
        """测试端到端流程"""
        # 创建 CustomModel
        model = CustomModel(
            vocab_size=50000,
            d_model=512,
            n_layers=6,
            n_heads=8
        )
        model.eval()

        # 创建语义编码器
        encoder = SemanticEncoder(d_model=512, semantic_dim=64)
        encoder.eval()

        # 模拟输入
        batch_size = 1
        seq_len = 50
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))

        # 前向传播
        with torch.no_grad():
            logits = model(input_ids)
            
            # 获取 hidden states（这里简化处理，实际应该从模型内部获取）
            # 使用 logits 作为近似
            hidden_states = logits[:, :seq_len, :512]  # 取前 512 维

        # 语义编码
        semantic = encoder(hidden_states)
        
        self.assertEqual(semantic.shape, (batch_size, 64))
        self.assertTrue(torch.all(semantic >= 0) and torch.all(semantic <= 1))


if __name__ == "__main__":
    unittest.main()
