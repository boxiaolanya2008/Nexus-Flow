# 语义编码器预训练脚本
# 使用对比学习训练 SemanticEncoder，使相似代码产生相似语义向量

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import ast
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.architecture.semantic_encoder import SemanticEncoder, ContrastiveTrainer
from backend.architecture import CustomModel
from backend.architecture.tokenizer import SimpleTokenizer


class CodeDataset(Dataset):
    """
    代码数据集：加载代码样本并生成训练对
    """
    
    def __init__(self, code_samples: List[Dict], max_length: int = 512):
        self.code_samples = code_samples
        self.max_length = max_length
    
    def __len__(self):
        return len(self.code_samples)
    
    def __getitem__(self, idx):
        return self.code_samples[idx]
    
    def get_similar_pair(self, idx: int) -> Tuple[Dict, Dict]:
        """获取相似代码对（同一类别）"""
        anchor = self.code_samples[idx]
        category = anchor.get("category", "general")
        
        # 找到同类别其他样本
        same_category = [
            s for i, s in enumerate(self.code_samples)
            if i != idx and s.get("category") == category
        ]
        
        if same_category:
            positive = random.choice(same_category)
        else:
            # 如果没有同类样本，使用自身增强
            positive = self._augment_code(anchor)
        
        return anchor, positive
    
    def get_dissimilar_pair(self, idx: int) -> Tuple[Dict, Dict]:
        """获取不相似代码对（不同类别）"""
        anchor = self.code_samples[idx]
        anchor_category = anchor.get("category", "general")
        
        # 找到不同类别样本
        different_category = [
            s for s in self.code_samples
            if s.get("category") != anchor_category
        ]
        
        if different_category:
            negative = random.choice(different_category)
        else:
            # 如果没有不同类样本，创建一个随机样本
            negative = {"code": "# random code\nprint('hello')", "category": "random"}
        
        return anchor, negative
    
    def _augment_code(self, sample: Dict) -> Dict:
        """对代码进行简单增强（变量重命名、添加注释等）"""
        code = sample.get("code", "")
        
        # 简单增强：添加随机注释
        lines = code.split("\n")
        if lines:
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, f"# Augmented: {random.randint(1000, 9999)}")
        
        augmented_code = "\n".join(lines)
        
        return {
            "code": augmented_code,
            "category": sample.get("category", "general"),
            "augmented": True
        }


class SyntheticCodeGenerator:
    """
    合成代码生成器：生成用于预训练的代码样本
    """
    
    CODE_TEMPLATES = {
        "algorithm_sorting": [
            """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
            """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
            """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result"""
        ],
        "algorithm_search": [
            """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
            """def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1"""
        ],
        "data_processing": [
            """import json

def process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for item in data:
        processed = {
            'id': item.get('id'),
            'name': item.get('name', '').upper(),
            'value': item.get('value', 0) * 2
        }
        results.append(processed)
    
    return results""",
            """def filter_records(records, min_value):
    filtered = []
    for record in records:
        if record.get('score', 0) >= min_value:
            filtered.append(record)
    return filtered"""
        ],
        "file_io": [
            """def read_config(path):
    config = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config""",
            """import csv

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        headers = next(reader)
        writer.writerow(headers)
        
        for row in reader:
            processed = [cell.upper() for cell in row]
            writer.writerow(processed)"""
        ],
        "network_io": [
            """import requests

def fetch_data(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None""",
            """import socket

def create_server(host='0.0.0.0', port=8080):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")
    
    while True:
        client, addr = server.accept()
        print(f"Connection from {addr}")
        handle_client(client)"""
        ],
        "class_definition": [
            """class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process(self, data):
        if data['id'] in self.cache:
            return self.cache[data['id']]
        
        result = self._transform(data)
        self.cache[data['id']] = result
        return result
    
    def _transform(self, data):
        return {k: v.upper() for k, v in data.items()}""",
            """class Logger:
    def __init__(self, level='INFO'):
        self.level = level
        self.handlers = []
    
    def add_handler(self, handler):
        self.handlers.append(handler)
    
    def log(self, message, level='INFO'):
        if self._should_log(level):
            for handler in self.handlers:
                handler.write(f"[{level}] {message}")
    
    def _should_log(self, level):
        levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        return levels.get(level, 1) >= levels.get(self.level, 1)"""
        ],
        "recursive": [
            """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
            """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)""",
            """def tree_traversal(node):
    if node is None:
        return []
    result = [node.value]
    result.extend(tree_traversal(node.left))
    result.extend(tree_traversal(node.right))
    return result"""
        ],
        "async": [
            """import asyncio

async def fetch_all(urls):
    tasks = [fetch_one(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

async def fetch_one(url):
    await asyncio.sleep(0.1)
    return f"Data from {url}""",
            """async def process_queue(queue):
    while not queue.empty():
        item = await queue.get()
        result = await process_item(item)
        await queue.put(result)"""
        ]
    }
    
    @classmethod
    def generate_dataset(cls, samples_per_category: int = 50) -> List[Dict]:
        """生成合成代码数据集"""
        dataset = []
        
        for category, templates in cls.CODE_TEMPLATES.items():
            for i in range(samples_per_category):
                # 随机选择模板并添加变化
                template = random.choice(templates)
                code = cls._vary_template(template, i)
                
                dataset.append({
                    "code": code,
                    "category": category,
                    "id": f"{category}_{i}"
                })
        
        return dataset
    
    @classmethod
    def _vary_template(cls, template: str, seed: int) -> str:
        """对模板进行微小变化"""
        random.seed(seed)
        
        lines = template.split("\n")
        
        # 随机添加注释
        if random.random() > 0.5:
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, f"# Generated variation {seed}")
        
        # 随机修改变量名（简单替换）
        code = "\n".join(lines)
        
        # 添加随机空行
        if random.random() > 0.7:
            code = code.replace("\n\n", "\n\n\n", 1)
        
        return code


class SemanticEncoderTrainer:
    """
    语义编码器训练器
    """
    
    def __init__(
        self,
        d_model: int = 512,
        semantic_dim: int = 64,
        device: str = "cpu",
        learning_rate: float = 1e-4
    ):
        self.device = torch.device(device)
        
        # 创建语义编码器
        self.encoder = SemanticEncoder(d_model=d_model, semantic_dim=semantic_dim).to(self.device)
        
        # 创建 CustomModel 用于生成 hidden states
        self.custom_model = CustomModel(
            vocab_size=50000,
            d_model=d_model,
            n_layers=6,
            n_heads=8
        ).to(self.device)
        
        # 冻结 CustomModel 参数（只训练语义编码器）
        for param in self.custom_model.parameters():
            param.requires_grad = False
        
        self.custom_model.eval()
        
        # 创建对比学习训练器
        self.contrastive_trainer = ContrastiveTrainer(self.encoder)
        
        # 优化器
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # 用共享的分词器替代原来的手动实现
        self.tokenizer = SimpleTokenizer(vocab_size=50000)
    
    def get_hidden_states(self, code: str) -> torch.Tensor:
        """通过 CustomModel 获取 hidden states"""
        input_ids = self.tokenizer.encode(code).to(self.device)
        
        with torch.no_grad():
            # 获取嵌入
            batch_size, seq_len = input_ids.shape
            token_emb = self.custom_model.token_embedding(input_ids)
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.custom_model.position_embedding(positions)
            x = self.custom_model.embedding_norm(token_emb + pos_emb)
            
            # 通过所有层
            for block in self.custom_model.blocks:
                x = block(x)
            
            # 最终层归一化
            hidden = self.custom_model.final_norm(x)
        
        return hidden
    
    def train_epoch(self, dataset: CodeDataset, batch_size: int = 8) -> float:
        """训练一个 epoch"""
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        # 随机打乱索引
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            anchor_hidden_list = []
            positive_hidden_list = []
            negative_hidden_list = []
            
            for idx in batch_indices:
                # 获取相似对
                anchor_sample, positive_sample = dataset.get_similar_pair(idx)
                anchor_hidden = self.get_hidden_states(anchor_sample["code"])
                positive_hidden = self.get_hidden_states(positive_sample["code"])
                
                # 获取不相似对
                _, negative_sample = dataset.get_dissimilar_pair(idx)
                negative_hidden = self.get_hidden_states(negative_sample["code"])
                
                anchor_hidden_list.append(anchor_hidden)
                positive_hidden_list.append(positive_hidden)
                negative_hidden_list.append(negative_hidden)
            
            # 拼接批次
            anchor_batch = torch.cat(anchor_hidden_list, dim=0)
            positive_batch = torch.cat(positive_hidden_list, dim=0)
            negative_batch = torch.cat(negative_hidden_list, dim=0)
            
            # 前向传播
            anchor_vec = self.encoder(anchor_batch)
            positive_vec = self.encoder(positive_batch)
            negative_vec = self.encoder(negative_batch)
            
            # 计算对比损失
            loss = self.contrastive_trainer.contrastive_loss(
                anchor_vec, positive_vec, negative_vec
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        
        return total_loss / max(num_batches, 1)
    
    def train(
        self,
        dataset: CodeDataset,
        epochs: int = 50,
        batch_size: int = 8,
        save_path: str = "semantic_encoder.pt",
        validate_every: int = 5
    ):
        """完整训练流程"""
        print(f"开始训练语义编码器...")
        print(f"数据集大小: {len(dataset)}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"语义维度: {self.encoder.semantic_dim}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataset, batch_size)
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 定期验证和保存
            if (epoch + 1) % validate_every == 0:
                self.validate(dataset)
                
                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save(save_path)
                    print(f"  -> 保存最佳模型 (loss: {best_loss:.4f})")
        
        # 保存最终模型
        self.save(save_path)
        print(f"\n训练完成！模型已保存到 {save_path}")
    
    def validate(self, dataset: CodeDataset, num_samples: int = 10):
        """验证语义编码器"""
        self.encoder.eval()
        
        print("\n  验证样本:")
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            hidden = self.get_hidden_states(sample["code"])
            
            with torch.no_grad():
                semantic = self.encoder.encode_to_dict(hidden)
            
            top_features = sorted(semantic.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    [{sample.get('category', 'unknown')}] Top: {top_features}")
        
        self.encoder.train()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "semantic_dim": self.encoder.semantic_dim,
            "d_model": self.encoder.d_model,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def main():
    """主函数：生成数据并训练"""
    # 生成合成代码数据集
    print("生成合成代码数据集...")
    code_samples = SyntheticCodeGenerator.generate_dataset(samples_per_category=100)
    print(f"生成了 {len(code_samples)} 个代码样本")
    
    # 创建数据集
    dataset = CodeDataset(code_samples)
    
    # 创建训练器
    trainer = SemanticEncoderTrainer(
        d_model=512,
        semantic_dim=64,
        device="cpu",  # 可根据需要改为 "cuda"
        learning_rate=1e-4
    )
    
    # 训练
    trainer.train(
        dataset=dataset,
        epochs=30,
        batch_size=4,  # 小批次以适应 CPU 训练
        save_path="backend/architecture/semantic_encoder.pt",
        validate_every=5
    )


if __name__ == "__main__":
    main()
