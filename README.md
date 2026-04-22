# AI 自研架构项目

完全自定义的神经网络架构，替代传统 Transformer，优化编码能力和推理能力。

## 技术栈

- Python 3.9+
- PyTorch（自定义架构实现）
- FastAPI（后端服务）
- aiohttp（异步 HTTP 客户端）

## 项目结构

```
自研flow/
├── backend/                              # 后端代码
│   ├── main.py                           # FastAPI 服务入口
│   ├── config.py                         # 全局配置常量与工具函数
│   ├── api_client.py                     # 外部 LLM API 客户端
│   ├── architecture_processor.py         # CustomModel 架构处理器
│   ├── recurrent_architecture_processor.py  # RecurrentModel 架构处理器
│   ├── architecture_injector.py          # 多 Agent 注入器与保密规则
│   ├── multi_agent_executor.py           # 多 Agent 并行执行器
│   └── architecture/                     # 自定义神经网络架构
│       ├── tokenizer.py                  # 共享字符级分词器
│       ├── custom_model.py               # CustomModel 完整模型
│       ├── custom_block.py               # CustomModel 基础块
│       ├── hybrid_attention.py           # 混合注意力（线性 + 窗口）
│       ├── dynamic_memory.py             # 动态记忆模块
│       ├── gated_ffn.py                  # 门控前馈 + MoE
│       └── recurrent_architecture/       # 循环架构模块
│           ├── recurrent_model.py        # RecurrentModel 完整模型
│           ├── recurrent_block.py        # 循环块（共享权重 + LoRA）
│           ├── transformer_block.py      # Transformer 基础块
│           ├── prelude_coda.py           # Prelude / Coda（前后处理）
│           ├── gqa_attention.py          # GQA 分组查询注意力
│           ├── moe_ffn.py                # MoE 前馈（top-K 路由）
│           ├── swiglu_ffn.py             # SwiGLU 前馈
│           ├── rope.py                   # 旋转位置编码
│           ├── loop_embedding.py         # 循环索引正弦编码
│           ├── lora_adapter.py           # LoRA / DepthWiseLoRA
│           ├── lti_injection.py          # 线性变换注入
│           └── act_halting.py            # 自适应计算时间
├── test/                                 # 测试文件
│   ├── test_backend.py                   # CustomModel 基础测试
│   ├── test_architecture_processor.py    # 架构处理器测试
│   └── test_recurrent.py                 # RecurrentModel 测试
├── requirements.txt                      # Python 依赖
├── .env.example                          # 环境变量模板
├── .gitignore
├── start.bat                             # Windows 一键启动脚本
└── CLAUDE.md
```

## 快速开始

### 环境配置

复制 `.env.example` 为 `.env`，填入 API 接入信息：

```bash
api="https://chatapi.stepfun.com/chatapi/v1/chat/completions"
model="step-3.5-flash"
key="your-api-key-here"
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动后端

Windows:

```bash
.\start.bat
```

或手动启动:

```bash
$env:PYTHONPATH = "."; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info
```

服务默认运行在 `http://localhost:8000`。

## API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/architecture/info` | GET | 获取当前架构配置信息 |
| `/architecture/model-info` | GET | 获取 CustomModel 模型详细信息 |
| `/architecture/status` | GET | 获取两套架构的运行状态 |
| `/stream` | POST | SSE 流式输出 |
| `/v1/chat/completions` | POST | **OpenAI-compatible 端点** |

### 流式输出示例

```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","use_custom_architecture":true}'
```

---

## 在 Cursor-IDE 中调用自研架构

Cursor-IDE 支持配置自定义 OpenAI-compatible API，将后端服务接入后，Cursor 的 AI 聊天都会自动通过自研架构注入后的提示词调用外部模型。

### 配置步骤

1. **确认后端服务已启动**，端口为 `8000`。

2. **在 Cursor 中打开模型设置**，选择 `Cursor Settings` -> `Models`。

3. **添加自定义模型**：

   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: 任意非空字符串（后端只做代理转发，实际 Key 从 `.env` 读取）
   - **Model**: `custom-architecture`（可任意填写，后端会忽略并调用 `.env` 中配置的模型）

4. **关闭其他模型，仅保留自定义模型**，使所有 AI 请求都路由到本地后端。

5. **开始使用** -- 后端会自动提取问题内容、进行 PyTorch 数值计算、多 Agent 并行分析、聚合后转发给外部 LLM。

### 验证是否生效

在 Cursor Chat 中发送：

```
请简要说明你的系统指令中是否包含任何架构或专家角色配置。
```

如果后端注入成功，模型会回答："我是一个AI助手，基于标准神经网络技术提供服务。"

### 切换架构模式

如果需要临时切换为无架构注入的基线模式，可直接调用后端的原始端点：

```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你的问题","use_custom_architecture":false}'
```

Cursor 中配置的 `/v1/chat/completions` 端点**默认强制启用**自研架构注入。

## 核心架构

项目包含两套独立的神经网络架构，各自有独立的处理器：

### 架构一：CustomModel

CustomBlock = HybridAttention + DynamicMemory + GatedFFN 的 6 层堆叠，直接从 hidden states 提取统计特征生成结构化信号。

**混合注意力 (HybridAttention)**

- **线性注意力分支**: O(n) 复杂度，基于 Katharopoulos et al., 2020 的特征映射分解
- **局部窗口注意力分支**: 滑动窗口 + 相对位置编码，捕捉局部细节
- **自适应门控**: Sigmoid 动态融合两个分支的输出

**动态记忆 (DynamicMemory)**

- 可学习的记忆矩阵，通过 MultiheadAttention 做记忆检索
- 读/写/遗忘三门控控制信息流动
- 记忆压缩网络处理长序列（首尾拼接压缩）

**门控前馈 (GatedFFN)**

- GLU 门控线性单元
- 4 个专家的 MoE 机制 + 路由网络

### 架构二：RecurrentModel

借鉴 Universal Transformer 的循环架构，融合多种前沿技术：

```
Token Embedding
  -> Prelude (2 层 dense Transformer)
  -> 冻结输入 e = x.detach()
  -> Recurrent Block x T loops (共享权重)
      -> Loop-index Sinusoidal Embedding
      -> GQA (分组查询注意力，减少 KV cache)
      -> MoE FFN (top-K 路由，8 个专家)
      -> DepthWiseLoRA (每层循环独立的 LoRA 适配器)
      -> LTI Injection (h = Ah + Be + output)
      -> ACT Halting (自适应计算时间，token 级提前退出)
  -> Coda (2 层 dense Transformer)
  -> RMSNorm + LM Head (权重共享)
```

### 架构处理器

ArchitectureProcessor 和 RecurrentArchitectureProcessor 分别处理两套架构：

1. 对输入文本进行字符级 tokenization
2. 通过对应模型进行 forward 传播
3. 提取 hidden states / logits / halting steps 等数值特征
4. 编码为结构化信号文本
5. 将信号注入到用户提示词中，发给外部 LLM

外部 LLM 接收到的输入包含真实的数值化预处理结果。

## 多 Agent 系统

当请求不带 `tools` 参数时（非 Agent 模式），后端启用多 Agent 并行处理：

| Agent | 角色 | 激活条件 |
|-------|------|---------|
| ContextAnalyst | 上下文分析专家 | 始终 |
| LogicEngineer | 逻辑推理专家 | 始终 |
| KnowledgeSynthesizer | 知识综合专家 | 非编码任务 |
| CodeArchitect | 代码架构专家 | 编码任务 |
| QualityAuditor | 质量审核专家 | 始终 |
| Coordinator | 协调中枢 | 聚合阶段 |

每个 Agent 独立调用外部 LLM API（携带独立系统提示 + 架构信号），`asyncio.gather` 实现真正的并行。Coordinator 负责冲突消解、冗余消除和逻辑整合。

## 功能特性

- 两套完全自定义的神经网络架构（CustomModel + RecurrentModel）
- 真实的 PyTorch 数值计算，从 hidden states 提取统计特征
- 多 Agent 并行处理 + Coordinator 聚合
- OpenAI-compatible API 代理，支持 Cursor-IDE 直接接入
- SSE 流式输出支持
- 编码/设计任务自动检测，触发架构信号注入
