# AI 自研架构项目

完全自定义的神经网络架构，替代传统 Transformer，优化编码能力和推理能力。

## 技术栈

- Python 3.9+
- PyTorch（自定义架构实现）
- FastAPI（后端服务）
- SQLite3（数据持久化）
- SQLAlchemy（ORM 框架）

## 项目结构

```
自研flow/
├── backend/                  # 后端代码
│   ├── architecture/         # 自定义神经网络架构
│   │   ├── custom_model.py   # 完整模型
│   │   ├── custom_block.py   # 基础块
│   │   ├── hybrid_attention.py   # 混合注意力
│   │   ├── dynamic_memory.py     # 动态记忆
│   │   └── gated_ffn.py          # 门控前馈
│   ├── main.py              # FastAPI 服务
│   ├── architecture_processor.py  # 架构处理器（真实数值计算）
│   ├── architecture_injector.py   # 架构注入器
│   ├── multi_agent_executor.py    # 多 Agent 执行器
│   └── api_client.py        # API 客户端
├── test/                    # 测试文件
├── requirements.txt         # Python 依赖
├── .env                     # 环境变量
└── CLAUDE.md                # Claude 角色配置
```

## 快速开始

### 环境配置

编辑 `.env` 文件配置 API 接入信息：

```bash
api="https://chatapi.stepfun.com/chatapi/v1/chat/completions"
model="step-3.5-flash"
key="your-api-key-here"
```

### 启动后端

```bash
cd backend
pip install -r ../requirements.txt
python main.py
```

服务默认运行在 `http://localhost:8000`。

## API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/architecture/info` | GET | 获取当前架构信息 |
| `/architecture/model-info` | GET | 获取模型详细信息 |
| `/architecture/status` | GET | 获取处理器运行状态 |
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

Cursor-IDE 支持配置自定义 OpenAI-compatible API，将后端服务接入后，Cursor 的 AI 聊天（Chat / Composer / Tab Completion）都会自动通过自研架构注入后的提示词调用外部模型。

### 配置步骤

1. **确认后端服务已启动**

   ```bash
   cd backend
   python main.py
   ```

   确保终端显示 `FastAPI 服务启动成功`，端口为 `8000`。

2. **在 Cursor 中打开模型设置**

   点击左下角设置图标，选择 `Cursor Settings` → `Models`。

3. **添加自定义模型**

   在 `OpenAI API` 或 `Custom` 区域配置：

   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: 任意非空字符串（后端只做代理转发，实际 Key 从 `.env` 读取）
   - **Model**: `custom-architecture`（可任意填写，后端会忽略并调用 `.env` 中配置的模型）

4. **关闭其他模型，仅保留自定义模型**

   取消勾选 Cursor 内置的其他模型，仅保留刚才添加的自定义端点。这样所有 AI 请求都会路由到本地后端。

5. **开始使用**

   在 Cursor 的 Chat 或 Composer 中正常提问即可。后端会自动：
   - 提取你的问题内容
   - 使用 CustomModel 进行真实 PyTorch 计算，提取 hidden states 数值特征
   - 多 Agent 并行分析（ContextAnalyst、LogicEngineer、CodeArchitect、QualityAuditor）
   - Coordinator 聚合所有 Agent 输出，生成统一回复
   - 转发给 `step-3.5-flash`（或 `.env` 中配置的其他模型）
   - 返回响应给 Cursor

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

而 Cursor 中配置的 `/v1/chat/completions` 端点**默认强制启用**自研架构注入。

## 核心架构

### 1. 混合注意力机制 (HybridAttention)

- **线性注意力分支**: O(n) 复杂度，使用特征映射避免显式计算注意力矩阵
- **局部窗口注意力分支**: 捕捉局部细节，支持相对位置编码
- **自适应门控**: 动态融合全局和局部信息

### 2. 动态记忆模块 (DynamicMemory)

- 可学习的记忆矩阵
- 读写门、遗忘门控制信息流动
- 记忆压缩机制处理长序列

### 3. 门控前馈网络 (GatedFFN)

- GLU (Gated Linear Unit) 变体
- 专家混合 (MoE)，4 个专家
- 路由网络智能选择专家

### 4. 架构处理器 (ArchitectureProcessor)

不同于传统的提示词装饰，ArchitectureProcessor 使用真实的 PyTorch 计算：

1. 对输入文本进行 tokenization
2. 通过 CustomModel 进行 forward 传播
3. 从各层提取真实的 hidden states、gate weights、memory states
4. 将数值特征编码为结构化信号
5. 将信号与用户问题拼接，发给外部 LLM

这样外部 LLM 接收到的输入已经被自定义架构做了数值化预处理。

## 多 Agent 系统

| Agent | 角色 | 职责 |
|-------|------|------|
| ContextAnalyst | 上下文分析专家 | 语义解析、意图识别、歧义消解 |
| LogicEngineer | 逻辑推理专家 | 因果分析、形式化推理、漏洞检测 |
| KnowledgeSynthesizer | 知识综合专家 | 知识映射、跨域关联、事实校验 |
| CodeArchitect | 代码架构专家 | 算法设计、代码生成、架构评审 |
| QualityAuditor | 质量审核专家 | 事实核查、偏见检测、安全审查 |
| Coordinator | 协调中枢 | 冲突仲裁、信息整合、最终输出 |

## 功能特性

- 完全自定义神经网络架构（混合注意力、动态记忆、门控机制）
- 真实的 PyTorch 数值计算，提取 hidden states 特征
- 多 Agent 并行处理，Coordinator 聚合
- OpenAI-compatible API 代理，支持 Cursor-IDE 直接接入
- SSE 流式输出支持
- 代码/设计任务自动触发优化框架
