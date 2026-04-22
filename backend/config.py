# 集中管理所有架构参数，避免魔法数字散落各处
# 改参数来这里改一处就行，不用满项目搜索

ARCHITECTURE_DEFAULTS = {
    "vocab_size": 50000,
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "window_size": 64,
    "memory_size": 128,
    "max_seq_len": 512,
    "dropout": 0.0,
    "num_experts": 4,
}

RECURRENT_DEFAULTS = {
    "vocab_size": 50000,
    "d_model": 512,
    "n_heads": 8,
    "n_kv_heads": 4,
    "max_loops": 4,
    "use_act": False,
    "max_seq_len": 8192,
}

SEMANTIC_ENCODER_DEFAULTS = {
    "d_model": 512,
    "semantic_dim": 64,
}

CODING_KEYWORDS = [
    "def ", "class ", "import ", "function", "代码", "编程",
    "bug", "debug", "报错", "React", "Vue", "组件", "UI", "前端",
]

DESIGN_KEYWORDS = [
    "设计", "UI", "界面", "布局", "样式", "组件", "React", "Vue", "Tailwind",
]


def detect_task_mode(user_content: str):
    # 统一的关键词检测，main.py 和 agent 模式共享同一套逻辑
    is_coding = any(kw in user_content for kw in CODING_KEYWORDS)
    is_design = any(kw in user_content for kw in DESIGN_KEYWORDS)
    return is_coding, is_design


def extract_user_content(messages: list) -> str:
    # 从 OpenAI 格式的 messages 中提取最后一条用户消息
    # 处理多模态 content（list 格式）和纯文本（str 格式）
    for msg in reversed(messages):
        if msg.get("role") == "user":
            raw = msg.get("content", "")
            if isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "\n".join(parts)
            return raw if isinstance(raw, str) else str(raw)
    return ""
