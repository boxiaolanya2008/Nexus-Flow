# API 调用模块：连接 step-3.5-flash
import aiohttp
import asyncio
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import time
import json
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class APIClient:
    """API 客户端，用于调用 step-3.5-flash"""
    
    def __init__(self):
        self.api_url = os.getenv('api')
        self.model = os.getenv('model')
        self.api_key = os.getenv('key')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_api(
        self,
        prompt: str,
        use_custom_architecture: bool = False,
        architecture_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        调用 API
        支持透传 tools（Agent 模式）和完整 messages 链
        """
        start_time = time.time()
        
        # 构建请求体
        if messages is not None:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            }
        
        # 如果使用自定义架构，注入架构参数
        if use_custom_architecture and architecture_config:
            payload["architecture"] = architecture_config
            payload["use_custom_architecture"] = True
        
        # Agent 模式：透传 tools
        if tools:
            payload["tools"] = tools
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            async with self.session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                response_time = (time.time() - start_time) * 1000  # 转换为毫秒
                
                if response.status == 200:
                    result = await response.json()
                    
                    # 记录原始响应以便调试
                    logger.info(f"Raw API response: {result}")
                    
                    # 提取响应内容
                    content = ""
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                    elif "data" in result and len(result["data"]) > 0:
                        # 尝试其他可能的响应格式
                        content = result["data"][0].get("content", "")
                    elif "content" in result:
                        # 直接的 content 字段
                        content = result["content"]
                    
                    logger.info(f"Extracted content: {content}")
                    
                    # 估算 token 数量（粗略估计：中文字符数）
                    tokens_generated = len(content)
                    throughput = tokens_generated / (response_time / 1000) if response_time > 0 else 0
                    
                    return {
                        "success": True,
                        "response": content,
                        "response_time": response_time,
                        "tokens_generated": tokens_generated,
                        "throughput": throughput,
                        "raw_response": result
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"API 调用失败: {response.status}, 错误详情: {error_text}")
                    return {
                        "success": False,
                        "error": f"API 调用失败: {response.status}",
                        "error_detail": error_text,
                        "response_time": response_time
                    }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"请求异常: {str(e)}")
            return {
                "success": False,
                "error": f"请求异常: {str(e)}",
                "response_time": response_time
            }
    
    async def call_api_stream(
        self,
        prompt: str,
        use_custom_architecture: bool = False,
        architecture_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ):
        """
        流式调用 API，返回异步生成器
        支持透传 tools（Agent 模式）和完整 messages 链
        """
        start_time = time.time()
        
        if messages is not None:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": True
            }
        
        if use_custom_architecture and architecture_config:
            payload["architecture"] = architecture_config
            payload["use_custom_architecture"] = True
        
        if tools:
            payload["tools"] = tools
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            async with self.session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    # 更健壮的 SSE 解析：支持 data: 前缀、JSON Lines、以及可能的格式变体
                    buffer = ""
                    has_finish = False
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line:
                                continue
                            raw_data = None
                            if line.startswith('data: '):
                                data_str = line[6:]
                                if data_str == '[DONE]':
                                    yield {"content": "", "done": True, "has_finish": has_finish, "response_time": (time.time() - start_time) * 1000}
                                    return
                                try:
                                    raw_data = json.loads(data_str)
                                except:
                                    continue
                            elif line.startswith('{'):
                                try:
                                    raw_data = json.loads(line)
                                except:
                                    continue
                            if raw_data and 'choices' in raw_data and len(raw_data.get('choices', [])) > 0:
                                choice = raw_data['choices'][0]
                                delta = choice.get('delta', {}) or {}
                                content = delta.get('content', '')
                                tool_calls = delta.get('tool_calls')
                                finish_reason = choice.get('finish_reason')
                                if content or tool_calls is not None or finish_reason is not None:
                                    if finish_reason:
                                        has_finish = True
                                    yield {
                                        "content": content,
                                        "tool_calls": tool_calls,
                                        "finish_reason": finish_reason,
                                        "done": False,
                                        "raw": raw_data,
                                        "response_time": (time.time() - start_time) * 1000
                                    }
                    # 处理 buffer 中剩余的内容
                    if buffer.strip():
                        line = buffer.strip()
                        raw_data = None
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str != '[DONE]':
                                try:
                                    raw_data = json.loads(data_str)
                                except:
                                    pass
                        elif line.startswith('{'):
                            try:
                                raw_data = json.loads(line)
                            except:
                                pass
                        if raw_data and 'choices' in raw_data and len(raw_data.get('choices', [])) > 0:
                            choice = raw_data['choices'][0]
                            delta = choice.get('delta', {}) or {}
                            content = delta.get('content', '')
                            tool_calls = delta.get('tool_calls')
                            finish_reason = choice.get('finish_reason')
                            if content or tool_calls is not None or finish_reason is not None:
                                if finish_reason:
                                    has_finish = True
                                yield {
                                    "content": content,
                                    "tool_calls": tool_calls,
                                    "finish_reason": finish_reason,
                                    "done": False,
                                    "raw": raw_data,
                                    "response_time": (time.time() - start_time) * 1000
                                }
                    yield {"content": "", "done": True, "has_finish": has_finish, "response_time": (time.time() - start_time) * 1000}
                else:
                    error_text = await response.text()
                    logger.error(f"流式 API 调用失败: {response.status}")
                    yield {"error": f"API 调用失败: {response.status}", "done": True}
        except Exception as e:
            logger.error(f"流式请求异常: {str(e)}")
            yield {"error": f"请求异常: {str(e)}", "done": True}

    async def batch_call(
        self,
        prompts: list,
        use_custom_architecture: bool = False,
        architecture_config: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        批量调用 API
        """
        tasks = [
            self.call_api(prompt, use_custom_architecture, architecture_config)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
