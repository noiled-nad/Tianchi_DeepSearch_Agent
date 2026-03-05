# -*- coding: utf-8 -*-
"""
配置管理
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _getenv(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    小工具：读取环境变量，并在 required=True 时做校验。
    """
    val = os.getenv(name, default)
    if required and (val is None or val.strip() == ""):
        raise RuntimeError(
            f"缺少环境变量 {name}。请在 .env 中配置，例如：\n"
            f"{name}=your_value"
        )
    return val


@dataclass()
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.2
    enable_thinking: bool = False
    max_tokens: int = 2000


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        model=_getenv("DEEPRESEARCH_MODEL", default="qwen3.5-plus"),
        api_key=_getenv("DASHSCOPE_API_KEY", required=True),
        base_url=_getenv(
            "DEEPRESEARCH_BASE_URL",
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=float(_getenv("DEEPRESEARCH_TEMPERATURE", default="0.2")),
        enable_thinking=_getenv("ENABLE_THINKING", default="false").lower() in ("true", "1", "yes"),
        max_tokens=int(_getenv("MAX_TOKENS", default="2000")),
    )


def create_llm():
    from langchain_openai import ChatOpenAI

    cfg = load_llm_config()
    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
    )


class ThinkingLLM:
    """支持 enable_thinking 的自定义 LLM"""

    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model
        self.enable_thinking = cfg.enable_thinking
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        # Convert LangChain messages to OpenAI format
        openai_messages = []
        if isinstance(messages, str):
            openai_messages = [{"role": "user", "content": messages}]
        else:
            for msg in messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type  # e.g., 'system', 'human', 'ai'
                    if role == 'human':
                        role = 'user'
                    elif role == 'ai':
                        role = 'assistant'
                    openai_messages.append({"role": role, "content": msg.content})
                else:
                    # Fallback if not BaseMessage
                    openai_messages.append({"role": "user", "content": str(msg)})

        extra_body = {"enable_thinking": True} if self.enable_thinking else {}
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            extra_body=extra_body,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        content = completion.choices[0].message.content
        return AIMessage(content=content)


def create_llm():
    cfg = load_llm_config()
    if cfg.enable_thinking:
        return ThinkingLLM(cfg)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
        )


def create_flash_llm():
    """
    创建 qwen-flash 模型实例 —— 用于 query_optimize 等轻量推理节点。
    快、便宜、适合 reflect/rollout 这种不需要深度推理的场景。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_getenv("FLASH_MODEL", default="qwen-flash"),
        api_key=_getenv("DASHSCOPE_API_KEY", required=True),
        base_url=_getenv(
            "DEEPRESEARCH_BASE_URL",
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=float(_getenv("FLASH_TEMPERATURE", default="0.3")),
    )


def enable_langsmith_tracing_from_env() -> None:
    """
    想开 LangSmith tracing，可以在 .env 里配置 :ENABLE_LANGSMITH=1
    需要时，在 app.py/agent.py 的初始化里手动调用一次即可。
    """
    enable = _getenv("ENABLE_LANGSMITH", default="0")
    if enable not in ("1", "true", "True", "YES", "yes"):
        return

    # 可配置
    os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_OTEL_ONLY"] = "true"
