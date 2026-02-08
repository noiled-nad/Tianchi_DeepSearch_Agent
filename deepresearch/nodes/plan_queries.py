# deepresearch/nodes/plan_queries.py
# -*- coding: utf-8 -*-
"""
节点2：plan_queries
根据 claims 生成多条检索 query，重点围绕“区分度最强”的约束。考虑结合问题，尽可能缩小每个query的搜索空间。
"""

from __future__ import annotations
import json
import os
import re
from typing import List,Callable

from langchain_core.messages import AIMessage## 自动为message配置role

from ..schemas import Claim
from ..state import DeepResearchState

def _safe_json_list(text: str) -> List[str]:
    """
    从文本中提取第一个 JSON 数组，解析为非空字符串列表。
    
    如果找不到或解析失败，抛出 ValueError。
    """
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    raise ValueError("无法解析 query JSON")

def make_plan_queries_node(llm)->Callable[[DeepResearchState],DeepResearchState]:
    async def plan_queries(state:DeepResearchState)->DeepResearchState:
        claims:List[Claim] = state.get("claims",[])
        question:str = state.get("question","")
        claims_text = "\n".join([f"- {c.id}: {c.description}" for c in claims])
        print(claims_text)
        prompt = (
            "你是检索策略助手。根据问题和约束，生成多用于网页搜索的查询语句。\n"
            "要求：\n"
            "1) 必须输出 JSON 数组，仅数组，不要输出其它文字。\n"
            "2) 生成五条queries，包括中文与英文。\n"
            "3) query 要尽量包含年份、金额阈值、专有名词、关键关系等强约束。\n\n"
            f"问题：{question}\n"
            f"约束：\n{claims_text}\n"
        )
        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        try:
            queries = _safe_json_list(raw)    ## 提取json数组解析返回的queries
        except Exception:  
            remember = question if question else "deep research"
            queries = [remember]
        
        progress_msg = AIMessage(content=f"[plan_queries] 生成了 {len(queries)} 条查询。")
        print(progress_msg)
        return {
            "queries":queries,
            "messages":[progress_msg]## 需要保持列表message，因为state是add reducer
        }
    return plan_queries



if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    class _DummyLLM:
        """演示用 LLM，返回固定的查询列表。"""

        async def ainvoke(self, prompt: str) -> AIMessage:
            _ = prompt
            sample = json.dumps(
                [
                    "艺术家A 2012 拍卖 超过500万",
                    "artist A auction 2021 over 20 million",
                    "school B named after writer C south china city",
                ]
            )
            return AIMessage(content=sample)

    async def _demo() -> None:
        # USE_REAL_LLM=1 可切换真实 LLM；默认使用 Dummy 便于离线验证
        from langchain_openai import ChatOpenAI

        if os.getenv("DASHSCOPE_API_KEY"):
            llm = ChatOpenAI(
                model=os.getenv("DEEPRESEARCH_MODEL", "qwen-plus"),
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv(
                    "DEEPRESEARCH_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
        else:
            print("USE_REAL_LLM=1 但缺少对应 API Key，回退到 Dummy LLM")
            llm = _DummyLLM()

        node = make_plan_queries_node(llm)

        test_question = (
            "某艺术家A，25岁毕业于某学校B，该学校B以某中国近代作家C命名，该作家C出生于中国南方城市。"
            "该艺术家作品在2012年和2021年分别拍出了超过500万和超过2000万的高价，请问该艺术家A是谁？"
        )

        # 直接复用 parse_claims 的示例拆解结果
        claims: List[Claim] = [
            Claim(id="1", description="艺术家A在25岁时从学校B毕业。", must=True),
            Claim(id="2", description="学校B以中国近代作家C的名字命名。", must=True),
            Claim(id="3", description="作家C出生于中国南方城市。", must=True),
            Claim(id="4", description="艺术家A的作品在2012年拍卖价格超过500万元人民币。", must=True),
            Claim(id="5", description="艺术家A的作品在2021年拍卖价格超过2000万元人民币。", must=True),
        ]

        state: DeepResearchState = {
            "question": test_question,
            "claims": claims,
        }

        try:
            new_state = await node(state)
        except Exception as exc:
            print(f"plan_queries 调用失败，回退 Dummy：{exc}")
            fallback_node = make_plan_queries_node(_DummyLLM())
            new_state = await fallback_node(state)

        print(f"question: {test_question}")
        print(f"queries: {len(new_state['queries'])}")
        for q in new_state["queries"]:
            print(f"- {q}")

    asyncio.run(_demo())
