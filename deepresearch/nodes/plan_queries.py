# deepresearch/nodes/plan_queries.py
# -*- coding: utf-8 -*-
"""
节点2：plan_queries
根据 claims 生成多条检索 query。利用 claim_type 做差异化检索，并读取 state.batch_size 控制 query 数量。
"""

from __future__ import annotations

import json
import re
import os
from typing import List, Callable, Dict, Any

from langchain_core.messages import AIMessage

from ..schemas import Claim
from ..state import DeepResearchState

def _safe_json_list(text: str) -> List[str]:
    """从文本中提取第一个 JSON 数组，解析为非空字符串列表。"""
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    raise ValueError("无法解析 query JSON")


def _claim_to_search_brief(c: Claim) -> str:
    """
    给 LLM 的“可检索摘要”：包含 claim_type/must + S/P/O/Q（比 description 更稳定）。
    """
    s = c.S.name if c.S else ""
    o = c.O.name if c.O else ""
    q = c.Q.model_dump() if c.Q else {}
    # 注意：Q.disambiguation 在你的 schema 里是 Optional[str]，保持原样输出即可
    return (
        f"[{c.id}] must={c.must} type={c.claim_type}\n"
        f"  S={s}\n"
        f"  P={c.P}\n"
        f"  O={o or '∅'}\n"
        f"  Q={json.dumps(q, ensure_ascii=False)}\n"
        f"  desc={c.description}"
    )

def make_plan_queries_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def plan_queries(state: DeepResearchState) -> DeepResearchState:
        print("\n============ plan_queries 阶段 ============")
        claims: List[Claim] = state.get("claims", [])
        question: str = state.get("question", "")

        # ✅ 读取调度参数（第 1 步你已加入）
        batch_size = int(state.get("batch_size", 5))
        # query 数：至少 5，最多 12
        num_queries = max(5, min(12, batch_size + 2))
        print(f"[plan_queries] claims={len(claims)}, batch_size={batch_size}, target_queries={num_queries}")

        # ✅ 简单优先级：disamb/anchor/output 先展示（更像 Lemon/GAIA 的策略）
        type_rank = {"disamb": 0, "anchor": 1, "output": 2, "bridge": 3, "other": 4}
        claims_sorted = sorted(
            claims,
            key=lambda c: (0 if c.must else 1, type_rank.get(c.claim_type, 9)),
        )

        # 只给 LLM 前 batch_size 条重点约束（避免 prompt 过长）
        focus_claims = claims_sorted[: max(3, min(len(claims_sorted), batch_size))]
        claims_text = "\n".join(_claim_to_search_brief(c) for c in focus_claims)

        # ✅ 新增：读取 missing_info，用于“失败诊断回灌”
        missing = state.get("missing_info", [])
        missing_text = "\n".join(f"- {x}" for x in missing) if missing else "（无）"
        print(f"[plan_queries] missing_info_count={len(missing)}")

        prompt = (
            "你是检索策略助手。目标：为一个“谜题式、多跳、强约束、同名消歧”的问题生成网页搜索 queries。\n\n"
            "硬性输出格式要求：\n"
            f"1) 必须输出 JSON 数组，仅数组，不要输出其它文字。\n"
            f"2) 输出 {num_queries} 条 query。\n"
            "3) 必须同时包含中文和英文 queries（至少各 2 条）。\n"
            "4) query 要尽量包含强约束：年份/金额阈值/专名/关键关系/地名/机构全称。\n"
            "5) 对不同 claim_type 使用不同策略：\n"
            "   - disamb：优先唯一识别特征（出生年、作品名、机构所在地、别名），避免泛词。\n"
            "   - anchor：优先权威来源词（官网/百科/学术/数据库）+ 实体全称。\n"
            "   - output：包含年份+拍卖+金额阈值+币种+作品/艺术家名线索。\n"
            "   - bridge：包含关系词（born in / named after / graduated from / auction record）。\n"
            "6) 不要把多个事实打包成一个 query；每条 query 聚焦一个最强约束组合。\n\n"
            f"问题：{question}\n\n"
            "重点约束（结构化 claims）：\n"
            f"{claims_text}\n\n"
            f"上一轮缺口（用于改写查询）：\n{missing_text}\n"
        )
        print("[plan_queries] --- prompt begin ---")
        print(prompt)
        print("[plan_queries] --- prompt end ---")

        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        print(f"[plan_queries] raw_response={raw}")

        try:
            queries = _safe_json_list(raw)
        except Exception:
            remember = question if question else "deep research"
            queries = [remember]

        progress_msg = AIMessage(content=f"[plan_queries] 生成了 {len(queries)} 条查询（目标 {num_queries}）。")
        print(f"[plan_queries] generated_queries={len(queries)}")
        for i, q in enumerate(queries, start=1):
            print(f"[plan_queries] q{i}: {q}")
        return {"queries": queries, "messages": [progress_msg]}

    return plan_queries



if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from dotenv import load_dotenv
    from deepresearch.schemas import Claim, EntityRef, Qualifier

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
            Claim(
                id="1",
                must=True,
                claim_type="bridge",
                S=EntityRef(name="艺术家A"),
                P="毕业于",
                O=EntityRef(name="学校B"),
                Q=Qualifier(),
                description="艺术家A在25岁时从学校B毕业。"
            ),
            Claim(
                id="2",
                must=True,
                claim_type="bridge",
                S=EntityRef(name="学校B"),
                P="以命名",
                O=EntityRef(name="作家C"),
                Q=Qualifier(),
                description="学校B以中国近代作家C的名字命名。"
            ),
            Claim(
                id="3",
                must=True,
                claim_type="disamb",
                S=EntityRef(name="作家C"),
                P="出生于",
                O=EntityRef(name="南方城市"),
                Q=Qualifier(),
                description="作家C出生于中国南方城市。"
            ),
            Claim(
                id="4",
                must=True,
                claim_type="output",
                S=EntityRef(name="艺术家A"),
                P="作品拍卖",
                O=None,
                Q=Qualifier(quantity=5000000, extra={"year": 2012, "currency": "CNY"}),
                description="艺术家A的作品在2012年拍卖价格超过500万元人民币。"
            ),
            Claim(
                id="5",
                must=True,
                claim_type="output",
                S=EntityRef(name="艺术家A"),
                P="作品拍卖",
                O=None,
                Q=Qualifier(quantity=20000000, extra={"year": 2021, "currency": "CNY"}),
                description="艺术家A的作品在2021年拍卖价格超过2000万元人民币。"
            ),
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
