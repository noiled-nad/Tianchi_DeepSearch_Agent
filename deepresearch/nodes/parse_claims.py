# deepresearch/nodes/parse_claims.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import json
import os
import re
from typing import Callable, List

from langchain_core.messages import AIMessage, BaseMessage

try:
    from ..schemas import Claim
    from ..state import DeepResearchState
except ImportError:  # pragma: no cover - fallback for direct script execution
    from deepresearch.schemas import Claim
    from deepresearch.state import DeepResearchState


def _extract_last_user_question(messages: List[BaseMessage]) -> str:
    # 从最后一条用户消息里取问题（简单假设：最后一条 human message 就是问题）
    # langchain_core 里 HumanMessage.content 是 str
    for m in reversed(messages):
        role = getattr(m, "type", "")
        if role == "human":
            return str(m.content).strip()
    # 如果没有 human message，就兜底取最后一条
    return str(messages[-1].content).strip() if messages else ""


def _safe_json_loads(text: str):
    """
    从模型输出中清洗得到json
    """
    # 找到第一个 '[' 到最后一个 ']' 的片段
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    # 或者找 '{...}'
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("无法从模型输出中解析 JSON")


def make_parse_claims_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    """
    创建一个“解析约束”的图节点。

    该节点负责从用户的对话历史中提取最后的问题，并指示 LLM 将其拆解为一组可验证的约束条件（Claims）。

    Args:
        llm: 用于执行提取任务的大语言模型对象 (LangChain Runnable)。

    Returns:
        Callable[[DeepResearchState], DeepResearchState]: 
            符合 LangGraph 签名的节点函数。

            输入 State:
            - messages (List[BaseMessage]): 必须包含用户的问题（通常是最后一条 HumanMessage）。

            输出 State 更新:
            - question (str): 提取出的用户问题文本。
            - claims (List[Claim]): 拆解得到的约束列表。
            - messages (List[BaseMessage]): 追加一条表示进度的 AIMessage。
    """
    async def parse_claims(state: DeepResearchState) -> DeepResearchState:
        print("\n============ parse_claims 阶段 ============")
        question = _extract_last_user_question(state.get("messages", []))
        print(f"[parse_claims] question_len={len(question)}")

        prompt = f"""你是一个“可验证结构化约束抽取器”。任务：将用户问题拆解为可检索、可判定、可闭环验证的 Claim 列表。
        只能使用题目中出现的信息；禁止引入外部常识、猜测、补全具体人名/片名/币种/平台等。

        【输出】
        - 只输出一个 JSON 数组（list[Claim]），不要输出任何额外文本或 Markdown。
        【Claim 结构】
        每条 Claim 必须包含：
        - id: "claim_1"...
        - must: bool
        - claim_type: "anchor" | "disamb" | "bridge" | "output"
        - S: {{"name": string, "type": string|null}}
        - P: string  (snake_case 英文动词短语，2~5词)
        - O: {{"name": string, "type": string|null}} 或 null
        - Q: {{
            "time": string|null,
            "location": string|null,
            "disambiguation": string|null,
            "extra": object
        }}
        - description: string

        占位符规则：
        - 未知人物/作品/机构/城市用 Person_A / Writer_A / Director_B / Spouse_C / School_B / Movie_D / Series_Y / City_X 等。
        - 题目没给出的信息不要补；若有“未知但可能重要”的字段放到 Q.extra 里标记 unknown。

        【第一性原则：先建“可跑通的研究图”】
        1) 明确 output 主体实体（例如 Writer_A）与最终要输出的变量（例如 Series_Y 的中文名）。
        2) 列出题目出现的关键实体节点（人物/作品/机构）与它们之间的显式关系。
        3) 生成 claims 时必须保证“图连通”：
        - 除 output 外的所有关键实体，必须通过 1~2 跳关系链连接到 output 主体实体。
        - 若题面同时出现主体人物与另一实体（学校/配偶/兄弟/作品），必须显式产出连接边（如 married_to / has_sibling / co_created / graduated_from / adapted_from 等）。
        - 禁止产生与主体完全无连接的实体子图。

        【拆分粒度：One-snippet-per-claim + 允许轻量打包】
        - One-snippet-per-claim：理想上每条 claim 可被单一证据片段直接支持或反驳。
        - 允许“轻量打包”以避免过度原子化：
        - 同一关系簇或同一记录簇中、强相关且常共现的 2~3 个要素可以合并在一条 claim 的 description/Q.disambiguation 中。
        - 数值+年份/阈值必须保持在同一条 claim 中，不要拆散。

        【信号强度（Signal）判定：决定 claim_type 的硬规则】
        对每条候选事实先估计 signal_strength（写入 Q.extra.signal_strength）：
        - high：包含专名/作品名/机构名/编号/明确年份+数值阈值/明确亲属或合作关系/明确改编关系
        - medium：可用于区分但常需上下文（出生地、任职机构、合作者、语言、平台等）
        - low：泛化或模糊（欧洲人、某区域、叙事结构描述、风格/轶事、只给年代无更多限定等）

        【claim_type 定义（按控制流功能）】
        A) output（目标变量）
        - 定义最终要输出的槽位与格式约束，不承载证据细节。
        - output 通常 must=true。
        - 若题目有格式要求（如“不要标点符号”），写入 Q.extra.format_constraints。

        B) anchor（第一轮可用的候选生成约束）
        满足任一即可：
        1) signal_strength=high 且能直接用于“找人/找作品/找记录”的检索启动；
        2) 主体与其他关键实体之间的“强关系边”：
        - 亲属/配偶/长期合作/改编关系/共同创作关系等，只要它是题面核心线索，就应作为 anchor（即便是关系）。
        3) 明确年份+数值阈值/明确年份+作品类型 等可直接定位数据库记录的约束。
        注意：
        - 低信息增益的身份碎片（如“欧洲人”“70年代出生”）不要单独写成多条 anchor；
        应合并为 1 条 identity bundle（仍可标 anchor 但 signal_strength 多为 medium/low，并通常 must=false）。

        C) disamb（候选收敛约束）
        - 用于在同名/多候选情况下收敛到唯一实体；
        - 通常 signal_strength=medium/high，但需要先有候选集或先确定某实体；
        - 必须在 Q.extra 中写：
        - depends_on: [claim_id,...]（先确定谁/哪部作品）
        - trigger: "candidate_count>1" 或 "name_ambiguous_risk"
        - 如果题面没有明显同名风险，disamb 多数应 must=false。

        D) bridge（后续补链/软验证/低增益约束）
        - 不能有效启动第一轮检索，或必须在已知实体后才能验证；
        - 或者信息增益低、更多用于交叉验证（如叙事结构、模糊地域“南方”、风格描述、轶事等）。
        - bridge 默认 must=false，除非题面明确“必须满足”的硬约束。

        【must 标注（避免死锁）】
        must=true 仅给同时满足：
        1) 对定位答案关键（缺了它很难找对人/作品）；
        2) 公开资料高概率可得（常出现在简介/条目/片尾字幕/权威数据库条目中）；
        3) 表述相对明确（不是“可能/大概/某区域/叙事感觉”）。
        否则设 must=false，并在 Q.extra.must_reason 写 "soft_verify" / "often_missing" / "ambiguous" 等。

        【建议在 Q.extra 补充最小编排信息（不改变主 schema）】
        - facet_bucket: "identity" | "relation" | "work" | "constraint" | "anecdote"
        - exec_phase: "phase1_anchor" | "phase2_disamb" | "phase3_bridge" | "phase4_verify"
        - bundle_id: 用于把同一主体的 2~3 条强 anchor 分组（一次检索支持多条判定）

        【输入】
        用户问题：{question}

        现在开始输出 JSON 数组。"""
        print("[parse_claims] --- prompt begin ---")
        print(prompt)
        print("[parse_claims] --- prompt end ---")
        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        print(f"[parse_claims] LLM 原始输出：{raw}")
        try:
            arr = _safe_json_loads(raw)
            claims = [Claim(**c) for c in arr]
        except Exception:
            # 兜底：如果解析失败，就给一个最简 claim，至少流程能跑通
            claims = [
                Claim(
                    id="c1",
                    must=True,
                    claim_type="other",
                    S={"name": "用户问题", "type": "doc", "aliases": []},
                    P="answer_question",
                    O={"name": "最终答案", "type": "doc", "aliases": []},
                    Q={"time": None, "location": None, "quantity": None, "disambiguation": None, "extra": {}},
                    description="从网页证据中回答该问题"
                )
            ]

        progress_msg = AIMessage(content=f"[parse_claims] 已解析出 {len(claims)} 条约束。")
        print(f"[parse_claims] claims_count={len(claims)}")
        return {
            "question": question,
            "claims": claims,
            "messages": [progress_msg],
        }

    return parse_claims


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage

    load_dotenv()

    class _DummyLLM:
        """无需外部服务的演示 LLM，返回固定 JSON。"""

        async def ainvoke(self, prompt: str) -> AIMessage:
            _ = prompt  # 演示用，不解析 prompt
            sample = (
                '[{"id":"c1","description":"示例约束：回答 Python 由谁创建","must":true},'
                '{"id":"c2","description":"示例约束：给出创建年份","must":true}]'
            )
            return AIMessage(content=sample)

    async def _demo() -> None:
        from langchain_openai import ChatOpenAI
        if os.getenv("DASHSCOPE_API_KEY"):
            print(os.getenv("DASHSCOPE_API_KEY"))
            llm = ChatOpenAI(
                model=os.getenv("DEEPRESEARCH_MODEL", "qwen-plus"),
                api_key=os.getenv("DASHSCOPE_API_KEY"),

                base_url=os.getenv(
                    "DEEPRESEARCH_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
            
        else:
            print("缺少对应 API Key，回退到 Dummy LLM")
            llm = _DummyLLM()

        node = make_parse_claims_node(llm)

        test_question = (
            "有一位 20 世纪 70 年代出生的欧洲编剧兼作家，其哥哥和妻子均为导演。他的哥哥曾依据他创作的短篇小说拍摄了分为两条叙事线展开的电影。而他与妻子共同创作并编剧了多部电视剧。请问，在 2024 年夫妻二人共同创作的电视剧的中文译名叫什么？不要加任何标点符号。"
        )

        state: DeepResearchState = {
            "messages": [HumanMessage(content=test_question)],
            "question": test_question,   ## 补充 question state
        }

        new_state = await node(state)

        print(f"question: {new_state.get('question')}")
        claims = new_state.get("claims", [])
        print(f"claims: {len(claims)}")

        for c in claims:
            print(f"\n- {c.id} [{c.claim_type}] must={c.must}")
            print(f"  desc: {c.description}")
            print(f"  S: {c.S.name} (type={c.S.type})")
            print(f"  P: {c.P}")
            if c.O is None:
                print(f"  O: null")
            else:
                print(f"  O: {c.O.name} (type={c.O.type})")
            q = c.Q
            print(f"  Q.time: {q.time}")
            print(f"  Q.location: {q.location}")
            print(f"  Q.quantity: {q.quantity}")
            print(f"  Q.disambiguation: {q.disambiguation}")
            if q.extra:
                print(f"  Q.extra: {q.extra}")

    asyncio.run(_demo())
