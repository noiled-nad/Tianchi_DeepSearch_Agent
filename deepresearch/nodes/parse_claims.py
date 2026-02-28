"""deepresearch/nodes/parse_claims.py

兼容旧文件名，但逻辑改为：
1) 提炼研究目标（research_brief）
2) 直接给出首轮查询（queries）

不再强依赖 SPOQ/Claim 拆分。
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List

from langchain_core.messages import AIMessage, BaseMessage

from ..state import DeepResearchState


def _extract_last_user_question(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            return str(m.content).strip()
    return str(messages[-1].content).strip() if messages else ""


def _safe_json_obj(text: str) -> Dict:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def _fallback_queries(question: str) -> List[str]:
    q = question.strip()
    if not q:
        return ["deep research answer"]
    return [q, f"{q} 官方 资料", f"{q} wikipedia"]


def make_parse_claims_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def parse_claims(state: DeepResearchState) -> DeepResearchState:
        print("\n============ parse_claims(brief) 阶段 ============")
        question = state.get("question") or _extract_last_user_question(state.get("messages", []))
        print(f"[parse_claims] question_len={len(question)}")

        prompt = (
            "你是 Deep Research 的任务分析器。请把问题转成“研究简报”，不要做 SPOQ 拆分。\n"
            "输出必须是 JSON 对象，字段固定如下：\n"
            "{\n"
            "  \"objective\": string,\n"
            "  \"answer_format\": string,\n"
            "  \"hard_constraints\": [string],\n"
            "  \"key_entities\": [string],\n"
            "  \"initial_queries\": [string],\n"
            "  \"done_criteria\": [string]\n"
            "}\n"
            "规则：\n"
            "1) initial_queries 给 5~8 条，短查询优先，覆盖中英双语。\n"
            "2) query 聚焦可检索锚点（年份、作品名、关系词、机构名）。\n"
            "3) 不要输出 markdown，不要额外解释。\n\n"
            f"问题：{question}"
        )

        try:
            resp = await llm.ainvoke(prompt)
            obj = _safe_json_obj(str(resp.content))
        except Exception:
            obj = {}

        queries = [str(x).strip() for x in obj.get("initial_queries", []) if str(x).strip()]
        if not queries:
            queries = _fallback_queries(question)

        brief = {
            "objective": str(obj.get("objective", "根据证据回答用户问题")).strip(),
            "answer_format": str(obj.get("answer_format", "简洁文本答案")).strip(),
            "hard_constraints": [str(x).strip() for x in obj.get("hard_constraints", []) if str(x).strip()],
            "key_entities": [str(x).strip() for x in obj.get("key_entities", []) if str(x).strip()],
            "done_criteria": [str(x).strip() for x in obj.get("done_criteria", []) if str(x).strip()],
        }

        progress_msg = AIMessage(content=f"[brief] 已生成研究简报，首轮查询 {len(queries)} 条。")
        return {
            "question": question,
            "research_brief": brief,
            "queries": queries,
            "query_history": queries[:],
            "research_gaps": [],
            "needs_followup": True,
            "iteration": int(state.get("iteration", 0)),
            "max_iterations": int(state.get("max_iterations", 4)),
            "claims": [],  # 兼容旧调用方
            "messages": [progress_msg],
        }

    return parse_claims
