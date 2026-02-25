# deepresearch/nodes/judge_claims.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple

from langchain_core.messages import AIMessage

from deepresearch.schemas import Claim, Evidence, Verdict
from deepresearch.state import DeepResearchState


def _safe_json_obj(text: str):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def _group_evidence(notes: List[Evidence]) -> Dict[str, List[Evidence]]:
    by = {}
    for e in notes:
        by.setdefault(e.claim_id, []).append(e)
    # relevance top-k
    for k in by:
        by[k].sort(key=lambda x: x.relevance, reverse=True)
        by[k] = by[k][:5]
    return by


def make_judge_claims_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    async def judge_claims(state: DeepResearchState) -> DeepResearchState:
        print("\n============ judge_claims 阶段 ============")
        claims: List[Claim] = state.get("claims", [])
        notes: List[Evidence] = state.get("notes", [])
        iteration = int(state.get("iteration", 0))
        state.setdefault("max_iterations", 6)
        print(f"[judge_claims] iteration={iteration}, claims={len(claims)}, notes={len(notes)}")

        by_claim = _group_evidence(notes)

        # 只判 must + 高价值类型（先最小闭环）
        focus = [c for c in claims if c.must or c.claim_type in ("disamb", "anchor", "output")]
        focus = focus[:20]
        print(f"[judge_claims] focus_claims={len(focus)}")
        for c in focus:
            print(f"[judge_claims] claim: id={c.id}, type={c.claim_type}, must={c.must}, desc={c.description}")

        claims_text = "\n".join(
            f"- {c.id} ({c.claim_type}, must={c.must}): S={c.S.name} P={c.P} O={(c.O.name if c.O else '∅')} Q={c.Q.model_dump()} desc={c.description}"
            for c in focus
        )

        ev_text_lines = []
        for c in focus:
            evs = by_claim.get(c.id, [])
            if not evs:
                continue
            ev_text_lines.append(f"## Evidence for {c.id}")
            for i, e in enumerate(evs, start=1):
                ev_text_lines.append(
                    f"[{c.id}-E{i}] url={e.url}\nquote={e.quote}\nstance={e.stance} relevance={e.relevance}"
                )
        ev_text = "\n\n".join(ev_text_lines) if ev_text_lines else "（无证据）"
        if ev_text_lines:
            print("[judge_claims] --- grouped evidence begin ---")
            print(ev_text)
            print("[judge_claims] --- grouped evidence end ---")

        prompt = (
            "你是 claim 判定器。你必须仅基于提供的 Evidence 判断每条 claim。\n"
            "输出严格 JSON 对象：\n"
            "{\n"
            '  "verdicts": {"claim_id": "supported|refuted|unknown", ...},\n'
            '  "missing_info": ["下一轮需要补查的缺口描述1", "缺口描述2", ...]\n'
            "}\n"
            "规则：\n"
            "1) 没有证据或证据不直接相关 -> unknown。\n"
            "2) Evidence quote 明确支持 -> supported；明确矛盾 -> refuted。\n"
            "3) missing_info 要具体到“查什么”而不是泛泛说‘需要更多证据’。\n\n"
            f"Claims:\n{claims_text}\n\n"
            f"Evidence:\n{ev_text}\n"
        )
        print("[judge_claims] --- prompt begin ---")
        print(prompt)
        print("[judge_claims] --- prompt end ---")

        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        print(f"[judge_claims] raw_response={raw}")

        verdicts: Dict[str, Verdict] = dict(state.get("claim_verdicts", {}))
        missing_info: List[str] = []

        try:
            obj = _safe_json_obj(raw)
            new_v = obj.get("verdicts", {})
            for cid, v in new_v.items():
                if v in ("supported", "refuted", "unknown"):
                    verdicts[cid] = v
            mi = obj.get("missing_info", [])
            if isinstance(mi, list):
                missing_info = [str(x).strip() for x in mi if str(x).strip()]
        except Exception:
            # 兜底：不更新 verdicts
            pass

        msg = AIMessage(content=f"[judge_claims] iteration={iteration} 已判定 {len(verdicts)} 条 claim，缺口 {len(missing_info)} 条。")
        print(f"[judge_claims] verdicts={len(verdicts)}, missing_info={len(missing_info)}")
        for cid, v in verdicts.items():
            print(f"[judge_claims] verdict: {cid} -> {v}")
        for i, miss in enumerate(missing_info, start=1):
            print(f"[judge_claims] missing_{i}: {miss}")
        return {"claim_verdicts": verdicts, "missing_info": missing_info, "iteration": iteration + 1, "messages": [msg]}

    return judge_claims


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from deepresearch.schemas import Claim, Evidence, EntityRef, Qualifier

    # Dummy LLM for testing
    class _DummyLLM:
        async def ainvoke(self, prompt: str):
            # Simulate LLM response with sample verdicts and missing info
            sample_response = {
                "verdicts": {
                    "claim_1": "supported",
                    "claim_2": "unknown",
                    "claim_3": "refuted"
                },
                "missing_info": [
                    "需要查艺术家A的具体毕业年份",
                    "确认学校B的命名来源"
                ]
            }
            from langchain_core.messages import AIMessage
            return AIMessage(content=str(sample_response).replace("'", '"'))

    async def _demo():
        # Create sample claims
        claims = [
            Claim(
                id="claim_1",
                must=True,
                claim_type="output",
                S=EntityRef(name="艺术家A"),
                P="毕业于",
                O=EntityRef(name="学校B"),
                Q=Qualifier(),
                description="艺术家A在25岁时从学校B毕业。"
            ),
            Claim(
                id="claim_2",
                must=True,
                claim_type="bridge",
                S=EntityRef(name="学校B"),
                P="以命名",
                O=EntityRef(name="作家C"),
                Q=Qualifier(),
                description="学校B以中国近代作家C的名字命名。"
            ),
            Claim(
                id="claim_3",
                must=False,
                claim_type="disamb",
                S=EntityRef(name="作家C"),
                P="出生于",
                O=EntityRef(name="南方城市"),
                Q=Qualifier(),
                description="作家C出生于中国南方城市。"
            )
        ]

        # Create sample evidence
        notes = [
            Evidence(
                claim_id="claim_1",
                url="https://example.com/artist",
                title="艺术家传记",
                quote="艺术家A于25岁毕业于学校B。",
                relevance=0.9,
                stance="supports"
            ),
            Evidence(
                claim_id="claim_2",
                url="https://example.com/school",
                title="学校历史",
                quote="学校B以作家C命名。",
                relevance=0.8,
                stance="supports"
            )
        ]

        # Initial state
        state: DeepResearchState = {
            "claims": claims,
            "notes": notes,
            "iteration": 0,
            "max_iterations": 6
        }

        # Create and run node
        llm = _DummyLLM()
        node = make_judge_claims_node(llm)
        new_state = await node(state)

        # Print results
        print("=== 输入状态 ===")
        print(f"Claims: {len(claims)} 条")
        for c in claims:
            print(f"  - {c.id}: {c.description}")
        print(f"Notes: {len(notes)} 条")
        for n in notes:
            print(f"  - {n.claim_id}: {n.quote[:50]}...")

        print("\n=== 输出状态 ===")
        verdicts = new_state.get("claim_verdicts", {})
        print(f"Verdicts: {verdicts}")
        missing_info = new_state.get("missing_info", [])
        print(f"Missing Info: {missing_info}")
        iteration = new_state.get("iteration", 0)
        print(f"Iteration: {iteration}")
        messages = new_state.get("messages", [])
        if messages:
            print(f"Message: {messages[0].content}")

    asyncio.run(_demo())
