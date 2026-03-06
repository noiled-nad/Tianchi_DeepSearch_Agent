# -*- coding: utf-8 -*-
"""
约束提取 Agent

专门用于从问题中提取原子化的、可验证的约束条件。
只在问题开始时执行一次，提取结果用于：
1. 引导 parse_claims 的子任务规划
2. 验证 agent 的答案验证
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Any

from langchain_core.messages import AIMessage
from .state import DeepResearchState
from .prompt_loader import load_prompt


CONSTRAINT_EXTRACT_PROMPT = """
你是约束提取专家。请从问题中提取所有可验证的约束条件。

## 原始问题
{question}

## 提取要求

请提取以下类型的约束：

### 1. 实体约束（entity_constraints）
- 问题中涉及的实体必须满足的属性
- 格式：{{"entity": "实体描述", "attribute": "属性名", "value": "期望值", "source": "原文依据"}}

### 2. 关系约束（relation_constraints）
- 实体之间的关系必须满足的条件
- 格式：{{"subject": "主体", "relation": "关系", "object": "客体", "condition": "附加条件"}}

### 3. 答案约束（answer_constraints）
- 最终答案必须满足的条件
- 格式：{{"constraint": "约束内容", "type": "must_satisfy|should_satisfy", "verifiable": true/false}}

### 4. 格式约束（format_constraint）
- 答案的格式要求（如：人名/年份/游戏名）
- 格式：{{"expected_type": "类型", "description": "描述"}}

## 提取原则

1. **原子化**：每个约束应该是最小可验证单元
2. **可验证**：约束应该能够通过事实核查来验证
3. **无歧义**：约束描述应该清晰明确
4. **完整覆盖**：提取所有对答案有影响的约束

## 输出格式（严格 JSON，不要 markdown）

{{
  "entity_constraints": [
    {{"entity": "亚洲母公司", "attribute": "业务领域", "value": "量子计算", "source": "涉足量子计算"}},
    {{"entity": "北美公司", "attribute": "产品", "value": "MOBA游戏", "source": "MOBA 公司"}}
  ],
  "relation_constraints": [
    {{"subject": "亚洲母公司", "relation": "收购", "object": "北美公司", "condition": "2010年代全资收购"}}
  ],
  "answer_constraints": [
    {{"constraint": "格斗手游由亚洲母公司代理发行", "type": "must_satisfy", "verifiable": true}},
    {{"constraint": "角色为年龄偏大的武术教官", "type": "must_satisfy", "verifiable": true}},
    {{"constraint": "角色名与FPS武器名相同", "type": "must_satisfy", "verifiable": true}}
  ],
  "format_constraint": {{"expected_type": "游戏名称", "description": "格斗手游的名称"}},
  "key_entities": ["实体1", "实体2"],
  "summary": "约束摘要（一句话）"
}}
"""


def _safe_json_obj(text: str) -> Dict:
    """从文本中提取 JSON 对象"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    t = t.strip()

    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 对象
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}

    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def make_constraint_extractor_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    """
    约束提取节点。

    从问题中提取原子化的约束条件，存储到 state["constraints"]。
    """

    async def extract_constraints(state: DeepResearchState) -> DeepResearchState:
        print("\n============ constraint_extract 阶段 ============")

        question = state.get("question", "")
        if not question:
            print("[constraint_extract] 无问题，跳过")
            return {"constraints": {}}

        print(f"[constraint_extract] question_len={len(question)}")

        prompt = CONSTRAINT_EXTRACT_PROMPT.format(question=question)
        print(f"[constraint_extract] prompt_len={len(prompt)}")

        try:
            resp = await llm.ainvoke(prompt)
            raw = str(resp.content).strip()
            print(f"[constraint_extract] raw_len={len(raw)}")

            constraints = _safe_json_obj(raw)

            if not constraints:
                print("[constraint_extract] JSON 解析失败，使用空约束")
                constraints = {}

            # 打印提取结果
            entity_constraints = constraints.get("entity_constraints", [])
            relation_constraints = constraints.get("relation_constraints", [])
            answer_constraints = constraints.get("answer_constraints", [])
            format_constraint = constraints.get("format_constraint", {})

            print(f"[constraint_extract] entity_constraints={len(entity_constraints)}")
            for c in entity_constraints:
                print(f"  - {c.get('entity')}.{c.get('attribute')} = {c.get('value')}")

            print(f"[constraint_extract] relation_constraints={len(relation_constraints)}")
            for c in relation_constraints:
                print(f"  - {c.get('subject')} {c.get('relation')} {c.get('object')}")

            print(f"[constraint_extract] answer_constraints={len(answer_constraints)}")
            for c in answer_constraints:
                print(f"  - [{c.get('type')}] {c.get('constraint')}")

            print(f"[constraint_extract] format: {format_constraint.get('expected_type', '未知')}")

            msg = AIMessage(
                content=f"[constraint_extract] 提取约束：{len(entity_constraints)} 实体约束，"
                       f"{len(relation_constraints)} 关系约束，{len(answer_constraints)} 答案约束"
            )

            return {
                "constraints": constraints,
                "messages": [msg],
            }

        except Exception as e:
            print(f"[constraint_extract] 出错: {e}")
            return {
                "constraints": {},
                "messages": [AIMessage(content=f"[constraint_extract] 出错: {e}")],
            }

    return extract_constraints


def get_constraints_summary(constraints: Dict) -> str:
    """获取约束的简短摘要"""
    if not constraints:
        return "无约束"

    parts = []

    entity_c = constraints.get("entity_constraints", [])
    if entity_c:
        parts.append(f"{len(entity_c)}个实体约束")

    relation_c = constraints.get("relation_constraints", [])
    if relation_c:
        parts.append(f"{len(relation_c)}个关系约束")

    answer_c = constraints.get("answer_constraints", [])
    if answer_c:
        parts.append(f"{len(answer_c)}个答案约束")

    return ", ".join(parts) if parts else "无约束"


def format_constraints_for_validation(constraints: Dict) -> str:
    """格式化约束用于验证 prompt"""
    if not constraints:
        return "（无明确约束）"

    lines = []

    entity_c = constraints.get("entity_constraints", [])
    if entity_c:
        lines.append("## 实体约束")
        for c in entity_c:
            lines.append(f"- {c.get('entity', '?')} 的 {c.get('attribute', '?')} 必须是 {c.get('value', '?')}")

    relation_c = constraints.get("relation_constraints", [])
    if relation_c:
        lines.append("\n## 关系约束")
        for c in relation_c:
            cond = f" ({c.get('condition')})" if c.get("condition") else ""
            lines.append(f"- {c.get('subject', '?')} {c.get('relation', '?')} {c.get('object', '?')}{cond}")

    answer_c = constraints.get("answer_constraints", [])
    if answer_c:
        lines.append("\n## 答案约束")
        for c in answer_c:
            marker = "✓" if c.get("type") == "must_satisfy" else "?"
            lines.append(f"- [{marker}] {c.get('constraint', '?')}")

    format_c = constraints.get("format_constraint", {})
    if format_c:
        lines.append(f"\n## 格式要求")
        lines.append(f"- 答案类型: {format_c.get('expected_type', '未知')}")

    return "\n".join(lines)
