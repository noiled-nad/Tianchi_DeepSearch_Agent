# -*- coding: utf-8 -*-
"""
答案验证 Agent

用于验证当前答案是否满足所有约束条件。
如果验证通过，可以提前结束迭代；否则生成 followup_queries 指导下一轮搜索。

核心改进：
- 使用 reasoning_chain（只包含与答案相关的推理过程）
- 逐步骤验证推理链的完整性和证据充分性
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List
from enum import Enum

from langchain_core.messages import AIMessage
from .state import DeepResearchState
from .constraint_extractor import format_constraints_for_validation


class ValidationResult(Enum):
    """验证结果"""
    PASSED = "passed"           # 通过验证，可以结束
    NEEDS_MORE = "needs_more"   # 需要更多信息
    FAILED = "failed"           # 验证失败，方向可能错误


VALIDATION_PROMPT = """
你是答案验证专家。请基于推理链验证当前答案是否满足所有约束条件。

## 原始问题
{question}

## 约束条件
{constraints_text}

## 最终答案
{final_answer}

## 推理链（只包含与答案相关的步骤）
{reasoning_chain_text}

## 验证要求

请逐一验证：

### 1. 推理链完整性检查
- 推理链是否覆盖了所有关键约束？
- 是否有缺失的推理步骤？

### 2. 每步证据充分性检查
- 每个步骤的 evidence 是否支持 conclusion？
- 证据来源是否可靠（有 [Dx] 引用）？

### 3. 最终答案验证
- 最终答案是否与推理链结论一致？
- 是否满足所有 must_satisfy 约束？

## 输出格式（严格 JSON，不要 markdown）

{{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "验证推理过程（简洁）",
  "step_validation": [
    {{
      "step": 1,
      "subtask_id": "ST1",
      "conclusion_valid": true/false,
      "evidence_sufficient": true/false,
      "note": "备注"
    }}
  ],
  "missing_constraints": ["未满足的约束1"],
  "missing_evidence_for_steps": ["ST2 缺少证据", "ST4 证据不充分"],
  "followup_queries": [
    "针对缺失约束的搜索词1",
    "针对缺失约束的搜索词2"
  ],
  "should_continue": true/false
}}

## followup_queries 生成规则（极重要！）

当 passed=false 时，必须为每个缺失项生成精准搜索词：

1. **精准锚点**：用已确认的实体名作为搜索锚点
2. **关系词**：加上缺失的关系词
3. **短小精悍**：每条 3-6 个词
4. **避免泛化**：不要用"相关信息"、"详细资料"等

注意：
- 只有当推理链完整且每步证据充分时，passed 才能为 true
- passed=false 时，followup_queries 必须有内容
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

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}

    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _format_reasoning_chain(reasoning_chain: List[Dict]) -> str:
    """格式化推理链用于验证 prompt"""
    if not reasoning_chain:
        return "（无推理链）"

    lines = []
    for step in reasoning_chain:
        step_num = step.get("step", "?")
        subtask_id = step.get("subtask_id", "?")
        conclusion = step.get("conclusion", "")
        evidence = step.get("evidence", "")
        confidence = step.get("confidence", 0)

        lines.append(f"### 步骤 {step_num} [{subtask_id}]")
        lines.append(f"**结论**: {conclusion}")
        lines.append(f"**证据**: {evidence}")
        lines.append(f"**置信度**: {confidence}")
        lines.append("")

    return "\n".join(lines)


def make_answer_validator_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    """
    答案验证节点。

    基于 reasoning_chain 验证答案是否满足所有约束条件。
    如果验证失败，输出 followup_queries 指导下一轮搜索。
    """

    async def validate_answer(state: DeepResearchState) -> DeepResearchState:
        print("\n============ answer_validator 阶段 ============")

        question = state.get("question", "")
        constraints = state.get("constraints", {}) or {}
        final_answer = state.get("final_answer", "")
        reasoning_chain = state.get("reasoning_chain", []) or []
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 50))

        print(f"[validator] iteration={iteration}/{max_iterations}")
        print(f"[validator] reasoning_chain_steps={len(reasoning_chain)}")

        if not final_answer:
            print("[validator] 无答案，需要继续检索")
            return {
                "validation_result": "needs_more",
                "validation_passed": False,
                "needs_followup": True,
                "queries": [],
            }

        # 格式化约束和推理链
        constraints_text = format_constraints_for_validation(constraints)
        reasoning_chain_text = _format_reasoning_chain(reasoning_chain)

        prompt = VALIDATION_PROMPT.format(
            question=question,
            constraints_text=constraints_text,
            final_answer=final_answer,
            reasoning_chain_text=reasoning_chain_text,
        )

        print(f"[validator] prompt_len={len(prompt)}")

        try:
            resp = await llm.ainvoke(prompt)
            raw = str(resp.content).strip()
            print(f"[validator] raw_len={len(raw)}")

            result = _safe_json_obj(raw)

            passed = bool(result.get("passed", False))
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", "")).strip()
            should_continue = bool(result.get("should_continue", True))
            missing_constraints = result.get("missing_constraints", [])
            missing_evidence_for_steps = result.get("missing_evidence_for_steps", [])
            followup_queries = result.get("followup_queries", [])
            step_validation = result.get("step_validation", [])

            # 打印验证结果
            print(f"[validator] passed={passed}, confidence={confidence:.2f}")
            print(f"[validator] reasoning: {reasoning[:200]}...")

            if missing_constraints:
                print(f"[validator] missing_constraints: {missing_constraints[:3]}")

            if missing_evidence_for_steps:
                print(f"[validator] missing_evidence_for_steps: {missing_evidence_for_steps[:3]}")

            if step_validation:
                print(f"[validator] step_validation: {len(step_validation)} steps checked")

            # 决定验证结果
            if passed and confidence >= 0.7:
                validation_result = ValidationResult.PASSED
                print(f"[validator] ✓ 验证通过，可以结束")
            elif confidence >= 0.3:
                validation_result = ValidationResult.NEEDS_MORE
                print(f"[validator] ⏳ 需要更多信息")
            else:
                validation_result = ValidationResult.FAILED
                print(f"[validator] ✗ 验证失败")

            # 决定是否继续
            has_room = iteration < max_iterations
            should_stop = (
                (passed and confidence >= 0.7) or  # 验证通过
                (not has_room) or                   # 无迭代空间
                (not should_continue and iteration >= 3)  # LLM 建议停止且至少尝试3轮
            )

            needs_followup = not should_stop

            # 如果需要继续，使用 followup_queries 作为下一轮查询
            queries = followup_queries if needs_followup else []

            msg = AIMessage(
                content=f"[validator] passed={passed}, confidence={confidence:.2f}, "
                       f"needs_followup={needs_followup}, queries={len(queries)}"
            )

            return {
                "validation_result": validation_result.value,
                "validation_passed": passed,
                "validation_confidence": confidence,
                "validation_reasoning": reasoning,
                "missing_evidence": missing_constraints + missing_evidence_for_steps,
                "validation_suggestions": followup_queries,
                "needs_followup": needs_followup,
                "queries": queries,
                "messages": [msg],
            }

        except Exception as e:
            print(f"[validator] 出错: {e}")
            return {
                "validation_result": "needs_more",
                "validation_passed": False,
                "needs_followup": True,
                "queries": [],
                "messages": [AIMessage(content=f"[validator] 出错: {e}")],
            }

    return validate_answer
