# -*- coding: utf-8 -*-
"""
答案验证 Agent

用于验证当前答案是否满足所有约束条件。
如果验证通过，可以提前结束迭代；否则生成 followup_queries 指导下一轮搜索。
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple
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
你是答案验证专家。请验证当前答案是否满足问题的所有约束条件。

## 原始问题
{question}

## 约束条件
{constraints_text}

## 当前答案
{final_answer}

## 子任务调查结果
{subtask_findings}

## 验证要求

请逐一检查每个约束条件是否被满足：

1. **实体约束检查**：答案中的实体是否满足属性要求？
2. **关系约束检查**：实体之间的关系是否满足条件？
3. **答案约束检查**：最终答案是否满足所有 must_satisfy 约束？
4. **格式检查**：答案格式是否符合要求？

## 评分标准

- **通过 (passed=true)**：所有 must_satisfy 约束都有明确证据支持
- **需要更多信息 (passed=false, confidence >= 0.3)**：部分约束有证据，但不够完整
- **失败 (passed=false, confidence < 0.3)**：关键约束无证据支持，或方向错误

## 输出格式（严格 JSON，不要 markdown）

{{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "验证推理过程（简洁）",
  "constraint_check": [
    {{"constraint": "约束内容", "satisfied": true/false, "evidence": "证据来源或'无'", "note": "备注"}}
  ],
  "missing_constraints": ["未满足的约束1", "未满足的约束2"],
  "followup_queries": [
    "针对缺失约束1的搜索词",
    "针对缺失约束2的搜索词"
  ],
  "should_continue": true/false
}}

## followup_queries 生成规则（极重要！）

当 passed=false 时，必须为每个未满足的约束生成 1-2 条精准搜索词：

1. **精准锚点**：用已确认的实体名/数字作为搜索锚点
2. **关系词**：加上约束中涉及的关系词
3. **短小精悍**：每条 3-6 个词
4. **避免泛化**：不要用"相关信息"、"详细资料"等泛化词

通用格式：
- 实体 + 关系词 + 目标属性
- 已知实体A + 关系 + 待确认实体B

注意：
- 只有当所有 must_satisfy 约束都有明确证据时，passed 才能为 true
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


def _format_subtask_findings(subtask_findings: Dict) -> str:
    """格式化子任务发现"""
    if not subtask_findings:
        return "（无子任务结果）"

    lines = []
    for st_id, findings in subtask_findings.items():
        if isinstance(findings, dict):
            candidates = findings.get("candidates", [])
            evidence = findings.get("evidence", [])
            confidence = findings.get("confidence", 0)

            best_answer = candidates[0] if candidates else "(未找到)"
            lines.append(f"[{st_id}] 答案: {best_answer} (置信度: {confidence})")
            if evidence:
                for e in evidence[:3]:
                    lines.append(f"    - {e}")
        else:
            lines.append(f"[{st_id}] {str(findings)[:150]}")

    return "\n".join(lines)


def make_answer_validator_node(llm) -> Callable[[DeepResearchState], DeepResearchState]:
    """
    答案验证节点。

    验证当前答案是否满足所有约束条件。
    如果验证失败，输出 followup_queries 指导下一轮搜索。
    """

    async def validate_answer(state: DeepResearchState) -> DeepResearchState:
        print("\n============ answer_validator 阶段 ============")

        question = state.get("question", "")
        constraints = state.get("constraints", {}) or {}
        final_answer = state.get("final_answer", "")
        subtask_findings = state.get("subtask_findings", {}) or {}
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 50))

        print(f"[validator] iteration={iteration}/{max_iterations}")

        if not final_answer:
            print("[validator] 无答案，需要继续检索")
            return {
                "validation_result": "needs_more",
                "validation_passed": False,
                "needs_followup": True,
                "queries": [],
            }

        # 格式化约束
        constraints_text = format_constraints_for_validation(constraints)
        findings_text = _format_subtask_findings(subtask_findings)

        prompt = VALIDATION_PROMPT.format(
            question=question,
            constraints_text=constraints_text,
            final_answer=final_answer,
            subtask_findings=findings_text,
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
            followup_queries = result.get("followup_queries", [])
            constraint_check = result.get("constraint_check", [])

            # 打印验证结果
            print(f"[validator] passed={passed}, confidence={confidence:.2f}")
            print(f"[validator] reasoning: {reasoning[:200]}...")

            if missing_constraints:
                print(f"[validator] missing_constraints: {missing_constraints[:3]}")

            if followup_queries:
                print(f"[validator] followup_queries: {followup_queries[:3]}")

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
                "missing_evidence": missing_constraints,
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
