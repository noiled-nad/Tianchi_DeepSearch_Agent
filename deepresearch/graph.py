# deepresearch/graph.py
# -*- coding: utf-8 -*-
"""
OAgents 风格子任务拆解流程（带约束验证）：

START -> constraint_extract -> parse_claims -> execute_subtasks -> finalize
                                                                       ↓
                                                        (需要验证？)
                                                         ↓      ↓
                                                        YES     NO
                                                         ↓      ↓
                                                   validator  继续迭代
                                                         ↓
                                                   (验证通过？)
                                                    ↓     ↓
                                                  通过   不通过
                                                   ↓      ↓
                                                  END   继续迭代

验证触发条件（三选一）：
1. 达到最大迭代次数
2. finalize confidence > 0.6
3. 连续 3 轮 confidence 无提升
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import END, START, StateGraph

from deepresearch.state import DeepResearchState
from deepresearch.nodes.parse_claims import make_parse_claims_node
from deepresearch.nodes.execute_subtasks import make_execute_subtasks_node
from deepresearch.nodes.finalize import make_finalize_node
from deepresearch.constraint_extractor import make_constraint_extractor_node
from deepresearch.answer_validator import make_answer_validator_node


# 存储历史 confidence 用于判断是否停滞
_confidence_history: list = []


def _should_validate(state: DeepResearchState) -> bool:
    """
    判断是否需要启动验证 agent。

    触发条件（三选一）：
    1. 达到最大迭代次数
    2. finalize confidence > 0.6
    3. 连续 3 轮 confidence 无提升（差距 < 0.05）
    """
    global _confidence_history

    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 50))

    # 从 messages 或 finalize 结果中提取 confidence
    confidence = 0.0
    for msg in reversed(state.get("messages", [])):
        content = str(getattr(msg, "content", ""))
        if "confidence=" in content.lower() or "置信度=" in content:
            # 尝试提取数字
            import re
            match = re.search(r"(?:confidence|置信度)[=：]\s*([\d.]+)", content, re.I)
            if match:
                confidence = float(match.group(1))
                break

    # 记录 confidence
    _confidence_history.append(confidence)
    if len(_confidence_history) > 10:
        _confidence_history = _confidence_history[-10:]

    # 条件1: 达到最大迭代次数
    if iteration >= max_iterations:
        print(f"[router] 达到最大迭代次数 {max_iterations}，触发验证")
        return True

    # 条件2: confidence 足够高
    if confidence >= 0.6:
        print(f"[router] confidence={confidence:.2f} >= 0.6，触发验证")
        return True

    # 条件3: 连续 3 轮无提升
    if len(_confidence_history) >= 3:
        recent = _confidence_history[-3:]
        max_diff = max(recent) - min(recent)
        if max_diff < 0.05:
            print(f"[router] 连续 3 轮 confidence 无提升 (max_diff={max_diff:.3f})，触发验证")
            return True

    return False


def _route_after_finalize(state: DeepResearchState) -> str:
    """
    finalize 后路由：
    - 需要验证 -> answer_validator
    - 不需要验证 -> execute_subtasks (继续迭代)
    - 无迭代空间 -> END
    """
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 50))
    needs_followup = bool(state.get("needs_followup", False))

    # 无迭代空间，直接结束（此时会强制验证）
    if iteration >= max_iterations:
        return "validate"

    # 不需要继续检索
    if not needs_followup:
        return "validate"

    # 判断是否需要验证
    if _should_validate(state):
        return "validate"

    # 继续迭代
    return "continue"


def _route_after_validator(state: DeepResearchState) -> str:
    """
    验证后路由：
    - 验证通过 -> END
    - 验证失败但还有迭代空间 -> execute_subtasks
    - 达到最大迭代 -> END
    """
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 50))
    validation_passed = bool(state.get("validation_passed", False))
    validation_confidence = float(state.get("validation_confidence", 0.0))

    # 验证通过
    if validation_passed and validation_confidence >= 0.7:
        print(f"[router] ✓ 验证通过，结束 (confidence={validation_confidence:.2f})")
        return "end"

    # 达到最大迭代次数
    if iteration >= max_iterations:
        print(f"[router] 达到最大迭代次数 {max_iterations}，结束")
        return "end"

    # 需要继续检索
    print(f"[router] 验证未通过，继续检索 (iteration={iteration}/{max_iterations})")
    return "continue"


def build_deepresearch_graph(llm, searcher, fetcher, flash_llm=None, max_iterations: int = 50):
    """
    构建深度研究图。
    """
    g = StateGraph(DeepResearchState)

    # ── 添加节点 ──
    g.add_node("constraint_extract", make_constraint_extractor_node(llm))
    g.add_node("parse_claims", make_parse_claims_node(llm))
    g.add_node("execute_subtasks", make_execute_subtasks_node(llm, flash_llm, searcher, fetcher))
    g.add_node("finalize", make_finalize_node(llm))
    g.add_node("answer_validator", make_answer_validator_node(llm))

    # ── 定义边 ──
    g.add_edge(START, "constraint_extract")
    g.add_edge("constraint_extract", "parse_claims")
    g.add_edge("parse_claims", "execute_subtasks")
    g.add_edge("execute_subtasks", "finalize")

    # finalize -> validator | execute_subtasks
    g.add_conditional_edges(
        "finalize",
        _route_after_finalize,
        {
            "validate": "answer_validator",
            "continue": "execute_subtasks",
        }
    )

    # validator -> END | execute_subtasks
    g.add_conditional_edges(
        "answer_validator",
        _route_after_validator,
        {
            "end": END,
            "continue": "execute_subtasks",
        }
    )

    return g


if __name__ == "__main__":
    # Dummy components for graph visualization
    class _DummyLLM:
        pass

    class _DummySearcher:
        pass

    class _DummyFetcher:
        pass

    # Build the graph
    llm = _DummyLLM()
    searcher = _DummySearcher()
    fetcher = _DummyFetcher()
    graph = build_deepresearch_graph(llm, searcher, fetcher)

    # Compile the graph
    compiled_graph = graph.compile()

    # Draw the graph as Mermaid diagram
    mermaid_code = compiled_graph.get_graph().draw_mermaid()
    print("=== DeepResearch Graph Mermaid Diagram ===")
    print(mermaid_code)

    # Save to file
    with open("deepresearch_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("Mermaid diagram saved to deepresearch_graph.mmd")
