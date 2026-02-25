# deepresearch/nodes/route_and_schedule.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Literal,Callable
from ..state import DeepResearchState

Difficulty = Literal["low", "mid", "high"]
def _schedule_from_claims(state:DeepResearchState)-> Dict[str, Any]:
    """
    根据问题中的 claims 分析任务难度，并生成相应的调度参数。

    该函数通过统计 claims 的类型和数量来评估问题的复杂度：
    - 如果有 'disamb' 类型 claims（需要消歧），认为是高难度。
    - 如果 must claims 数量 >=7，或有 'anchor' 但无 'output'，认为是中高难度。
    - 否则为低难度。

    调度参数包括：
    - difficulty: 难度等级 ("low", "mid", "high")
    - batch_size: 批处理大小（影响单次处理的文档数量）
    - parallelism: 并行度（影响并发搜索/抓取的数量）

    Args:
        state (DeepResearchState): 当前的深度研究状态，包含 claims 列表。

    Returns:
        Dict[str, Any]: 调度参数字典，包含以下键：
            - "difficulty" (str): 难度等级。
            - "batch_size" (int): 批处理大小。
            - "parallelism" (int): 并行度。

    Examples:
        >>> claims = [Claim(claim_type="disamb", must=True)]
        >>> state = {"claims": claims}
        >>> _schedule_from_claims(state)
        {'difficulty': 'high', 'batch_size': 3, 'parallelism': 5}
    """
    claims = state.get("claims", [])
    if not claims:
        return {"difficulty": "low", "batch_size": 4, "parallelism": 2}
    # 统计 claim_type
    type_counts = {}
    must_count = 0
    for c in claims:
        ct = getattr(c, "claim_type", "other")  # Claim 是 Pydantic 模型
        type_counts[ct] = type_counts.get(ct, 0) + 1
        if getattr(c, "must", True):
            must_count += 1

    has_disamb = type_counts.get("disamb", 0) > 0
    has_anchor = type_counts.get("anchor", 0) > 0
    has_output = type_counts.get("output", 0) > 0

    # 1) 有消歧：高难 → 小 batch + 高并发（更快铺候选）
    if has_disamb:
        return {"difficulty": "high", "batch_size": 3, "parallelism": 5}

    # 2) must 很多或需要多跳：中高难 → 较大 batch + 中并发
    if must_count >= 7 or (has_anchor and not has_output):
        return {"difficulty": "mid", "batch_size": 6, "parallelism": 3}
    
    return {"difficulty": "low", "batch_size": 4, "parallelism": 2}


def make_route_and_schedule_node()-> Callable[[DeepResearchState], Dict[str, Any]]:
    """
    创建路由和调度节点函数。

    该函数返回一个 LangGraph 节点函数，用于在工作流中执行路由和调度逻辑。
    节点函数会调用 _schedule_from_claims 来分析状态并返回调度参数。

    Returns:
        Callable[[DeepResearchState], Dict[str, Any]]: 
            节点函数，接受 DeepResearchState 并返回调度参数字典。
            符合 LangGraph 节点签名，用于图中的节点执行。

    Examples:
        >>> node = make_route_and_schedule_node()
        >>> state = {"claims": []}
        >>> node(state)
        {'difficulty': 'low', 'batch_size': 4, 'parallelism': 2}
    """
    def _node(state: DeepResearchState) -> Dict[str, Any]:
        print("\n============ route_and_schedule 阶段 ============")
        result = _schedule_from_claims(state)
        print(
            f"[route_and_schedule] difficulty={result.get('difficulty')}, "
            f"batch_size={result.get('batch_size')}, parallelism={result.get('parallelism')}"
        )
        return result
    return _node