# deepresearch/memory.py
# -*- coding: utf-8 -*-
"""
轻量级执行记忆系统

设计哲学：
- 记忆 = 结构化的执行轨迹 + LLM 压缩的阶段摘要
- 每个子任务完成后记录一条 StepRecord（轻量，无 LLM 调用）
- 每个并行组完成后，用 flash_llm 生成一段 group_summary（压缩推理链）
- memory.to_context() 输出给后续所有 prompt 用，替代原来零散的 deps_context/chain_memory

核心类：
- StepRecord: 一个子任务执行完毕后的结构化快照
- ExecutionMemory: 管理所有 StepRecord + group_summary 的容器
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ───────── StepRecord ─────────

@dataclass
class StepRecord:
    """一个子任务执行完毕后的结构化快照"""
    
    # 基本信息
    step_id: str                    # e.g. "ST1"
    title: str                      # 子任务标题（已替换占位符后）
    group_index: int = 0            # 所在并行组编号
    
    # 执行结果
    candidates: List[str] = field(default_factory=list)
    best_answer: str = ""           # candidates[0] 的快照
    evidence: List[str] = field(default_factory=list)
    reasoning_trace: str = ""       # 模型的推理过程
    confidence: float = 0.0
    
    # 执行元信息
    queries_used: List[str] = field(default_factory=list)
    attempt_count: int = 1          # 实际尝试了几次
    success: bool = True            # 是否拿到了有效 candidates
    
    # 错误路径（如果失败）
    failure_reason: str = ""        # 为何失败的简要原因
    
    # 时间戳
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_s(self) -> float:
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "group_index": self.group_index,
            "candidates": self.candidates,
            "best_answer": self.best_answer,
            "evidence": self.evidence,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
            "queries_used": self.queries_used,
            "attempt_count": self.attempt_count,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "duration_s": self.duration_s,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        rec = cls(
            step_id=d.get("step_id", ""),
            title=d.get("title", ""),
            group_index=d.get("group_index", 0),
            candidates=d.get("candidates", []),
            best_answer=d.get("best_answer", ""),
            evidence=d.get("evidence", []),
            reasoning_trace=d.get("reasoning_trace", ""),
            confidence=d.get("confidence", 0.0),
            queries_used=d.get("queries_used", []),
            attempt_count=d.get("attempt_count", 1),
            success=d.get("success", True),
            failure_reason=d.get("failure_reason", ""),
        )
        return rec


# ───────── ExecutionMemory ─────────

class ExecutionMemory:
    """
    执行记忆管理器。
    
    职责：
    1. 存储每个子任务的 StepRecord
    2. 存储每个并行组完成后 flash_llm 生成的 group_summary
    3. 提供 to_context() → 生成给 prompt 用的上下文字符串
    4. 提供 to_dict() / from_dict() 支持序列化到 State
    """
    
    def __init__(self):
        self.steps: Dict[str, StepRecord] = {}       # step_id → StepRecord
        self.group_summaries: Dict[int, str] = {}     # group_index → LLM 生成的摘要
        self.execution_order: List[str] = []          # 按执行顺序记录 step_id
        self.iteration: int = 0                       # 当前迭代轮次
    
    # ── 写入 ──
    
    def record_step(self, record: StepRecord) -> None:
        """记录一个子任务的执行结果"""
        self.steps[record.step_id] = record
        if record.step_id not in self.execution_order:
            self.execution_order.append(record.step_id)
    
    def set_group_summary(self, group_index: int, summary: str) -> None:
        """记录一个并行组的 LLM 压缩摘要"""
        self.group_summaries[group_index] = summary
    
    # ── 读取：给 prompt 用的上下文 ──
    
    def to_context_for_subtask(self, current_subtask: Dict, detail_level: str = "full") -> str:
        """
        为当前子任务生成记忆上下文字符串。
        
        策略：
        - 直接依赖的 step：输出详细信息（answer + evidence + reasoning）
        - 非依赖但同迭代的 step：输出简要信息（answer only）
        - 历史迭代的 group_summary：原样附上（已被 LLM 压缩过）
        
        Args:
            current_subtask: 当前子任务字典
            detail_level: "full" 完整 | "brief" 仅答案
        """
        if not self.steps and not self.group_summaries:
            return ""
        
        parts = []
        deps = set(current_subtask.get("depends_on", []))
        current_group = -1
        
        # 找出当前子任务所在的 group（用于区分"之前的组"和"同组"）
        for sid, rec in self.steps.items():
            if sid in deps:
                current_group = max(current_group, rec.group_index)
        
        # 1. 输出历史迭代的压缩摘要（如果有 followup 场景）
        # group_summary 已经是 LLM 压缩过的，直接用
        past_summaries = []
        for gi in sorted(self.group_summaries.keys()):
            # 只输出早于当前依赖组的摘要
            if gi < current_group or (current_group == -1 and self.group_summaries):
                past_summaries.append(self.group_summaries[gi])
        
        if past_summaries:
            parts.append("## 前序阶段推理摘要")
            parts.extend(past_summaries)
        
        # 2. 直接依赖：输出详细信息
        dep_parts = []
        for dep_id in deps:
            rec = self.steps.get(dep_id)
            if not rec:
                continue
            
            line = f"  [{rec.step_id}] {rec.title}"
            line += f"\n    → 结论: {rec.best_answer or '(未确定)'}"
            if len(rec.candidates) > 1:
                line += f" （其他候选: {', '.join(rec.candidates[1:])}）"
            if rec.evidence:
                line += f"\n    证据: {'; '.join(rec.evidence[:3])}"
            if rec.reasoning_trace:
                line += f"\n    推理: {rec.reasoning_trace[:300]}"
            if not rec.success:
                line += f"\n    ⚠️ 失败: {rec.failure_reason}"
            dep_parts.append(line)
        
        if dep_parts:
            parts.append("## 直接前置任务（详细）")
            parts.extend(dep_parts)
        
        # 3. 非依赖的已完成任务：简要摘要
        other_parts = []
        for sid in self.execution_order:
            if sid in deps:
                continue
            rec = self.steps.get(sid)
            if not rec:
                continue
            answer = rec.best_answer or '(未确定)'
            status = "✓" if rec.success else "✗"
            other_parts.append(f"  [{rec.step_id}] {status} {rec.title}: {answer}")
        
        if other_parts:
            parts.append("## 其他已完成任务（摘要）")
            parts.extend(other_parts)
        
        if not parts:
            return ""
        
        return "【执行记忆】\n" + "\n".join(parts)
    
    def to_context_for_finalize(self) -> str:
        """
        为 finalize 阶段生成完整的推理链上下文。
        比 to_context_for_subtask 更完整：包含所有 step 的推理过程。
        """
        if not self.steps:
            return ""
        
        parts = ["## 完整执行轨迹与推理链"]
        
        # 按组输出
        max_group = max((r.group_index for r in self.steps.values()), default=0)
        for gi in range(max_group + 1):
            group_steps = [r for r in self.steps.values() if r.group_index == gi]
            if not group_steps:
                continue
            
            parts.append(f"\n### 阶段 {gi}")
            
            for rec in group_steps:
                status = "✓" if rec.success else "✗"
                parts.append(f"\n[{rec.step_id}] {status} {rec.title}")
                parts.append(f"  结论: {rec.best_answer or '(未确定)'}")
                if len(rec.candidates) > 1:
                    parts.append(f"  候选: {', '.join(rec.candidates)}")
                if rec.reasoning_trace:
                    parts.append(f"  推理: {rec.reasoning_trace[:400]}")
                if rec.evidence:
                    parts.append(f"  证据: {'; '.join(rec.evidence[:3])}")
                if not rec.success:
                    parts.append(f"  ⚠️ 失败原因: {rec.failure_reason}")
                parts.append(f"  置信度: {rec.confidence:.1f} | 尝试次数: {rec.attempt_count}")
            
            # 附上该组的压缩摘要（如果有）
            if gi in self.group_summaries:
                parts.append(f"\n  📋 阶段 {gi} 摘要: {self.group_summaries[gi]}")
        
        return "\n".join(parts)
    
    # ── 序列化 ──
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "group_summaries": self.group_summaries,
            "execution_order": self.execution_order,
            "iteration": self.iteration,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionMemory":
        mem = cls()
        for k, v in d.get("steps", {}).items():
            mem.steps[k] = StepRecord.from_dict(v)
        mem.group_summaries = {int(k): v for k, v in d.get("group_summaries", {}).items()}
        mem.execution_order = d.get("execution_order", [])
        mem.iteration = d.get("iteration", 0)
        return mem
    
    # ── 调试 ──
    
    def print_summary(self) -> None:
        """打印记忆摘要（调试用）"""
        print(f"\n  ═══ 执行记忆状态 ═══")
        print(f"  已记录 {len(self.steps)} 个步骤, {len(self.group_summaries)} 个组摘要")
        for sid in self.execution_order:
            rec = self.steps.get(sid)
            if rec:
                status = "✓" if rec.success else "✗"
                print(f"  [{rec.step_id}] {status} {rec.best_answer or '(空)'} (conf={rec.confidence:.1f}, attempts={rec.attempt_count})")
        for gi, summary in sorted(self.group_summaries.items()):
            print(f"  [组{gi}摘要] {summary[:120]}")


# ───────── Group Summary 生成 ─────────

GROUP_SUMMARY_PROMPT = """你是研究进度总结专家。请将本阶段多个子任务的执行结果压缩成一段简洁的推理链摘要。

原始问题：{question}

本阶段完成的子任务：
{steps_text}

请用 2-4 句话总结本阶段的关键发现和推理进展，重点包括：
1. 确认了哪些关键实体/事实
2. 建立了哪些实体间的关系
3. 当前推理链推进到了什么位置
4. 有哪些不确定点需要后续验证

直接输出摘要文本，不需要JSON格式。"""


async def generate_group_summary(
    flash_llm,
    question: str,
    group_records: List[StepRecord],
) -> str:
    """
    用 flash_llm 为一个并行组生成压缩摘要。
    
    这是记忆系统中唯一需要 LLM 调用的地方。
    目的是将多个子任务的详细结果压缩为一段连贯的推理链描述，
    供后续组/迭代使用时不会因为 token 爆炸。
    """
    if not group_records:
        return ""
    
    steps_parts = []
    for rec in group_records:
        status = "✓" if rec.success else "✗"
        part = f"[{rec.step_id}] {status} {rec.title}"
        part += f"\n  结论: {rec.best_answer or '(未确定)'}"
        if rec.evidence:
            part += f"\n  证据: {'; '.join(rec.evidence[:3])}"
        if rec.reasoning_trace:
            part += f"\n  推理: {rec.reasoning_trace[:200]}"
        if not rec.success:
            part += f"\n  失败: {rec.failure_reason}"
        steps_parts.append(part)
    
    steps_text = "\n\n".join(steps_parts)
    
    prompt = GROUP_SUMMARY_PROMPT.format(
        question=question,
        steps_text=steps_text,
    )
    
    try:
        resp = await asyncio.wait_for(flash_llm.ainvoke(prompt), timeout=30)
        summary = str(resp.content).strip()
        # 限制长度，避免摘要本身过长
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return summary
    except Exception as e:
        # 降级：手动拼接简要摘要
        fallback_parts = []
        for rec in group_records:
            ans = rec.best_answer or '(未知)'
            fallback_parts.append(f"[{rec.step_id}] {rec.title}: {ans}")
        return "; ".join(fallback_parts)
