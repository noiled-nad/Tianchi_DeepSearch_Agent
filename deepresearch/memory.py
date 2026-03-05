# deepresearch/memory.py
# -*- coding: utf-8 -*-
"""
Memory 体系（OAgents 风格）

提供结构化的执行轨迹记录，支持：
1. 完整的步骤记录（输入、输出、工具调用、耗时等）
2. to_messages() 转换为可注入 prompt 的消息格式
3. 上下文压缩/摘要功能
4. 调试和可观测性支持
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ToolCall:
    """工具调用记录"""
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class ResearchStep:
    """
    研究步骤记录（结构化）。

    记录完整的执行过程，支持转换为 prompt 可用的消息格式。
    """
    # 基本信息
    step_type: Literal["plan", "subtask", "search", "fetch", "extract", "finalize", "reflect"]
    step_id: str                              # 如 "ST1", "ST1_search", "finalize_0"

    # 输入输出
    input_summary: str = ""                   # 简短输入描述
    output_summary: str = ""                  # 简短输出描述
    raw_input: Optional[Dict[str, Any]] = None    # 原始输入（调试用）
    raw_output: Optional[Dict[str, Any]] = None   # 原始输出

    # 工具调用
    tool_calls: List[ToolCall] = field(default_factory=list)

    # 执行信息
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    # 反思
    score: Optional[float] = None             # 0.0 ~ 1.0
    reflection: Optional[str] = None

    # 子步骤引用（用于层级结构）
    sub_steps: List[str] = field(default_factory=list)  # 子步骤 ID 列表

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "step_type": self.step_type,
            "step_id": self.step_id,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "score": self.score,
            "reflection": self.reflection,
            "sub_steps": self.sub_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchStep":
        """从字典恢复"""
        return cls(
            step_type=data.get("step_type", "subtask"),
            step_id=data.get("step_id", ""),
            input_summary=data.get("input_summary", ""),
            output_summary=data.get("output_summary", ""),
            raw_input=data.get("raw_input"),
            raw_output=data.get("raw_output"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            success=data.get("success", True),
            error=data.get("error"),
            score=data.get("score"),
            reflection=data.get("reflection"),
            sub_steps=data.get("sub_steps", []),
        )

    def to_messages(self, summary_mode: bool = False, include_reflection: bool = True) -> List[Dict[str, Any]]:
        """
        转换为可注入 prompt 的消息格式。

        Args:
            summary_mode: 是否使用摘要模式（更简洁）
            include_reflection: 是否包含反思内容

        Returns:
            消息列表，每个消息包含 role 和 content
        """
        messages = []

        # 基本状态标记
        status_mark = "✓" if self.success else "✗"
        if summary_mode:
            # 简洁模式：只输出关键信息
            content = f"[{self.step_id}] {status_mark} {self.output_summary[:200]}"
            if self.duration_ms > 0:
                content += f" ({self.duration_ms:.0f}ms)"
            messages.append({
                "role": "assistant",
                "content": content,
            })
        else:
            # 详细模式：完整信息
            header = f"### 步骤 [{self.step_id}] {self.step_type.upper()} {status_mark}"
            if self.duration_ms > 0:
                header += f" ({self.duration_ms:.0f}ms)"

            parts = [header]

            if self.input_summary:
                parts.append(f"**输入**: {self.input_summary}")

            if self.output_summary:
                parts.append(f"**输出**: {self.output_summary}")

            if self.tool_calls:
                tool_summaries = []
                for tc in self.tool_calls:
                    if tc.error:
                        tool_summaries.append(f"- {tc.name}: ❌ {tc.error}")
                    else:
                        result_preview = (tc.result or "")[:100]
                        tool_summaries.append(f"- {tc.name}: {result_preview}")
                parts.append(f"**工具调用**:\n" + "\n".join(tool_summaries))

            if not self.success and self.error:
                parts.append(f"**错误**: {self.error}")

            if include_reflection and self.reflection:
                parts.append(f"**反思**: {self.reflection}")

            messages.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })

        return messages

    def summarize(self) -> str:
        """生成本步骤摘要（用于压缩上下文）"""
        status = "成功" if self.success else "失败"
        summary = f"[{self.step_id}] {self.step_type}: {status}"
        if self.output_summary:
            summary += f" - {self.output_summary[:100]}"
        return summary


class ResearchMemory:
    """
    研究记忆管理器。

    管理所有执行步骤，支持：
    - 添加/获取步骤
    - 转换为消息格式
    - 压缩/摘要长上下文
    - 调试回放
    """

    def __init__(self):
        self.steps: List[ResearchStep] = []
        self._step_index: Dict[str, ResearchStep] = {}  # step_id -> ResearchStep

    def add_step(self, step: ResearchStep) -> None:
        """添加步骤"""
        self.steps.append(step)
        self._step_index[step.step_id] = step

    def get_step(self, step_id: str) -> Optional[ResearchStep]:
        """根据 ID 获取步骤"""
        return self._step_index.get(step_id)

    def get_steps_by_type(self, step_type: str) -> List[ResearchStep]:
        """根据类型获取步骤"""
        return [s for s in self.steps if s.step_type == step_type]

    def get_successful_steps(self) -> List[ResearchStep]:
        """获取成功的步骤"""
        return [s for s in self.steps if s.success]

    def get_failed_steps(self) -> List[ResearchStep]:
        """获取失败的步骤"""
        return [s for s in self.steps if not s.success]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchMemory":
        """从字典恢复"""
        memory = cls()
        for step_data in data.get("steps", []):
            step = ResearchStep.from_dict(step_data)
            memory.add_step(step)
        return memory

    def to_messages(
        self,
        summary_mode: bool = False,
        max_steps: Optional[int] = None,
        include_reflection: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        转换为消息格式。

        Args:
            summary_mode: 是否使用摘要模式
            max_steps: 最大步骤数（用于限制上下文长度）
            include_reflection: 是否包含反思内容

        Returns:
            消息列表
        """
        messages = []

        steps_to_process = self.steps
        if max_steps is not None and len(steps_to_process) > max_steps:
            # 只保留最近的步骤，前面的压缩
            compressed = self._compress_older_steps(len(steps_to_process) - max_steps)
            if compressed:
                messages.append({
                    "role": "assistant",
                    "content": f"### 前序步骤摘要\n{compressed}",
                })
            steps_to_process = steps_to_process[-(max_steps):]

        for step in steps_to_process:
            messages.extend(step.to_messages(
                summary_mode=summary_mode,
                include_reflection=include_reflection,
            ))

        return messages

    def _compress_older_steps(self, count: int) -> str:
        """压缩较早的步骤为摘要"""
        if count <= 0 or not self.steps:
            return ""

        to_compress = self.steps[:count]
        summaries = [s.summarize() for s in to_compress]
        return "\n".join(summaries)

    def compress(self, max_steps: int = 10) -> "ResearchMemory":
        """
        压缩记忆，返回新的压缩后的 Memory。

        保留最近 max_steps 个步骤的完整信息，
        较早的步骤压缩为摘要。
        """
        if len(self.steps) <= max_steps:
            return self

        compressed = ResearchMemory()

        # 较早的步骤压缩为一个摘要步骤
        older_steps = self.steps[:-max_steps]
        if older_steps:
            summary_step = ResearchStep(
                step_type="plan",  # 使用 plan 类型作为摘要
                step_id="compressed_summary",
                input_summary=f"压缩了 {len(older_steps)} 个前序步骤",
                output_summary=self._compress_older_steps(len(older_steps)),
                success=True,
            )
            compressed.add_step(summary_step)

        # 保留最近的步骤
        for step in self.steps[-max_steps:]:
            compressed.add_step(step)

        return compressed

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        total = len(self.steps)
        success = len(self.get_successful_steps())
        failed = len(self.get_failed_steps())

        total_duration = sum(s.duration_ms for s in self.steps)
        total_tool_calls = sum(len(s.tool_calls) for s in self.steps)

        # 按类型统计
        by_type: Dict[str, int] = {}
        for s in self.steps:
            by_type[s.step_type] = by_type.get(s.step_type, 0) + 1

        return {
            "total_steps": total,
            "success_count": success,
            "failed_count": failed,
            "success_rate": success / total if total > 0 else 0,
            "total_duration_ms": total_duration,
            "total_tool_calls": total_tool_calls,
            "by_type": by_type,
        }

    def replay(self, detailed: bool = False) -> str:
        """
        生成可读的回放文本（用于调试）。

        Args:
            detailed: 是否包含详细信息
        """
        lines = ["=" * 60, "Research Memory Replay", "=" * 60]

        stats = self.get_statistics()
        lines.append(f"\n统计: {stats['total_steps']} 步, "
                     f"成功率 {stats['success_rate']*100:.0f}%, "
                     f"耗时 {stats['total_duration_ms']/1000:.1f}s")

        for i, step in enumerate(self.steps, 1):
            status = "✓" if step.success else "✗"
            lines.append(f"\n--- Step {i}: [{step.step_id}] {step.step_type} {status} ---")

            if detailed:
                if step.input_summary:
                    lines.append(f"输入: {step.input_summary[:200]}")
                if step.output_summary:
                    lines.append(f"输出: {step.output_summary[:200]}")
                if step.tool_calls:
                    lines.append(f"工具: {[tc.name for tc in step.tool_calls]}")
                if step.error:
                    lines.append(f"错误: {step.error}")
                if step.reflection:
                    lines.append(f"反思: {step.reflection[:100]}")
            else:
                lines.append(step.summarize())

        return "\n".join(lines)


# ───────── 辅助函数 ─────────

def create_step(
    step_type: str,
    step_id: str,
    input_summary: str = "",
    output_summary: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    success: bool = True,
    error: Optional[str] = None,
    score: Optional[float] = None,
    reflection: Optional[str] = None,
) -> ResearchStep:
    """
    创建 ResearchStep 的便捷函数。

    自动设置时间戳。
    """
    now = time.time()
    return ResearchStep(
        step_type=step_type,
        step_id=step_id,
        input_summary=input_summary,
        output_summary=output_summary,
        tool_calls=tool_calls or [],
        start_time=now,
        end_time=now,
        duration_ms=0.0,
        success=success,
        error=error,
        score=score,
        reflection=reflection,
    )


def create_tool_call(
    name: str,
    arguments: Dict[str, Any],
    result: Optional[str] = None,
    error: Optional[str] = None,
    duration_ms: float = 0.0,
) -> ToolCall:
    """创建 ToolCall 的便捷函数"""
    return ToolCall(
        name=name,
        arguments=arguments,
        result=result,
        error=error,
        duration_ms=duration_ms,
    )
