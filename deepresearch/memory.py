# deepresearch/memory.py
# -*- coding: utf-8 -*-
"""
Memory 体系（OAgents 风格）+ 智能记忆管理

提供结构化的执行轨迹记录，支持：
1. 完整的步骤记录（输入、输出、工具调用、耗时等）
2. to_messages() 转换为可注入 prompt 的消息格式
3. 智能压缩：基于 LLM 评估的语义压缩
4. 错误路径标注：避免重复犯错
5. 调试和可观测性支持
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple


# ───────── 错误路径和记忆评估相关类型 ─────────

class MemoryValueCategory(Enum):
    """记忆价值类别"""
    HIGH_WORTH = "high_worth"        # 高价值：核心证据、成功路径
    MEDIUM_WORTH = "medium_worth"    # 中价值：辅助信息
    LOW_WORTH = "low_worth"          # 低价值：边缘信息
    NEGATIVE_WORTH = "negative_worth"  # 负价值：已验证的错误路径


@dataclass
class MemoryAssessment:
    """记忆评估结果"""
    step_id: str
    category: MemoryValueCategory

    # 评分（0-1）
    relevance_score: float      # 与目标相关性
    quality_score: float        # 信息质量
    citation_count: int         # 被引用次数

    # 压缩建议
    should_keep: bool           # 是否完整保留
    compression_level: Literal["none", "summary", "conclusion", "error_tag"]

    # 压缩后的内容（如果需要压缩）
    compressed_content: Optional[str] = None

    # 错误路径标注（如果是负价值）
    error_advice: Optional[str] = None  # 例如："该查询方向未找到有效信息"

    # 评估理由（用于调试）
    reasoning: str = ""


# ───────── ToolCall（工具调用记录）────────

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


# ───────── ResearchStep（研究步骤记录）────────

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

    # ── 新增：错误路径标注 ──
    is_error_path: bool = False               # 是否是错误路径
    error_advice: Optional[str] = None        # 错误建议（如"避免混合多个实体查询"）

    # ── 新增：依赖关系 ──
    depends_on_steps: List[str] = field(default_factory=list)  # 依赖的步骤ID

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
            "is_error_path": self.is_error_path,
            "error_advice": self.error_advice,
            "depends_on_steps": self.depends_on_steps,
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
            is_error_path=data.get("is_error_path", False),
            error_advice=data.get("error_advice"),
            depends_on_steps=data.get("depends_on_steps", []),
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

        # 错误路径特殊标记
        status_mark = "⚠" if self.is_error_path else ("✓" if self.success else "✗")

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

            if self.is_error_path and self.error_advice:
                parts.append(f"**错误路径**: {self.error_advice}")

            if include_reflection and self.reflection:
                parts.append(f"**反思**: {self.reflection}")

            messages.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })

        return messages

    def summarize(self) -> str:
        """生成本步骤摘要（用于压缩上下文）"""
        if self.is_error_path:
            return f"[{self.step_id}] ⚠ 错误路径: {self.error_advice or self.output_summary[:80]}"

        status = "成功" if self.success else "失败"
        summary = f"[{self.step_id}] {self.step_type}: {status}"
        if self.output_summary:
            summary += f" - {self.output_summary[:100]}"
        return summary


# ───────── ResearchMemory（记忆管理器）────────

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

    def get_error_paths(self) -> List[ResearchStep]:
        """获取错误路径步骤"""
        return [s for s in self.steps if s.is_error_path]

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
        error_paths = len(self.get_error_paths())

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
            "error_paths_count": error_paths,
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
                     f"错误路径 {stats['error_paths_count']} 个, "
                     f"耗时 {stats['total_duration_ms']/1000:.1f}s")

        for i, step in enumerate(self.steps, 1):
            if step.is_error_path:
                status = "⚠"
            elif step.success:
                status = "✓"
            else:
                status = "✗"
            lines.append(f"\n--- Step {i}: [{step.step_id}] {step.step_type} {status} ---")

            if detailed:
                if step.input_summary:
                    lines.append(f"输入: {step.input_summary[:200]}")
                if step.output_summary:
                    lines.append(f"输出: {step.output_summary[:200]}")
                if step.is_error_path and step.error_advice:
                    lines.append(f"错误建议: {step.error_advice}")
                if step.tool_calls:
                    lines.append(f"工具: {[tc.name for tc in step.tool_calls]}")
                if step.error:
                    lines.append(f"错误: {step.error}")
                if step.reflection:
                    lines.append(f"反思: {step.reflection[:100]}")
            else:
                lines.append(step.summarize())

        return "\n".join(lines)


# ───────── MemoryManagerAgent（记忆管理 Agent）────────

class MemoryManagerAgent:
    """
    记忆管理 Agent：负责评估和压缩记忆

    核心能力：
    1. 评估每个记忆片段的价值（多维度）
    2. 识别错误路径并标注
    3. 智能压缩低价值记忆
    4. 保留高价值记忆的完整信息
    """

    def __init__(self, llm, flash_llm=None):
        self.llm = llm
        self.flash_llm = flash_llm or llm

        # 评估缓存（避免重复评估）
        self._assessment_cache: Dict[str, MemoryAssessment] = {}

        # 错误模式注册表（用于反馈到 parse_claims）
        self._error_patterns: List[str] = []

    def add_error_pattern(self, pattern: str) -> None:
        """添加错误模式"""
        if pattern and pattern not in self._error_patterns:
            self._error_patterns.append(pattern)

    def get_error_patterns(self) -> List[str]:
        """获取所有错误模式"""
        return self._error_patterns.copy()

    def clear_error_patterns(self) -> None:
        """清空错误模式"""
        self._error_patterns = []

    async def assess_memory(
        self,
        memory: ResearchMemory,
        current_question: str,
        current_iteration: int,
    ) -> List[MemoryAssessment]:
        """
        评估所有记忆片段的价值

        Args:
            memory: 要评估的记忆
            current_question: 当前问题
            current_iteration: 当前迭代轮次

        Returns:
            每个步骤的评估结果列表
        """
        assessments = []

        # 第一阶段：快速规则过滤（避免调用太多 LLM）
        for step in memory.steps:
            quick_assessment = self._quick_assess(step, current_iteration)
            if quick_assessment is not None:
                assessments.append(quick_assessment)
                continue

            # 第二阶段：需要 LLM 深度评估的步骤
            llm_assessment = await self._llm_assess(
                step, memory, current_question, current_iteration
            )
            assessments.append(llm_assessment)

        # 第三阶段：分析引用关系（被引用次数）
        assessments = self._augment_with_citation_count(assessments, memory)

        return assessments

    def _quick_assess(
        self,
        step: ResearchStep,
        current_iteration: int,
    ) -> Optional[MemoryAssessment]:
        """
        快速规则评估（避免 LLM 调用）

        Returns:
            如果能确定评估结果则返回，否则返回 None
        """
        # 规则1：最近的 finalize 步骤默认保留
        if step.step_type == "finalize" and f"finalize_{current_iteration}" in step.step_id:
            return MemoryAssessment(
                step_id=step.step_id,
                category=MemoryValueCategory.HIGH_WORTH,
                relevance_score=1.0,
                quality_score=step.score if step.score else 0.7,
                citation_count=0,
                should_keep=True,
                compression_level="none",
                reasoning="当前迭代的最终步骤，默认保留"
            )

        # 规则2：明显失败的步骤标记为低价值
        if not step.success and ("未找到" in step.output_summary or "失败" in step.output_summary):
            return MemoryAssessment(
                step_id=step.step_id,
                category=MemoryValueCategory.LOW_WORTH,
                relevance_score=0.1,
                quality_score=0.0,
                citation_count=0,
                should_keep=False,
                compression_level="conclusion",
                compressed_content=f"[{step.step_id}] 搜索未找到相关信息",
                reasoning="搜索失败，无有效信息"
            )

        # 规则3：已有的错误路径标签
        if step.is_error_path:
            return MemoryAssessment(
                step_id=step.step_id,
                category=MemoryValueCategory.NEGATIVE_WORTH,
                relevance_score=0.0,
                quality_score=0.0,
                citation_count=0,
                should_keep=True,  # 负价值也要保留，用于避免重蹈覆辙
                compression_level="error_tag",
                error_advice=step.error_advice or "已验证的错误路径",
                reasoning="已标注的错误路径"
            )

        # 规则4：反思步骤中的负面建议标记为负价值
        if step.step_type == "reflect" and step.reflection:
            try:
                reflection_data = json.loads(step.reflection) if step.reflection.startswith("[") else {}
                if isinstance(reflection_data, dict):
                    suggestions = reflection_data.get("suggestions", [])
                    weaknesses = reflection_data.get("weaknesses", [])
                else:
                    suggestions = []
                    weaknesses = []

                if any("避免" in s or "不要" in s or "不" in s for s in suggestions + weaknesses):
                    # 提取错误建议
                    error_advice = self._extract_error_advice_from_reflection(reflection_data)
                    if error_advice:
                        self.add_error_pattern(error_advice)

                    return MemoryAssessment(
                        step_id=step.step_id,
                        category=MemoryValueCategory.NEGATIVE_WORTH,
                        relevance_score=0.0,
                        quality_score=0.0,
                        citation_count=0,
                        should_keep=True,
                        compression_level="error_tag",
                        error_advice=error_advice or "已验证的错误路径",
                        reasoning="反思步骤中包含错误路径警示"
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        # 无法快速判断，需要 LLM 深度评估
        return None

    async def _llm_assess(
        self,
        step: ResearchStep,
        memory: ResearchMemory,
        current_question: str,
        current_iteration: int,
    ) -> MemoryAssessment:
        """使用 LLM 深度评估步骤价值"""

        # 获取该步骤的上下文（依赖关系）
        context_parts = []
        if step.depends_on_steps:
            for dep_id in step.depends_on_steps:
                dep_step = memory.get_step(dep_id)
                if dep_step:
                    context_parts.append(f"依赖 [{dep_id}]: {dep_step.output_summary[:100]}")

        context = "\n".join(context_parts) if context_parts else "无依赖"

        prompt = f"""请评估以下研究步骤对于回答问题的价值。

## 当前问题
{current_question}

## 步骤信息
- ID: {step.step_id}
- 类型: {step.step_type}
- 输入: {step.input_summary[:200]}
- 输出: {step.output_summary[:300]}
- 是否成功: {step.success}
- 评分: {step.score if step.score else '无'}

## 步骤上下文
{context}

## 评估维度
请从以下维度评估该步骤的价值（0-1分）：

1. **目标相关性**：该步骤的产出是否直接有助于回答最终问题？
   - 1.0 = 核心证据（如关键实体、核心关系）
   - 0.5 = 辅助信息
   - 0.0 = 与问题无关

2. **信息质量**：该步骤产出的信息质量如何？
   - 1.0 = 高质量、可信的发现
   - 0.5 = 中等质量
   - 0.0 = 低质量、错误或未找到信息

3. **路径价值**：该步骤代表了什么样的路径？
   - 正向路径：成功的查询方向
   - 负向路径：已验证的错误方向（需要标注避免重复）
   - 中性路径：不确定

请输出 JSON（不要 markdown）：
{{
  "relevance_score": 0.0-1.0,
  "quality_score": 0.0-1.0,
  "path_type": "positive" | "negative" | "neutral",
  "reasoning": "评估理由（简短）",
  "error_advice": "如果是负向路径，如何避免？（简短）",
  "key_findings": ["该步骤的关键发现，最多2条"]
}}
"""

        try:
            resp = await self.flash_llm.ainvoke(prompt)
            result = json.loads(self._extract_json(str(resp.content)))

            # 根据 LLM 评估结果确定类别
            path_type = result.get("path_type", "neutral")
            relevance = result.get("relevance_score", 0.5)
            quality = result.get("quality_score", 0.5)

            if path_type == "negative":
                category = MemoryValueCategory.NEGATIVE_WORTH
                compression_level = "error_tag"
                error_advice = result.get("error_advice", "已验证的错误路径")
                if error_advice and error_advice != "已验证的错误路径":
                    self.add_error_pattern(error_advice)
                should_keep = True
            elif relevance >= 0.7 and quality >= 0.7:
                category = MemoryValueCategory.HIGH_WORTH
                compression_level = "none"
                should_keep = True
                error_advice = None
            elif relevance >= 0.4 or quality >= 0.4:
                category = MemoryValueCategory.MEDIUM_WORTH
                compression_level = "summary"
                should_keep = False
                error_advice = None
            else:
                category = MemoryValueCategory.LOW_WORTH
                compression_level = "conclusion"
                should_keep = False
                error_advice = None

            return MemoryAssessment(
                step_id=step.step_id,
                category=category,
                relevance_score=relevance,
                quality_score=quality,
                citation_count=0,  # 后续补充
                should_keep=should_keep,
                compression_level=compression_level,
                compressed_content=self._compress_content(step, result),
                error_advice=error_advice,
                reasoning=result.get("reasoning", "")
            )

        except Exception as e:
            # LLM 评估失败，返回保守评估
            return MemoryAssessment(
                step_id=step.step_id,
                category=MemoryValueCategory.MEDIUM_WORTH,
                relevance_score=0.5,
                quality_score=0.5,
                citation_count=0,
                should_keep=False,
                compression_level="summary",
                reasoning=f"LLM评估失败: {e}"
            )

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        if text.startswith("{"):
            return text
        m = re.search(r"\{[\s\S]*\}", text)
        return m.group(0) if m else "{}"

    def _compress_content(self, step: ResearchStep, llm_result: Dict) -> str:
        """根据评估结果压缩内容"""
        key_findings = llm_result.get("key_findings", [])
        if key_findings:
            return f"[{step.step_id}] " + "; ".join(key_findings)
        return f"[{step.step_id}] {step.output_summary[:100]}"

    def _extract_error_advice_from_reflection(self, reflection_data: Dict) -> str:
        """从反思数据中提取错误建议"""
        suggestions = reflection_data.get("suggestions", [])
        weaknesses = reflection_data.get("weaknesses", [])

        # 优先提取包含"避免"、"不要"的建议
        error_suggestions = [s for s in suggestions + weaknesses if any(w in s for w in ["避免", "不要", "不"])]

        if error_suggestions:
            return error_suggestions[0]

        # 如果没有明确的错误建议，返回弱点
        if weaknesses:
            return f"问题: {weaknesses[0]}"

        return "已验证的错误路径"

    def _augment_with_citation_count(
        self,
        assessments: List[MemoryAssessment],
        memory: ResearchMemory,
    ) -> List[MemoryAssessment]:
        """补充被引用次数"""
        citation_count: Dict[str, int] = {}
        for step in memory.steps:
            for dep_id in step.depends_on_steps:
                citation_count[dep_id] = citation_count.get(dep_id, 0) + 1

        for assessment in assessments:
            assessment.citation_count = citation_count.get(assessment.step_id, 0)

        return assessments

    async def compress_memory(
        self,
        memory: ResearchMemory,
        current_question: str,
        current_iteration: int,
        max_full_steps: int = 5,
    ) -> ResearchMemory:
        """
        压缩记忆：根据评估结果进行智能压缩

        Args:
            memory: 原始记忆
            current_question: 当前问题
            current_iteration: 当前迭代
            max_full_steps: 最多保留多少个完整步骤

        Returns:
            压缩后的新记忆
        """
        # 评估所有步骤
        assessments = await self.assess_memory(memory, current_question, current_iteration)

        # 按价值排序
        high_worth = [a for a in assessments if a.category == MemoryValueCategory.HIGH_WORTH]
        medium_worth = [a for a in assessments if a.category == MemoryValueCategory.MEDIUM_WORTH]
        low_worth = [a for a in assessments if a.category == MemoryValueCategory.LOW_WORTH]
        negative_worth = [a for a in assessments if a.category == MemoryValueCategory.NEGATIVE_WORTH]

        # 保留策略
        # 1. 高价值：完整保留（最多 max_full_steps 个）
        # 2. 中价值：压缩为摘要
        # 3. 低价值：压缩为结论
        # 4. 负价值：保留错误标签

        compressed = ResearchMemory()
        kept_full = 0

        # 先处理高价值
        for assessment in high_worth:
            if kept_full < max_full_steps:
                step = memory.get_step(assessment.step_id)
                if step:
                    compressed.add_step(step)
                    kept_full += 1
            else:
                # 超出限制，也压缩为摘要
                self._add_compressed_step(compressed, memory, assessment)

        # 中价值：摘要
        for assessment in medium_worth:
            self._add_compressed_step(compressed, memory, assessment)

        # 低价值：结论
        for assessment in low_worth:
            self._add_compressed_step(compressed, memory, assessment)

        # 负价值：错误标签
        for assessment in negative_worth:
            step = memory.get_step(assessment.step_id)
            if step:
                error_step = create_step(
                    step_type="reflect",
                    step_id=f"{assessment.step_id}_error_tag",
                    input_summary=f"错误路径: {step.input_summary[:50]}",
                    output_summary=assessment.error_advice or "已验证的错误路径",
                    success=True,
                )
                error_step.is_error_path = True
                error_step.error_advice = assessment.error_advice
                compressed.add_step(error_step)

        return compressed

    def _add_compressed_step(
        self,
        compressed: ResearchMemory,
        original: ResearchMemory,
        assessment: MemoryAssessment,
    ):
        """添加压缩后的步骤"""
        step = original.get_step(assessment.step_id)
        if not step:
            return

        if assessment.compression_level == "summary":
            # 摘要模式
            compressed_step = create_step(
                step_type=step.step_type,
                step_id=f"{assessment.step_id}_summary",
                input_summary=step.input_summary[:100],
                output_summary=step.output_summary[:200],
                success=step.success,
            )
            compressed.add_step(compressed_step)
        elif assessment.compression_level == "conclusion":
            # 结论模式
            compressed_step = create_step(
                step_type=step.step_type,
                step_id=f"{assessment.step_id}_conclusion",
                input_summary=assessment.step_id,
                output_summary=assessment.compressed_content or step.output_summary[:100],
                success=step.success,
            )
            compressed.add_step(compressed_step)


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
