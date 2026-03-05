"""deepresearch/plan_tips.py

OAgents 风格 Plan Tips —— 启发式经验规则库。

论文结论：Plan Tips 是所有组件中贡献最大的（+14.54%），
其本质是把"历史执行轨迹中反复出现的错误模式和成功策略"
提炼为软约束，注入 Planning Prompt。

本模块提供：
1. 内置规则库（手工提炼自 GAIA/BrowseComp 类题目的常见模式）
2. get_plan_tips(question) —— 根据问题特征选取相关 tips
3. format_tips_for_prompt(tips) —— 格式化为可直接拼入 prompt 的文本
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ─────────────────────────────────────────────
# 规则库：(tag_list, tip_text)
# tag_list 用于匹配问题特征，tip_text 是注入 prompt 的具体建议
# ─────────────────────────────────────────────

TIPS_DB: List[Tuple[List[str], str]] = [
    # ── 消歧 / 同名问题 ──
    (
        ["disambig", "same_name", "person", "who"],
        "当问题涉及可能重名的人物时，优先搜索其出生年份、国籍、"
        "所属机构或代表作品等唯一识别特征来消歧，而非仅用姓名搜索。"
    ),
    # ── 多跳推理 ──
    (
        ["multi_hop", "bridge", "indirect"],
        "多跳问题必须拆成独立子查询，每个子查询只聚焦一个事实跳转。"
        "先锁定桥接实体，再用桥接实体搜索目标信息。"
    ),
    # ── 数值 / 金额 / 统计 ──
    (
        ["number", "amount", "price", "auction", "revenue", "GDP",
         "population", "statistic", "how_many", "how_much"],
        "涉及金额/数值的查询必须包含币种（USD/CNY/EUR）和时间范围。"
        "拍卖类问题加入 auction record / sold for 等关键词。"
        "统计数据优先查官方来源（政府统计局、世界银行、UN）。"
    ),
    # ── 时间 / 年份约束 ──
    (
        ["year", "date", "when", "century", "era", "period"],
        "有明确年份约束时，必须在查询中保留年份数字。"
        "对于历史问题，尝试同时搜索精确年份和年代范围（如 1972 和 1970s）。"
        "如果目标文档可能在互联网存档中，考虑使用 site:jstor.org 或 site:archive.org。"
    ),
    # ── 学术 / 论文 / 期刊 ──
    (
        ["journal", "paper", "article", "author", "publish", "academic",
         "research", "study", "methodology"],
        "学术文献查询策略：用论文标题关键词 + 作者姓氏 + 发表年份搜索。"
        "优先查 JSTOR、Google Scholar、PubMed、arXiv。"
        "期刊问题要搜索 journal name + volume + issue + year。"
    ),
    # ── 地理 / 地名 ──
    (
        ["location", "city", "country", "where", "region", "capital",
         "geography", "born_in", "located"],
        "地名查询要同时使用当地语言名和英文名。"
        "行政区划可能历史变更，注意搜索当时的地名。"
    ),
    # ── 作品 / 艺术 / 文化 ──
    (
        ["work", "book", "film", "movie", "painting", "song", "album",
         "novel", "artwork", "artist", "composer", "director"],
        "作品类查询：用作品原名（含原语言）搜索，不要只用翻译名。"
        "同时搜索 creator/artist/author + work title + year。"
    ),
    # ── 组织 / 机构 ──
    (
        ["organization", "company", "institution", "university", "school",
         "founded", "established", "headquarter"],
        "机构查询用全称而非缩写。同时搜索中英文官方名称。"
        "成立时间问题加入 founded / established + year。"
    ),
    # ── 比较 / 排名 ──
    (
        ["compare", "rank", "most", "largest", "first", "oldest",
         "highest", "best", "longest"],
        "排名/比较类问题要指明排名维度和数据来源年份。"
        "搜索时加入 ranking / list / top 等关键词。"
    ),
    # ── 通用查询质量 ──
    (
        ["_always"],
        "每条查询只包含一个核心约束组合，不要把多个独立事实塞进一条查询。"
        "中英双语查询都要覆盖。"
        "如果首轮搜索结果不理想，换用同义词或更宽泛的表述重试。"
    ),
    # ── 定义 / 概念 ──
    (
        ["what_is", "definition", "meaning", "concept", "explain"],
        "定义类问题优先查 Wikipedia 和领域百科。"
        "用 'X is' 或 'definition of X' 格式搜索。"
    ),
    # ── 因果 / 原因 ──
    (
        ["why", "cause", "reason", "because", "effect", "impact",
         "result", "consequence"],
        "因果类问题需要找到明确的因果链条。"
        "搜索时用 'why did X' / 'cause of X' / 'X because' 等模式。"
    ),
    # ── 殖民史 / 拉美史 (领域特定) ──
    (
        ["colonial", "encomienda", "hacienda", "prosopography",
         "historiography", "latin_america", "spanish_america"],
        "殖民史相关：搜索英文学术关键词如 prosopography, encomienda system, "
        "hacienda evolution。同时用 HAHR (Hispanic American Historical Review) "
        "和 Latin American Research Review 等期刊名缩写搜索。"
    ),
]


def _detect_tags(question: str) -> List[str]:
    """
    根据问题文本检测匹配的标签。
    使用关键词匹配 + 简单模式识别。
    """
    q_lower = question.lower()
    detected: List[str] = ["_always"]  # 通用 tips 始终注入

    # 关键词 → 标签映射
    keyword_to_tags = {
        # 消歧
        "who": ["person", "who", "disambig"],
        "whose": ["person", "who", "disambig"],
        "person": ["person", "disambig"],
        "name": ["person", "disambig"],
        # 多跳
        "which": ["multi_hop"],
        "that": ["multi_hop", "bridge"],
        # 数值
        "how many": ["number", "how_many"],
        "how much": ["number", "how_much", "amount"],
        "price": ["number", "price", "amount"],
        "cost": ["number", "amount"],
        "auction": ["number", "auction", "amount"],
        "revenue": ["number", "revenue"],
        "gdp": ["number", "GDP", "statistic"],
        "population": ["number", "population", "statistic"],
        "million": ["number", "amount"],
        "billion": ["number", "amount"],
        "percent": ["number", "statistic"],
        "$": ["number", "amount"],
        "usd": ["number", "amount"],
        "rmb": ["number", "amount"],
        "cny": ["number", "amount"],
        # 时间
        "year": ["year", "date"],
        "when": ["year", "when", "date"],
        "century": ["year", "century"],
        "decade": ["year", "period"],
        "founded": ["year", "founded", "organization"],
        "established": ["year", "established", "organization"],
        # 学术
        "journal": ["journal", "academic"],
        "paper": ["paper", "academic"],
        "article": ["article", "academic"],
        "published": ["publish", "academic"],
        "author": ["author", "academic"],
        "study": ["study", "academic", "research"],
        "methodology": ["methodology", "academic"],
        "research": ["research", "academic"],
        # 地理
        "where": ["location", "where"],
        "city": ["location", "city"],
        "country": ["location", "country"],
        "capital": ["location", "capital"],
        "born in": ["location", "born_in", "person"],
        "located": ["location", "located"],
        # 作品
        "book": ["work", "book"],
        "film": ["work", "film"],
        "movie": ["work", "movie"],
        "painting": ["work", "painting", "artist"],
        "novel": ["work", "novel"],
        "song": ["work", "song"],
        "album": ["work", "album"],
        "artist": ["artist", "work"],
        "composer": ["composer", "work"],
        "director": ["director", "work"],
        # 机构
        "university": ["organization", "university", "school"],
        "school": ["organization", "school"],
        "company": ["organization", "company"],
        "institution": ["organization", "institution"],
        "organization": ["organization"],
        # 排名
        "first": ["rank", "first"],
        "largest": ["rank", "largest"],
        "most": ["rank", "most"],
        "oldest": ["rank", "oldest"],
        "highest": ["rank", "highest"],
        "longest": ["rank", "longest"],
        # 定义
        "what is": ["what_is", "definition"],
        "define": ["definition"],
        "meaning": ["meaning", "definition"],
        # 因果
        "why": ["why", "cause"],
        "cause": ["cause"],
        "because": ["cause", "because"],
        "effect": ["effect", "impact"],
        "impact": ["impact", "effect"],
        "result": ["result", "consequence"],
        # 领域特定
        "colonial": ["colonial"],
        "encomienda": ["encomienda", "colonial"],
        "hacienda": ["hacienda", "colonial"],
        "prosopography": ["prosopography", "colonial", "academic"],
        "historiography": ["historiography", "colonial", "academic"],
        "latin america": ["latin_america", "colonial"],
        "spanish america": ["spanish_america", "colonial"],
    }

    for keyword, tags in keyword_to_tags.items():
        if keyword in q_lower:
            detected.extend(tags)

    # 年份模式检测：4位数字
    if re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", question):
        detected.extend(["year", "date"])

    # 金额模式检测
    if re.search(r"\$[\d,]+|\d+\s*(million|billion|万|亿|USD|EUR|CNY)", q_lower):
        detected.extend(["number", "amount"])

    # 句子复杂度检测：多个从句 → 多跳
    clause_markers = ["that", "which", "whose", "where", "who", "whom"]
    clause_count = sum(1 for m in clause_markers if f" {m} " in f" {q_lower} ")
    if clause_count >= 2:
        detected.extend(["multi_hop", "bridge", "indirect"])

    return list(set(detected))


def get_plan_tips(question: str) -> List[str]:
    """
    根据问题特征，从规则库中选取相关的 Plan Tips。

    Returns:
        匹配到的 tip 文本列表（已去重，通常 3~8 条）
    """
    tags = _detect_tags(question)
    tag_set = set(tags)

    matched: List[str] = []
    seen: set = set()

    for rule_tags, tip_text in TIPS_DB:
        # 规则匹配：rule_tags 中任意一个出现在检测到的 tag_set 中
        if tag_set & set(rule_tags):
            if tip_text not in seen:
                seen.add(tip_text)
                matched.append(tip_text)

    return matched


def format_tips_for_prompt(tips: List[str]) -> str:
    """
    将 tips 列表格式化为可直接拼入 prompt 的文本块。
    """
    if not tips:
        return ""

    lines = ["Plan Tips（基于历史经验的搜索策略建议，请在生成查询时参考）："]
    for i, tip in enumerate(tips, 1):
        lines.append(f"  {i}) {tip}")
    return "\n".join(lines)
