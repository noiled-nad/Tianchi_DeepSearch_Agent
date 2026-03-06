# -*- coding: utf-8 -*-
"""
目标类型定义和动态分析模块

用于在 extract_findings 阶段动态分析子任务目标，
并选择合适的信息提取策略。

优化版本：
1. 修复匹配优先级问题
2. 改进验证逻辑（识别"未找到"类回答）
3. 新增目标类型（列表提取、名称匹配）
4. 精简提取指令
"""

from __future__ import annotations
import re
from enum import Enum
from typing import Optional, List, Tuple


class GoalType(Enum):
    """子任务目标类型"""
    EXTRACT_ENTITY = "extract_entity"       # 提取具体实体名称（人名/武器名/公司名）
    EXTRACT_LIST = "extract_list"           # 提取列表（多个名称/选项）
    EXTRACT_MATCH = "extract_match"         # 名称匹配（找出A和B的共同点/匹配项）
    EXTRACT_RELATION = "extract_relation"   # 提取关系（A收购B）
    EXTRACT_NUMBER = "extract_number"       # 提取数字（年份/金额）
    CONFIRM_EXISTENCE = "confirm_existence" # 确认是否存在
    GATHER_INFO = "gather_info"             # 收集背景信息（默认）


# ──────────────── 匹配优先级（按优先级排序）────────────────

# 注意：列表顺序决定匹配优先级，高优先级在前
GOAL_PRIORITY: List[GoalType] = [
    GoalType.EXTRACT_MATCH,      # 最高优先级：名称匹配问题
    GoalType.EXTRACT_NUMBER,     # 数字提取
    GoalType.EXTRACT_RELATION,   # 关系提取
    GoalType.EXTRACT_LIST,       # 列表提取
    GoalType.EXTRACT_ENTITY,     # 实体提取
    GoalType.CONFIRM_EXISTENCE,  # 存在性确认
    GoalType.GATHER_INFO,        # 默认
]


# ──────────────── 目标类型特征（关键词规则）────────────────

GOAL_PATTERNS = {
    GoalType.EXTRACT_MATCH: {
        "keywords": [
            "名称.*相同", "名称.*一样", "名称.*匹配", "名称.*一致",
            "与.*同名", "和.*同名", "同名",
            "匹配", "对应", "交叉",
            "same name", "matching name", "corresponding"
        ],
        "patterns": [
            r"名称.*(?:相同|一样|一致)",
            r"(?:与|和).*同名",
            r"匹配.*名称",
            r"武器.*角色.*名.*同"
        ]
    },

    GoalType.EXTRACT_NUMBER: {
        "keywords": [
            "年份", "哪一年", "何时", "时间", "日期",
            "金额", "多少钱", "价格", "费用",
            "数量", "多少个", "几个",
            "when", "what year", "how much", "how many"
        ],
        "patterns": [
            r"哪一年",
            r"何时",
            r"多少(?:年|钱|个)",
            r"(?:年份|时间|金额|数量|价格)"
        ]
    },

    GoalType.EXTRACT_RELATION: {
        "keywords": [
            "收购", "被.*收购", "母公司", "子公司", "拥有",
            "关系", "联系", "属于",
            "acquired", "acquisition", "owns?", "parent company", "subsidiary"
        ],
        "patterns": [
            r"(?:谁|哪个).*(?:收购|拥有)",
            r"被.*(?:收购|拥有)",
            r"(?:母公司|子公司)",
            r"(?:收购|拥有).*关系"
        ]
    },

    GoalType.EXTRACT_LIST: {
        "keywords": [
            "列表", "名单", "所有", "全部", "清单",
            "有哪些", "列出", "枚举",
            "list", "all", "items"
        ],
        "patterns": [
            r"(?:所有|全部|列出).*(?:名称|列表|名单)",
            r"有哪些",
            r"(?:武器|角色|游戏).*列表"
        ]
    },

    GoalType.EXTRACT_ENTITY: {
        "keywords": [
            "名称", "叫什么", "是什么", "名叫", "名为", "名字",
            "找出", "锁定", "确认.*是", "具体是", "到底是",
            "title", "name"
        ],
        "patterns": [
            r"(?:武器|角色|人物|公司|游戏|组织).*(?:叫|称|名|是)",
            r"找出.*(?:名称|名字)",
            r"(?:叫|名|称).*什么"
        ]
    },

    GoalType.CONFIRM_EXISTENCE: {
        "keywords": [
            "是否存在", "有没有", "是否.*有",
            "exist", "whether.*exist", "is there"
        ],
        "patterns": [
            r"是否(?:存在|有)",
            r"有(?:没有|无)"
        ]
    },
}


# ──────────────── 提取指令模板（精简版）────────────────

EXTRACTION_INSTRUCTIONS = {
    GoalType.EXTRACT_ENTITY: """\
【目标：提取具体名称】
- 必须写出实体的完整名称（人名/武器名/公司名/游戏名）
- 用《》或""标注名称
- 若文档中无此信息，回答"文档中未提及"
- 示例：武器名称是《魂舞者》""",

    GoalType.EXTRACT_LIST: """\
【目标：提取名称列表】
- 列出文档中提到的所有相关名称
- 每个名称单独一行，用序号或符号标注
- 若无相关信息，回答"文档中未提及"
- 示例：1.《名称A》 2.《名称B》 3.《名称C》""",

    GoalType.EXTRACT_MATCH: """\
【目标：找出名称匹配项】
- 分别列出两侧的名称（如：武器名列表、角色名列表）
- 找出名称相同或对应的项
- 必须明确写出匹配的名称是什么
- 示例：武器《魂舞者》与角色"魂舞者"名称相同""",

    GoalType.EXTRACT_RELATION: """\
【目标：提取实体关系】
- 明确说明：A 与 B 的关系（收购/拥有/母子公司等）
- 若有时间，一并说明
- 示例：腾讯于2011年收购了Riot Games""",

    GoalType.EXTRACT_NUMBER: """\
【目标：提取具体数字】
- 必须写出具体数字（年份/金额/数量）
- 注意单位
- 示例：收购年份是2011年""",

    GoalType.CONFIRM_EXISTENCE: """\
【目标：确认是否存在】
- 明确回答：存在/不存在/无法确认
- 简要说明依据""",

    GoalType.GATHER_INFO: """\
【目标：收集背景信息】
- 总结3~8个关键要点
- 标注来源 [D1] [D2] 等""",
}


# ──────────────── 目标类型分析函数 ────────────────

def _match_by_keywords(text: str) -> Optional[GoalType]:
    """按优先级顺序匹配目标类型"""
    text_lower = text.lower()

    # 按优先级顺序遍历
    for goal_type in GOAL_PRIORITY:
        if goal_type not in GOAL_PATTERNS:
            continue

        patterns = GOAL_PATTERNS[goal_type]

        # 先检查正则模式（更精确）
        for pattern in patterns.get("patterns", []):
            if re.search(pattern, text, re.IGNORECASE):
                return goal_type

        # 再检查关键词
        for keyword in patterns.get("keywords", []):
            if keyword.lower() in text_lower:
                return goal_type

    return None


def analyze_goal_type(
    subtask_title: str,
    subtask_reason: str = "",
    question: str = "",
) -> GoalType:
    """
    分析子任务的目标类型。

    优先级：
    1. 正则模式匹配（更精确）
    2. 关键词匹配
    3. 默认返回 GATHER_INFO
    """
    # 组合文本进行分析（标题权重最高）
    combined_text = f"{subtask_title} {subtask_title} {subtask_reason} {question}"

    # 尝试匹配
    matched = _match_by_keywords(combined_text)
    if matched:
        return matched

    return GoalType.GATHER_INFO


def get_extraction_instructions(goal_type: GoalType) -> str:
    """获取目标类型对应的提取指令"""
    return EXTRACTION_INSTRUCTIONS.get(goal_type, EXTRACTION_INSTRUCTIONS[GoalType.GATHER_INFO])


def get_goal_type_name(goal_type: GoalType) -> str:
    """获取目标类型的可读名称"""
    names = {
        GoalType.EXTRACT_ENTITY: "提取实体名称",
        GoalType.EXTRACT_LIST: "提取列表",
        GoalType.EXTRACT_MATCH: "名称匹配",
        GoalType.EXTRACT_RELATION: "提取关系",
        GoalType.EXTRACT_NUMBER: "提取数字",
        GoalType.CONFIRM_EXISTENCE: "确认存在性",
        GoalType.GATHER_INFO: "收集信息",
    }
    return names.get(goal_type, "未知类型")


# ──────────────── 验证函数（优化版）────────────────

# 表示"未找到信息"的有效短语
NOT_FOUND_PHRASES = [
    "未提及", "未找到", "未提到", "无法确认", "无法确定",
    "文档中没有", "没有提到", "没有提及",
    "not mentioned", "not found", "no information"
]

# 表示具体名称的标记
ENTITY_MARKERS = ["《", "「", "\"", "「", "『"]


def validate_findings_by_goal(
    findings: str,
    goal_type: GoalType,
) -> Tuple[bool, str]:
    """
    根据目标类型验证 findings 是否符合要求。

    返回: (is_valid, error_message)

    注意：如果 findings 明确表示"未找到信息"，则视为有效（这是诚实的回答）。
    """
    if not findings or findings.strip() == "":
        return False, "findings 为空"

    findings_lower = findings.lower()

    # 检查是否是"未找到信息"的有效回答
    is_not_found = any(phrase in findings for phrase in NOT_FOUND_PHRASES)

    if goal_type == GoalType.EXTRACT_ENTITY:
        # 如果明确说"未找到"，这是有效回答
        if is_not_found:
            return True, ""

        # 检查是否包含具体名称标记
        has_entity_marker = any(mark in findings for mark in ENTITY_MARKERS)
        # 检查是否有明确的命名动词
        has_naming_verb = any(v in findings for v in ["是《", "名为", "名叫", "叫《", "名称是"])

        if has_entity_marker or has_naming_verb:
            return True, ""

        # 检查是否只是模糊描述
        vague_only = any(phrase in findings_lower for phrase in ["存在一个", "有一个", "有某个"])
        if vague_only:
            return False, "只包含模糊描述，未提取到具体实体名称"

        return True, ""  # 默认有效，避免过度严格

    elif goal_type == GoalType.EXTRACT_NUMBER:
        # 如果明确说"未找到"，这是有效回答
        if is_not_found:
            return True, ""
        # 检查是否包含数字
        has_digit = any(c.isdigit() for c in findings)
        if not has_digit:
            return False, "未找到数字"
        return True, ""

    elif goal_type == GoalType.EXTRACT_MATCH:
        # 如果明确说"未找到"，这是有效回答
        if is_not_found:
            return True, ""
        # 检查是否包含匹配说明
        has_match = any(word in findings for word in ["相同", "一样", "匹配", "对应", "同名", "same"])
        if not has_match:
            return False, "未说明名称匹配关系"
        return True, ""

    elif goal_type == GoalType.CONFIRM_EXISTENCE:
        # 检查是否给出明确结论
        confirm_phrases = ["存在", "不存在", "无法确认", "有", "没有"]
        has_confirmation = any(phrase in findings for phrase in confirm_phrases)
        if not has_confirmation:
            return False, "未给出明确的存否结论"
        return True, ""

    return True, ""


# ──────────────── 测试代码 ────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("目标类型分析测试")
    print("=" * 60)

    test_cases = [
        {"title": "查找北美公司 FPS 游戏近战武器名称", "reason": "", "expected": GoalType.EXTRACT_ENTITY},
        {"title": "确认北美游戏公司及其亚洲母公司", "reason": "需要找出公司名称", "expected": GoalType.EXTRACT_ENTITY},
        {"title": "腾讯何时收购Riot Games", "reason": "", "expected": GoalType.EXTRACT_NUMBER},
        {"title": "确认母公司收购信息", "reason": "了解腾讯是否收购了Riot Games", "expected": GoalType.EXTRACT_RELATION},
        {"title": "武器名称与格斗手游角色名称的匹配验证", "reason": "", "expected": GoalType.EXTRACT_MATCH},
        {"title": "列出所有近战武器", "reason": "", "expected": GoalType.EXTRACT_LIST},
        {"title": "收集ELO系统的背景信息", "reason": "了解其历史和应用", "expected": GoalType.GATHER_INFO},
    ]

    for i, case in enumerate(test_cases, 1):
        result = analyze_goal_type(
            subtask_title=case["title"],
            subtask_reason=case["reason"],
            question=""
        )
        expected = case["expected"]
        status = "✓" if result == expected else "✗"
        print(f"\n测试 {i}: {status}")
        print(f"  标题: {case['title']}")
        print(f"  预期: {get_goal_type_name(expected)}")
        print(f"  实际: {get_goal_type_name(result)}")

    print("\n" + "=" * 60)
    print("findings 验证测试（优化版）")
    print("=" * 60)

    validation_tests = [
        {"findings": "该武器名称是《魂舞者》", "goal_type": GoalType.EXTRACT_ENTITY, "expected_valid": True},
        {"findings": "文档中有一个近战武器", "goal_type": GoalType.EXTRACT_ENTITY, "expected_valid": False},
        {"findings": "文档中未提到具体公司名称", "goal_type": GoalType.EXTRACT_ENTITY, "expected_valid": True},  # 诚实回答
        {"findings": "武器《魂舞者》与角色\"魂舞者\"名称相同", "goal_type": GoalType.EXTRACT_MATCH, "expected_valid": True},
        {"findings": "列出了武器列表但没有匹配", "goal_type": GoalType.EXTRACT_MATCH, "expected_valid": False},
        {"findings": "收购年份是2011年", "goal_type": GoalType.EXTRACT_NUMBER, "expected_valid": True},
        {"findings": "腾讯收购了Riot Games", "goal_type": GoalType.EXTRACT_NUMBER, "expected_valid": False},
    ]

    for i, test in enumerate(validation_tests, 1):
        is_valid, error = validate_findings_by_goal(test["findings"], test["goal_type"])
        expected = test["expected_valid"]
        status = "✓" if is_valid == expected else "✗"
        print(f"\n验证测试 {i}: {status}")
        print(f"  findings: {test['findings'][:50]}...")
        print(f"  目标类型: {get_goal_type_name(test['goal_type'])}")
        print(f"  预期: {'有效' if expected else '无效'}")
        print(f"  实际: {'有效' if is_valid else '无效'}")
        if error:
            print(f"  错误信息: {error}")
