"""
护栏验证器 (Guardrail Validator) — 三层生成后验证机制。

第一层: 退款政策护栏 (确定性校验)
    - 如果草稿中提到了退款条款，则将每一个退款相关的声明
      与权威的 refund_policy.md 知识块进行交叉比对。

第二层: 接地性/事实依据检查 (由大模型作为裁判)
    - 调用独立的大模型来询问：“草稿中的每一个声明是否都有上下文依据？”

第三层: 违禁内容扫描 (正则表达式/确定性校验)
    - 拦截未在上下文中出现过的具体金额、编造的截止日期等。

如果任何一层验证失败 → 将处理动作转交人工，并附带违规详情。
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.state import EmailState
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 第一层 — 退款政策护栏
# ---------------------------------------------------------------------------

# 触发退款检查的正则表达式模式
_REFUND_TRIGGER_PATTERN = re.compile(
    r"\b(refund|money\s+back|return|cancell?ation|reimbursement|reimburse)\b",
    re.IGNORECASE,
)

# 必须在退款文档中找到依据的可疑声明模式
_REFUND_CLAIM_PATTERNS = [
    re.compile(r"\b\d+[-\s]?(business\s+)?days?\b", re.IGNORECASE), # 处理天数
    re.compile(r"\b\d+\s*%\b"),  # 百分比
    re.compile(r"\b(eligible|not\s+eligible)\b", re.IGNORECASE), # 是否符合资格
    re.compile(r"\$\d+", re.IGNORECASE), # 具体的美元金额
]


def _extract_refund_policy_text(chunks: list[dict]) -> str:
    """提取退款政策知识块中的纯文本内容并拼接。"""
    parts = [
        c["content"]
        for c in chunks
        if c.get("metadata", {}).get("is_refund_policy") is True
    ]
    return " ".join(parts).lower()


def _check_refund_guardrail(draft: str, chunks: list[dict]) -> list[str]:
    """
    检查退款相关的声明是否具有文档依据。
    返回违规列表 (空列表表示通过)。
    只有在草稿包含退款相关词汇时才会运行。
    """
    if not _REFUND_TRIGGER_PATTERN.search(draft):
        return []

    refund_text = _extract_refund_policy_text(chunks)
    if not refund_text:
        return [
            "草稿包含退款声明，但未检索到任何退款政策知识块。无法验证事实依据。"
        ]

    violations = []
    for pattern in _REFUND_CLAIM_PATTERNS:
        for match in pattern.finditer(draft):
            claim = match.group(0).strip()
            # 检查这个特定的声明是否出现在检索到的退款政策文本中
            if claim.lower() not in refund_text:
                violations.append(
                    f"在政策文档中未找到对应的退款声明: '{claim}'"
                )
    return violations


# ---------------------------------------------------------------------------
# 第二层 — 由大模型作为裁判的事实依据检查 (LLM-as-judge)
# ---------------------------------------------------------------------------

class GroundingVerdict(BaseModel):
    is_grounded: bool = Field(
        description="如果草稿中的每一个事实声明都有上下文的支持，则为 True。"
    )
    ungrounded_statements: list[str] = Field(
        default_factory=list,
        description="草稿中明确不被上下文支持的具体声明列表。",
    )
    verdict_reasoning: str = Field(
        description="用一句话解释整体判决的理由。"
    )


# 裁判系统提示词
_JUDGE_SYSTEM = """\
你是一个客户支持系统的严格事实核查员。你的工作是\
验证起草的回复是否完全建立在提供的上下文基础之上 (接地性)。

规则:
- 如果一个声明在上下文中被明确表述或直接暗示，则该声明是有依据的。
- 如果一个声明引入了任何未在上下文中出现的事实、政策细节、时间线、价格\
或程序，则该声明是无依据的 (幻觉)。
- 诸如问候语或“如果您还有其他问题，请告诉我们”之类的通用陈述\
始终被视为有依据的 (它们不构成事实性声明)。
"""

_JUDGE_USER = """\
## 检索到的上下文

{context}

---
## 待验证的回复草稿

{draft}

---
上述回复草稿中的每一个事实声明是否都得到了检索到的上下文的支持？
"""

_judge_llm = ChatGoogleGenerativeAI(
    model=settings.google_model_judge,
    temperature=0, # 确保证实核查的稳定性
    google_api_key=settings.google_api_key,
)
_judge_chain = (
    ChatPromptTemplate.from_messages(
        [("system", _JUDGE_SYSTEM), ("human", _JUDGE_USER)]
    )
    | _judge_llm.with_structured_output(GroundingVerdict)
)


def _check_grounding(draft: str, chunks: list[dict]) -> list[str]:
    """返回来自大模型裁判的违规列表 (空列表表示通过)。"""
    if not chunks:
        # 没有上下文 → 无法验证；标记为需要人工审核
        return ["No context chunks available — cannot verify grounding."]

    context_str = "\n\n".join(c["content"] for c in chunks)
    verdict: GroundingVerdict = _judge_chain.invoke(
        {"context": context_str, "draft": draft}
    )

    if verdict.is_grounded:
        return []
    return [f"[LLM-judge] {s}" for s in verdict.ungrounded_statements] or [
        f"[LLM-judge] Grounding check failed: {verdict.verdict_reasoning}"
    ]


# ---------------------------------------------------------------------------
# 第三层 — 违禁内容扫描 (基于正则表达式)
# ---------------------------------------------------------------------------

# 匹配具体美元金额，但不属于 "X business days" 形式的数字 (那些已在第一层被捕获)
_DOLLAR_AMOUNT_RE = re.compile(r"\$\d[\d,]*(?:\.\d{2})?")
# 匹配可能涉及伪造联系方式的模式
_FAKE_CONTACT_RE = re.compile(
    r"\b(call\s+us\s+at|phone|1-\d{3}-\d{3}-\d{4}|ext\s*\.\s*\d+)\b", re.IGNORECASE
)
# 匹配非文档来源的法律术语声明
_LEGAL_CLAIM_RE = re.compile(
    r"\b(legally\s+obligated|guaranteed\s+by\s+law|your\s+legal\s+right)\b",
    re.IGNORECASE,
)


def _check_prohibited_content(draft: str, chunks: list[dict]) -> list[str]:
    """返回违规列表 (空列表表示通过)。"""
    all_context = " ".join(c["content"] for c in chunks)
    violations = []

    for match in _DOLLAR_AMOUNT_RE.finditer(draft):
        amount = match.group(0)
        if amount not in all_context:
            violations.append(
                f"在检索到的上下文中未找到美元金额 '{amount}' — 可能是模型幻觉。"
            )

    if _FAKE_CONTACT_RE.search(draft):
        contact_snippet = _FAKE_CONTACT_RE.search(draft).group(0)
        if contact_snippet.lower() not in all_context.lower():
            violations.append(
                f"检测到伪造的联系方式信息: '{contact_snippet}'"
            )

    if _LEGAL_CLAIM_RE.search(draft):
        violations.append("草稿中包含不受文档支持的法律声明。")

    return violations


# ---------------------------------------------------------------------------
# 工作流节点函数 (Node function)
# ---------------------------------------------------------------------------

def validate_guardrails(state: EmailState) -> EmailState:
    """
    LangGraph 节点函数：对起草的回复执行所有三层护栏验证。

    输入状态包含: email_id, draft_response, retrieved_chunks
    输出状态追加: guardrail_passed, guardrail_violations, final_action (视条件而定),
            assigned_agent_queue, audit_log
    """
    email_id = state["email_id"]
    draft = state.get("draft_response", "")
    chunks = state.get("retrieved_chunks", [])

    all_violations: list[str] = []

    # 运行第一层
    layer1 = _check_refund_guardrail(draft, chunks)
    if layer1:
        logger.warning("[%s] Layer 1 (refund guardrail) violations: %s", email_id, layer1)
    all_violations.extend(layer1)

    # 运行第二层
    layer2 = _check_grounding(draft, chunks)
    if layer2:
        logger.warning("[%s] Layer 2 (LLM judge) violations: %s", email_id, layer2)
    all_violations.extend(layer2)

    # 运行第三层
    layer3 = _check_prohibited_content(draft, chunks)
    if layer3:
        logger.warning("[%s] Layer 3 (prohibited content) violations: %s", email_id, layer3)
    all_violations.extend(layer3)

    # 如果所有违规列表为空，则验证通过
    guardrail_passed = len(all_violations) == 0
    updates: dict = {
        "guardrail_passed": guardrail_passed,
        "guardrail_violations": all_violations,
    }

    # 如果验证失败，则强制修改最终动作转交人工处理
    if not guardrail_passed:
        updates["final_action"] = "route_to_human"
        updates["assigned_agent_queue"] = state.get("assigned_agent_queue", "human_review")

    # 记录运行日志
    log_entry = (
        f"validate_guardrails: passed={guardrail_passed} "
        f"violations={len(all_violations)}"
    )
    logger.info("[%s] %s", email_id, log_entry)

    # 返回更新后的状态字典，传递给下一个节点
    return {
        **state,
        **updates,
        "audit_log": [*state.get("audit_log", []), log_entry],
    }
