"""
分类代理程序 (Classifier Agent) — 使用结构化的大语言模型输出 (通过 Pydantic 强制进行模式校验)
来对收到的客户支持邮件进行分类。
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.state import EmailState
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 结构化输出模型 (Structured output schema)
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """
    大语言模型必须强制返回的结构化输出数据。
    这里使用了大模型的函数调用 (Function-calling) 能力来确保输出符合格式。
    """

    category: Literal["billing", "technical", "feedback", "general"] = Field(
        description=(
            "邮件的分类类别。"
            "'billing' (账单) = 支付、发票、退款、扣费问题。"
            "'technical' (技术) = 漏洞、错误、登录问题、性能问题。"
            "'feedback' (反馈) = 功能请求、表扬、产品建议。"
            "'general' (通用) = 其他所有不属于上述类别的内容。"
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分类的置信度评分，范围在 0.0 到 1.0 之间。"
    )
    detected_keywords: list[str] = Field(
        description="促使模型做出该分类决策的邮件关键单词或短语。"
    )
    reasoning: str = Field(
        description="用一句话解释为什么选择这个分类类别。"
    )


# ---------------------------------------------------------------------------
# 提示词模板 (Prompt template)
# ---------------------------------------------------------------------------

# 系统提示词，用于设定 AI 的角色并提供分类示例 (Few-Shot Prompting)
SYSTEM_PROMPT = """\
你是一个专业的客户支持邮件分类专家。你的工作是阅读客户发送的支持邮件，并将其准确分类为以下类别之一：

- billing: 支付问题、退款请求、发票问题、意外扣费、订阅管理。
- technical: 软件漏洞、登录失败、错误代码、性能问题、集成问题、数据丢失。
- feedback: 功能请求、产品表扬、可用性建议、对产品方向的一般性抱怨。
- general: 账户管理 (非计费类)、入门引导问题、任何不属于上述类别的内容。

请参考下面的少样本示例 (Few-Shot Examples) 来校准你的分类标准。

## 少样本示例

### 示例 1
主题: I was charged twice this month
正文: Hello, I noticed two charges of $79 on my credit card for the same billing cycle. Please refund one.
→ category: billing | confidence: 0.97 | keywords: charged twice, refund

### 示例 2
主题: App crashes on startup
正文: Since the last update, your app crashes immediately when I open it on my iPhone 15. Error code: 500.
→ category: technical | confidence: 0.95 | keywords: crashes, error code 500

### 示例 3
主题: Would love a dark mode
正文: The app is great but it would be amazing to have a dark mode option. Many of us work at night!
→ category: feedback | confidence: 0.92 | keywords: dark mode, feature request

### 示例 4
主题: How do I add a team member?
正文: Hi, I just signed up for the Professional plan. How do I invite my colleague?
→ category: general | confidence: 0.88 | keywords: add team member, invite
"""

# 用户提示词模板，用于动态插入具体的邮件数据
USER_TEMPLATE = """\
请对以下客户支持邮件进行分类。

主题: {subject}
发件人: {sender_email}
正文:
{body}
"""

# 组装完整的聊天提示词模板
_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)]
)

# ---------------------------------------------------------------------------
# 大语言模型设置 (LLM setup)
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(
    model=settings.google_model_classifier,
    temperature=0,  # 使用 temperature=0 确保每次分类结果是确定性的
    google_api_key=settings.google_api_key,
)

# 构建处理链：将提示词传递给大模型，并强制其输出 ClassificationResult 格式的数据
_chain = _prompt | _llm.with_structured_output(ClassificationResult)

# ---------------------------------------------------------------------------
# 工作流节点函数 (Node function)
# ---------------------------------------------------------------------------

def classify_email(state: EmailState) -> EmailState:
    """
    LangGraph 节点函数：对传入的邮件进行分类，并更新状态字典。

    输入状态包含: email_id, sender_email, subject, body
    输出状态追加: category, confidence, detected_keywords, classification_reasoning, audit_log
    """
    logger.info("[%s] 正在分类邮件: %r", state["email_id"], state["subject"])

    # 调用大模型执行分类推理
    result: ClassificationResult = _chain.invoke(
        {
            "subject": state["subject"],
            "sender_email": state["sender_email"],
            "body": state["body"],
        }
    )

    # 记录运行日志
    log_entry = (
        f"classify_email: category={result.category!r} "
        f"confidence={result.confidence:.2f} "
        f"keywords={result.detected_keywords}"
    )
    logger.info("[%s] %s", state["email_id"], log_entry)

    # 返回更新后的状态字典，传递给下一个节点
    return {
        **state,
        "category": result.category,
        "confidence": result.confidence,
        "detected_keywords": result.detected_keywords,
        "classification_reasoning": result.reasoning,
        "audit_log": [*state.get("audit_log", []), log_entry],
    }
