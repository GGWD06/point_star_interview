"""
LangGraph StateGraph 编排模块。

定义了完整的智能代理管道，作为一个编译后的图：
  classify_email (分类邮件) → check_escalation (检查是否升级)
    ├─ [is_critical=True] → route_to_human (转交人工) → write_audit_log (写入审计日志) → END
    └─ [is_critical=False] → retrieve_context (检索上下文) → draft_response (起草回复)
                              → validate_guardrails (验证护栏)
                                ├─ [pass] → send_response (发送回复) → write_audit_log → END
                                └─ [fail] → route_to_human (转交人工) → write_audit_log → END
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from agents.classifier import classify_email
from agents.escalation_checker import check_escalation
from agents.guardrails import validate_guardrails
from agents.response_drafter import draft_response
from agents.retrieve_context import retrieve_context
from agents.state import EmailState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 终端节点函数 (Terminal node functions)
# ---------------------------------------------------------------------------

def route_to_human(state: EmailState) -> EmailState:
    """将邮件标记为转交人工，并记录最终的审计日志条目。"""
    email_id = state["email_id"]
    queue = state.get("assigned_agent_queue", "human_review")
    reasons = state.get("escalation_reasons", []) + state.get("guardrail_violations", [])

    log_entry = f"route_to_human: queue={queue!r} reasons={reasons}"
    logger.warning("[%s] %s", email_id, log_entry)

    return {
        **state,
        "final_action": "route_to_human",
        "assigned_agent_queue": queue,
        "audit_log": [*state.get("audit_log", []), log_entry],
    }


def send_response(state: EmailState) -> EmailState:
    """将邮件标记为已自动回复 (这里是与 SMTP/SendGrid 集成的挂载点)。"""
    email_id = state["email_id"]
    log_entry = "send_response: draft approved — auto-reply dispatched"
    logger.info("[%s] %s", email_id, log_entry)
    # 注意：这里可以触发外部服务发送真实的邮件。
    return {
        **state,
        "final_action": "send_auto_reply",
        "audit_log": [*state.get("audit_log", []), log_entry],
    }


def write_audit_log(state: EmailState) -> EmailState:
    """最终节点：给审计日志盖上完成的时间戳。"""
    timestamp = datetime.now(timezone.utc).isoformat()
    final_log_entry = (
        f"audit_complete: action={state.get('final_action')!r} "
        f"at={timestamp}"
    )
    return {
        **state,
        "audit_log": [*state.get("audit_log", []), final_log_entry],
    }


# ---------------------------------------------------------------------------
# 条件边路由函数 (Conditional edge functions)
# ---------------------------------------------------------------------------

def _route_after_escalation(state: EmailState) -> str:
    """在检查升级条件后，决定下一步是转交人工还是检索上下文。"""
    if state.get("is_critical"):
        return "route_to_human"
    return "retrieve_context"


def _route_after_guardrail(state: EmailState) -> str:
    """在执行护栏验证后，决定下一步是发送回复还是由于失败转交人工。"""
    if state.get("guardrail_passed"):
        return "send_response"
    return "route_to_human"


# ---------------------------------------------------------------------------
# 图定义 (Graph definition)
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """构建并编译 LangGraph StateGraph (状态图)。"""
    graph = StateGraph(EmailState)

    # 注册所有的节点
    graph.add_node("classify_email", classify_email)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("draft_response", draft_response)
    graph.add_node("validate_guardrails", validate_guardrails)
    graph.add_node("route_to_human", route_to_human)
    graph.add_node("send_response", send_response)
    graph.add_node("write_audit_log", write_audit_log)

    # 建立边 (连接各个节点)
    graph.add_edge(START, "classify_email")
    graph.add_edge("classify_email", "check_escalation")

    # 根据 _route_after_escalation 的返回值进行条件分支跳转
    graph.add_conditional_edges(
        "check_escalation",
        _route_after_escalation,
        {
            "route_to_human": "route_to_human",
            "retrieve_context": "retrieve_context",
        },
    )

    graph.add_edge("retrieve_context", "draft_response")
    graph.add_edge("draft_response", "validate_guardrails")

    # 根据 _route_after_guardrail 的返回值进行条件分支跳转
    graph.add_conditional_edges(
        "validate_guardrails",
        _route_after_guardrail,
        {
            "send_response": "send_response",
            "route_to_human": "route_to_human",
        },
    )

    graph.add_edge("route_to_human", "write_audit_log")
    graph.add_edge("send_response", "write_audit_log")
    graph.add_edge("write_audit_log", END)

    return graph


# ---------------------------------------------------------------------------
# 编译后的图单例 (Compiled graph singleton)
# ---------------------------------------------------------------------------

_compiled_graph = None


def get_compiled_graph():
    """返回编译好的 LangGraph 实例 (使用懒加载并缓存)。"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


def process_email(initial_state: EmailState) -> EmailState:
    """
    顶层入口点：将邮件送入完整的管道进行处理。

    参数:
        initial_state: 必须包含 email_id, sender_email, subject, body, received_at。

    返回:
        所有节点执行完毕后的最终 EmailState。
    """
    graph = get_compiled_graph()
    result = graph.invoke(initial_state)
    return result
