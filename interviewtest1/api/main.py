"""
FastAPI 应用程序 — Agentic 客户支持邮件处理系统。

API 端点 (Endpoints):
  POST /api/v1/emails/process          提交邮件以进行完整的管道处理
  GET  /api/v1/emails/{id}/status      获取处理状态 (此处为占位符 — 可扩展为异步)
  POST /api/v1/knowledge-base/ingest   触发知识库重新注入 (Ingestion)
  GET  /api/v1/health                  健康检查 (检查 Redis, ChromaDB 的连接)

运行命令:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from agents.graph import process_email
from agents.state import EmailState
from api.models import (
    ClassificationInfo,
    EmailRequest,
    EscalationInfo,
    GuardrailInfo,
    HealthResult,
    IngestRequest,
    IngestResult,
    ProcessingResult,
)
from config.settings import settings
from rag.ingest import ingest
from services.contact_tracker import get_contact_count, record_contact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内存结果存储字典 (在生产环境中请扩展为 Redis 或数据库)
# ---------------------------------------------------------------------------

_results_store: dict[str, ProcessingResult] = {}

# ---------------------------------------------------------------------------
# API 密钥身份验证
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(_api_key_header)) -> str:
    """验证 HTTP 请求头中是否包含正确的 API 密钥。"""
    if key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺失的 API 密钥。请传递 X-API-Key 请求头。",
        )
    return key


# ---------------------------------------------------------------------------
# 应用程序生命周期管理
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 应用的启动和关闭事件。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(name)s — %(message)s")
    logger.info("Email Processing System starting up...")
    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI 应用实例定义
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic 客户支持邮件处理系统",
    description=(
        "一个生产级别的智能系统，能自动对客户支持邮件进行分类、路由，"
        "并起草回复草稿 — 拥有防止幻觉的硬性安全护栏和确定性的升级逻辑。"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 允许跨域资源共享，方便前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 路由端点定义 (Endpoints)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/emails/process",
    response_model=ProcessingResult,
    status_code=status.HTTP_200_OK,
    summary="通过完整的 Agentic 管道处理一封支持邮件",
    tags=["Email Processing"],
)
async def process_email_endpoint(
    request: EmailRequest,
    _: str = Depends(verify_api_key),
) -> ProcessingResult:
    """
    提交一封客户支持邮件进行处理。

    这封邮件将流经以下节点：
    1. **分类 (Classification)** — 通过结构化的 LLM 输出获取类别和置信度
    2. **升级检查 (Escalation Check)** — 基于关键词或频率规则的确定性路由
    3. **RAG 检索 (RAG Retrieval)** — 从知识库获取有事实依据的上下文
    4. **回复起草 (Response Drafting)** — 带有反幻觉约束的回复草稿生成
    5. **护栏验证 (Guardrail Validation)** — 交付前的三层事实核查

    返回完整的处理结果，包括分类、路由决策、回复草稿（如果被批准的话）以及完整的审计追踪记录。
    """
    # 记录该发件人的联系次数，用于频率追踪
    record_contact(email=request.sender_email)

    # 构建初始的状态图输入
    initial_state: EmailState = {
        "email_id": request.email_id,
        "sender_email": request.sender_email,
        "subject": request.subject,
        "body": request.body,
        "received_at": request.received_at or datetime.now(timezone.utc).isoformat(),
        "audit_log": [],
    }

    try:
        # 调用核心管道进行处理
        final_state = process_email(initial_state)
    except Exception as exc:
        logger.exception("[%s] 管道处理错误: %s", request.email_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"管道处理错误: {exc}",
        )

    # 将管道返回的状态转换为对外暴露的 Pydantic 模型
    result = ProcessingResult(
        email_id=final_state["email_id"],
        final_action=final_state.get("final_action", "route_to_human"),
        assigned_agent_queue=final_state.get("assigned_agent_queue"),
        classification=ClassificationInfo(
            category=final_state.get("category", "general"),
            confidence=final_state.get("confidence", 0.0),
            detected_keywords=final_state.get("detected_keywords", []),
            reasoning=final_state.get("classification_reasoning", ""),
        ),
        escalation=EscalationInfo(
            is_critical=final_state.get("is_critical", False),
            reasons=final_state.get("escalation_reasons", []),
            contact_count_7d=final_state.get("contact_count_7d", 0),
        ),
        guardrail=GuardrailInfo(
            passed=final_state.get("guardrail_passed", False),
            violations=final_state.get("guardrail_violations", []),
        ),
        draft_response=final_state.get("draft_response"),
        citations=final_state.get("citations", []),
        audit_log=final_state.get("audit_log", []),
    )

    # 存入内存缓存
    _results_store[request.email_id] = result
    return result


@app.get(
    "/api/v1/emails/{email_id}/status",
    response_model=ProcessingResult,
    summary="获取之前提交邮件的处理结果",
    tags=["Email Processing"],
)
async def get_email_status(
    email_id: str,
    _: str = Depends(verify_api_key),
) -> ProcessingResult:
    """从缓存中检索以前处理过的邮件结果。"""
    result = _results_store.get(email_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到 email_id={email_id!r} 的结果。请先通过 POST /api/v1/emails/process 提交处理。",
        )
    return result


@app.post(
    "/api/v1/knowledge-base/ingest",
    response_model=IngestResult,
    summary="触发知识库重构与注入",
    tags=["Knowledge Base"],
)
async def ingest_knowledge_base(
    request: IngestRequest = IngestRequest(),
    _: str = Depends(verify_api_key),
) -> IngestResult:
    """
    将指定目录中的文档重新加载并存入 ChromaDB 向量数据库。

    当您在知识库中添加或更新文档时，请使用此端点。
    """
    try:
        # 调用注入脚本将文档分块并向量化
        chunks = ingest(request.docs_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("知识注入错误: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识注入错误: {exc}",
        )
    return IngestResult(
        chunks_stored=chunks,
        message=f"成功从 '{request.docs_dir}' 注入了 {chunks} 个数据块。",
    )


@app.get(
    "/api/v1/health",
    response_model=HealthResult,
    summary="健康状态检查",
    tags=["System"],
)
async def health_check() -> HealthResult:
    """检查与 Redis 和 ChromaDB 数据库的连接状态。"""
    redis_ok = False
    chroma_ok = False

    # 检查 Redis 连通性
    if settings.redis_url:
        try:
            import redis as redis_lib
            client = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
            client.ping()
            redis_ok = True
        except Exception:
            pass

    # 检查 ChromaDB 连通性
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.heartbeat()
        chroma_ok = True
    except Exception:
        pass

    # Redis 是可选的，主要看 ChromaDB
    overall = "ok" if (chroma_ok) else "degraded"
    return HealthResult(
        status=overall,
        redis_connected=redis_ok,
        chroma_connected=chroma_ok,
    )

