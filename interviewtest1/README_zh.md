# Agentic 客户支持邮件处理系统

这是一个基于 **LangGraph** 和 **FastAPI** 构建的生产级智能化系统。它能自动完成客户支持邮件的分类、路由分配以及回复起草工作。系统内置了严格的“防幻觉”安全护栏和确定性的升级逻辑。

## 核心功能

- **图结构工作流编排**：使用 LangGraph 的 StateGraph 进行节点调度。
- **结构化的大模型输出**：在邮件分类环节强制使用 Pydantic 结构化输出。
- **确定性的升级路由**：对安全或关键问题采用纯规则引擎（不依赖大模型）进行路由分配。
- **结合知识库的 RAG 检索**：基于本地 ChromaDB 和公司文档生成可靠上下文。
- **三层安全护栏验证**：在发送任何自动回复之前进行严格的事实验证。
- **Redis 追踪联系频率**：内置滑动窗口追踪用户联系频率（并提供内存备用方案）。
- **FastAPI 服务层**：提供标准 API 接口并包含 API Key 认证功能。

## 快速开始

```bash
# 1. 复制环境配置模板
cp .env.example .env
# 编辑 .env 文件并填入你的 Google Gemini API 密钥: GOOGLE_API_KEY

# 2. 安装依赖
pip install -e ".[dev]"

# 3. 注入知识库
python -m rag.ingest

# 4. 运行 API 服务器
uvicorn api.main:app --reload

# 5. 运行自动化测试
pytest tests/ -v
```

## 系统架构

完整的系统架构图及组件详细设计，请参阅实现计划文档 (implementation plan)。
