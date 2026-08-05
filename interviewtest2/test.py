import logging
from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 导入 Playwright 用于执行 JavaScript，抓取动态渲染的网页
from playwright.sync_api import sync_playwright
# 导入 BeautifulSoup 用于解析 HTML 并提取纯文本
from bs4 import BeautifulSoup

# 配置日志记录器，用于在控制台输出运行状态
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. 护栏定义 (使用代码级别的验证器)
# ---------------------------------------------------------
# 定义大模型输出的结构化数据模型
class ConciseSummary(BaseModel):
    # Field 描述会作为提示词传递给大模型，指导其输出
    summary: str = Field(description="网页的核心摘要，绝对不能超过50个字。")
    key_points: List[str] = Field(description="最多提取3个核心要点。")

    # 护栏机制：严格校验摘要的长度，如果不符合要求则抛出异常，触发重试或失败
    @field_validator('summary')
    @classmethod
    def validate_summary_length(cls, v: str) -> str:
        # 使用简单的按空格分词来估算字数/词数
        words = v.split()
        if len(words) > 50:
            raise ValueError(f"护栏校验失败：摘要过长 ({len(words)} 个词)。必须小于等于 50 个词。")
        return v

    # 护栏机制：严格校验提取的要点数量
    @field_validator('key_points')
    @classmethod
    def validate_key_points(cls, v: List[str]) -> List[str]:
        if len(v) > 3:
            raise ValueError(f"护栏校验失败：提取的要点过多 ({len(v)} 个)。必须小于等于 3 个。")
        return v

# ---------------------------------------------------------
# 2. 核心抓取与清理逻辑 (解决单页应用和 JS 渲染的瓶颈)
# ---------------------------------------------------------
def fetch_and_clean_html(url: str) -> str:
    logger.info(f"正在抓取 URL (将执行 JavaScript): {url}")
    try:
        # 启动 Playwright 上下文
        with sync_playwright() as p:
            # 启动无头浏览器 (后台运行，不显示界面)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # 访问指定 URL，wait_until="networkidle" 确保网页中动态加载的 JS 内容渲染完毕
            page.goto(url, wait_until="networkidle", timeout=15000)
            html_content = page.content()
            browser.close()

        # 使用 BeautifulSoup 解析获取到的 HTML 内容
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 移除无用的 HTML 标签 (如脚本、样式表、导航栏等)，减少发送给大模型的噪音
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
            element.decompose()
            
        # 提取纯文本，使用换行符分隔，并去除多余空白字符
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        logger.error(f"网页抓取失败: {e}")
        return ""

# ---------------------------------------------------------
# 3. 摘要生成与长内容处理逻辑
# ---------------------------------------------------------
def generate_concise_summary(text: str) -> ConciseSummary:
    # 鉴于当前的大模型 (如 Gemini 1.5 Flash) 拥有超长上下文窗口 (超过100万 Token)，
    # 我们可以直接传入全部文本，而不需要像传统做法那样进行文本分块 (Chunking)。
    # 作为安全兜底策略，如果文本长得离谱，我们截取前 100,000 个字符。
    safe_text = text[:100000] if len(text) > 100000 else text
    
    # 实例化大模型，temperature=0 保证输出的确定性和稳定性
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 使用 with_structured_output 方法绑定 Pydantic 模型，
    # 这样 LangChain 会在底层自动处理格式化提示词以及解析错误时的自动重试。
    structured_llm = llm.with_structured_output(ConciseSummary)
    
    # 构建提示词模板
    prompt = PromptTemplate.from_template(
        "请总结以下网页内容。你必须严格遵守输出格式和长度限制。\n\n网页内容:\n{context}\n"
    )
    
    # 将提示词和结构化大模型组合成处理链
    chain = prompt | structured_llm
    
    logger.info("正在调用大模型并执行护栏验证...")
    # structured_llm 会根据之前定义的 validate_summary_length 等验证器自动校验结果
    result = chain.invoke({"context": safe_text})
    return result

# ---------------------------------------------------------
# 4. 测试运行入口
# ---------------------------------------------------------
if __name__ == "__main__":
    # 注意：运行前需要安装依赖：pip install playwright 并且执行 playwright install chromium
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    clean_text = fetch_and_clean_html(test_url)
    
    if clean_text:
        try:
            # 执行摘要生成流程
            summary_result = generate_concise_summary(clean_text)
            print("\n--- 最终摘要 (已通过护栏校验) ---")
            print(f"摘要 ({len(summary_result.summary.split())} 个词): {summary_result.summary}")
            print(f"核心要点 ({len(summary_result.key_points)} 项):")
            for pt in summary_result.key_points:
                print(f"- {pt}")
        except Exception as e:
            # 如果大模型生成的摘要最终仍然不符合护栏要求，则会捕获到这里的异常
            logger.error(f"护栏拦截或生成失败: {e}")
