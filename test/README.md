# Nova Dynamics Conversational QA Agent

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

### Project Overview

This is an interactive Question & Answer Agent built using **LangGraph** and the **Google Gemini API**.

**To better demonstrate the practicality and real-world application of this Agent, we have intentionally provided it with a specific background and identity:** 
It is configured to act as the official AI customer support assistant, named **Nova**, for a fictional smart-home company called **Nova Dynamics Inc.**

By providing it with an internal company knowledge base (`sample_document.txt`, which includes company background, product pricing, refund policies, etc.), the Agent can simulate real customer support scenarios. It can answer product inquiries, calculate order discounts, and converse naturally with users across multiple turns.

### Core Features

1. **Roleplaying & Identity**:
   * Possesses a complete customer service persona, maintaining a polite and professional tone throughout the conversation.
   * Strictly avoids hallucination. For questions not covered in the company documents (e.g., grocery prices, local computer files), it clearly states that the request is outside its service scope.
2. **Conversation Memory**:
   * Capable of remembering the user's name and conversational context. For example, if you state your name in the first message, it will remember it in all subsequent interactions.
3. **Tool Use (Function Calling)**:
   * **Document Search (`search_document`)**: When users ask about products, refund policies, or warranty terms, the Agent automatically scans the knowledge base to extract information.
   * **Math Calculator (`calculator`)**: When users ask for the total price of multiple items combined with employee discounts, the Agent automatically retrieves the prices from the document and uses the calculator to provide an exact figure.

### How to Run

#### 1. Install Dependencies
Ensure you have Python 3.10+ installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

#### 2. Configure API Key
Copy the environment variable template:
```bash
cp .env.example .env
```
Then, insert your Google Gemini API Key into the `.env` file:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```
*(Note: This project uses `gemini-3.5-flash` to bypass free tier constraints)*

#### 3. Start Chatting
Run the following command to start the interactive CLI chat:
```bash
python agent.py
```

### File Structure

* `agent.py`: The core program containing the LangGraph ReAct architecture, System Prompt, tool functions, and the CLI interaction loop.
* `sample_document.txt`: The Agent's "brain/knowledge base", containing all core details about Nova Dynamics.
* `requirements.txt`: Project dependencies.
* `.env.example`: Environment variable template.

---

<a id="中文"></a>
## 中文

### 项目简介 (Project Overview)

这是一个基于 **LangGraph** 和 **Google Gemini API** 构建的交互式问答 Agent。

**为了更好地演示该 Agent 的实用性和落地场景，我们特意为它设定了一个具体的背景与身份：**
它被设定为一家虚构的智能家居公司（**Nova Dynamics Inc.**）的官方 AI 客服助手，名为 **Nova**。

通过为其提供一份公司的“内部知识库”（即 `sample_document.txt`，包含公司简介、产品价格、退款政策等），该 Agent 可以模拟真实的客服场景，回答用户的产品咨询、计算订单折扣、并自然地与用户进行多轮对话。

### 核心特性 (Features)

1. **角色扮演与身份设定 (Roleplaying & Identity)**：
   * 具备完整的客服人格，会在对话中保持礼貌、专业的态度。
   * 绝不凭空捏造（Hallucination），对于未包含在公司文档中的问题（例如生鲜价格、本地电脑文件），会清晰地表示超出服务范围。
2. **长效记忆 (Conversation Memory)**：
   * 能够记住用户的名字和上下文。例如，你在第一句话告诉它你的名字，后续对话中它都会记得。
3. **按需调用工具 (Tool Use / Function Calling)**：
   * **文档检索工具 (`search_document`)**：当用户询问产品、退款政策或保修条款时，Agent 会自动扫描知识库提取信息。
   * **数学计算器 (`calculator`)**：当用户询问购买多件商品并叠加员工折扣的价格时，Agent 会自动提取文档中的价格，并调用计算器得出精确数值。

### 如何运行 (How to Run)

#### 1. 安装依赖
确保你已经安装了 Python 3.10+，然后安装依赖库：
```bash
pip install -r requirements.txt
```

#### 2. 配置 API Key
复制环境变量模板文件：
```bash
cp .env.example .env
```
然后在 `.env` 文件中填入你的 Google Gemini API Key：
```env
GOOGLE_API_KEY=你的真实_API_KEY
```
*(注意：本项目已将模型切换为 `gemini-3.5-flash` 以避免免费额度限制)*

#### 3. 启动对话
运行以下命令启动终端聊天界面：
```bash
python agent.py
```

### 文件结构 (File Structure)

* `agent.py`：核心程序，包含了 LangGraph 的 ReAct 架构、系统提示词（System Prompt）、工具函数以及终端交互循环。
* `sample_document.txt`：Agent 的“大脑知识库”，包含了 Nova Dynamics 公司的所有核心设定。
* `requirements.txt`：项目依赖。
* `.env.example`：环境变量模板。
