# 视觉问答 + RAG 知识库系统

基于通义千问视觉模型、RAG（检索增强生成）和 **LangGraph** 状态图编排的统一问答 Web 应用。
支持**可选**上传图片进行视觉分析，同时自动检索文档知识库增强回答，并具备**自省纠错**能力。

## 功能

- 🖼️ **视觉问答** — 可选上传图片，AI 分析和回答（Qwen-VL）
- 📚 **RAG 知识库** — 上传 PDF/DOCX/TXT 等文档，构建 FAISS 向量索引
- 🔍 **检索增强** — 提问时自动检索知识库中相关内容，标注来源
- ⚡ **快捷提问** — 一键提问模板
- 💡 **知识库管理** — 在线新增/删除文档，实时更新
- 🔄 **LangGraph 状态图编排** — 多节点推理流程，条件路由，自动重试
- 🧠 **自省纠错** — LLM 自评答案质量，低置信度时自动修正
- 📊 **LangSmith 可观测** — 可选集成，全链路 tracing

## 技术栈

| 组件 | 技术 |
|------|------|
| **后端框架** | FastAPI + Uvicorn |
| **推理引擎** | LangGraph StateGraph |
| **前端** | 原生 HTML/CSS/JavaScript |
| **视觉模型** | Qwen-VL（通义千问视觉语言模型） |
| **文本模型** | Qwen-Plus + text-embedding-v3 |
| **向量检索** | FAISS（余弦相似度） |
| **文档解析** | PyMuPDF / python-docx / openpyxl |
| **可观测** | LangSmith（可选，条件启用） |

## 架构

### LangGraph 推理流程图

```
用户输入（问题 ± 图片）
        │
        ▼
┌───────────────────────┐
│  classify_question     │  ← 判断问题类型
│  (direct_visual /      │     (direct_visual / general /
│   logical / ocr_heavy) │      logical / ocr_heavy)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  rag_retrieve          │  ← FAISS 检索知识库
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  build_prompt          │  ← 拼接 system prompt + RAG 上下文
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  call_model            │  ← 调 DashScope API
│  (有图→VLM / 无图→LLM)│     有图走 Qwen-VL，无图走 Qwen-Plus
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  self_critic           │  ← LLM 自评答案质量
│                        │
│  ┌─ 高置信度 ──────┐  │
│  │  → format_answer │  │
│  │  → END           │  │
│  └─────────────────┘  │
│  ┌─ 低置信度 ──────┐  │
│  │  → call_model    │  │  ← 重试，带修正指令
│  │  (重试)           │  │
│  └─────────────────┘  │
└───────────────────────┘
```

### 核心节点

| 节点 | 职责 |
|------|------|
| `classify_question` | LLM 分类问题类型，路由到不同处理路径 |
| `rag_retrieve` | FAISS 检索知识库，返回 top-5 相关片段 |
| `build_prompt` | 组装 system prompt，支持 retry 时带修正指令 |
| `call_model` | 调通义千问 API（VLM 或 LLM） |
| `self_critic` | LLM 自评答案质量；visual 类型跳过（模型无法看图） |
| `format_answer` | 最终格式化输出 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 3. 启动服务

```bash
# 方式一：直接运行
python app.py

# 方式二：Uvicorn 显式启动
uvicorn app:app --host 0.0.0.0 --port 5000
```

访问 http://localhost:5000

### 可选：启用 LangSmith 追踪

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
# 重启服务即可自动开启 tracing
```

## 使用指南

### 视觉问答
1. 可选地拖拽或点击上传一张图片
2. 在输入框输入问题
3. 点击「提问」按钮

### RAG 知识库
1. 展开「知识库管理」面板
2. 点击上传文档（PDF / DOCX / XLSX / TXT / MD）
3. 文档自动切片、嵌入、存入 FAISS 索引
4. 提问时自动检索相关内容并标注来源

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/ask` | 统一问答（表单） |
| GET | `/api/graph/visualize` | LangGraph 图结构（JSON） |
| POST | `/api/knowledge/upload` | 上传文档 |
| GET | `/api/knowledge/docs` | 文档列表 |
| POST | `/api/knowledge/remove` | 删除文档 |
| GET | `/api/knowledge/stats` | 知识库统计 |

## 项目结构

```
vqa-system/
├── app.py                 # FastAPI 后端（统一问答 + 知识库 API）
├── app_flask.py           # 原 Flask 版本（备份）
├── vqa_graph.py           # LangGraph 状态图定义（推理引擎核心）
├── rag_manager.py         # RAG 引擎（文档解析 + FAISS 索引 + 检索）
├── embeddings.py          # DashScope Embedding 封装
├── templates/
│   └── index.html         # 统一前端界面
├── static/
│   └── uploads/           # 上传的图片
├── data/
│   ├── docs/              # 上传的文档
│   └── index/             # FAISS 向量索引
├── requirements.txt       # Python 依赖
└── README.md
```

## 版本历史

- **v2.0** — FastAPI + LangGraph 重构，集成自省纠错与 LangSmith
- **v1.0** — Flask 版本，基础 VQA + RAG

## 许可证

MIT
