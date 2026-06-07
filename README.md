# VQA + RAG · LangGraph 多 Agent 智能问答系统

基于通义千问视觉模型、RAG（检索增强生成）和 **LangGraph 多 Agent 架构**的统一问答系统。
由 **Supervisor Agent** 协调 5 个专家 Agent 协同工作，支持视觉问答、知识库检索、自省纠错重试，
集成 **LangSmith** 全链路可观测。

---

## 功能

| 能力 | 说明 |
|------|------|
| 🤖 **多 Agent 编排** | Supervisor Agent 调度 5 个 Specialist Agent 协同推理 |
| 🖼️ **视觉问答** | 上传图片，Vision Analyst Agent 调用 Qwen-VL 分析 |
| 📚 **RAG 知识库** | Retriever Agent 搜索 FAISS 向量索引增强回答 |
| 🧠 **自省纠错** | Critic Agent 评估答案质量，低置信度 Supervisor 指派 Reasoner 重试 |
| 📊 **LangSmith 可观测** | 每个 Agent 独立 span，全链路 tracing |
| 🌙 **深色模式** | 跟随系统自动切换，支持手动切换 |

---

## 多 Agent 架构

### Agent 团队

```
┌──────────────────────────────────────────────────┐
│                  Supervisor Agent                 │
│  🧑‍💼 决策者：路由任务、追踪进度、决定何时完成      │
└──┬──────────┬──────────┬──────────┬──────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Classifier│ │Retriever│ │ Vision │ │Reasoner│
│ 🔍 分析师│ │ 📚 检索师│ │ 👁️ 视觉师│ │ 🧠 推理师│
│ 判定问题  │ │ 搜索知识  │ │ 分析图片 │ │ 生成回答 │
│ 类型     │ │ 库      │ │ 内容    │ │        │
└────────┘ └────────┘ └────────┘ └────────┘
                                        │
                                        ▼
                                  ┌────────┐
                                  │  Critic │
                                  │ ✅ 评审师│
                                  │ 评估回答│
                                  │ 质量    │
                                  └────────┘
```

### 工作流程

```
用户输入（问题 ± 图片）
       │
       ▼
┌────────────────────┐
│ Supervisor Agent    │ → 指派 Classifier
│ 第1轮：分类问题     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Classifier Agent   │ → 返回问题类型
│ 输出：general       │    supervisor
└────────┬───────────┘
         │
┌────────────────────┐
│ Supervisor Agent    │ → 指派 Retriever
│ 第2轮：检索         │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Retriever Agent    │ → FAISS 检索 top-5
│ 输出：RAG 上下文     │    返回 supervisor
└────────┬───────────┘
         │
┌────────────────────┐
│ Supervisor Agent    │ → 有图？→ Vision
│ 第3轮：分析/推理     │ → 否则 → Reasoner
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Reasoner Agent     │ → 整合所有上下文
│ 输出：回答           │    生成回答
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Supervisor Agent    │ → 指派 Critic
│ 第4轮：评估         │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Critic Agent       │ → 评估质量
│ 输出：置信度/问题    │    返回 supervisor
└────────┬───────────┘
         │
    ┌────┴────┐
    ▼         ▼
 高置信度   低置信度
    │         │
 ┌──┴──┐  ┌──┴──┐
 │ END │  │Retry│──→ Reasoner Agent (修正重试)
 └─────┘  └─────┘
```

### 每个 Agent 的职责

| Agent | 角色 | 工具/模型 | 输出 |
|-------|------|-----------|------|
| 🧑‍💼 **Supervisor** | 协调主管 | Python 路由逻辑 | 下一个 Agent 或 FINISH |
| 🔍 **Classifier** | 问题分类师 | Qwen-Plus (LLM) | question_type |
| 📚 **Retriever** | 知识检索师 | FAISS 向量搜索 | RAG 上下文 |
| 👁️ **Vision Analyst** | 视觉分析师 | Qwen-VL (VLM) | 图片描述 |
| 🧠 **Reasoner** | 推理回答师 | Qwen-VL / Qwen-Plus | 最终回答 |
| ✅ **Critic** | 质量评审师 | Qwen-Plus (LLM) | 置信度 / 问题 |

---

## 前端特性

- **多 Agent 流水线可视化** — 实时展示哪个 Agent 正在工作 🧑‍💼→🔍→📚→👁️→🧠→✅
- **Agent 协作链路** — 回答后展示完整 Agent 调用链
- **答案元信息** — 置信度、问题类型、修正次数
- **深色模式** — 跟随系统，支持手动切换
- **知识库管理** — 在线增删文档
- **响应式** — 桌面和移动端自适应

---

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

# 方式二：Uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000
```

访问 http://localhost:5000

### 可选：启用 LangSmith 全链路可观测

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
```

重启服务后访问 https://smith.langchain.com 选择 `vqa-system` 项目，每个 Agent 调用独立 span：

```
run_multi_agent_vqa
 ├─ supervisor_agent    [chain]
 ├─ classifier_agent    [chain]  → question_type
 ├─ supervisor_agent    [chain]
 ├─ retriever_agent     [chain]  → FAISS results
 ├─ supervisor_agent    [chain]
 ├─ reasoner_agent      [llm]   → answer + tokens
 ├─ supervisor_agent    [chain]
 ├─ critic_agent        [llm]   → confidence
 ├─ supervisor_agent    [chain]
 └─ format_output       [chain]  → final
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/ask` | 统一问答（表单，支持图片上传） |
| GET | `/api/graph/visualize` | 多 Agent 图结构（JSON） |
| POST | `/api/knowledge/upload` | 上传文档到知识库 |
| GET | `/api/knowledge/docs` | 文档列表 |
| POST | `/api/knowledge/remove` | 删除文档 |
| GET | `/api/knowledge/stats` | 知识库统计 |

---

## 项目结构

```
vqa-system/
├── app.py                 # FastAPI 后端（统一问答 + 知识库 API）
├── multi_agent.py         # 多 Agent 系统（Supervisor + 5 个 Specialist Agent）
├── vqa_graph.py           # 原单 Agent 工作流版本（备份）
├── app_flask.py           # 原 Flask 版本（备份）
├── rag_manager.py         # RAG 引擎（文档解析 + FAISS 索引 + 检索）
├── embeddings.py          # DashScope Embedding 封装
├── templates/
│   └── index.html         # 多 Agent 流水线可视化前端
├── static/
│   └── uploads/           # 上传的图片
├── data/
│   ├── docs/              # 上传的文档
│   └── index/             # FAISS 向量索引
├── requirements.txt       # Python 依赖
└── README.md
```

---

## 版本历史

- **v3.0** — 多 Agent 架构：Supervisor + 5 个 Specialist Agent
- **v2.1** — 前端流水线可视化 + LangSmith 全链路 tracing
- **v2.0** — FastAPI + LangGraph 单 Agent 工作流重构
- **v1.0** — Flask 版本，基础 VQA + RAG

---

## 许可证

MIT
