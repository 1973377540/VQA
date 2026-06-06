# 视觉问答 + RAG 知识库系统

基于通义千问视觉模型和 RAG（检索增强生成）的统一问答 Web 应用。
支持**可选**上传图片进行视觉分析，同时自动检索文档知识库增强回答。

## 功能

- 🖼️ **视觉问答** — 可选上传图片，AI 分析和回答
- 📚 **RAG 知识库** — 上传 PDF/DOCX/TXT 等文档，构建向量索引
- 🔍 **检索增强** — 提问时自动检索知识库中相关内容
- ⚡ **快捷提问** — 一键提问模板
- 💡 **知识库管理** — 在线新增/删除文档，实时更新

## 技术栈

| 组件 | 技术 |
|------|------|
| **后端** | Python Flask + DashScope SDK |
| **前端** | 原生 HTML/CSS/JavaScript |
| **视觉模型** | Qwen-VL（通义千问视觉语言模型） |
| **文本模型** | Qwen-Plus + text-embedding-v3 |
| **向量检索** | FAISS（余弦相似度） |
| **文档解析** | PyMuPDF / python-docx / openpyxl |

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
python app.py
```

访问 http://localhost:5000

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

### 阿里云端口放行

需要放行 **5000** 端口：

1. 进入 ECS 控制台 → 安全组
2. 添加入方向规则：
   - 端口范围：5000/5000
   - 授权对象：0.0.0.0/0（或指定 IP）
   - 协议：TCP

## 项目结构

```
vqa-system/
├── app.py                 # Flask 后端（统一问答 + 知识库 API）
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

## 许可证

MIT
