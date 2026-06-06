# 视觉问答系统 (VQA)

基于通义千问视觉模型 (Qwen-VL) 的图片问答 Web 应用。

## 功能

- 📷 拖拽或点击上传图片
- 💬 输入自然语言问题
- 🤖 AI 自动分析图片并回答
- ⚡ 快捷问题一键提问

## 技术栈

- **后端**: Python Flask + DashScope SDK
- **前端**: 原生 HTML/CSS/JavaScript
- **模型**: Qwen-VL (通义千问视觉语言模型)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

设置环境变量（可选，代码中已内置默认 Key）：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### 3. 启动服务

```bash
python app.py
```

访问 http://localhost:5000

## 阿里云端口放行

如果使用阿里云 ECS，需要在安全组中放行 **5000** 端口：

1. 进入 ECS 控制台 → 安全组
2. 添加入方向规则：
   - 端口范围：5000/5000
   - 授权对象：0.0.0.0/0（或指定 IP）
   - 协议：TCP

## 项目结构

```
vqa-system/
├── app.py                 # Flask 后端
├── templates/
│   └── index.html         # 前端页面
├── static/
│   └── uploads/           # 上传图片存储
├── requirements.txt       # Python 依赖
└── README.md
```

## 截图

上传任意图片，输入问题即可获得 AI 解读。
