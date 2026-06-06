"""视觉问答系统 (VQA) + RAG 知识库 - Flask 后端"""

import os
import uuid
import json
import tempfile
import shutil
from flask import Flask, request, render_template, jsonify, url_for, send_file
from werkzeug.utils import secure_filename

import dashscope
from dashscope import MultiModalConversation, Generation

from rag_manager import RagManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['DOCS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'docs')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

# DashScope 配置
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', 'sk-717044f3c7914b50b16a85c21f405662')

VISION_MODEL = 'qwen-vl-plus'
RAG_LLM_MODEL = 'qwen-plus'

ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
ALLOWED_DOC_EXT = {'pdf', 'docx', 'xlsx', 'xls', 'txt', 'md', 'csv', 'log'}

# 初始化 RAG Manager
rag = RagManager(api_key=DASHSCOPE_API_KEY, base_dir=os.path.dirname(os.path.abspath(__file__)))


def _allowed_ext(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


# ============ 页面 ============

@app.route('/')
def index():
    return render_template('index.html')


# ============ 视觉问答 (VQA) ============

@app.route('/ask', methods=['POST'])
def ask():
    """图片 + 问题 → 视觉问答"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    question = request.form.get('question', '').strip()

    if not question:
        return jsonify({'error': '请输入问题'}), 400

    if file.filename == '':
        return jsonify({'error': '没有选择图片'}), 400

    if not _allowed_ext(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({'error': f'不支持的格式，支持: {", ".join(sorted(ALLOWED_IMAGE_EXT))}'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image_url = url_for('static', filename=f'uploads/{filename}')

    try:
        image_input = f"file://{filepath}"
        messages = [
            {"role": "user", "content": [
                {"image": image_input},
                {"text": question}
            ]}
        ]

        response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=VISION_MODEL,
            messages=messages
        )

        if response.status_code == 200:
            answer = response.output.choices[0].message.content[0]["text"]
            return jsonify({'success': True, 'answer': answer, 'image_url': image_url})
        else:
            return jsonify({'error': f'API 调用失败: {response.code} - {response.message}'}), 500

    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


# ============ RAG 知识库 ============

@app.route('/api/knowledge/upload', methods=['POST'])
def knowledge_upload():
    """上传文档到知识库"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if not _allowed_ext(file.filename, ALLOWED_DOC_EXT):
        return jsonify({'error': f'不支持的格式，支持: {", ".join(sorted(ALLOWED_DOC_EXT))}'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = os.path.join(app.config['DOCS_FOLDER'], unique_name)
    file.save(filepath)

    result = rag.add_document(filepath)
    if not result['success']:
        # 上传失败时删除文件
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


@app.route('/api/knowledge/docs', methods=['GET'])
def knowledge_docs():
    """获取知识库文档列表"""
    return jsonify(rag.get_all_docs())


@app.route('/api/knowledge/remove', methods=['POST'])
def knowledge_remove():
    """移除知识库中的文档"""
    data = request.get_json(silent=True) or {}
    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({'error': '缺少 doc_id'}), 400

    return jsonify(rag.remove_document(doc_id))


@app.route('/api/knowledge/stats', methods=['GET'])
def knowledge_stats():
    """获取知识库统计"""
    return jsonify(rag.get_stats())


@app.route('/api/knowledge/search', methods=['POST'])
def knowledge_search():
    """纯检索，用于调试"""
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    results = rag.search(query, top_k=top_k)
    return jsonify({'results': results})


@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    """RAG 问答：基于知识库回答问题（可附带图片）"""
    data = request.form
    question = data.get('question', '').strip()
    image_file = request.files.get('image')

    if not question:
        return jsonify({'error': '请输入问题'}), 400

    # 1. RAG 检索
    rag_results = rag.search(question, top_k=5)
    context = ""
    if rag_results:
        context_parts = []
        for i, r in enumerate(rag_results, 1):
            context_parts.append(f"[文档: {r['metadata'].get('filename', '未知')}]\n{r['content']}")
        context = "\n\n---\n\n".join(context_parts)

    try:
        # 2. 构建 prompt
        if context:
            system_prompt = (
                "你是一个智能助手。请基于以下参考资料回答问题。\n"
                "如果参考资料中有相关信息，请引用并回答；\n"
                "如果没有相关信息，请根据你的知识回答，并说明参考资料中未提及。\n\n"
                f"## 参考资料\n\n{context}\n\n## 用户问题"
            )
        else:
            system_prompt = "你是一个智能助手。请回答用户的问题。"

        # 3. 如果有图片，使用视觉模型；否则使用纯文本模型
        if image_file and image_file.filename:
            if not _allowed_ext(image_file.filename, ALLOWED_IMAGE_EXT):
                return jsonify({'error': f'不支持的图片格式'}), 400

            ext = image_file.filename.rsplit('.', 1)[1].lower()
            img_name = f"{uuid.uuid4().hex}.{ext}"
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            image_file.save(img_path)
            image_url = url_for('static', filename=f'uploads/{img_name}')

            messages = [
                {"role": "user", "content": [
                    {"text": f"{system_prompt}\n\n{question}"},
                    {"image": f"file://{img_path}"},
                ]}
            ]

            response = MultiModalConversation.call(
                api_key=DASHSCOPE_API_KEY,
                model=VISION_MODEL,
                messages=messages
            )
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            response = Generation.call(
                api_key=DASHSCOPE_API_KEY,
                model=RAG_LLM_MODEL,
                messages=messages,
                result_format='message'
            )

        if response.status_code == 200:
            if hasattr(response.output.choices[0].message, 'content'):
                content = response.output.choices[0].message.content
                if isinstance(content, list):
                    answer = content[0].get('text', str(content))
                else:
                    answer = content
            else:
                answer = response.output.text

            return jsonify({
                'success': True,
                'answer': answer,
                'has_rag_context': bool(context),
                'rag_doc_count': len(rag_results),
                'image_url': image_url if (image_file and image_file.filename) else None,
            })
        else:
            return jsonify({'error': f'API 调用失败: {response.code} - {response.message}'}), 500

    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
