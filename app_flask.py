"""视觉问答 + RAG 知识库 - 统一后端"""

import os
import uuid
from flask import Flask, request, render_template, jsonify, url_for

import dashscope
from dashscope import MultiModalConversation, Generation

from rag_manager import RagManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['DOCS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'docs')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', 'sk-71719159a0784e08aa71e66ae09a5662')
VISION_MODEL = 'qwen-vl-plus'
LLM_MODEL = 'qwen-plus'

ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
ALLOWED_DOC_EXT = {'pdf', 'docx', 'xlsx', 'xls', 'txt', 'md', 'csv', 'log'}

rag = RagManager(api_key=DASHSCOPE_API_KEY, base_dir=os.path.dirname(os.path.abspath(__file__)))


def _allowed(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


# ============ 页面 ============

@app.route('/')
def index():
    return render_template('index.html')


# ============ 统一问答接口 (VQA + RAG 合并) ============

@app.route('/ask', methods=['POST'])
def ask():
    """
    统一问答：图片可选 + 知识库自动检索
    - question (必填)
    - image (可选)
    - 自动从知识库检索相关内容
    """
    question = request.form.get('question', '').strip()
    image_file = request.files.get('image')
    search_only = request.form.get('search_only', '') == 'true'

    if not question:
        return jsonify({'error': '请输入问题'}), 400

    # 可选：仅检索不回答（前端用于展示参考片段）
    if search_only:
        rag_results = rag.search(question, top_k=5)
        return jsonify({
            'success': True,
            'rag_results': rag_results,
            'rag_doc_count': len(rag_results),
        })

    # 处理图片
    saved_image_path = None
    image_url = None
    if image_file and image_file.filename:
        if not _allowed(image_file.filename, ALLOWED_IMAGE_EXT):
            return jsonify({'error': f'不支持的图片格式'}), 400
        ext = image_file.filename.rsplit('.', 1)[1].lower()
        img_name = f"{uuid.uuid4().hex}.{ext}"
        saved_image_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        image_file.save(saved_image_path)
        image_url = url_for('static', filename=f'uploads/{img_name}')

    try:
        # 1. RAG 检索
        rag_results = rag.search(question, top_k=5)
        context = ""
        if rag_results:
            context_parts = []
            for i, r in enumerate(rag_results, 1):
                context_parts.append(f"[{i}] 来自文档「{r['metadata'].get('filename', '未知')}」:\n{r['content']}")
            context = "\n\n---\n\n".join(context_parts)

        # 2. 构建 system prompt
        system_prompt = "你是一个智能助手，擅长分析图片和文档。"
        if context:
            system_prompt += (
                "\n请基于以下参考资料回答用户问题。引用时请标注来源编号。"
                "如果参考资料中没有相关信息，请根据你的知识回答，并说明参考资料中未提及。\n\n"
                f"## 参考资料\n\n{context}"
            )

        # 3. 调用 API
        if saved_image_path:
            messages = [{
                "role": "user", "content": [
                    {"text": f"{system_prompt}\n\n问题：{question}"},
                    {"image": f"file://{saved_image_path}"},
                ]
            }]
            response = MultiModalConversation.call(
                api_key=DASHSCOPE_API_KEY,
                model=VISION_MODEL,
                messages=messages
            )
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"问题：{question}"},
            ]
            response = Generation.call(
                api_key=DASHSCOPE_API_KEY,
                model=LLM_MODEL,
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
                'image_url': image_url,
            })
        else:
            return jsonify({'error': f'API 调用失败: {response.code} - {response.message}'}), 500

    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


# ============ 知识库管理 API ============

@app.route('/api/knowledge/upload', methods=['POST'])
def knowledge_upload():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    if not _allowed(file.filename, ALLOWED_DOC_EXT):
        return jsonify({'error': f'不支持的格式，支持: {", ".join(sorted(ALLOWED_DOC_EXT))}'}), 400

    safe_name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = os.path.join(app.config['DOCS_FOLDER'], unique_name)
    file.save(filepath)

    try:
        result = rag.add_document(filepath)
        if not result['success'] and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify(result)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'文档处理失败: {str(e)}'}), 500


@app.route('/api/knowledge/docs', methods=['GET'])
def knowledge_docs():
    return jsonify(rag.get_all_docs())


@app.route('/api/knowledge/remove', methods=['POST'])
def knowledge_remove():
    data = request.get_json(silent=True) or {}
    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({'error': '缺少 doc_id'}), 400
    return jsonify(rag.remove_document(doc_id))


@app.route('/api/knowledge/stats', methods=['GET'])
def knowledge_stats():
    return jsonify(rag.get_stats())


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
