"""视觉问答系统 (VQA) - Flask 后端"""

import os
import uuid
import json
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import dashscope

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# DashScope 配置
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', 'sk-71719159a0784e08aa71e66ae09a5662')
MODEL = 'qwen-vl-plus'  # 使用视觉模型

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """处理图片上传和问题"""
    # 检查是否有文件
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    question = request.form.get('question', '').strip()

    if not question:
        return jsonify({'error': '请输入问题'}), 400

    if file.filename == '':
        return jsonify({'error': '没有选择图片'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的图片格式，支持: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # 保存上传的图片
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 构建图片 URL（相对路径）
    image_url = url_for('static', filename=f'uploads/{filename}')

    # 调用 DashScope 多模态 API
    try:
        # 使用本地文件路径
        image_input = f"file://{filepath}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_input},
                    {"text": question}
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=MODEL,
            messages=messages
        )

        if response.status_code == 200:
            answer = response.output.choices[0].message.content[0]["text"]
            return jsonify({
                'success': True,
                'answer': answer,
                'image_url': image_url
            })
        else:
            return jsonify({
                'error': f'API 调用失败: {response.code} - {response.message}'
            }), 500

    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/history', methods=['GET'])
def history():
    """获取历史问答记录（简化版，从 uploads 目录读取）"""
    uploads_dir = app.config['UPLOAD_FOLDER']
    files = sorted(os.listdir(uploads_dir), reverse=True)[:10]
    return jsonify({
        'images': [url_for('static', filename=f'uploads/{f}') for f in files if allowed_file(f)]
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
