"""
FastAPI + LangGraph VQA System
================================
Replaces the original Flask app with FastAPI, integrating a LangGraph state-graph
for the VQA + RAG pipeline. LangSmith tracing is enabled when LANGSMITH_API_KEY
is set in the environment.

Endpoints:
  GET  /                        → HTML page
  POST /ask                     → VQA (question ± image, auto RAG + self-critique)
  POST /ask_sync                → VQA (synchronous, returns once ready)
  GET  /api/graph/visualize     → Graph structure (JSON)
  POST /api/knowledge/upload    → Upload document into RAG
  GET  /api/knowledge/docs      → List documents
  POST /api/knowledge/remove    → Remove document
  GET  /api/knowledge/stats     → Knowledge-base statistics
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import Request

# Use Jinja2 directly to avoid Starlette 1.2.1 ↔ Jinja2 3.1.x cache-version incompatibility
from jinja2 import Environment, FileSystemLoader, select_autoescape

from multi_agent import rag, run_multi_agent_vqa as run_vqa

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
DOCS_DIR = BASE_DIR / "data" / "docs"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_DOC_EXT = {"pdf", "docx", "xlsx", "xls", "txt", "md", "csv", "log"}

# ── App lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="VQA + RAG System", version="2.0.0", lifespan=lifespan)

# ── Static files & template engine ───────────────────────────────
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def _render_template(name: str, request: Request, **extra) -> str:
    """Render a Jinja2 template with request context."""
    tpl = _jinja_env.get_template(name)
    return tpl.render(request=request, **extra)


# ══════════════════════════════════════════════════════════════════
#  Pydantic models
# ══════════════════════════════════════════════════════════════════
class AskResponse(BaseModel):
    success: bool = True
    answer: str
    has_rag_context: bool = False
    rag_doc_count: int = 0
    image_url: Optional[str] = None
    confidence: float = 0.0
    critique: str = ""
    question_type: str = "general"
    retry_count: int = 0
    agent_history: list = []


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


class DocResponse(BaseModel):
    success: bool
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None
    remaining_docs: Optional[int] = None


# ══════════════════════════════════════════════════════════════════
#  Helper
# ══════════════════════════════════════════════════════════════════
def _allowed(filename: str, allowed_set: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# ══════════════════════════════════════════════════════════════════
#  Routes — Page
# ══════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    html = _render_template("index.html", request)
    return HTMLResponse(html)


# ══════════════════════════════════════════════════════════════════
#  Routes — VQA
# ══════════════════════════════════════════════════════════════════
@app.post("/ask", response_model=AskResponse)
async def ask(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
    search_only: Optional[str] = Form(""),
):
    """
    Unified Q&A endpoint:
    - question (required)
    - image (optional)
    - Uses LangGraph state-graph with RAG, self-critique & optional retry
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入问题")

    # ── search_only mode (frontend may want just RAG snippets) ──
    if search_only == "true":
        rag_results = rag.search(question, top_k=5)
        return JSONResponse({
            "success": True,
            "rag_results": rag_results,
            "rag_doc_count": len(rag_results),
        })

    # ── Save uploaded image ─────────────────────────────────────
    saved_image_path: Optional[str] = None
    image_url: Optional[str] = None

    if image and image.filename:
        if not _allowed(image.filename, ALLOWED_IMAGE_EXT):
            raise HTTPException(status_code=400, detail="不支持的图片格式")

        ext = image.filename.rsplit(".", 1)[1].lower()
        img_name = f"{uuid.uuid4().hex}.{ext}"
        dest = UPLOAD_DIR / img_name

        content = await image.read()
        dest.write_bytes(content)

        saved_image_path = str(dest)
        image_url = f"/static/uploads/{img_name}"

    # ── Run LangGraph ───────────────────────────────────────────
    thread_id = f"ask_{uuid.uuid4().hex[:12]}"

    result = run_vqa(
        question=question,
        image_path=saved_image_path,
        image_url=image_url,
        thread_id=thread_id,
    )

    return AskResponse(
        answer=result["answer"],
        has_rag_context=result["has_rag_context"],
        rag_doc_count=result["rag_doc_count"],
        image_url=image_url,
        confidence=result["confidence"],
        critique=result["critique"],
        question_type=result["question_type"],
        retry_count=result["retry_count"],
        agent_history=result.get("agent_history", []),
    )


# ══════════════════════════════════════════════════════════════════
#  Routes — Graph introspection
# ══════════════════════════════════════════════════════════════════
@app.get("/api/graph/visualize")
async def graph_visualize():
    """Return the graph structure as JSON for debugging."""
    from multi_agent import multi_agent_graph

    graph_def = multi_agent_graph.get_graph()
    return JSONResponse({
        "nodes": list(graph_def.nodes.keys()),
        "edges": [
            {"source": e[0], "target": e[1]}
            for e in graph_def.edges
        ],
    })


# ══════════════════════════════════════════════════════════════════
#  Routes — Knowledge base management
# ══════════════════════════════════════════════════════════════════
@app.post("/api/knowledge/upload")
async def knowledge_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="没有选择文件")
    if not _allowed(file.filename, ALLOWED_DOC_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的格式，支持: {', '.join(sorted(ALLOWED_DOC_EXT))}",
        )

    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = DOCS_DIR / unique_name

    content = await file.read()
    filepath.write_bytes(content)

    try:
        result = rag.add_document(str(filepath))
        if not result.get("success") and filepath.exists():
            filepath.unlink()
        return result
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {e}")


@app.get("/api/knowledge/docs")
async def knowledge_docs():
    return rag.get_all_docs()


@app.post("/api/knowledge/remove")
async def knowledge_remove(data: dict):
    doc_id = data.get("doc_id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="缺少 doc_id")
    return rag.remove_document(doc_id)


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    return rag.get_stats()


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
