"""
LangGraph VQA StateGraph — Visual Question Answering + RAG + Self-Critique

Constructs a directed graph with the following flow:

    __start__ → classify_question → rag_retrieve → build_prompt
        → call_model → self_critic ──[low conf]──→ call_model (retry)
                                   └─[high/retry exhausted]─→ format_answer → END
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langsmith import traceable

import dashscope
from dashscope import Generation, MultiModalConversation

from rag_manager import RagManager


# LangGraph 内置的 LangSmith 回调已在环境变量中配置
# @traceable 装饰器为每个节点添加独立 span

# ── DashScope 配置 ──────────────────────────────────────────────
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY",
    "sk-71719159a0784e08aa71e66ae09a5662",
)
VISION_MODEL = "qwen-vl-plus"
LLM_MODEL = "qwen-plus"

# ── RAG 单例 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag = RagManager(api_key=DASHSCOPE_API_KEY, base_dir=BASE_DIR)


# ==================================================================
#  State
# ==================================================================
class VQAState(TypedDict):
    question: str
    image_path: Optional[str]
    image_url: Optional[str]
    image_present: bool

    # RAG
    rag_context: str
    rag_results: List[Dict[str, Any]]
    has_rag_context: bool

    # Question classification
    question_type: str  # "direct_visual" | "logical" | "ocr_heavy" | "general"

    # Prompt building
    system_prompt: str

    # Answer
    answer: str
    confidence: float
    critique: str

    # Control
    retry_count: int
    max_retries: int
    last_error: str


# ==================================================================
#  Node: classify_question
# ==================================================================
@traceable(name="classify_question", run_type="chain")
def _classify_question(state: VQAState) -> dict:
    """Use LLM to classify the question into one of 4 types."""
    q = state["question"]
    has_img = state["image_present"]

    prompt = (
        "你是一个问答分类器。判断以下问题的类型，只返回一个词。\n\n"
        f"问题：{q}\n"
        f"是否提供了图片：{'是' if has_img else '否'}\n\n"
        "类型选项：\n"
        "- direct_visual：需要直接看图才能回答（颜色、形状、人物识别、场景描述等）\n"
        "- logical：需要多步推理、逻辑判断、对比分析\n"
        "- ocr_heavy：图片中的文字识别（截图、文档照片、路牌等）\n"
        "- general：纯文本知识问答，不需要看图也能回答\n\n"
        "只返回类型名称，不要其他内容。"
    )

    try:
        resp = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
        )
        if resp.status_code == 200:
            txt = resp.output.choices[0].message.content or ""
            txt = txt.strip().lower()
            for t in ("direct_visual", "logical", "ocr_heavy", "general"):
                if t in txt:
                    return {"question_type": t}
    except Exception:
        pass
    return {"question_type": "general"}


# ==================================================================
#  Node: rag_retrieve
# ==================================================================
@traceable(name="rag_retrieve", run_type="chain")
def _rag_retrieve(state: VQAState) -> dict:
    """Search the FAISS knowledge base."""
    results = rag.search(state["question"], top_k=5)
    if not results:
        return {"rag_context": "", "rag_results": [], "has_rag_context": False}

    parts = []
    for i, r in enumerate(results, 1):
        fn = r["metadata"].get("filename", "未知")
        parts.append(f"[{i}] 来自文档「{fn}」:\n{r['content']}")
    context = "\n\n---\n\n".join(parts)

    return {
        "rag_context": context,
        "rag_results": results,
        "has_rag_context": True,
    }


# ==================================================================
#  Node: build_prompt
# ==================================================================
SYSTEM_BASE = "你是一个智能助手，擅长分析图片和文档。"


@traceable(name="build_prompt", run_type="chain")
def _build_prompt(state: VQAState) -> dict:
    """Assemble the system prompt using RAG context."""
    prompt = SYSTEM_BASE
    if state["has_rag_context"]:
        prompt += (
            "\n请基于以下参考资料回答用户问题。引用时请标注来源编号。"
            "如果参考资料中没有相关信息，请根据你的知识回答，并说明参考资料中未提及。\n\n"
            f"## 参考资料\n\n{state['rag_context']}"
        )

    # 如果是 retry （自我修正模式），加修正指令
    rt = state["retry_count"]
    if rt > 0:
        prompt += (
            f"\n\n## 修正说明\n"
            f"这是第 {rt} 次修正尝试。之前的答案存在以下问题：\n"
            f"{state['critique']}\n"
            "请修正上述问题，提供更准确的回答。"
        )

    return {"system_prompt": prompt}


# ==================================================================
#  Node: call_model
# ==================================================================
@traceable(name="call_model", run_type="llm")
def _call_model(state: VQAState) -> dict:
    """Call Qwen-VL (with image) or Qwen-Plus (text-only)."""
    question = state["question"]
    sp = state["system_prompt"]

    try:
        if state["image_present"] and state["image_path"]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": f"{sp}\n\n问题：{question}"},
                        {"image": f"file://{state['image_path']}"},
                    ],
                }
            ]
            resp = MultiModalConversation.call(
                api_key=DASHSCOPE_API_KEY, model=VISION_MODEL, messages=messages
            )
        else:
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": f"问题：{question}"},
            ]
            resp = Generation.call(
                api_key=DASHSCOPE_API_KEY,
                model=LLM_MODEL,
                messages=messages,
                result_format="message",
            )

        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.code}: {resp.message}")

        # Parse content
        if hasattr(resp.output.choices[0].message, "content"):
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                answer = content[0].get("text", str(content))
            else:
                answer = content
        else:
            answer = resp.output.text

        return {"answer": answer, "last_error": ""}

    except Exception as e:
        return {"answer": f"[系统错误] {e}", "last_error": str(e)}


# ==================================================================
#  Node: self_critic
# ==================================================================
CRITIQUE_PROMPT = """你是答案质量审查员。请严格评审以下 Q&A：

## 问题
{question}

## 问题类型
{question_type}

## 参考资料（知识库检索结果，不一定与图片相关）
{context}

## 答案
{answer}

请从以下维度评分（0.0 ~ 1.0）：

1. **内部一致性** — 答案本身是否逻辑自洽，没有自相矛盾
2. **回答完整度** — 是否完整回答了问题
3. **相关性** — 是否切题

评分规则：
- **general 类型（纯文本问答）** → 有参考资料时严格对照事实准确性
- **logical 类型** → 重点评估推理链条是否完整自洽
- **ocr_heavy 类型** → 重点评估文字提取的准确性

最后综合给出一个 overall_confidence 分数。

以 JSON 格式返回，不要其他内容：
{{
    "overall_confidence": 0.xx,
    "issues": "存在的具体问题（如无问题写'无'）",
    "needs_revision": true/false
}}"""


@traceable(name="self_critic", run_type="llm")
def _self_critic(state: VQAState) -> dict:
    """Run self-critique on the answer. Return updated confidence + critique."""
    if not state["answer"] or state["answer"].startswith("[系统错误]"):
        return {"confidence": 0.0, "critique": "系统级错误，无需自评", "retry_count": state["retry_count"] + 1}

    qtype = state.get("question_type", "general")

    # direct_visual 类型：自评器看不到图片，无法验证视觉细节准确性
    # → 跳过 LLM 自评，直接给高置信度
    if qtype == "direct_visual":
        return {
            "confidence": 0.9,
            "critique": "视觉问答，需人工验证图片细节",
            "retry_count": state["retry_count"] + 1,
        }

    context_snippet = state["rag_context"][:600] if state["rag_context"] else "无"
    prompt = CRITIQUE_PROMPT.format(
        question=state["question"],
        question_type=qtype,
        context=context_snippet,
        answer=state["answer"],
    )

    try:
        resp = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
        )
        if resp.status_code == 200:
            txt = resp.output.choices[0].message.content or "{}"
            # Extract JSON from possible markdown fence
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                data = json.loads(m.group())
            else:
                data = json.loads(txt)

            confidence = float(data.get("overall_confidence", 0.5))
            issues = data.get("issues", "")
            needs_revision = data.get("needs_revision", confidence < 0.5)

            # Clamp
            confidence = max(0.0, min(1.0, confidence))

            return {
                "confidence": confidence,
                "critique": issues,
                "retry_count": state["retry_count"] + 1,
            }
    except Exception:
        pass

    # Fallback: moderate confidence, no revision
    return {"confidence": 0.5, "critique": "自评失败，采用默认评分", "retry_count": state["retry_count"] + 1}


# ==================================================================
#  Node: format_answer
# ==================================================================
@traceable(name="format_answer", run_type="chain")
def _format_answer(state: VQAState) -> dict:
    """Final formatting / post-processing of the answer."""
    answer = state["answer"]
    if not answer:
        answer = "抱歉，我没能生成有效答案。请换个方式再问一次。"
    return {"answer": answer.strip()}


# ==================================================================
#  Conditional edge: should_retry
# ==================================================================
def _should_retry(state: VQAState) -> Literal["call_model", "format_answer"]:
    """Decide whether to retry or finish."""
    if (
        state["confidence"] < 0.5
        and state["retry_count"] <= state["max_retries"]
        and not state["answer"].startswith("[系统错误]")
    ):
        return "call_model"
    return "format_answer"


# ==================================================================
#  Build & compile graph
# ==================================================================
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# LangGraph 内置 LangSmith 回调：设置 LANGCHAIN_TRACING_V2=true 后自动生效
_LC_KEY = os.environ.get("LANGSMITH_API_KEY", "")
_LANGCHAIN_CALLBACKS = bool(_LC_KEY)
if _LC_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "vqa-system")


def _create_vqa_graph() -> Any:
    """Build and compile the VQA StateGraph."""
    builder = StateGraph(VQAState)

    builder.add_node("classify_question", _classify_question)
    builder.add_node("rag_retrieve", _rag_retrieve)
    builder.add_node("build_prompt", _build_prompt)
    builder.add_node("call_model", _call_model)
    builder.add_node("self_critic", _self_critic)
    builder.add_node("format_answer", _format_answer)

    builder.set_entry_point("classify_question")

    builder.add_edge("classify_question", "rag_retrieve")
    builder.add_edge("rag_retrieve", "build_prompt")
    builder.add_edge("build_prompt", "call_model")
    builder.add_edge("call_model", "self_critic")

    builder.add_conditional_edges(
        "self_critic",
        _should_retry,
        {"call_model": "call_model", "format_answer": "format_answer"},
    )
    builder.add_edge("format_answer", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


vqa_graph = _create_vqa_graph()


# ==================================================================
#  Public convenience: run_vqa
# ==================================================================
@traceable(name="run_vqa", run_type="chain")
def run_vqa(
    question: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """Run the VQA graph and return structured results."""
    initial: VQAState = {
        "question": question,
        "image_path": image_path,
        "image_url": image_url,
        "image_present": bool(image_path),
        "rag_context": "",
        "rag_results": [],
        "has_rag_context": False,
        "question_type": "general",
        "system_prompt": "",
        "answer": "",
        "confidence": 0.5,
        "critique": "",
        "retry_count": 0,
        "max_retries": 1,
        "last_error": "",
    }

    result = vqa_graph.invoke(initial, {"configurable": {"thread_id": thread_id}})

    return {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "critique": result.get("critique", ""),
        "has_rag_context": result.get("has_rag_context", False),
        "rag_doc_count": len(result.get("rag_results", [])),
        "question_type": result.get("question_type", "general"),
        "retry_count": result.get("retry_count", 0),
    }
