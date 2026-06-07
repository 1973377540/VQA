"""
Multi-Agent VQA System — LangGraph Supervisor + Specialist Agents
=================================================================
Architecture:

    ┌──────────────────────────────────────────────────┐
    │                  Supervisor Agent                 │
    │  (Routes tasks, tracks progress, decides when     │
    │   done. Falls back to Python logic if LLM call    │
    │   fails.)                                         │
    └──┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │Classifier│ │Retriever│ │ Vision │ │ Reasoner│
  │ Agent   │ │ Agent   │ │ Analyst│ │ Agent   │
  │(LLM)    │ │(FAISS)  │ │(QwenVL)│ │(Qwen+)  │
  └────────┘ └────────┘ └────────┘ └────────┘
                                        │
                                        ▼
                                  ┌────────┐
                                  │ Critic │
                                  │ Agent  │
                                  │(LLM)   │
                                  └────────┘

Each specialist has its own persona, tools, and state.
The Supervisor routes work and decides when to finish.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

import dashscope
from dashscope import Generation, MultiModalConversation
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from rag_manager import RagManager

# ── Configuration ──────────────────────────────────────────────
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
VISION_MODEL = "qwen-vl-plus"
LLM_MODEL = "qwen-plus"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rag = RagManager(api_key=DASHSCOPE_API_KEY, base_dir=BASE_DIR)

# ── LangSmith ─────────────────────────────────────────────────
_LANGKEY = os.environ.get("LANGSMITH_API_KEY", "")
if _LANGKEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "vqa-system")
    os.environ["LANGSMITH_API_KEY"] = _LANGKEY


# ==================================================================
#  Agent Identity Prompts
# ==================================================================
SUPERVISOR_PROMPT = """你是多 Agent 系统的协调主管 (Supervisor)。
你管理一个专家团队，根据当前任务状态分派工作给合适的 Agent。

你的团队成员：
- **classifier**: 分析问题类型（direct_visual / logical / ocr_heavy / general）
- **retriever**: 在知识库中检索相关信息（当需要文档参考时）
- **vision_analyst**: 分析图片内容（当有图片需要解读时）
- **reasoner**: 基于所有上下文生成最终回答
- **critic**: 评估回答质量，给出置信度评分

当前节点应返回一个 JSON 格式的下一步指令，包含：
{{"next": "<agent_name>", "reason": "<简短原因>"}}
或者当所有工作完成时：
{{"next": "FINISH", "reason": "<原因>", "answer": "<最终答案>"}}

请分析当前状态，决定下一步行动。"""

CLASSIFIER_PROMPT = """你是问题分类专家。分析用户的问题并确定类型。
返回 JSON: {{"question_type": "direct_visual|logical|ocr_heavy|general"}}"""

VISION_ANALYST_PROMPT = """你是视觉分析专家。你擅长详细描述图片内容。
使用 Qwen-VL 模型分析图片，返回详细的图片描述。"""

REASONER_PROMPT = """你是推理回答专家。你擅长整合各种信息生成准确、
全面、有条理的答案。基于所有上下文信息（RAG 检索结果、图片分析结果等）
来回答问题。如果信息不足，如实说明。"""

CRITIC_PROMPT = """你是质量评审专家。评估回答的质量并给出评分。
返回 JSON: {{"confidence": 0.0-1.0, "issues": "...", "needs_revision": true/false}}"""


# ==================================================================
#  State
# ==================================================================
class AgentState(TypedDict):
    """Shared state for the multi-agent VQA system."""

    # Input
    question: str
    image_path: Optional[str]
    image_url: Optional[str]
    image_present: bool

    # RAG
    rag_context: str
    rag_results: List[Dict[str, Any]]
    has_rag_context: bool

    # Agent outputs
    question_type: Optional[str]
    vision_analysis: Optional[str]
    answer: Optional[str]
    confidence: float
    critique: str
    retry_count: int

    # Routing
    next_agent: Optional[str]
    supervisor_reason: str
    agent_history: List[str]
    round_count: int

    # Final
    final_answer: Optional[str]


# ==================================================================
#  Tool: call LLM
# ==================================================================
def _call_llm(
    system_prompt: str,
    user_message: str,
    model: str = LLM_MODEL,
    max_retries: int = 2,
) -> Optional[str]:
    """Call DashScope LLM with retry."""
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            resp = Generation.call(
                api_key=DASHSCOPE_API_KEY,
                model=model,
                messages=messages,
                result_format="message",
            )
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                if isinstance(content, list):
                    return content[0].get("text", str(content))
                return content
        except Exception:
            if attempt == max_retries - 1:
                return None
    return None


def _call_vlm(
    system_prompt: str, question: str, image_path: str
) -> Optional[str]:
    """Call DashScope multi-modal model."""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": f"{system_prompt}\n\n问题：{question}"},
                    {"image": f"file://{image_path}"},
                ],
            }
        ]
        resp = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY, model=VISION_MODEL, messages=messages
        )
        if resp.status_code == 200:
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                return content[0].get("text", str(content))
            return content
    except Exception:
        return None
    return None


def _parse_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response (with or without markdown fences)."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ==================================================================
#  Agent: Supervisor
# ==================================================================
@traceable(name="supervisor_agent", run_type="chain")
def _supervisor_agent(state: AgentState) -> dict:
    """Supervisor decides which agent to dispatch next."""
    q = state["question"]
    history = list(state.get("agent_history", []))
    has_image = state["image_present"]
    qtype = state.get("question_type")
    answer = state.get("answer")
    confidence = state.get("confidence", 0.0)
    retry = state.get("retry_count", 0)

    # ── Phase 1: Classify first if not done ──
    if "classifier" not in history:
        return {"next_agent": "classifier", "supervisor_reason": "需要先分析问题类型"}

    # ── Phase 2: Retrieve if not done (always, for context) ──
    if "retriever" not in history:
        return {"next_agent": "retriever", "supervisor_reason": "检索知识库获取参考信息"}

    # ── Phase 3: Vision analysis if image and not done ──
    if has_image and "vision_analyst" not in history:
        return {"next_agent": "vision_analyst", "supervisor_reason": "需要分析图片内容"}

    # ── Phase 4: Reason if not done ──
    if "reasoner" not in history:
        return {"next_agent": "reasoner", "supervisor_reason": "生成回答"}

    # ── Phase 5: Critic if answer exists ──
    if answer and "critic" not in history:
        return {"next_agent": "critic", "supervisor_reason": "评估回答质量"}

    # ── Phase 6: Retry if needed ──
    if answer and confidence < 0.5 and retry < 2:
        return {
            "next_agent": "reasoner",
            "supervisor_reason": f"置信度 {round(confidence,2)} 过低，需要修正重试",
            "retry_count": retry + 1,
        }

    # ── Phase 7: Finish ──
    return {
        "next_agent": "FINISH",
        "supervisor_reason": "所有工作完成",
        "final_answer": answer or "",
        "retry_count": retry,
    }


# ==================================================================
#  Agent: Classifier
# ==================================================================
@traceable(name="classifier_agent", run_type="chain")
def _classifier_agent(state: AgentState) -> dict:
    """Determine question type using LLM."""
    q = state["question"]
    has_img = state["image_present"]

    prompt = (
        f"问题：{q}\n"
        f"是否提供了图片：{'是' if has_img else '否'}\n\n"
        "只返回JSON：{\"question_type\": \"direct_visual|logical|ocr_heavy|general\"}"
    )

    result = _call_llm(CLASSIFIER_PROMPT, prompt)
    parsed = _parse_json(result) if result else None

    qtype = "general"
    if parsed and parsed.get("question_type"):
        qtype = parsed["question_type"]

    history = list(state.get("agent_history", [])) + ["classifier"]
    return {"question_type": qtype, "next_agent": None, "agent_history": history}


# ==================================================================
#  Agent: Retriever
# ==================================================================
@traceable(name="retriever_agent", run_type="chain")
def _retriever_agent(state: AgentState) -> dict:
    """Search the FAISS knowledge base."""
    results = rag.search(state["question"], top_k=5)
    if not results:
        history = list(state.get("agent_history", [])) + ["retriever"]
        return {"rag_context": "", "rag_results": [], "has_rag_context": False, "next_agent": None, "agent_history": history}

    parts = []
    for i, r in enumerate(results, 1):
        fn = r["metadata"].get("filename", "未知")
        parts.append(f"[{i}] 来自文档「{fn}」:\n{r['content']}")
    context = "\n\n---\n\n".join(parts)

    history = list(state.get("agent_history", [])) + ["retriever"]
    return {
        "rag_context": context,
        "rag_results": results,
        "has_rag_context": True,
        "next_agent": None,
        "agent_history": history,
    }


# ==================================================================
#  Agent: Vision Analyst
# ==================================================================
@traceable(name="vision_analyst_agent", run_type="llm")
def _vision_analyst_agent(state: AgentState) -> dict:
    """Analyze image using Qwen-VL."""
    if not state["image_present"] or not state["image_path"]:
        return {"vision_analysis": "", "next_agent": None}

    result = _call_vlm(VISION_ANALYST_PROMPT, state["question"], state["image_path"])
    history = list(state.get("agent_history", [])) + ["vision_analyst"]
    return {"vision_analysis": result or "图片分析失败", "next_agent": None, "agent_history": history}


# ==================================================================
#  Agent: Reasoner
# ==================================================================
@traceable(name="reasoner_agent", run_type="llm")
def _reasoner_agent(state: AgentState) -> dict:
    """Generate answer using all available context."""
    q = state["question"]
    qtype = state.get("question_type", "general")
    rag_ctx = state.get("rag_context", "")
    vision = state.get("vision_analysis", "")
    retry = state["retry_count"]
    prev_critique = state.get("critique", "")

    context_parts = []

    if rag_ctx:
        context_parts.append(f"【知识库参考信息】\n{rag_ctx}")

    if vision:
        context_parts.append(f"【图片分析结果】\n{vision}")

    context_str = "\n\n".join(context_parts) if context_parts else "无额外参考信息。"

    system_prompt = REASONER_PROMPT
    if retry > 0 and prev_critique:
        system_prompt += (
            f"\n\n这是第 {retry} 次修正尝试。前次回答的问题：\n{prev_critique}\n请修正。"
        )

    user_msg = (
        f"问题类型：{qtype}\n\n"
        f"参考信息：\n{context_str}\n\n"
        f"问题：{q}\n\n"
        "请给出全面、准确的回答。"
    )

    # If image present, use VLM; otherwise use LLM
    if state["image_present"] and state["image_path"]:
        answer = _call_vlm(system_prompt, user_msg, state["image_path"])
    else:
        answer = _call_llm(system_prompt, user_msg)

    history = list(state.get("agent_history", [])) + ["reasoner"]
    return {"answer": answer or "抱歉，无法生成有效答案。", "next_agent": None, "agent_history": history}


# ==================================================================
#  Agent: Critic
# ==================================================================
@traceable(name="critic_agent", run_type="llm")
def _critic_agent(state: AgentState) -> dict:
    """Evaluate answer quality."""
    answer = state.get("answer", "")
    history = list(state.get("agent_history", [])) + ["critic"]

    if not answer:
        return {"confidence": 0.0, "critique": "无答案可评估", "next_agent": None, "agent_history": history}

    qtype = state.get("question_type", "general")

    # For visual questions, skip LLM critic (can't see image)
    if qtype == "direct_visual":
        return {"confidence": 0.9, "critique": "视觉问答，需人工验证", "next_agent": None, "agent_history": history}

    context_snippet = state["rag_context"][:500] if state["rag_context"] else "无"

    prompt = (
        f"问题：{state['question']}\n"
        f"参考资料：{context_snippet}\n"
        f"答案：{answer}\n\n"
        f"评分维度：\n"
        f"- 事实准确性（是否与参考资料或常识一致）\n"
        f"- 回答完整度（是否完整回答了问题）\n"
        f"- 相关性（是否切题）\n\n"
        f"只返回JSON：{{\"confidence\": 0.xx, \"issues\": \"...\", \"needs_revision\": true/false}}"
    )

    result = _call_llm(CRITIC_PROMPT, prompt)
    parsed = _parse_json(result) if result else None

    if parsed:
        history = list(state.get("agent_history", [])) + ["critic"]
        return {
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "critique": parsed.get("issues", "无"),
            "next_agent": None,
            "agent_history": history,
        }

    history = list(state.get("agent_history", [])) + ["critic"]
    return {"confidence": 0.6, "critique": "自评失败，采用默认评分", "next_agent": None}


# ==================================================================
#  Routing
# ==================================================================
def _supervisor_routing(state: AgentState) -> str:
    """Route based on supervisor's decision."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "format_output"
    return next_agent


# ==================================================================
#  Output formatting
# ==================================================================
@traceable(name="format_output", run_type="chain")
def _format_output(state: AgentState) -> dict:
    """Format final output with metadata."""
    answer = state.get("final_answer") or state.get("answer", "")
    return {
        "final_answer": answer.strip() if answer else "抱歉，我没能生成有效答案。"
    }


# ==================================================================
#  Build Graph
# ==================================================================
def _create_multi_agent_graph() -> StateGraph:
    """Build and compile the multi-agent VQA graph."""
    builder = StateGraph(AgentState)

    # Register all agents as nodes
    builder.add_node("supervisor", _supervisor_agent)
    builder.add_node("classifier", _classifier_agent)
    builder.add_node("retriever", _retriever_agent)
    builder.add_node("vision_analyst", _vision_analyst_agent)
    builder.add_node("reasoner", _reasoner_agent)
    builder.add_node("critic", _critic_agent)
    builder.add_node("format_output", _format_output)

    # Entry point
    builder.set_entry_point("supervisor")

    # Supervisor routes to specialists
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_routing,
        {
            "classifier": "classifier",
            "retriever": "retriever",
            "vision_analyst": "vision_analyst",
            "reasoner": "reasoner",
            "critic": "critic",
            "format_output": "format_output",
        },
    )

    # All specialists return to supervisor for next decision
    for agent in ["classifier", "retriever", "vision_analyst", "reasoner", "critic"]:
        builder.add_edge(agent, "supervisor")

    # Output formatting → end
    builder.add_edge("format_output", END)

    # MemorySaver for checkpointing
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Compiled graph singleton
multi_agent_graph = _create_multi_agent_graph()


# ==================================================================
#  Public API
# ==================================================================
def run_multi_agent_vqa(
    question: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the multi-agent VQA system and return structured results."""
    thread_id = thread_id or f"ma_{uuid.uuid4().hex[:12]}"

    initial: AgentState = {
        "question": question,
        "image_path": image_path,
        "image_url": image_url,
        "image_present": bool(image_path),
        "rag_context": "",
        "rag_results": [],
        "has_rag_context": False,
        "question_type": None,
        "vision_analysis": None,
        "answer": None,
        "confidence": 0.5,
        "critique": "",
        "retry_count": 0,
        "next_agent": None,
        "supervisor_reason": "",
        "agent_history": [],
        "round_count": 0,
        "final_answer": None,
    }

    # Record agent history during execution via callback
    # We'll track it manually from the result
    result = multi_agent_graph.invoke(initial, {"configurable": {"thread_id": thread_id}})

    return {
        "answer": result.get("final_answer") or result.get("answer") or "",
        "confidence": result.get("confidence", 0.0),
        "critique": result.get("critique", ""),
        "has_rag_context": result.get("has_rag_context", False),
        "rag_doc_count": len(result.get("rag_results", [])),
        "question_type": result.get("question_type", "general"),
        "retry_count": result.get("retry_count", 0),
        "vision_analysis": result.get("vision_analysis"),
        "agent_history": list(result.get("agent_history", [])),
    }
