# rag_core.py (معدّل للعمل مع DB فقط)
from typing import List, Dict, Any
from dataclasses import dataclass
from db import search_drug  # الدالة اللي بتجيب بيانات الدواء من PostgreSQL

# ---------- إعدادات عامة ----------
TOP_K = 5  # عدد النتائج المرجعية لو حابين نستخدم أكثر من واحد

_SYSTEM_PROMPT = {
    "name": "SafeMeds Assistant",
    "role": "Medical Information Assistant",
    "instructions": (
        "أنت SafeMeds Assistant، مساعد معلومات دوائية موثوق. "
        "هدفك الأساسي هو الإجابة على الأسئلة المتعلقة بالأدوية بعد عمليات التكميم، "
        "بناءً على بيانات موثقة من قاعدة البيانات فقط. "
        "إذا لم تكفِ المعلومات المتاحة، أخبر المستخدم صراحة بذلك. "
        "استخدم لغة واضحة، دقيقة، ومبسطة."
    )
}

@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    score: float | None = None

# ---------- استرجاع البيانات من DB ----------
def retrieve_context(drug_name: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """تجيب بيانات الدواء كـ RetrievedChunk من قاعدة البيانات."""
    data = search_drug(drug_name)
    if not data:
        return []

    chunk_text = (
        f"Drug Name: {data['drug_name']}\n"
        f"Generic: {data['generic_name']}\n"
        f"Class: {data['drug_class']}\n"
        f"Indication: {data['indication']}\n"
        f"Status After Sleeve: {data['status_after_sleeve']}\n"
        f"Reason: {data['reason']}\n"
        f"Dose Adjustment: {data['dose_adjustment_notes']}\n"
        f"Form: {data['administration_form']}\n"
        f"Notes: {data['notes']}\n"
    )

    return [RetrievedChunk(
        id=data['drug_name'],
        text=chunk_text,
        source="SafeMeds DB"
    )]

# ---------- دالة للإجابة ----------
def answer_question(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    chunks = retrieve_context(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "لا توجد معلومات كافية في قاعدة البيانات للإجابة على سؤالك.",
            "citations": [],
            "used_chunks": []
        }

    # أبسط إجابة: نجمع النصوص من الـ chunks مباشرة
    raw_answer = "\n".join(c.text for c in chunks)

    # أبسط شكل للاستشهادات: [1], [2], ... بناءً على ترتيب المقاطع
    citations = [f"[{i}]" for i in range(1, len(chunks) + 1)]

    return {
        "answer": raw_answer,
        "citations": citations,
        "used_chunks": [c.__dict__ for c in chunks]
    }
