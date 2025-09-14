# vector_store_db.py (مبسط للعمل مع meds DB فقط)
from typing import List, Dict, Any
from db import search_drug

# ---------- إعدادات عامة ----------
TOP_K = 5  # لو حابين نرجع أكثر من نتيجة

class RetrievedChunk:
    def __init__(self, id: str, content: str, source: str):
        self.id = id
        self.content = content
        self.source = source

def query_similar(drug_name: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    بديل query_similar: ترجع البيانات من جدول meds مباشرة بدل vector store.
    """
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

    return [{
        "id": data["drug_name"],
        "content": chunk_text,
        "metadata": {"source": "SafeMeds DB"},
        "score": None  # score غير موجود هنا لأننا مش بنعمل embedding
    }]

# ---------- مثال استخدام ----------
if __name__ == "__main__":
    results = query_similar("paracetamol")
    for r in results:
        print(r["content"])
