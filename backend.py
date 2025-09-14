# backend.py
import os
import requests
import time
import logging
import hashlib
from typing import Dict, List
from prompting import prompt_config   # ملف البرومبت بتاعك
from db import search_drug            # دالة بتسحب بيانات الدواء من PostgreSQL

# ======================
#   إعدادات Gemini API
# ======================
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCmyBnUkNvm70VpbJLLvIaFdv6YB7t2JwA")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ======================
#   Cache Layer
# ======================
class SimpleCache:
    def __init__(self):
        self.cache: Dict[str, str] = {}

    def _hash(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()

    def get(self, query: str):
        key = self._hash(query)
        return self.cache.get(key)

    def set(self, query: str, response: str):
        key = self._hash(query)
        self.cache[key] = response

cache = SimpleCache()

# ======================
#   كلمات مفتاحية للأعراض
# ======================
SYMPTOMS_KEYWORDS = {
    "وجع المعدة": ["وجع", "بطن", "معدة", "حرقة", "ألم في البطن"],
    "صداع": ["صداع", "راس وجع", "وجع راس"],
    "قيء": ["ترجيع", "سخنية", "غثيان"],
    "إسهال": ["اسهال", "إسهال", "إسهال مائي"],
    "تعب": ["تعب", "ارهاق", "ضعف"],
    "تنميل": ["تنميل", "وخز", "خدر"],
}

# ======================
#   Mapping الأعراض ↔ أسماء الأدوية الحقيقية في DB
# ======================
SYMPTOM_TO_DRUG = {
    "وجع المعدة": ["Omeprazole", "Pantoprazole", "Ranitidine"],
    "صداع": ["Paracetamol", "Ibuprofen", "Aspirin"],
    "قيء": ["Metoclopramide", "Domperidone"],
    "إسهال": ["Loperamide", "Oral Rehydration Salt"],
    "تعب": ["Multivitamins", "Iron Supplement", "Vitamin B Complex"],
    "تنميل": ["Vitamin B Complex", "Magnesium"]
}

# ======================
#   دالة لمطابقة الأعراض
# ======================
def match_symptom(user_input: str):
    user_input = user_input.lower()
    for symptom, keywords in SYMPTOMS_KEYWORDS.items():
        for kw in keywords:
            if kw in user_input:
                return symptom
    return None

# ======================
#   الدالة الرئيسية للرد
# ======================
def gemini_chat_wrapper(message: str, history: List = []):
    """
    message: سؤال المستخدم
    history: list من dict أو tuple -> [{'role':'user','message':'...'}, ...] أو [('user','...'), ...]
    """

    # 🔍 أولاً: نشوف لو الرد متخزن في الكاش
    cached = cache.get(message)
    if cached:
        return cached

    # نجيب بيانات الدواء من قاعدة البيانات
    drug_info = search_drug(message)

    if drug_info:
        relevant_context = f"""
اسم الدواء: {drug_info['drug_name']}
الاسم العلمي: {drug_info['generic_name']}
الفئة الدوائية: {drug_info['drug_class']}
دواعي الاستعمال: {drug_info['indication']}
الحالة بعد التكميم: {drug_info.get('status_after_sleeve', 'غير محدد')}
السبب: {drug_info.get('reason', 'غير محدد')}
ملاحظات تعديل الجرعة: {drug_info.get('dose_adjustment_notes', 'لا توجد')}
شكل الدواء: {drug_info.get('administration_form', 'غير محدد')}
ملاحظات عامة: {drug_info.get('notes', 'لا توجد')}
"""
    else:
        # لو مفيش تطابق مباشر، نحاول نربط بالأعراض
        symptom = match_symptom(message)
        if symptom:
            meds_list = SYMPTOM_TO_DRUG.get(symptom, [])
            meds_info = []
            for med in meds_list:
                info = search_drug(med)
                if info:
                    meds_info.append(f"{info['drug_name']} ({info['generic_name']}): {info['indication']}")
            if meds_info:
                relevant_context = f"بخصوص {symptom}، الأدوية المناسبة:\n- " + "\n- ".join(meds_info)
            else:
                relevant_context = f"واضح إن عندك {symptom}. حاول تاخد راحة، اشرب مياه كويسة، ولو استمر الألم أو زاد راجع دكتور."
        else:
            relevant_context = "❌ مفيش تطابق مباشر للدواء أو العرض ده في قاعدة البيانات."

    # نجمع history لو موجود
    history_text = "\n".join([
        f"{h['role']}: {h['message']}" if isinstance(h, dict) else f"{h[0]}: {h[1]}"
        for h in history
    ])

    # بناء البرومبت
    final_prompt = f"""
{prompt_config['instructions']}

HISTORY:
{history_text}

CONTEXT:
{relevant_context}

QUESTION:
{message}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": final_prompt},
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        return f"❌ مشكلة في الشبكة: {e}"

    if response.status_code == 200:
        data = response.json()
        try:
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            cache.set(message, answer)
            return answer
        except Exception as e:
            return f"⚠️ شكل الرد غير متوقع: {e}\n{data}"
    else:
        return f"❌ خطأ: {response.status_code} - {response.text}"

# ======================
#   Logging
# ======================
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    start = time.time()
    # مثال: سؤال طبي بالمصري عن أعراض
    print(gemini_chat_wrapper("معدتي وجعاني وبردو عندي اسهال"))
    end = time.time()
    logging.info(f"Execution time: {end - start:.2f} seconds")
