import os
import requests
import time
import logging
import hashlib
from typing import Dict, List
from prompting import prompt_config   # Importing prompt configuration file
from db import search_drug, insert_drug   # Import DB functions

# ======================
#   Gemini API Settings
# ======================
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCmyBnUkNvm70VpbJLLvIaFdv6YB7t2JwA")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ======================
#   Simple Cache Layer
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
#   Symptom Keywords
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
#   Mapping Symptoms ↔ Drugs
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
#   Add drug from text
# ======================
def add_drug_from_text(message: str):
    """
    Extracts drug info from user message and inserts into DB.
    Expected format:
    'عايز اضيف دواء اسمه باراسيتامول
     لازمتة: مسكن ألم
     هو ايه: Analgesic'
    """
    try:
        lines = [l.strip() for l in message.split("\n") if l.strip()]
        entry = {
            "drug_name": None,
            "generic_name": None,
            "drug_class": None,
            "indication": None,
            "status_after_sleeve": None,
            "reason": None,
            "dose_adjustment_notes": None,
            "administration_form": None,
            "interactions": None,
            "evidence_level": None,
            "source_links": None,
            "notes": None,
        }

        for line in lines:
            if "اسم" in line:
                entry["drug_name"] = line.split(":", 1)[-1].strip()
            elif "لازمت" in line or "ليه" in line:
                entry["indication"] = line.split(":", 1)[-1].strip()
            elif "هو ايه" in line or "فئة" in line:
                entry["drug_class"] = line.split(":", 1)[-1].strip()
            elif "ملاحظة" in line:
                entry["notes"] = line.split(":", 1)[-1].strip()

        if not entry["drug_name"] or not entry["indication"]:
            return "⚠️ لازم تكتب على الأقل اسم الدواء ولازمته عشان اقدر أضيفه."

        insert_drug(entry)
        return f"✅ تمت إضافة الدواء '{entry['drug_name']}' بنجاح!"
    except Exception as e:
        return f"❌ حصل خطأ أثناء الإضافة: {e}"

# ======================
#   Match Symptom
# ======================
def match_symptom(user_input: str):
    user_input = user_input.lower()
    for symptom, keywords in SYMPTOMS_KEYWORDS.items():
        for kw in keywords:
            if kw in user_input:
                return symptom
    return None

# ======================
#   Main Chat Wrapper
# ======================
def gemini_chat_wrapper(message: str, history: List = []):
    # 0️⃣ Check if user wants to add a drug
    if "اضيف" in message or "أدخل" in message or "اضافة" in message:
        return add_drug_from_text(message)

    # 1️⃣ Cache lookup
    cached = cache.get(message)
    if cached:
        return cached

    # 2️⃣ DB search
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
        # 3️⃣ Symptom mapping
        symptom = match_symptom(message)
        if symptom:
            meds_list = SYMPTOM_TO_DRUG.get(symptom, [])
            meds_info = []
            for med in meds_list:
                info = search_drug(med)
                if info:
                    meds_info.append(f"{info['drug_name']} ({info['generic_name']}): {info['indication']}")
            if meds_info:
                relevant_context = f"Regarding {symptom}, suitable medications:\n- " + "\n- ".join(meds_info)
            else:
                relevant_context = f"It seems you have {symptom}. Try resting, stay hydrated, and consult a doctor if symptoms persist."
        else:
            relevant_context = "❌ No direct match found for this drug or symptom in the database."

    history_text = "\n".join([
        f"{h['role']}: {h['message']}" if isinstance(h, dict) else f"{h[0]}: {h[1]}"
        for h in history
    ])

    final_prompt = f"""
{prompt_config['instructions']}

HISTORY:
{history_text}

CONTEXT:
{relevant_context}

QUESTION:
{message}
"""

    payload = {"contents": [{"parts": [{"text": final_prompt}]}]}

    try:
        response = requests.post(url, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {e}"

    if response.status_code == 200:
        data = response.json()
        try:
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            cache.set(message, answer)
            return answer
        except Exception as e:
            return f"⚠️ Unexpected response format: {e}\n{data}"
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# ======================
#   Logging
# ======================
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    start = time.time()
    print(gemini_chat_wrapper("عايز اضيف دواء اسمه باراسيتامول\nلازمتة: مسكن ألم\nهو ايه: Analgesic"))
    end = time.time()
    logging.info(f"Execution time: {end - start:.2f} seconds")


