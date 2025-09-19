# backend.py
import os
import requests
import time
import logging
import hashlib
from typing import Dict, List
from prompting import prompt_config   # Importing prompt configuration file
from db import search_drug            # Function to fetch drug info from PostgreSQL

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
        # Dictionary to store cached responses {hash: response}
        self.cache: Dict[str, str] = {}

    def _hash(self, query: str) -> str:
        # Generate MD5 hash of the query (case-insensitive)
        return hashlib.md5(query.lower().encode()).hexdigest()

    def get(self, query: str):
        # Retrieve response if query exists in cache
        key = self._hash(query)
        return self.cache.get(key)

    def set(self, query: str, response: str):
        # Save query-response pair in cache
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
#   Function to Match Symptoms
# ======================
def match_symptom(user_input: str):
    user_input = user_input.lower()
    # Loop through keywords to find a matching symptom
    for symptom, keywords in SYMPTOMS_KEYWORDS.items():
        for kw in keywords:
            if kw in user_input:
                return symptom
    return None

# ======================
#   Main Chat Wrapper
# ======================
def gemini_chat_wrapper(message: str, history: List = []):
    """
    Handles user queries and integrates:
    - Cache lookup
    - Database drug search
    - Symptom-to-drug mapping
    - Gemini API response generation
    
    message: User query (str)
    history: List of conversation history items (dict or tuple)
    """

    # 1️⃣ Check if response exists in cache
    cached = cache.get(message)
    if cached:
        return cached

    # 2️⃣ Try to fetch drug info directly from DB
    drug_info = search_drug(message)

    if drug_info:
        # If drug found in DB, prepare detailed context
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
        # 3️⃣ If no direct drug match, try symptom matching
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
            # 4️⃣ If no match at all, return fallback message
            relevant_context = "❌ No direct match found for this drug or symptom in the database."

    # 5️⃣ Build conversation history string
    history_text = "\n".join([
        f"{h['role']}: {h['message']}" if isinstance(h, dict) else f"{h[0]}: {h[1]}"
        for h in history
    ])

    # 6️⃣ Construct final prompt
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

    # 7️⃣ Send request to Gemini API
    try:
        response = requests.post(url, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {e}"

    # 8️⃣ Handle Gemini API response
    if response.status_code == 200:
        data = response.json()
        try:
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            cache.set(message, answer)  # Save response to cache
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
    # Example test query
    print(gemini_chat_wrapper("معدتي وجعاني وبردو عندي اسهال"))
    end = time.time()
    logging.info(f"Execution time: {end - start:.2f} seconds")

