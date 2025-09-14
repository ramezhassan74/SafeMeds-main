from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from backend import gemini_chat_wrapper  # استدعاء من backend.py

app = FastAPI(title="SafeMeds RAG Chatbot API", version="1.0")


class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[str] = []

class ChatHistoryItem(BaseModel):
    role: str 
    message: str

chat_histories: Dict[str, List[ChatHistoryItem]] = {}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # نجيب history القديم لو موجود
    history = chat_histories.setdefault(request.user_id, [])
    history.append(ChatHistoryItem(role="user", message=request.query))

    # 🔗 هنا نستدعي الجيميني + قاعدة البيانات
    answer = gemini_chat_wrapper(request.query, history=[h.dict() for h in history])

    # تقدر لو حابب تضيف citations لو backend بيرجع مصادر
    citations = ["[SafeMeds DB]"]

    history.append(ChatHistoryItem(role="assistant", message=answer))

    return ChatResponse(answer=answer, citations=citations)


@app.get("/chat/history/{user_id}", response_model=List[ChatHistoryItem])
def get_history(user_id: str):
    return chat_histories.get(user_id, [])
