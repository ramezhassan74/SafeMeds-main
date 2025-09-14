# ui.py
import gradio as gr
from backend import gemini_chat_wrapper   # استدعاء الفنكشن من backend

# واجهة Gradio
demo = gr.ChatInterface(
    fn=gemini_chat_wrapper,
    title="SafeMeds RAG Chatbot 🤖",
    description="Ask me anything from our information. I'm powered by SafeMeds RAG Chatbot 🤖",
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True لو عايز لينك خارجي

