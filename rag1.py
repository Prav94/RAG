import os
import glob
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from pathlib import Path
from langchain_openai import ChatOpenAI

MODEL = "gpt-4o-mini"

load_dotenv(find_dotenv(), override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

MODEL = "gpt-4o-mini"
chatOpenAI = ChatOpenAI(model=MODEL, temperature=0.7, streaming=True)


knowledge = {}
employees = glob.glob("./knowledge-base/employees/*")

for employee in employees:
    name = employee.split(" ")[-1][:-3]
    doc = ""
    with open(employee, "r", encoding="utf-8") as f:
        knowledge[name] = f.read()

products = glob.glob('knowledge-base/products/*')
for product in products:
    name = Path(product).stem
    doc = ""
    with open(product, "r", encoding="utf-8") as f:
        knowledge[name] = f.read()

system_mesage = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.
Relevant context:
"""

def get_relevant_context(message):
    relevant_knowledge = []
    for knowledge_title, knowledge_details in knowledge.items():
        # message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        if knowledge_title in message:
            relevant_knowledge.append(knowledge_details)
    return relevant_knowledge

def add_context(message):
    relevant_knowledge = get_relevant_context(message)
    if relevant_knowledge:
        message += "\n\nThe following additional context might be relevant in anwering the question:\n\n"
        for relevant in relevant_knowledge:
            message += relevant + "\n\n"
    return message

def chat(message, history):
    '''
    Gradio passes: 
    message: latest user message (str)
    history: list[dict] of {"role": ..., "content": ...}
    '''
    # Start with the system message

    messages = [{"role": "system", "content": system_mesage}]

    # Append the prior conversation

    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    
    user_input = add_context(message)
    # Append the latest user input
    messages.append({"role": "user", "content": user_input})

    # Stream Response
    response = ""
    updated_history = history + [{"role": "user", "content": message}]  # show plain user input

    for chunk in chatOpenAI.stream(messages):
        if chunk.content:
            response +=chunk.content
            yield updated_history + [{"role": "assistant", "content": response}]    


with gr.Blocks() as demo:
    gr.Markdown("### 🤖 InsureLLM AI Assistant")
    chatbot = gr.Chatbot(type="messages", label="AI Assistant", height=500)
    msg = gr.Textbox(placeholder="Type your message...", label="Your Message")
    msg.submit(chat, [msg, chatbot], [chatbot])
    msg.submit(chat, [msg, chatbot], [chatbot]).then(
    lambda: "", None, [msg]
    )#clear text box after submit

demo.launch()