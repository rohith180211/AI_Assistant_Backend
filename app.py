import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"


def upload_file(file):
    files = {"file": open(file.name, "rb")}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()


def chat_with_bot(message, history):
    response = requests.post(
        f"{API_URL}/query",
        json={"question": message}
    )

    data = response.json()

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    # Better formatting
    formatted_answer = f"{answer}\n\n📚 Sources:\n"
    for i, src in enumerate(sources):
        formatted_answer += f"\n{i+1}. {src[:200]}..."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": formatted_answer})

    return "", history


with gr.Blocks() as app:
    gr.Markdown("# 🤖 AI Knowledge Retrieval System")

    # 📄 Upload Section
    with gr.Row():
        file_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Upload")

    upload_output = gr.Textbox(label="Upload Status")

    upload_btn.click(
        upload_file,
        inputs=file_input,
        outputs=upload_output
    )

    gr.Markdown("---")

    # 💬 Chat Section
    chatbot = gr.Chatbot(height=400)

    msg = gr.Textbox(
        placeholder="Ask a question about your documents...",
        label="Your Question"
    )

    clear = gr.Button("Clear Chat")

    msg.submit(chat_with_bot, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

app.launch(share=True)