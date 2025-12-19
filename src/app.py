import gradio as gr
from rag import rag_answer


def run_rag(question):
    answer, docs = rag_answer(question)

    results = []
    for i, d in enumerate(docs, start=1):
        text = d["text"][:500] + ("..." if len(d["text"]) > 500 else "")
        results.append(
            f"### [{i}] (score={d['score']:.4f})\n"
            f"**Title**: {d['title']}\n\n"
            f"{text}"
        )

    return answer, "\n\n---\n\n".join(results)


with gr.Blocks(title="Korean News RAG Chatbot") as demo:
    gr.Markdown("# 📰 Korean News RAG Chatbot")

    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(
                label="질문",
                placeholder="예: AI 반도체 시장 전망은?",
                lines=2,
            )
            submit = gr.Button("검색 + 답변")

        with gr.Column(scale=2):
            answer = gr.Markdown(label="답변")

    docs = gr.Markdown(label="Top-5 검색 문서")

    submit.click(run_rag, inputs=query, outputs=[answer, docs])

demo.launch()