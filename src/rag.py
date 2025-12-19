import os
import yaml
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def load_cfg(path="configs/app.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(question: str, docs: List[dict]) -> List[dict]:
    """
    RAG 프롬프트 (간결 + 근거 기반)
    """
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        block = f"[{i}]\n{d['text']}"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    system = (
        "너는 한국어 뉴스 질의응답 도우미다.\n"
        "아래 제공된 문서 근거를 바탕으로만 답변하라.\n"
        "근거가 부족하면 추측하지 말고 모른다고 말하라.\n"
        "답변에는 관련 근거 번호([1], [2] 등)를 포함하라."
    )

    user = f"""
질문:
{question}

근거 문서:
{context}

위 근거를 바탕으로 질문에 답하라.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rag_answer(question: str):
    cfg = load_cfg(os.environ.get("APP_CONFIG", "configs/app.yaml"))

    # ---- Embedding ----
    embedder = SentenceTransformer(
        cfg["embedding"]["model_name"],
        device=cfg["embedding"].get("device", "cpu"),
    )

    qvec = embedder.encode(
        ["query: " + question],
        normalize_embeddings=True,
    )[0].tolist()

    # ---- Retrieval ----
    client = QdrantClient(url=cfg["qdrant"]["url"])
    top_k = int(cfg.get("top_k", 1))

    res = client.query_points(
        collection_name=cfg["qdrant"]["collection"],
        query=qvec,
        limit=top_k,
        with_payload=True,
    )

    docs = []
    for p in res.points:
        payload = p.payload or {}
        docs.append(
            {
                "score": p.score,
                "title": payload.get("title", ""),
                "summary": payload.get("summary", ""),
                "text": payload.get("text", ""),
            }
        )

    # ---- Generation (Ollama: OpenAI compatible) ----
    llm = OpenAI(
        base_url=cfg["llm"]["base_url"],
        api_key="dummy",
    )

    messages = build_prompt(question, docs)

    resp = llm.chat.completions.create(
        model=cfg["llm"]["model"],
        messages=messages,
        temperature=cfg["llm"].get("temperature", 0.2),
        max_tokens=cfg["llm"].get("max_tokens", 40960),
    )

    print("finish_reason:", resp.choices[0].finish_reason)
    
    answer = resp.choices[0].message.content
    return answer, docs