import os
import yaml
from typing import List, Optional, Tuple
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def load_cfg(path="configs/app.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(question: str, docs: List[dict]) -> List[dict]:
    """
    RAG 프롬프트 (간결 + 근거 기반) - 키워드 답변 강제
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
        "너는 질문에 해당하는 답변만 간결하게 제공하면 된다.\n"
        "불필요한 수식어나 설명은 하지 마라.\n"
        "문장으로 답변하지 마라. 간결한 키워드 형태로 답변하라.\n"
        "출력은 한 줄로만 하라.\n"
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


def _get_point_vector(p) -> Optional[np.ndarray]:
    v = getattr(p, "vector", None)
    if v is not None:
        return np.asarray(v, dtype=np.float32)
    vs = getattr(p, "vectors", None)
    if isinstance(vs, dict) and len(vs) > 0:
        first_key = next(iter(vs.keys()))
        return np.asarray(vs[first_key], dtype=np.float32)
    return None


def mmr_rerank(
    cand_scores: np.ndarray,   # (K,)
    cand_vecs: np.ndarray,     # (K,D) normalized
    top_k: int,
    lam: float,
) -> List[int]:
    K = cand_scores.shape[0]
    if K == 0:
        return []

    selected = []
    remaining = list(range(K))

    while remaining and len(selected) < top_k:
        best_i = None
        best_score = -1e18

        for i in remaining:
            rel = float(cand_scores[i])
            if not selected:
                div_pen = 0.0
            else:
                div_pen = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected)

            score = lam * rel - (1.0 - lam) * div_pen
            if score > best_score:
                best_score = score
                best_i = i

        selected.append(best_i)
        remaining.remove(best_i)

    return selected


def rag_answer(
    question: str,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    rerank: Optional[str] = None,     # None or "mmr"
    mmr_lambda: float = 0.85,
) -> Tuple[str, List[dict]]:
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

    # defaults
    if top_k is None:
        top_k = int(cfg.get("top_k", 1))
    if fetch_k is None:
        fetch_k = int(cfg.get("fetch_k", max(30, top_k)))

    # ---- Retrieval ----
    client = QdrantClient(url=cfg["qdrant"]["url"])

    with_vectors = (rerank == "mmr")
    res = client.query_points(
        collection_name=cfg["qdrant"]["collection"],
        query=qvec,
        limit=fetch_k,
        with_payload=True,
        with_vectors=with_vectors,
    )

    c_docs = []
    c_scores = []
    c_vecs = []

    for p in res.points:
        payload = p.payload or {}
        doc = {
            "score": float(p.score),
            "title": payload.get("title", ""),
            "summary": payload.get("summary", ""),
            "text": payload.get("text", ""),
            "docid": payload.get("docid", payload.get("article_id")),
        }
        c_docs.append(doc)
        c_scores.append(doc["score"])

        if with_vectors:
            v = _get_point_vector(p)
            if v is None:
                with_vectors = False
            else:
                c_vecs.append(v)

    # ---- Select docs ----
    if rerank == "mmr" and with_vectors and len(c_vecs) == len(c_docs):
        cand_scores = np.asarray(c_scores, dtype=np.float32)
        cand_vecs = np.vstack(c_vecs).astype(np.float32)
        norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12
        cand_vecs = cand_vecs / norms

        pick = mmr_rerank(cand_scores, cand_vecs, top_k=top_k, lam=float(mmr_lambda))
        docs = [c_docs[i] for i in pick]
    else:
        docs = c_docs[:top_k]

    # ---- Generation ----
    llm = OpenAI(
        base_url=cfg["llm"]["base_url"],
        api_key="dummy",
    )

    messages = build_prompt(question, docs)

    resp = llm.chat.completions.create(
        model=cfg["llm"]["model"],
        messages=messages,
        temperature=cfg["llm"].get("temperature", 0.2),
        max_tokens=cfg["llm"].get("max_tokens", 256),  # 키워드 답변이면 짧게
    )

    print("finish_reason:", resp.choices[0].finish_reason)

    answer = resp.choices[0].message.content.strip()
    return answer, docs