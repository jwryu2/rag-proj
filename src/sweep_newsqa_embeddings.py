import os
import json
import re
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from statistics import mean

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------
# Config presets (edit here)
# -------------------------
@dataclass
class EmbCfg:
    key: str
    model_name: str
    query_prefix: str = "query: "
    doc_prefix: str = "passage: "  # e5 계열이면 passage 권장, 아니면 ""로 둬도 됨

EMBEDDINGS = [
    EmbCfg("e5_large", "intfloat/multilingual-e5-large", query_prefix="query: ", doc_prefix="passage: "),
    EmbCfg("bge_m3",   "BAAI/bge-m3",                   query_prefix="",        doc_prefix=""),
    EmbCfg("koe5",     "nlpai-lab/KoE5",                query_prefix="query: ", doc_prefix="passage: "),
    EmbCfg("e5_base",  "intfloat/multilingual-e5-base", query_prefix="query: ", doc_prefix="passage: "),
]

# -------------------------
# Utils: newsqa parsing
# -------------------------
def parse_qa_pair(qa_pair):
    if isinstance(qa_pair, dict):
        return qa_pair
    if isinstance(qa_pair, str):
        return json.loads(qa_pair)
    raise TypeError(f"Unexpected qa_pair type: {type(qa_pair)}")

def load_newsqa(path: str, limit: Optional[int] = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    skipped = 0
    for i, ex in enumerate(data):
        try:
            qa = parse_qa_pair(ex["qa_pair"])
            out.append({
                "qid": ex.get("qid"),
                "docid": ex.get("docid"),
                "question": qa["question"],
                "answer": qa["answer"],
            })
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"[SKIP newsqa] idx={i} reason={e}")

    if limit:
        out = out[:limit]
    print(f"[newsqa] loaded={len(out)} skipped={skipped}")
    return out

# -------------------------
# Metrics: EM/F1
# -------------------------
def normalize_ko(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "")
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_ko(pred) == normalize_ko(gold))

def token_f1(pred: str, gold: str) -> float:
    p = normalize_ko(pred).split()
    g = normalize_ko(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)

# -------------------------
# Chunking (fixed 800/160 default)
# -------------------------
def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    step = max(1, chunk_chars - overlap)
    i = 0
    while i < len(text):
        c = text[i:i+chunk_chars]
        if c.strip():
            chunks.append(c)
        i += step
    return chunks

# -------------------------
# Prompt (keyword-only)
# -------------------------
def build_prompt(question: str, docs: List[Dict[str, Any]]) -> List[dict]:
    ctx = "\n\n".join([f"[{i+1}]\n{d['text']}" for i, d in enumerate(docs)])
    system = (
        "너는 한국어 뉴스 질의응답 도우미다.\n"
        "아래 제공된 문서 근거를 바탕으로만 답변하라.\n"
        "근거가 부족하면 추측하지 말고 모른다고 말하라.\n"
        "너는 질문에 해당하는 답변만 간결하게 제공하면 된다.\n"
        "불필요한 수식어나 설명은 하지 마라.\n"
        "문장으로 답변하지 마라. 간결한 키워드 형태로 답변하라.\n"
        "출력은 한 줄로만 하라.\n"
    )
    user = f"질문:\n{question}\n\n근거 문서:\n{ctx}\n\n위 근거를 바탕으로 질문에 답하라."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# -------------------------
# MMR (optional)
# -------------------------
def mmr_pick(q_vec: np.ndarray, cand_vecs: np.ndarray, cand_scores: np.ndarray, top_k: int, lam: float) -> List[int]:
    # all normalized. relevance uses q_vec dot cand_vec
    q_sims = cand_vecs @ q_vec
    selected = []
    remaining = list(range(len(cand_scores)))
    while remaining and len(selected) < top_k:
        best_i, best_val = None, -1e18
        for i in remaining:
            rel = float(q_sims[i])
            if not selected:
                div = 0.0
            else:
                div = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected)
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best_val = val
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

# -------------------------
# Load HF dataset (robust column detection)
# -------------------------
def load_corpus_hf(dataset_name: str, split: str, limit_docs: Optional[int] = None):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    if limit_docs:
        ds = ds.select(range(min(limit_docs, len(ds))))

    cols = set(ds.column_names)
    # 후보 칼럼들: 너 데이터셋 스키마가 달라도 최대한 잡히게
    text_candidates = ["document", "article", "content", "text", "body", "news", "src", "source"]
    title_candidates = ["title", "headline"]
    docid_candidates = ["docid", "id", "article_id", "news_id"]

    def pick_first(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    text_col = pick_first(text_candidates)
    title_col = pick_first(title_candidates)
    docid_col = pick_first(docid_candidates)

    if text_col is None:
        raise RuntimeError(f"Cannot find text column. Available columns: {ds.column_names}")

    print(f"[corpus] columns: text={text_col} title={title_col} docid={docid_col}")

    rows = []
    for i, r in enumerate(ds):
        text = r.get(text_col, "") or ""
        title = r.get(title_col, "") if title_col else ""
        docid = r.get(docid_col, i) if docid_col else i  # fallback: row index
        rows.append({"docid": int(docid) if isinstance(docid, (int, np.integer)) or str(docid).isdigit() else str(docid),
                     "title": title, "text": text})
    return rows

# -------------------------
# Qdrant: create + upsert
# -------------------------
def recreate_collection(client: QdrantClient, name: str, dim: int):
    # delete if exists
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def upsert_chunks(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    emb_cfg: EmbCfg,
    corpus_rows: List[Dict[str, Any]],
    chunk_chars: int,
    overlap: int,
    batch_size: int,
):
    pid = 0
    points: List[PointStruct] = []
    t0 = time.perf_counter()

    for ridx, row in enumerate(corpus_rows, 1):
        docid = row["docid"]
        title = row.get("title", "")
        text = row.get("text", "")

        chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
        if not chunks:
            continue

        # embed chunks (passage prefix if needed)
        passages = [(emb_cfg.doc_prefix + c) if emb_cfg.doc_prefix else c for c in chunks]
        vecs = embedder.encode(passages, normalize_embeddings=True, batch_size=batch_size)
        vecs = np.asarray(vecs, dtype=np.float32)

        for j, (c, v) in enumerate(zip(chunks, vecs)):
            payload = {
                "docid": docid,
                "title": title,
                "text": c,
                "chunk_id": f"{docid}_{j}",
            }
            points.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))
            pid += 1

            if len(points) >= batch_size:
                client.upsert(collection_name=collection, points=points)
                points = []

        if ridx % 200 == 0:
            dt = time.perf_counter() - t0
            print(f"[upsert] docs={ridx}/{len(corpus_rows)} points={pid} elapsed={dt:.1f}s")

    if points:
        client.upsert(collection_name=collection, points=points)

    dt = time.perf_counter() - t0
    print(f"[upsert done] collection={collection} total_points={pid} elapsed={dt:.1f}s")
    return pid

# -------------------------
# Retrieval + Generate + Eval
# -------------------------
def retrieve(
    client: QdrantClient,
    collection: str,
    qvec: List[float],
    top_k: int,
    fetch_k: int,
    use_mmr: bool,
    mmr_lambda: float,
):
    res = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=fetch_k if use_mmr else top_k,
        with_payload=True,
        with_vectors=use_mmr,
    )
    points = res.points

    docs = []
    if not use_mmr:
        for p in points[:top_k]:
            pl = p.payload or {}
            docs.append({"docid": pl.get("docid"), "title": pl.get("title",""), "text": pl.get("text",""), "score": float(p.score)})
        return docs

    # MMR: need candidate vectors
    cand_payloads = []
    cand_vecs = []
    cand_scores = []
    for p in points:
        pl = p.payload or {}
        v = getattr(p, "vector", None)
        if v is None:
            # cannot mmr without vectors
            continue
        cand_payloads.append(pl)
        cand_vecs.append(np.asarray(v, dtype=np.float32))
        cand_scores.append(float(p.score))

    if not cand_vecs:
        return []

    cand_vecs = np.vstack(cand_vecs)
    # normalize safety
    cand_vecs /= (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
    cand_scores = np.asarray(cand_scores, dtype=np.float32)

    # qvec is already normalized from embedder.encode(normalize_embeddings=True)
    q = np.asarray(qvec, dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-12)

    pick = mmr_pick(q, cand_vecs, cand_scores, top_k=top_k, lam=mmr_lambda)
    for i in pick:
        pl = cand_payloads[i]
        docs.append({"docid": pl.get("docid"), "title": pl.get("title",""), "text": pl.get("text",""), "score": None})
    return docs

def run_eval(
    client: QdrantClient,
    llm: OpenAI,
    llm_model: str,
    embedder: SentenceTransformer,
    emb_cfg: EmbCfg,
    collection: str,
    newsqa: List[Dict[str, Any]],
    top_k: int,
    fetch_k: int,
    use_mmr: bool,
    mmr_lambda: float,
    temperature: float,
    max_tokens: int,
    print_each: bool,
):
    ems, f1s, hits, lats = [], [], [], []

    for i, ex in enumerate(newsqa, 1):
        q = ex["question"]
        gold = ex["answer"]
        gold_docid = ex["docid"]

        # embed query
        qtext = (emb_cfg.query_prefix + q) if emb_cfg.query_prefix else q
        qvec = embedder.encode([qtext], normalize_embeddings=True)[0].tolist()

        t0 = time.perf_counter()
        docs = retrieve(client, collection, qvec, top_k=top_k, fetch_k=fetch_k, use_mmr=use_mmr, mmr_lambda=mmr_lambda)
        messages = build_prompt(q, docs)

        resp = llm.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ans = (resp.choices[0].message.content or "").strip()
        dt = (time.perf_counter() - t0) * 1000.0

        # metrics
        ems.append(exact_match(ans, gold))
        f1s.append(token_f1(ans, gold))
        lats.append(dt)

        docids = [d.get("docid") for d in docs]
        hits.append(int(gold_docid in docids))

        if print_each:
            print("=" * 80)
            print(f"[{i}/{len(newsqa)}] {'MMR' if use_mmr else 'BASE'} {emb_cfg.key} collection={collection}")
            print("Q:", q)
            print("GOLD:", gold)
            print("PRED:", ans)
            print("HIT@k:", hits[-1], "DOCIDS:", docids[:min(5, len(docids))])
            print("finish_reason:", resp.choices[0].finish_reason, "lat(ms):", f"{dt:.1f}")
            print("=" * 80)

        if i % 20 == 0:
            print(f"[{i}/{len(newsqa)}] EM={mean(ems):.3f} F1={mean(f1s):.3f} hit@{top_k}={mean(hits):.3f} lat(ms)={mean(lats):.1f}")

    return {
        "embedding": emb_cfg.key,
        "collection": collection,
        "mmr": use_mmr,
        "mmr_lambda": (mmr_lambda if use_mmr else None),
        "EM": mean(ems),
        "F1": mean(f1s),
        f"hit@{top_k}": mean(hits),
        "latency_ms_avg": mean(lats),
        "n": len(newsqa),
    }

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant_url", default="http://127.0.0.1:6333")
    ap.add_argument("--dataset", default="daekeun-ml/naver-news-summarization-ko")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit_docs", type=int, default=2000)
    ap.add_argument("--chunk_chars", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--newsqa", default="newsqa.json")
    ap.add_argument("--limit_qa", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--fetch_k", type=int, default=30)

    ap.add_argument("--use_mmr", action="store_true")
    ap.add_argument("--mmr_lambdas", default="0.70,0.85,0.95")

    ap.add_argument("--llm_base_url", required=True)  # ex) http://127.0.0.1:11434/v1
    ap.add_argument("--llm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=128)

    ap.add_argument("--print_each", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url)

    # load corpus once
    corpus = load_corpus_hf(args.dataset, args.split, limit_docs=args.limit_docs)

    # load eval once
    newsqa = load_newsqa(args.newsqa, limit=args.limit_qa)

    llm = OpenAI(base_url=args.llm_base_url, api_key="dummy")

    mmr_lams = [float(x) for x in args.mmr_lambdas.split(",") if x.strip()]

    results = []

    for emb in EMBEDDINGS:
        print(f"\n=== [EMBED] {emb.key}: {emb.model_name} ===")
        embedder = SentenceTransformer(emb.model_name, device="cpu")  # 필요하면 cuda로 바꿔도 됨
        dim = embedder.get_sentence_embedding_dimension()
        collection = f"navernews_{emb.key}_ch{args.chunk_chars}_ov{args.overlap}"

        print(f"[collection] recreate {collection} dim={dim}")
        recreate_collection(client, collection, dim=dim)

        print(f"[index] upsert chunks: docs={len(corpus)}")
        upsert_chunks(
            client=client,
            collection=collection,
            embedder=embedder,
            emb_cfg=emb,
            corpus_rows=corpus,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )

        # baseline eval
        print(f"\n--- EVAL baseline (no MMR) : {emb.key} ---")
        r0 = run_eval(
            client=client,
            llm=llm,
            llm_model=args.llm_model,
            embedder=embedder,
            emb_cfg=emb,
            collection=collection,
            newsqa=newsqa,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            use_mmr=False,
            mmr_lambda=0.85,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            print_each=args.print_each,
        )
        results.append(r0)

        if args.use_mmr:
            for lam in mmr_lams:
                print(f"\n--- EVAL MMR lam={lam} : {emb.key} ---")
                r1 = run_eval(
                    client=client,
                    llm=llm,
                    llm_model=args.llm_model,
                    embedder=embedder,
                    emb_cfg=emb,
                    collection=collection,
                    newsqa=newsqa,
                    top_k=args.top_k,
                    fetch_k=args.fetch_k,
                    use_mmr=True,
                    mmr_lambda=lam,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    print_each=args.print_each,
                )
                results.append(r1)

    # save results
    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = "sweep_newsqa_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== DONE ===")
    print(df[["embedding","mmr","mmr_lambda","EM","F1",f"hit@{args.top_k}","latency_ms_avg","n"]])
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()