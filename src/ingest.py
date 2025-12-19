import os, time, json
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# -------------------------
# Config
# -------------------------
DATASET_NAME = "daekeun-ml/naver-news-summarization-ko"
SPLIT = "test"          # 가능하면 "validation" 먼저
EVAL_SIZE = 1000        # 처음엔 200~500 추천, 확인 후 늘리기

# Dense candidates
MODELS = {
    "e5_large": "intfloat/multilingual-e5-large",
    "bge_m3": "BAAI/bge-m3",
    "koe5": "nlpai-lab/KoE5",
}

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_PREFIX = "navernews_dense_"

# Chunking (Retriever 비교에서 통제변수로 고정)
CHUNK_CHARS = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 200

# Evaluation
K_LIST = [5, 10]
OUT_DIR = "retrieval_experiments/results_qdrant"

# Bulk upload tuning
UPLOAD_BATCH_SIZE = 512
UPLOAD_PARALLEL = 4  # 환경에 따라 2~8로 튜닝

# -------------------------
# Helpers
# -------------------------
def format_query(model_key: str, text: str) -> str:
    # E5 계열은 prefix 권장
    if model_key.startswith("e5"):
        return "query: " + text
    return text

def format_doc(model_key: str, text: str) -> str:
    if model_key.startswith("e5"):
        return "passage: " + text
    return text

def chunk_text(text: str, chunk_chars: int, overlap: int):
    text = text.strip()
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def recall_at_k(hit_ranks, k):
    hits = [(r is not None and r <= k) for r in hit_ranks]
    return float(np.mean(hits))

def mrr_at_k(hit_ranks, k):
    rr = [(1.0 / r) if (r is not None and r <= k) else 0.0 for r in hit_ranks]
    return float(np.mean(rr))

def ensure_collection(client: QdrantClient, name: str, dim: int):
    # 실험 반복 안정성을 위해 항상 새로 생성
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load dataset
    ds = load_dataset(DATASET_NAME)
    if SPLIT not in ds:
        raise ValueError(f"Split '{SPLIT}' not found. Available: {list(ds.keys())}")
    data = ds[SPLIT]
    if EVAL_SIZE:
        data = data.select(range(min(EVAL_SIZE, len(data))))

    # 2) Build chunks + queries + gold mapping
    all_chunks = []
    chunk_article_ids = []
    queries = []
    query_article_ids = []

    for idx, row in enumerate(tqdm(data, desc="Building chunks/queries")):
        title = str(row["title"]).strip()
        doc = str(row["document"]).strip()

        queries.append(title)
        query_article_ids.append(idx)

        chunks = chunk_text(doc, CHUNK_CHARS, CHUNK_OVERLAP)
        for c in chunks:
            all_chunks.append(c)
            chunk_article_ids.append(idx)

    chunk_article_ids = np.array(chunk_article_ids, dtype=np.int32)
    print(f"#queries={len(queries)}  #chunks={len(all_chunks)}")

    # 3) Qdrant client
    client = QdrantClient(url=QDRANT_URL)

    results_rows = []

    for key, hf_model in MODELS.items():
        print(f"\n=== Evaluating {key}: {hf_model} ===")
        st = SentenceTransformer(hf_model)

        # 3-1) Embed docs (chunks)
        t0 = time.time()
        docs_for_embed = [format_doc(key, c) for c in all_chunks]
        doc_emb = st.encode(
            docs_for_embed,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")
        embed_time = time.time() - t0
        dim = doc_emb.shape[1]
        print(f"Doc embedding: {doc_emb.shape}, time={embed_time:.1f}s")

        # 3-2) Create collection for this model
        collection = COLLECTION_PREFIX + key
        ensure_collection(client, collection, dim)

        # 3-3) Build PointStruct list
        # ✅ payload는 article_id만 (text 제거로 속도/용량 개선)
        points = [
            PointStruct(
                id=i,
                vector=doc_emb[i].tolist(),
                payload={"article_id": int(chunk_article_ids[i])},
            )
            for i in range(len(all_chunks))
        ]

        # ✅ Bulk upload (upload_points) 사용
        t_up = time.time()
        client.upload_points(
            collection_name=collection,
            points=points,
            batch_size=UPLOAD_BATCH_SIZE,
            parallel=UPLOAD_PARALLEL,
        )
        upsert_time = time.time() - t_up
        print(f"Bulk upload done: time={upsert_time:.1f}s")

        # 3-4) Embed queries
        t1 = time.time()
        q_for_embed = [format_query(key, q) for q in queries]
        q_emb = st.encode(
            q_for_embed,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")
        q_embed_time = time.time() - t1
        print(f"Query embedding: {q_emb.shape}, time={q_embed_time:.1f}s")

        # 3-5) Search + hit ranks
        max_k = max(K_LIST)
        hit_ranks = []
        latencies = []

        for i in tqdm(range(len(queries)), desc=f"Searching ({key})"):
            qv = q_emb[i].tolist()
            target_aid = query_article_ids[i]

            t_s = time.time()
            res = client.search(
                collection_name=collection,
                query_vector=qv,
                limit=max_k,
                with_payload=["article_id"],  # payload만 받아서 가볍게
            )
            latencies.append((time.time() - t_s) * 1000.0)

            first = None
            for r, point in enumerate(res, start=1):
                if point.payload.get("article_id") == target_aid:
                    first = r
                    break
            hit_ranks.append(first)

        # 3-6) Aggregate metrics
        row = {
            "model_key": key,
            "hf_model": hf_model,
            "eval_split": SPLIT,
            "eval_size": len(queries),
            "num_chunks": len(all_chunks),
            "chunk_chars": CHUNK_CHARS,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_chars": MIN_CHUNK_CHARS,
            "doc_embed_time_s": embed_time,
            "upload_time_s": upsert_time,
            "query_embed_time_s": q_embed_time,
            "latency_ms_avg": float(np.mean(latencies)),
        }
        for k in K_LIST:
            row[f"recall@{k}"] = recall_at_k(hit_ranks, k)
        row["mrr@10"] = mrr_at_k(hit_ranks, 10)
        results_rows.append(row)

        with open(os.path.join(OUT_DIR, f"{key}_hit_ranks.json"), "w", encoding="utf-8") as f:
            json.dump(hit_ranks, f, ensure_ascii=False)

        print(row)

    df = pd.DataFrame(results_rows)
    out_csv = os.path.join(OUT_DIR, "dense_retriever_eval.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== FINAL RESULTS ===")
    print(df[["model_key", "recall@5", "recall@10", "mrr@10", "latency_ms_avg"]])
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()