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
SPLIT = "test"
EVAL_SIZE = 1000  # 먼저 200~500으로 sanity-check 권장, 이후 1000~2000

# Retriever (고정)
MODEL_KEY = "e5_large"
HF_MODEL = "intfloat/multilingual-e5-large"
DEVICE = "cpu"  # "cpu" or "cuda" (GPU 안 쓰면 None)

# Chunk size ablation (overlap은 20% 고정)
CHUNK_CONFIGS = [
    {"name": "sz500",  "chunk_chars": 500,  "overlap": 100},
    {"name": "sz800",  "chunk_chars": 800,  "overlap": 160},  # baseline
    {"name": "sz1200", "chunk_chars": 1200, "overlap": 240},
]
MIN_CHUNK_CHARS = 200

# Qdrant
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_PREFIX = "navernews_chunksize_"

# Eval
K_LIST = [5, 10]
OUT_DIR = "retrieval_experiments/results_chunking"
UPLOAD_BATCH_SIZE = 512
UPLOAD_PARALLEL = 2


# -------------------------
# Helpers
# -------------------------
def format_query(text: str) -> str:
    # E5 prefix
    return "query: " + text

def format_doc(text: str) -> str:
    return "passage: " + text

def chunk_text(text: str, chunk_chars: int, overlap: int):
    text = text.strip()
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        c = text[i:j].strip()
        if len(c) >= MIN_CHUNK_CHARS:
            chunks.append(c)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def recall_at_k(hit_ranks, k):
    return float(np.mean([(r is not None and r <= k) for r in hit_ranks]))

def mrr_at_k(hit_ranks, k):
    return float(np.mean([(1.0 / r) if (r is not None and r <= k) else 0.0 for r in hit_ranks]))

def ensure_collection(client: QdrantClient, name: str, dim: int):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def unwrap_query_points_result(out):
    # out이 wrapper이면 out.points, 아니면 out 자체가 iterable
    return out.points if hasattr(out, "points") else out

def extract_payload(item):
    # item이 tuple이면 dict payload 찾아서 반환
    if isinstance(item, tuple):
        return next((x for x in item if isinstance(x, dict)), {})
    return getattr(item, "payload", {}) or {}

def qdrant_query_points(client: QdrantClient, collection: str, qv, limit: int):
    out = client.query_points(
        collection_name=collection,
        query=qv,              # ✅ 너 환경 기준
        limit=limit,
        with_payload=True,
    )
    return unwrap_query_points_result(out)


# -------------------------
# Main
# -------------------------
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load dataset + build queries once (공통)
    ds = load_dataset(DATASET_NAME)
    if SPLIT not in ds:
        raise ValueError(f"Split '{SPLIT}' not found. Available: {list(ds.keys())}")
    data = ds[SPLIT]
    data = data.select(range(min(EVAL_SIZE, len(data))))

    queries = []
    query_article_ids = []
    documents = []

    for idx, row in enumerate(tqdm(data, desc="Load queries/docs")):
        queries.append(str(row["title"]).strip())
        query_article_ids.append(idx)
        documents.append(str(row["document"]).strip())

    # 2) Load embedder once (고정)
    print(f"\n=== Embedder fixed: {MODEL_KEY} / {HF_MODEL} ===")
    if DEVICE:
        st = SentenceTransformer(HF_MODEL, device=DEVICE)
    else:
        st = SentenceTransformer(HF_MODEL)

    # query embeddings는 chunking과 무관하므로 한 번만 만든다
    q_for_embed = [format_query(q) for q in queries]
    print(f"[STAGE] Query embedding start: n={len(q_for_embed)}")
    t_q = time.time()
    q_emb = st.encode(
        q_for_embed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    print(f"[STAGE] Query embedding done: {q_emb.shape}, time={time.time()-t_q:.1f}s")

    # 3) Qdrant client
    client = QdrantClient(url=QDRANT_URL)

    # 4) For each chunk config: build chunks -> doc emb -> upload -> search -> metrics
    rows = []
    for cfg in CHUNK_CONFIGS:
        name = cfg["name"]
        chunk_chars = cfg["chunk_chars"]
        overlap = cfg["overlap"]

        print(f"\n=== Chunk config: {name} (chunk={chunk_chars}, overlap={overlap}) ===")

        # 4-1) Build chunks
        all_chunks = []
        chunk_article_ids = []
        for aid, doc in enumerate(tqdm(documents, desc=f"Chunking({name})")):
            chunks = chunk_text(doc, chunk_chars, overlap)
            for c in chunks:
                all_chunks.append(c)
                chunk_article_ids.append(aid)

        chunk_article_ids = np.asarray(chunk_article_ids, dtype=np.int32)
        print(f"#chunks={len(all_chunks)}")

        # 4-2) Doc embedding
        docs_for_embed = [format_doc(c) for c in all_chunks]
        print(f"[STAGE] Doc embedding start: n={len(docs_for_embed)}")
        t_d = time.time()
        doc_emb = st.encode(
            docs_for_embed,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")
        doc_embed_time = time.time() - t_d
        dim = doc_emb.shape[1]
        print(f"[STAGE] Doc embedding done: {doc_emb.shape}, time={doc_embed_time:.1f}s")

        # 4-3) Create + upload to Qdrant (config별 collection)
        collection = f"{COLLECTION_PREFIX}{MODEL_KEY}_{name}"
        ensure_collection(client, collection, dim)

        points = [
            PointStruct(
                id=i,
                vector=doc_emb[i].tolist(),
                payload={"article_id": int(chunk_article_ids[i])},
            )
            for i in range(len(all_chunks))
        ]

        print("[STAGE] Upload start")
        t_up = time.time()
        client.upload_points(
            collection_name=collection,
            points=points,
            batch_size=UPLOAD_BATCH_SIZE,
            parallel=UPLOAD_PARALLEL,
        )
        upload_time = time.time() - t_up
        print(f"[STAGE] Upload done: time={upload_time:.1f}s")

        # 4-4) Search + hit ranks
        max_k = max(K_LIST)
        hit_ranks = []
        lat = []

        for i in tqdm(range(len(queries)), desc=f"Search({name})"):
            target_aid = query_article_ids[i]
            qv = q_emb[i].tolist()

            t0 = time.time()
            res = qdrant_query_points(client, collection, qv, limit=max_k)
            lat.append((time.time() - t0) * 1000.0)

            first = None
            for r, item in enumerate(res, start=1):
                payload = extract_payload(item)
                if payload.get("article_id") == target_aid:
                    first = r
                    break
            hit_ranks.append(first)

        row = {
            "chunk_config": name,
            "chunk_chars": chunk_chars,
            "overlap": overlap,
            "eval_size": len(queries),
            "num_chunks": int(len(all_chunks)),
            "recall@5": recall_at_k(hit_ranks, 5),
            "recall@10": recall_at_k(hit_ranks, 10),
            "mrr@10": mrr_at_k(hit_ranks, 10),
            "latency_ms_avg": float(np.mean(lat)),
            "doc_embed_time_s": float(doc_embed_time),
            "upload_time_s": float(upload_time),
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows).sort_values(["recall@10", "mrr@10"], ascending=False)
    out_csv = os.path.join(OUT_DIR, f"chunk_size_ablation_{MODEL_KEY}.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== FINAL CHUNK SIZE RESULTS ===")
    print(df[["chunk_config", "chunk_chars", "overlap", "num_chunks", "recall@5", "recall@10", "mrr@10", "latency_ms_avg"]])
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()