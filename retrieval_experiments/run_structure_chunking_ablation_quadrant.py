import os, time
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
EVAL_SIZE = 1000  # 먼저 200으로 sanity → 이후 1000~2000

# Retriever fixed
MODEL_KEY = "e5_large"
HF_MODEL = "intfloat/multilingual-e5-large"

# Baseline fixed-size params (너가 채택한 값)
BASE_CHUNK_CHARS = 800
BASE_OVERLAP = 160
MIN_CHUNK_CHARS = 200

# Structure-aware params
# - 문단 우선으로 묶되, 너무 길면 max_len 기준으로 분할
STRUCT_MAX_CHARS = 800
STRUCT_OVERLAP = 160  # max split시에만 사용
STRUCT_JOIN_PARAS_UNTIL = 800  # 작은 문단들을 합칠 때 목표 길이

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_PREFIX = "navernews_structchunk_"

# Eval
K_LIST = [5, 10]
OUT_DIR = "retrieval_experiments/results_chunking"
UPLOAD_BATCH_SIZE = 512
UPLOAD_PARALLEL = 2


# -------------------------
# Helpers
# -------------------------
def format_query(text: str) -> str:
    return "query: " + text  # E5 prefix

def format_doc(text: str) -> str:
    return "passage: " + text  # E5 prefix

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
    return out.points if hasattr(out, "points") else out

def extract_payload(item):
    if isinstance(item, tuple):
        return next((x for x in item if isinstance(x, dict)), {})
    return getattr(item, "payload", {}) or {}

def qdrant_query_points(client: QdrantClient, collection: str, qv, limit: int):
    out = client.query_points(
        collection_name=collection,
        query=qv,          # ✅ 너 환경 기준
        limit=limit,
        with_payload=True,
    )
    return unwrap_query_points_result(out)


# -------------------------
# Chunkers
# -------------------------
def chunk_fixed(text: str, chunk_chars: int, overlap: int):
    """기존 고정 길이 chunking"""
    text = text.strip()
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        c = text[i:j].strip()
        if len(c) >= MIN_CHUNK_CHARS:
            chunks.append(c)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def _split_paragraphs(text: str):
    """
    뉴스 본문에서 문단을 대충 분리:
    - 빈 줄(연속 개행) 기준 우선
    - 없으면 단일 개행 기준
    """
    t = text.strip()
    if not t:
        return []
    # 빈 줄 기준
    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    if len(paras) >= 2:
        return paras
    # 단일 개행 기준 fallback
    paras = [p.strip() for p in t.split("\n") if p.strip()]
    return paras if paras else [t]

def chunk_structure_paragraph(text: str):
    """
    구조 기반(문단 우선) chunking:
    1) 문단 단위로 분리
    2) 짧은 문단들은 목표 길이(STRUCT_JOIN_PARAS_UNTIL)까지 이어붙여 chunk로 만듦
    3) 매우 긴 문단/덩어리는 max_len 기준으로 고정길이 분할(오버랩 적용)
    """
    paras = _split_paragraphs(text)

    chunks = []
    buf = ""

    def flush_buf():
        nonlocal buf
        b = buf.strip()
        if len(b) >= MIN_CHUNK_CHARS:
            chunks.append(b)
        buf = ""

    for p in paras:
        p = p.strip()
        if not p:
            continue

        # 문단이 너무 길면: 버퍼 먼저 flush 후, 문단 자체를 고정 분할
        if len(p) > STRUCT_MAX_CHARS:
            flush_buf()
            chunks.extend(chunk_fixed(p, STRUCT_MAX_CHARS, STRUCT_OVERLAP))
            continue

        # 문단이 적당한 길이면: 버퍼에 누적
        if not buf:
            buf = p
        else:
            candidate = buf + "\n" + p
            if len(candidate) <= STRUCT_JOIN_PARAS_UNTIL:
                buf = candidate
            else:
                flush_buf()
                buf = p

    flush_buf()
    return chunks


# -------------------------
# Main
# -------------------------
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load dataset
    ds = load_dataset(DATASET_NAME)
    data = ds[SPLIT].select(range(min(EVAL_SIZE, len(ds[SPLIT]))))

    queries, query_article_ids, documents = [], [], []
    for idx, row in enumerate(tqdm(data, desc="Load queries/docs")):
        queries.append(str(row["title"]).strip())
        query_article_ids.append(idx)
        documents.append(str(row["document"]).strip())

    # Embedder fixed
    print(f"\n=== Embedder fixed: {MODEL_KEY} / {HF_MODEL} ===")
    st = SentenceTransformer(HF_MODEL)

    # Query embeddings (공통)
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

    client = QdrantClient(url=QDRANT_URL)

    # Two configs: baseline fixed vs structure-aware
    EXPERIMENTS = [
        {
            "name": "fixed_800_160",
            "chunker": lambda txt: chunk_fixed(txt, BASE_CHUNK_CHARS, BASE_OVERLAP),
        },
        {
            "name": "para_max800_join800",
            "chunker": lambda txt: chunk_structure_paragraph(txt),
        },
    ]

    rows = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        chunker = exp["chunker"]

        print(f"\n=== Experiment: {name} ===")

        # Build chunks
        all_chunks = []
        chunk_article_ids = []

        for aid, doc in enumerate(tqdm(documents, desc=f"Chunking({name})")):
            chunks = chunker(doc)
            for c in chunks:
                all_chunks.append(c)
                chunk_article_ids.append(aid)

        chunk_article_ids = np.asarray(chunk_article_ids, dtype=np.int32)
        print(f"#chunks={len(all_chunks)}")

        # Doc embedding
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

        # Qdrant upload
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

        # Search + metrics
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
            "experiment": name,
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
    out_csv = os.path.join(OUT_DIR, f"struct_chunking_ablation_{MODEL_KEY}.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== FINAL STRUCTURE CHUNKING RESULTS ===")
    print(df[["experiment", "num_chunks", "recall@5", "recall@10", "mrr@10", "latency_ms_avg"]])
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()