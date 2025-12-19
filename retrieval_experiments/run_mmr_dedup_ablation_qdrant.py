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
EVAL_SIZE = 1000  # 먼저 200으로 sanity -> 이후 1000~2000

# Fixed retriever + chunking (너가 채택한 최종 설정)
HF_MODEL = "intfloat/multilingual-e5-large"
CHUNK_CHARS = 800
OVERLAP = 160
MIN_CHUNK_CHARS = 200

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION = "navernews_fixed800_for_mmr"

# Retrieval / MMR
FETCH_K = 30      # Qdrant에서 먼저 넉넉히 가져오기
FINAL_K = 10      # 평가 K (Recall@10, MRR@10)
MMR_LAMBDA = 0.5  # 0.0=다양성만, 1.0=관련성만 (보통 0.3~0.7)

# Upload
UPLOAD_BATCH_SIZE = 512
UPLOAD_PARALLEL = 2

OUT_DIR = "retrieval_experiments/results_postretrieval"


# -------------------------
# Helpers
# -------------------------
def format_query(text: str) -> str:
    return "query: " + text

def format_doc(text: str) -> str:
    return "passage: " + text

def chunk_text(text: str, chunk_chars: int, overlap: int):
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

def ensure_collection(client: QdrantClient, name: str, dim: int):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def unwrap_query_points_result(out):
    return out.points if hasattr(out, "points") else out

def extract_payload(item):
    # item이 tuple인 경우 payload(dict) 찾기
    if isinstance(item, tuple):
        return next((x for x in item if isinstance(x, dict)), {})
    return getattr(item, "payload", {}) or {}

def extract_id(item):
    # item이 tuple인 경우 id(int) 찾기
    if isinstance(item, tuple):
        # 흔히 id가 int로 들어있음
        for x in item:
            if isinstance(x, (int, np.integer)):
                return int(x)
        return None
    # object case
    return int(getattr(item, "id"))

def qdrant_query_points(client: QdrantClient, collection: str, qv, limit: int):
    out = client.query_points(
        collection_name=collection,
        query=qv,          # ✅ 너 환경
        limit=limit,
        with_payload=True,
    )
    return unwrap_query_points_result(out)

def recall_at_k(hit_ranks, k):
    return float(np.mean([(r is not None and r <= k) for r in hit_ranks]))

def mrr_at_k(hit_ranks, k):
    return float(np.mean([(1.0 / r) if (r is not None and r <= k) else 0.0 for r in hit_ranks]))

def cosine_sim(a, b):
    # embeddings는 normalize_embeddings=True로 만들었으니 dot = cosine
    return float(np.dot(a, b))

def mmr_select(query_vec, cand_ids, doc_emb, k, lam):
    """
    MMR:
    score = lam * sim(q, d) - (1-lam) * max_{s in selected} sim(d, s)
    """
    if not cand_ids:
        return []

    # precompute q-sim
    q = query_vec
    q_sims = {cid: cosine_sim(q, doc_emb[cid]) for cid in cand_ids}

    selected = []
    remaining = cand_ids[:]

    while remaining and len(selected) < k:
        best_id = None
        best_score = -1e9

        for cid in remaining:
            rel = q_sims[cid]
            if not selected:
                div_pen = 0.0
            else:
                # diversity penalty: max similarity to already selected
                div_pen = max(cosine_sim(doc_emb[cid], doc_emb[sid]) for sid in selected)

            score = lam * rel - (1.0 - lam) * div_pen
            if score > best_score:
                best_score = score
                best_id = cid

        selected.append(best_id)
        remaining.remove(best_id)

    return selected

def redundancy_avg_pairwise_sim(ids, doc_emb):
    # top-k 내 평균 pairwise cosine similarity (중복/유사도 지표)
    if len(ids) < 2:
        return 0.0
    sims = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sims.append(cosine_sim(doc_emb[ids[i]], doc_emb[ids[j]]))
    return float(np.mean(sims))


# -------------------------
# Main
# -------------------------
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load dataset
    ds = load_dataset(DATASET_NAME)
    data = ds[SPLIT].select(range(min(EVAL_SIZE, len(ds[SPLIT]))))

    queries, query_article_ids, documents = [], [], []
    for idx, row in enumerate(tqdm(data, desc="Load queries/docs")):
        queries.append(str(row["title"]).strip())
        query_article_ids.append(idx)
        documents.append(str(row["document"]).strip())

    # 2) Build chunks (fixed 800/160)
    all_chunks = []
    chunk_article_ids = []
    for aid, doc in enumerate(tqdm(documents, desc="Chunking(fixed_800_160)")):
        chunks = chunk_text(doc, CHUNK_CHARS, OVERLAP)
        for c in chunks:
            all_chunks.append(c)
            chunk_article_ids.append(aid)
    chunk_article_ids = np.asarray(chunk_article_ids, dtype=np.int32)
    print(f"#queries={len(queries)}  #chunks={len(all_chunks)}")

    # 3) Embedder
    st = SentenceTransformer(HF_MODEL, device="cpu")  # 너 환경에서는 CPU 강제 권장
    # Query embedding (once)
    t0 = time.time()
    q_emb = st.encode(
        [format_query(q) for q in queries],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    print("q_emb:", q_emb.shape, "time:", time.time() - t0)

    # Doc embedding (once)
    t1 = time.time()
    doc_emb = st.encode(
        [format_doc(c) for c in all_chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    print("doc_emb:", doc_emb.shape, "time:", time.time() - t1)

    # 4) Qdrant index
    client = QdrantClient(url=QDRANT_URL)
    dim = doc_emb.shape[1]
    ensure_collection(client, COLLECTION, dim)

    points = [
        PointStruct(
            id=i,  # 중요: id == doc_emb index
            vector=doc_emb[i].tolist(),
            payload={"article_id": int(chunk_article_ids[i])},
        )
        for i in range(len(all_chunks))
    ]
    t_up = time.time()
    client.upload_points(
        collection_name=COLLECTION,
        points=points,
        batch_size=UPLOAD_BATCH_SIZE,
        parallel=UPLOAD_PARALLEL,
    )
    print("upload_time_s:", time.time() - t_up)

    # 5) Evaluate: baseline(top10) vs mmr(top10 from fetch30)
    hit_baseline = []
    hit_mmr = []
    lat_search = []
    lat_mmr = []
    red_base = []
    red_mmr = []

    for i in tqdm(range(len(queries)), desc="Search+MMR eval"):
        qv = q_emb[i]
        target_aid = query_article_ids[i]

        # search
        ts = time.time()
        res = qdrant_query_points(client, COLLECTION, qv.tolist(), limit=FETCH_K)
        lat_search.append((time.time() - ts) * 1000.0)

        # extract ids + payload order
        cand_ids = []
        cand_payloads = []
        for item in res:
            pid = extract_id(item)
            if pid is None:
                continue
            cand_ids.append(pid)
            cand_payloads.append(extract_payload(item))

        # baseline: take first FINAL_K
        base_ids = cand_ids[:FINAL_K]
        red_base.append(redundancy_avg_pairwise_sim(base_ids, doc_emb))

        first_b = None
        for r, pid in enumerate(base_ids, start=1):
            # payload는 result에서 가져오되, pid->payload 매핑이 필요
            # (cand_ids와 cand_payloads 동일 순서)
            payload = cand_payloads[cand_ids.index(pid)]
            if payload.get("article_id") == target_aid:
                first_b = r
                break
        hit_baseline.append(first_b)

        # MMR rerank/select
        tm = time.time()
        mmr_ids = mmr_select(qv, cand_ids, doc_emb, k=FINAL_K, lam=MMR_LAMBDA)
        lat_mmr.append((time.time() - tm) * 1000.0)
        red_mmr.append(redundancy_avg_pairwise_sim(mmr_ids, doc_emb))

        first_m = None
        # mmr_ids는 pid 리스트이므로 payload를 Qdrant 결과에서 찾아야 함(동일 cand 목록 내)
        for r, pid in enumerate(mmr_ids, start=1):
            payload = cand_payloads[cand_ids.index(pid)]
            if payload.get("article_id") == target_aid:
                first_m = r
                break
        hit_mmr.append(first_m)

    # 6) Aggregate
    rows = [
        {
            "method": "baseline_top10",
            "fetch_k": FETCH_K,
            "final_k": FINAL_K,
            "mmr_lambda": None,
            "recall@10": recall_at_k(hit_baseline, 10),
            "mrr@10": mrr_at_k(hit_baseline, 10),
            "latency_search_ms_avg": float(np.mean(lat_search)),
            "latency_rerank_ms_avg": 0.0,
            "redundancy_avg_sim@10": float(np.mean(red_base)),
        },
        {
            "method": "mmr_top10",
            "fetch_k": FETCH_K,
            "final_k": FINAL_K,
            "mmr_lambda": MMR_LAMBDA,
            "recall@10": recall_at_k(hit_mmr, 10),
            "mrr@10": mrr_at_k(hit_mmr, 10),
            "latency_search_ms_avg": float(np.mean(lat_search)),
            "latency_rerank_ms_avg": float(np.mean(lat_mmr)),
            "redundancy_avg_sim@10": float(np.mean(red_mmr)),
        },
    ]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "mmr_dedup_ablation.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== FINAL MMR DEDUP RESULTS ===")
    print(df[["method", "recall@10", "mrr@10", "latency_search_ms_avg", "latency_rerank_ms_avg", "redundancy_avg_sim@10"]])
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()