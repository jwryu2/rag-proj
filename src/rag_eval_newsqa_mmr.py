import json
import re
import time
from collections import Counter
from statistics import mean

# -----------------------------
# Config (필요하면 여기만 수정)
# -----------------------------
NEWSQA_PATH = "newsqa.json"   # rag-proj 폴더 기준
LIMIT = 200                  # 먼저 200으로 sanity -> 이후 1000
TOP_K = 10
FETCH_K = 30                 # MMR 후보 풀
LAMBDAS = [None, 0.70, 0.85, 0.95]  # None = baseline

# 너희 프로젝트에 rag_answer가 있으면 그걸 우선 사용
USE_PROJECT_RAG = True
PROJECT_RAG_IMPORT = ("rag", "rag_answer")  # from rag import rag_answer

# 모드 B(extractive)용: Qdrant/Embedding 설정
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "navernews_fixed800_for_mmr_eval"
EMBED_MODEL = "intfloat/multilingual-e5-large"
DEVICE = None  # None이면 자동: cuda 가능하면 cuda, 아니면 cpu


# -----------------------------
# Utils: normalize + metrics
# -----------------------------
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

def load_newsqa(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for ex in data:
        qa = json.loads(ex["qa_pair"])
        out.append({
            "qid": ex.get("qid"),
            "docid": ex.get("docid"),
            "question": qa["question"],
            "answer": qa["answer"],
        })
    return out


# -----------------------------
# Mode A: call your project rag_answer
# rag_answer(question, top_k, fetch_k, rerank, mmr_lambda) -> dict
#   dict = {"answer": str, "contexts": [{"docid":..., "text":...}, ...]}
# -----------------------------
def try_import_project_rag():
    mod_name, fn_name = PROJECT_RAG_IMPORT
    mod = __import__(mod_name, fromlist=[fn_name])
    fn = getattr(mod, fn_name)
    return fn


# -----------------------------
# Mode B: Extractive RAG (LLM 없이도 실행 가능)
# - Qdrant에서 contexts 뽑고
# - (baseline or MMR)로 top_k contexts 선택
# - 답은 contexts에서 "가장 질문과 유사한 문장"을 뽑아 반환
# -----------------------------
def pick_device_simple():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def format_query_e5(q: str) -> str:
    return "query: " + q

def cosine(a, b):
    import numpy as np
    return float(np.dot(a, b))

def mmr_select(q_vec, cand_vecs, cand_payloads, k_final=10, lam=0.85):
    # q_vec: (D,), cand_vecs: (K,D) assumed normalized
    import numpy as np
    K = cand_vecs.shape[0]
    if K == 0:
        return []

    q_sims = cand_vecs @ q_vec
    selected_idx = []
    remaining = list(range(K))

    while remaining and len(selected_idx) < k_final:
        best_i, best_score = None, -1e18
        for i in remaining:
            rel = float(q_sims[i])
            if not selected_idx:
                div_pen = 0.0
            else:
                div_pen = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected_idx)
            score = lam * rel - (1.0 - lam) * div_pen
            if score > best_score:
                best_score, best_i = score, i
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [cand_payloads[i] for i in selected_idx]

def extractive_answer(question: str, contexts: list[str]) -> str:
    """
    아주 단순한 extractive baseline:
    - 문장 단위로 쪼개고
    - 질문과 공유 토큰이 많은 문장을 답으로 선택
    """
    q_tokens = set(normalize_ko(question).split())
    best = ""
    best_score = -1
    for ctx in contexts:
        # 한국어 문장 분리 (간단히 . ! ? \n 기준)
        sents = re.split(r"[\.!\?\n]+", ctx)
        for s in sents:
            s = s.strip()
            if not s:
                continue
            toks = set(normalize_ko(s).split())
            score = len(q_tokens & toks)
            if score > best_score:
                best_score = score
                best = s
    return best if best else (contexts[0] if contexts else "")

def rag_answer_extractive(question: str, top_k=10, fetch_k=30, rerank=None, mmr_lambda=0.85):
    import numpy as np
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    device = DEVICE or pick_device_simple()
    st = SentenceTransformer(EMBED_MODEL, device=device)
    client = QdrantClient(url=QDRANT_URL)

    # query embedding
    q_vec = st.encode([format_query_e5(question)], normalize_embeddings=True)[0].astype("float32")

    # retrieve with vectors for MMR
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        limit=fetch_k,
        with_payload=True,
        with_vectors=True,
    )
    points = res.points if hasattr(res, "points") else (res[0] if isinstance(res, tuple) else res)

    cand_payloads = []
    cand_vecs = []
    for p in points:
        payload = getattr(p, "payload", None) or {}
        vec = getattr(p, "vector", None)
        if vec is None:
            # older clients might store vectors differently; skip if missing
            continue
        cand_payloads.append(payload)
        cand_vecs.append(np.asarray(vec, dtype=np.float32))

    if len(cand_vecs) == 0:
        contexts = []
    else:
        cand_vecs = np.vstack(cand_vecs)
        if rerank == "mmr":
            selected = mmr_select(q_vec, cand_vecs, cand_payloads, k_final=top_k, lam=mmr_lambda)
        else:
            selected = cand_payloads[:top_k]
        contexts = [{"docid": pl.get("docid", pl.get("article_id")), "text": pl.get("text", "")} for pl in selected]

    answer = extractive_answer(question, [c["text"] for c in contexts])
    return {"answer": answer, "contexts": contexts}


# -----------------------------
# Evaluation runner
# -----------------------------
def eval_setting(rag_fn, ds, top_k, fetch_k, rerank, mmr_lambda):
    ems, f1s, hits, lats = [], [], [], []

    for i, ex in enumerate(ds, 1):
        t0 = time.perf_counter()
        out = rag_fn(
            ex["question"],
            top_k=top_k,
            fetch_k=fetch_k,
            rerank=rerank,
            mmr_lambda=mmr_lambda
        )
        dt = (time.perf_counter() - t0) * 1000.0
        lats.append(dt)

        pred = out.get("answer", "")
        ctxs = out.get("contexts", []) or []

        ems.append(exact_match(pred, ex["answer"]))
        f1s.append(token_f1(pred, ex["answer"]))

        # hit@k (docid가 payload에 있으면 의미 있음)
        ctx_docids = [c.get("docid") for c in ctxs]
        hits.append(int(ex["docid"] in ctx_docids))

        if i % 20 == 0:
            print(f"[{i}/{len(ds)}] EM={mean(ems):.3f} F1={mean(f1s):.3f} hit@{top_k}={mean(hits):.3f} lat(ms)={mean(lats):.1f}")

    return {
        "rerank": rerank or "baseline",
        "mmr_lambda": mmr_lambda,
        "EM": mean(ems),
        "F1": mean(f1s),
        f"hit@{top_k}": mean(hits),
        "latency_ms_avg": mean(lats),
        "n": len(ds),
    }


def main():
    ds = load_newsqa(NEWSQA_PATH)
    ds = ds[:LIMIT]

    # choose rag function
    rag_fn = None
    if USE_PROJECT_RAG:
        try:
            rag_fn = try_import_project_rag()
            print("[mode] Using project RAG:", PROJECT_RAG_IMPORT)
        except Exception as e:
            print("[mode] Failed to import project RAG -> fallback to extractive. reason:", type(e).__name__, e)

    if rag_fn is None:
        print("[mode] Using extractive RAG (no LLM).")
        rag_fn = rag_answer_extractive

    results = []
    for lam in LAMBDAS:
        if lam is None:
            results.append(eval_setting(rag_fn, ds, TOP_K, FETCH_K, rerank=None, mmr_lambda=None))
        else:
            results.append(eval_setting(rag_fn, ds, TOP_K, FETCH_K, rerank="mmr", mmr_lambda=lam))

    print("\n=== FINAL NEWSQA EVAL (Baseline vs MMR) ===")
    for r in results:
        print(r)

    # save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    out_path = "eval_newsqa_mmr.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()