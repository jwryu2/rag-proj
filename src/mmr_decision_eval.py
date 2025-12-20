import os
import re
import json
import time
import argparse
from collections import Counter
from statistics import mean
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# -----------------------
# Text normalization utils
# -----------------------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "")
    return s

def token_f1(pred: str, gold: str) -> float:
    p = norm(pred).split()
    g = norm(gold).split()
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

def exact_match(pred: str, gold: str) -> int:
    return int(norm(pred) == norm(gold))

def answer_in_context(gold: str, docs: List[Dict[str, Any]]) -> int:
    g = norm(gold)
    if not g:
        return 0
    ctx = norm("\n".join([d.get("text", "") for d in docs]))
    return int(g in ctx)


# -----------------------
# Prompt (keyword-only)
# -----------------------
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


# -----------------------
# MMR + Diversity metrics
# -----------------------
def _vec_from_point(p) -> Optional[np.ndarray]:
    v = getattr(p, "vector", None)
    if v is not None:
        return np.asarray(v, dtype=np.float32)
    vs = getattr(p, "vectors", None)
    if isinstance(vs, dict) and len(vs) > 0:
        first_key = next(iter(vs.keys()))
        return np.asarray(vs[first_key], dtype=np.float32)
    return None

def redundancy_avg_sim(vecs: np.ndarray) -> float:
    # vecs: (K,D) normalized
    if vecs.shape[0] <= 1:
        return 0.0
    sim = vecs @ vecs.T
    k = sim.shape[0]
    # 평균 pairwise (i<j)
    triu = sim[np.triu_indices(k, k=1)]
    return float(np.mean(triu))

def ild(vecs: np.ndarray) -> float:
    # Intra-list diversity = 1 - avg_sim
    return 1.0 - redundancy_avg_sim(vecs)

def mmr_select(q: np.ndarray, cand: np.ndarray, top_k: int, lam: float) -> List[int]:
    # q: (D,) normalized, cand: (N,D) normalized
    N = cand.shape[0]
    if N == 0:
        return []
    qsim = cand @ q
    selected = []
    remaining = list(range(N))
    while remaining and len(selected) < top_k:
        best_i, best_val = None, -1e18
        for i in remaining:
            rel = float(qsim[i])
            if not selected:
                div = 0.0
            else:
                div = max(float(cand[i] @ cand[j]) for j in selected)
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


# -----------------------
# Data loader: newsqa.json
# -----------------------
def parse_qa_pair(qa_pair):
    if isinstance(qa_pair, dict):
        return qa_pair
    if isinstance(qa_pair, str):
        return json.loads(qa_pair)
    raise TypeError(type(qa_pair))

def load_newsqa(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out, skipped = [], 0
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
                print("[SKIP]", i, e)
    if limit:
        out = out[:limit]
    print(f"[newsqa] loaded={len(out)} skipped={skipped}")
    return out


# -----------------------
# Core: retrieve docs
# -----------------------
def retrieve_docs(
    client: QdrantClient,
    collection: str,
    qvec: List[float],
    top_k: int,
    fetch_k: int,
    want_vectors: bool,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    res = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=fetch_k if want_vectors else top_k,
        with_payload=True,
        with_vectors=want_vectors,
    )
    docs = []
    vecs = []
    for p in res.points:
        pl = p.payload or {}
        docs.append({
            "docid": pl.get("docid"),
            "title": pl.get("title", ""),
            "text": pl.get("text", ""),
            "score": float(p.score),
        })
        if want_vectors:
            v = _vec_from_point(p)
            if v is not None:
                vecs.append(v)
    if want_vectors and vecs:
        V = np.vstack(vecs).astype(np.float32)
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        return docs, V
    return docs, None


# -----------------------
# Main evaluation loop
# -----------------------
def eval_setting(
    setting_name: str,
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    llm: OpenAI,
    llm_model: str,
    ds: List[Dict[str, Any]],
    top_k: int,
    fetch_k: int,
    mmr_mode: str,         # "off" | "on" | "conditional"
    mmr_lambda: float,
    mmr_redun_threshold: float,
    temperature: float,
    max_tokens: int,
    print_each: bool,
):
    ems, f1s, aics, hits = [], [], [], []
    reduns, ilds, uniqs = [], [], []
    lat_total, lat_retr, lat_mmr, lat_llm = [], [], [], []

    for i, ex in enumerate(ds, 1):
        q = ex["question"]
        gold = ex["answer"]
        gold_docid = ex["docid"]

        # 1) embed query
        t0 = time.perf_counter()
        qvec = embedder.encode(["query: " + q], normalize_embeddings=True)[0].tolist()

        # 2) retrieve candidates (+vectors for diversity/MMR decisions)
        t_retr0 = time.perf_counter()
        cand_docs, cand_vecs = retrieve_docs(
            client, collection, qvec,
            top_k=top_k, fetch_k=fetch_k,
            want_vectors=True
        )
        t_retr1 = time.perf_counter()

        # compute baseline diversity on top_k of candidates
        base_top_docs = cand_docs[:top_k]
        base_top_vecs = None
        if cand_vecs is not None and cand_vecs.shape[0] >= top_k:
            base_top_vecs = cand_vecs[:top_k]
        if base_top_vecs is not None:
            base_redun = redundancy_avg_sim(base_top_vecs)
            base_ild = ild(base_top_vecs)
        else:
            base_redun, base_ild = None, None

        # 3) decide MMR apply
        use_mmr = False
        if mmr_mode == "on":
            use_mmr = True
        elif mmr_mode == "conditional":
            # if cannot compute redundancy, fall back to "off"
            use_mmr = (base_redun is not None and base_redun >= mmr_redun_threshold)

        # 4) MMR rerank (if enabled)
        t_mmr0 = time.perf_counter()
        if use_mmr and cand_vecs is not None:
            qv = np.asarray(qvec, dtype=np.float32)
            qv /= (np.linalg.norm(qv) + 1e-12)
            pick = mmr_select(qv, cand_vecs, top_k=top_k, lam=float(mmr_lambda))
            docs = [cand_docs[j] for j in pick]
            vecs = cand_vecs[pick]
        else:
            docs = base_top_docs
            vecs = base_top_vecs
        t_mmr1 = time.perf_counter()

        # diversity stats on final docs
        if vecs is not None and vecs.shape[0] >= 2:
            rdn = redundancy_avg_sim(vecs)
            ildv = ild(vecs)
        else:
            rdn, ildv = (base_redun if base_redun is not None else 0.0), (base_ild if base_ild is not None else 0.0)

        uniq_docids = len(set([d.get("docid") for d in docs]))
        reduns.append(rdn)
        ilds.append(ildv)
        uniqs.append(uniq_docids / max(1, len(docs)))

        # 5) LLM answer
        t_llm0 = time.perf_counter()
        messages = build_prompt(q, docs)
        resp = llm.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ans = (resp.choices[0].message.content or "").strip()
        t_llm1 = time.perf_counter()

        # metrics
        ems.append(exact_match(ans, gold))
        f1s.append(token_f1(ans, gold))
        aics.append(answer_in_context(gold, docs))

        docids = [d.get("docid") for d in docs]
        hits.append(int(gold_docid in docids))

        # latency
        t1 = time.perf_counter()
        lat_total.append((t1 - t0) * 1000.0)
        lat_retr.append((t_retr1 - t_retr0) * 1000.0)
        lat_mmr.append((t_mmr1 - t_mmr0) * 1000.0)
        lat_llm.append((t_llm1 - t_llm0) * 1000.0)

        if print_each:
            print("=" * 90)
            print(f"[{i}/{len(ds)}] {setting_name}  finish_reason={resp.choices[0].finish_reason}")
            print("Q:", q)
            print("GOLD:", gold)
            print("PRED:", ans)
            print(f"hit@{top_k}={hits[-1]}  AIC={aics[-1]}  redun={rdn:.3f}  uniq_doc={uniq_docids}/{len(docs)}")
            print("docids(top):", docids[:min(5, len(docids))])
            print(f"lat(ms): total={lat_total[-1]:.1f} retr={lat_retr[-1]:.1f} mmr={lat_mmr[-1]:.1f} llm={lat_llm[-1]:.1f}")
            print("=" * 90)

        if i % 20 == 0:
            print(f"[{i}/{len(ds)}] EM={mean(ems):.3f} F1={mean(f1s):.3f} AIC={mean(aics):.3f} "
                  f"hit@{top_k}={mean(hits):.3f} redun={mean(reduns):.3f} lat(ms)={mean(lat_total):.1f}")

    return {
        "setting": setting_name,
        "mmr_mode": mmr_mode,
        "mmr_lambda": (mmr_lambda if mmr_mode != "off" else None),
        "mmr_redun_threshold": (mmr_redun_threshold if mmr_mode == "conditional" else None),
        "EM": mean(ems),
        "F1": mean(f1s),
        "AIC": mean(aics),                      # Answer-in-Context rate
        f"hit@{top_k}": mean(hits),
        "redundancy@k": mean(reduns),
        "ild@k": mean(ilds),
        "unique_doc_ratio@k": mean(uniqs),
        "lat_total_ms": mean(lat_total),
        "lat_retr_ms": mean(lat_retr),
        "lat_mmr_ms": mean(lat_mmr),
        "lat_llm_ms": mean(lat_llm),
        "n": len(ds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant_url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--newsqa", default="newsqa.json")
    ap.add_argument("--limit_qa", type=int, default=200)

    ap.add_argument("--embed_model", required=True)  # must match collection dim
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--fetch_k", type=int, default=30)

    ap.add_argument("--llm_base_url", required=True)  # e.g. http://127.0.0.1:11434/v1
    ap.add_argument("--llm_model", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=128)

    ap.add_argument("--lambdas", default="0.60,0.70,0.80,0.85,0.90,0.95")
    ap.add_argument("--conditional_threshold", type=float, default=0.85)
    ap.add_argument("--print_each", action="store_true")

    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    ds = load_newsqa(args.newsqa, limit=args.limit_qa)

    embedder = SentenceTransformer(args.embed_model, device=args.device)
    llm = OpenAI(base_url=args.llm_base_url, api_key="dummy")

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    results = []

    # 1) Baseline (no MMR)
    results.append(eval_setting(
        setting_name="baseline_topk",
        client=client,
        collection=args.collection,
        embedder=embedder,
        llm=llm,
        llm_model=args.llm_model,
        ds=ds,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        mmr_mode="off",
        mmr_lambda=0.85,
        mmr_redun_threshold=args.conditional_threshold,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        print_each=args.print_each,
    ))

    # 2) MMR always-on sweep
    for lam in lambdas:
        results.append(eval_setting(
            setting_name=f"mmr_on_lam{lam}",
            client=client,
            collection=args.collection,
            embedder=embedder,
            llm=llm,
            llm_model=args.llm_model,
            ds=ds,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            mmr_mode="on",
            mmr_lambda=lam,
            mmr_redun_threshold=args.conditional_threshold,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            print_each=args.print_each,
        ))

    # 3) Conditional MMR sweep
    for lam in lambdas:
        results.append(eval_setting(
            setting_name=f"mmr_cond_lam{lam}_thr{args.conditional_threshold}",
            client=client,
            collection=args.collection,
            embedder=embedder,
            llm=llm,
            llm_model=args.llm_model,
            ds=ds,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            mmr_mode="conditional",
            mmr_lambda=lam,
            mmr_redun_threshold=args.conditional_threshold,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            print_each=args.print_each,
        ))

    df = pd.DataFrame(results)
    out = "mmr_decision_eval.csv"
    df.to_csv(out, index=False)

    show_cols = ["setting","mmr_mode","mmr_lambda","mmr_redun_threshold",
                 "AIC","EM","F1",f"hit@{args.top_k}",
                 "redundancy@k","ild@k","unique_doc_ratio@k",
                 "lat_total_ms","lat_retr_ms","lat_mmr_ms","lat_llm_ms","n"]
    print("\n=== MMR DECISION RESULTS ===")
    print(df[show_cols].sort_values(["AIC","F1","EM"], ascending=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()