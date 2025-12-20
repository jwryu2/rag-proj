import os
import re
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean
from collections import Counter

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# -------------------------
# newsqa loader
# -------------------------
def parse_qa_pair(qa_pair):
    if isinstance(qa_pair, dict):
        return qa_pair
    if isinstance(qa_pair, str):
        return json.loads(qa_pair)
    raise TypeError(type(qa_pair))

def load_newsqa(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
                print("[SKIP newsqa]", i, e)
    if limit:
        out = out[:limit]
    print(f"[newsqa] loaded={len(out)} skipped={skipped}")
    return out


# -------------------------
# normalization & metrics (NO EM/F1)
# -------------------------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "")
    return s

REFUSAL_PATTERNS = [
    r"\b모르겠", r"\b모른", r"\b알 수 없", r"\b확인할 수 없", r"\b근거(가)? 부족",
    r"\b제공된 문서(로|로는) 알 수 없", r"\b정보가 없",
]

def is_refusal(ans: str) -> int:
    a = norm(ans)
    if not a:
        return 1
    return int(any(re.search(p, a) for p in REFUSAL_PATTERNS))

def answer_in_context(gold: str, docs: List[Dict[str, Any]]) -> int:
    g = norm(gold)
    if not g:
        return 0
    ctx = norm("\n".join([d.get("text", "") for d in docs]))
    return int(g in ctx)

def jaccard_tokens(a: str, b: str) -> float:
    A = set(norm(a).split())
    B = set(norm(b).split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# -------------------------
# prompt (keyword-only)
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
# retrieval
# -------------------------
def retrieve_docs(
    client: QdrantClient,
    collection: str,
    qvec: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    res = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=top_k,
        with_payload=True,
    )
    docs = []
    for p in res.points:
        pl = p.payload or {}
        docs.append({
            "docid": pl.get("docid"),
            "title": pl.get("title", ""),
            "text": pl.get("text", ""),
            "score": float(p.score),
        })
    return docs


# -------------------------
# main eval for one config
# -------------------------
def eval_one(
    ds: List[Dict[str, Any]],
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    llm: OpenAI,
    llm_model: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
    n_gen: int,          # stability check: generate n times per question
    print_each: bool,
):
    aics, refusals, out_lens, lat_total = [], [], [], []
    stability = []  # average pairwise jaccard among n_gen answers (higher => more stable)

    for i, ex in enumerate(ds, 1):
        q = ex["question"]
        gold = ex["answer"]

        t0 = time.perf_counter()

        # embed query (keep consistent with your retriever setting; e5 style prefix)
        qvec = embedder.encode(["query: " + q], normalize_embeddings=True)[0].tolist()

        # retrieve
        docs = retrieve_docs(client, collection, qvec, top_k=top_k)

        # AIC (independent of generation)
        aic = answer_in_context(gold, docs)
        aics.append(aic)

        # generate n times to measure stability
        answers = []
        for _ in range(max(1, n_gen)):
            messages = build_prompt(q, docs)
            resp = llm.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ans = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
            answers.append(ans)

        # refusal / output length
        # use first answer for refusal/len, and n_gen for stability
        ans0 = answers[0]
        refusals.append(is_refusal(ans0))
        out_lens.append(len(ans0))

        # stability: mean pairwise jaccard
        if len(answers) >= 2:
            sims = []
            for a in range(len(answers)):
                for b in range(a + 1, len(answers)):
                    sims.append(jaccard_tokens(answers[a], answers[b]))
            stability.append(float(np.mean(sims)))
        else:
            stability.append(1.0)

        t1 = time.perf_counter()
        lat_total.append((t1 - t0) * 1000.0)

        if print_each:
            print("=" * 90)
            print(f"[{i}/{len(ds)}] temp={temperature} max_tokens={max_tokens} n_gen={n_gen}")
            print("Q:", q)
            print("GOLD:", gold)
            print("AIC:", aic, "REFUSAL:", refusals[-1], "LEN:", out_lens[-1], "STAB:", f"{stability[-1]:.3f}")
            print("ANS:", ans0)
            print("=" * 90)

        if i % 20 == 0:
            print(f"[{i}/{len(ds)}] AIC={mean(aics):.3f} refusal={mean(refusals):.3f} "
                  f"len={mean(out_lens):.1f} stab={mean(stability):.3f} lat(ms)={mean(lat_total):.1f}")

    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "AIC": mean(aics),
        "refusal_rate": mean(refusals),
        "avg_answer_chars": mean(out_lens),
        "stability": mean(stability),
        "latency_ms_avg": mean(lat_total),
        "n": len(ds),
        "top_k": top_k,
        "n_gen": n_gen,
        "collection": collection,
        "llm_model": llm_model,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant_url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", required=True)

    ap.add_argument("--embed_model", required=True)   # must match collection dim
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--newsqa", default="newsqa.json")
    ap.add_argument("--limit_qa", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10)

    ap.add_argument("--llm_base_url", required=True)  # e.g. http://127.0.0.1:11434/v1
    ap.add_argument("--llm_model", required=True)

    ap.add_argument("--temps", default="0.0,0.1,0.2,0.4")
    ap.add_argument("--max_tokens_list", default="16,32,64,128")

    ap.add_argument("--n_gen", type=int, default=1)   # set 3 to measure stability more meaningfully
    ap.add_argument("--print_each", action="store_true")
    args = ap.parse_args()

    ds = load_newsqa(args.newsqa, limit=args.limit_qa)
    client = QdrantClient(url=args.qdrant_url)
    embedder = SentenceTransformer(args.embed_model, device=args.device)
    llm = OpenAI(base_url=args.llm_base_url, api_key="dummy")

    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    max_tokens_list = [int(x) for x in args.max_tokens_list.split(",") if x.strip()]

    results = []
    for t in temps:
        for mt in max_tokens_list:
            print(f"\n=== Eval temp={t} max_tokens={mt} ===")
            results.append(eval_one(
                ds=ds,
                client=client,
                collection=args.collection,
                embedder=embedder,
                llm=llm,
                llm_model=args.llm_model,
                top_k=args.top_k,
                temperature=t,
                max_tokens=mt,
                n_gen=args.n_gen,
                print_each=args.print_each,
            ))

    df = pd.DataFrame(results)

    out = "genparam_eval_noemf1.csv"
    df.to_csv(out, index=False)

    # sort suggestion: maximize AIC, then minimize refusal (or keep moderate), then minimize latency
    # Depending on your preference you may want refusal not too low (avoid hallucination).
    print("\n=== RESULTS (sorted) ===")
    show = ["temperature","max_tokens","AIC","refusal_rate","avg_answer_chars","stability","latency_ms_avg","n_gen","top_k"]
    print(df[show].sort_values(["AIC","stability","latency_ms_avg"], ascending=[False, False, True]))
    print("Saved:", out)


if __name__ == "__main__":
    main()