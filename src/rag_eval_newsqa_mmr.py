import json
import re
import time
from collections import Counter
from statistics import mean
import pandas as pd

from rag import rag_answer

NEWSQA_PATH = "newsqa.json"
LIMIT = 200           # 먼저 200, 이후 1000
TOP_K = 10
FETCH_K = 30
LAMBDAS = [None, 0.70, 0.85, 0.95]


def parse_qa_pair(qa_pair):
    if isinstance(qa_pair, dict):
        return qa_pair
    if isinstance(qa_pair, str):
        return json.loads(qa_pair)
    raise TypeError(f"Unexpected qa_pair type: {type(qa_pair)}")


def load_newsqa(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    skipped = 0
    for idx, ex in enumerate(data):
        try:
            qa = parse_qa_pair(ex["qa_pair"])
            items.append({
                "qid": ex.get("qid"),
                "docid": ex.get("docid"),
                "question": qa["question"],
                "answer": qa["answer"],
            })
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"[SKIP] idx={idx} qid={ex.get('qid')} reason={e}")

    print(f"[newsqa] loaded={len(items)} skipped={skipped}")
    return items


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


def eval_setting(ds, rerank, mmr_lambda):
    ems, f1s, lats, hits = [], [], [], []

    for i, ex in enumerate(ds, 1):
        t0 = time.perf_counter()
        ans, docs = rag_answer(
            ex["question"],
            top_k=TOP_K,
            fetch_k=FETCH_K,
            rerank=rerank,
            mmr_lambda=mmr_lambda if mmr_lambda is not None else 0.85,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        lats.append(dt)

        ems.append(exact_match(ans, ex["answer"]))
        f1s.append(token_f1(ans, ex["answer"]))

        # hit@k: payload에 docid가 있으면 의미 있음
        docids = [d.get("docid") for d in (docs or [])]
        hits.append(int(ex["docid"] in docids))

        if i % 20 == 0:
            print(f"[{i}/{len(ds)}] EM={mean(ems):.3f} F1={mean(f1s):.3f} hit@{TOP_K}={mean(hits):.3f} lat(ms)={mean(lats):.1f}")

    return {
        "method": "baseline" if rerank is None else f"mmr(lam={mmr_lambda})",
        "rerank": rerank or "baseline",
        "mmr_lambda": mmr_lambda,
        "EM": mean(ems),
        "F1": mean(f1s),
        f"hit@{TOP_K}": mean(hits),
        "latency_ms_avg": mean(lats),
        "n": len(ds),
    }


def main():
    ds = load_newsqa(NEWSQA_PATH)[:LIMIT]

    results = []
    for lam in LAMBDAS:
        if lam is None:
            results.append(eval_setting(ds, rerank=None, mmr_lambda=None))
        else:
            results.append(eval_setting(ds, rerank="mmr", mmr_lambda=lam))

    df = pd.DataFrame(results)
    print("\n=== FINAL NEWSQA RAG EVAL (Baseline vs MMR) ===")
    print(df[["method", "EM", "F1", f"hit@{TOP_K}", "latency_ms_avg"]])

    out_csv = "eval_newsqa_mmr.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()