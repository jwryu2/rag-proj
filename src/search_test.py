import os
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def load_cfg(path="configs/app.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg(os.environ.get("APP_CONFIG", "configs/app.yaml"))
    top_k = int(cfg.get("top_k", 5))

    qdrant_url = cfg["qdrant"]["url"]
    collection = cfg["qdrant"]["collection"]

    embed_model = cfg["embedding"]["model_name"]
    device = cfg["embedding"].get("device", "cpu")

    client = QdrantClient(url=qdrant_url)
    embedder = SentenceTransformer(embed_model, device=device)

    query = input("Query> ").strip()
    qvec = embedder.encode(["query: " + query], normalize_embeddings=True)[0].tolist()

    hits = client.query_points(
        collection_name=collection,
        query=qvec,      # dense vector
        limit=top_k,
        with_payload=True,
    )

    # query_points는 응답이 hits.points에 들어있음
    print("\n=== Top-k Results ===")
    for rank, p in enumerate(hits.points, start=1):
        payload = p.payload or {}
        score = p.score
        title = payload.get("title", "")
        summary = payload.get("summary", "")
        text = payload.get("text", "")

        print(f"\n[{rank}] score={score:.4f}")
        if title:
            print(f"Title: {title}")
        if summary:
            print(f"Summary: {summary[:180]}{'...' if len(summary)>180 else ''}")
        print(f"Chunk: {text[:300]}{'...' if len(text)>300 else ''}")


if __name__ == "__main__":
    main()