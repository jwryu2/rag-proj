from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name="__smoke_test__",
    vectors_config=VectorParams(size=3, distance=Distance.COSINE),
)

pts = [
    PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"x": 1}),
    PointStruct(id=2, vector=[0.0, 1.0, 0.0], payload={"x": 2}),
]
client.upsert(collection_name="__smoke_test__", points=pts, wait=True)

res = client.query_points(
    collection_name="__smoke_test__",
    query=[1.0, 0.0, 0.0],
    limit=2,
    with_payload=True,
)
print([p for p in res])