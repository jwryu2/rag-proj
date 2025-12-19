from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy"  # 실제 키 필요 없음
)

resp = client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[
        {"role": "user", "content": "한국어로 간단히 RAG를 설명해줘"}
    ],
    temperature=0.2,
)

print(resp.choices[0].message.content)