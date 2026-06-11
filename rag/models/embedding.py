from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

_embedding_client = OpenAI(
	api_key=os.getenv("EMBEDDING_KEY"),
	base_url=os.getenv("EMBEDDING_URL"),
)

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
	"""批量生成 embedding，返回与输入列表顺序一致的 embedding 列表"""
	rsp = _embedding_client.embeddings.create(
		model="text-embedding-v4",
		input=texts,
		dimensions=1024,
		encoding_format="float",
	)
	sorted_data = sorted(rsp.data, key=lambda x: x.index)
	return [item.embedding for item in sorted_data]