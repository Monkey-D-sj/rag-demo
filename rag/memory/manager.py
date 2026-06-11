import math

from rag.models.embedding import generate_embeddings_batch


class MemoryManager:
	"""内存管理器 — 基于向量相似度的记忆存储与检索"""

	def __init__(self):
		self._memories: list[dict] = []

	def add(self, text: str, metadata: dict | None = None) -> None:
		"""添加一条记忆，自动生成 embedding"""
		embeddings = generate_embeddings_batch([text])
		self._memories.append({
			"text": text,
			"embedding": embeddings[0],
			"metadata": metadata or {},
		})

	def search(self, query: str, top_k: int = 5) -> list[dict]:
		"""根据查询文本搜索最相关的记忆，返回 top_k 条"""
		if not self._memories:
			return []

		query_embeddings = generate_embeddings_batch([query])
		query_vec = query_embeddings[0]

		scored = []
		for mem in self._memories:
			score = self._cosine_similarity(query_vec, mem["embedding"])
			scored.append({
				"text": mem["text"],
				"score": score,
				"metadata": mem["metadata"],
			})

		scored.sort(key=lambda x: x["score"], reverse=True)
		return scored[:top_k]

	@staticmethod
	def _cosine_similarity(a: list[float], b: list[float]) -> float:
		"""计算两个向量的余弦相似度"""
		dot = sum(x * y for x, y in zip(a, b))
		norm_a = math.sqrt(sum(x * x for x in a))
		norm_b = math.sqrt(sum(y * y for y in b))
		if norm_a == 0 or norm_b == 0:
			return 0.0
		return dot / (norm_a * norm_b)
