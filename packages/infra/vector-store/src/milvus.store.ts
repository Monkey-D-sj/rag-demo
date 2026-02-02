import { MilvusClient } from "@zilliz/milvus2-sdk-node"

interface VectorRecord {
	embedding: number[];
	content: string;
	metadata: Record<
		string,
		string | number | string[] | number[] | boolean | unknown
	>;
}

interface VectorSearchResult {
	content: string;
	score: number;
	metadata?: Record<string, any>;
}

export class MilvusVectorStore {
	constructor(
		private readonly client: MilvusClient,
		private readonly collectionName: string,
	) {}

	async upsert(records: VectorRecord[]) {
		await this.client.upsert({
			collection_name: this.collectionName,
			data: records.map((r) => ({
				embedding: r.embedding,
				content: r.content,
				metadata: r.metadata,
			})),
		});
	}

	async similaritySearch(
		queryEmbedding: number[],
		topK: number,
	): Promise<VectorSearchResult[]> {
		const res = await this.client.search({
			collection_name: this.collectionName,
			data: [queryEmbedding],
			top_k: topK,
			output_fields: ['content', 'metadata'],
		});

		return res.results.map((r) => ({
			content: r.content,
			score: r.score,
			metadata: r.metadata,
		}));
	}
}
