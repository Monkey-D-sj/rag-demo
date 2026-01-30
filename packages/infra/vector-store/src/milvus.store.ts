import { MilvusClient, RowData } from '@zilliz/milvus2-sdk-node';

interface VectorSearchResult {
	id: string;
	score: number;
	metadata?: Record<string, any>;
}

export class MilvusVectorStore {
	constructor(
		private readonly client: MilvusClient,
		private readonly collectionName: string,
	) {}

	async upsert(records: RowData[]) {
		await this.client.upsert({
			collection_name: this.collectionName,
			data: records,
		});
	}

	async similaritySearch(
		query: number[],
		topK: number,
	): Promise<VectorSearchResult[]> {
		const res = await this.client.search({
			collection_name: this.collectionName,
			data: [query],
			anns_field: 'vector',
			param: {
				metric_type: 'L2',
				params: {
					nprobe: 10,
				},
			},
			limit: topK,
		});
		return res.results.map((item) => ({
			id: item.id,
			score: item.score,
			metadata: item.entity?.metadata,
		}));
	}
}
