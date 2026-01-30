import { DataType } from '@zilliz/milvus2-sdk-node';
import { Collection } from './index';

export const KnowledgeCollection: Collection = {
	name: 'knowledge_chunks',
	dimension: 1536,

	fields: [
		{
			name: 'id',
			data_type: DataType.Int64,
			is_primary_key: true,
			autoID: true,
		},
		{
			name: 'embedding',
			data_type: DataType.FloatVector,
			dim: 1536,
		},
		{
			name: 'doc_id',
			data_type: DataType.VarChar,
			max_length: 64,
		},
		{
			name: 'content',
			data_type: DataType.VarChar,
			max_length: 4096,
		},
		{
			name: 'metadata',
			data_type: DataType.JSON,
		},
	],

	index: {
		collection_name: 'knowledge_chunks',
		field_name: 'embedding',
		index_type: 'HNSW',
		metric_type: 'COSINE',
		params: {
			M: 16,
			efConstruction: 200,
		},
	},
};
