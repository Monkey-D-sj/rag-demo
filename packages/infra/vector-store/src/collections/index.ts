import { FieldType } from '@zilliz/milvus2-sdk-node/dist/milvus/types/Collection';
import { CreateIndexSimpleReq } from '@zilliz/milvus2-sdk-node/dist/milvus/types/MilvusIndex';
import { KnowledgeCollection } from './knowledge.collection';

export interface Collection {
	name: string;
	dimension: number;
	fields: FieldType[];
	index: CreateIndexSimpleReq;
}

export const collectionList = [KnowledgeCollection];
export const collectionNameList = collectionList.map(
	(collection) => collection.name,
);
