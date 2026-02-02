import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { Collection } from './collections';

export class MilvusMigrator {
	constructor(private readonly client: MilvusClient) {}

	async migrate(collection: Collection) {
		const { value } = await this.client.hasCollection({
			collection_name: collection.name,
		});

		if (value) return;

		await this.client.createCollection({
			collection_name: collection.name,
			fields: collection.fields,
		});

		await this.client.createIndex(collection.index);
	}
}
