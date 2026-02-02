import { MilvusClient } from "@zilliz/milvus2-sdk-node"
import 'dotenv/config'

export function createMilvusClient() {
	return new MilvusClient({
		address: process.env.MILVUS_ADDR!,
	});
}
