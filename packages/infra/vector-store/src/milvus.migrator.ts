import { Collection, collectionList } from "./collections";

export class MilvusMigrator {
  async migrateAll() {
    for (const collection of collectionList) {
      await this.ensureCollection(collection)
    }
  }

  private async ensureCollection(collection: Collection) {
    const { value } = await this.client.hasCollection({
      collection_name: collection.name
    })

    if (!value) {
      await this.client.createCollection(...)
      await this.client.createIndex(...)
    }
  }
}
