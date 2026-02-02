export abstract class LLMBase {
  private model: any;
  async *stream(prompt: string): AsyncIterable<string> {
    const response = await this.model.stream(prompt);
    for await (const chunk of response) {
      yield chunk;
    }
  }
}
