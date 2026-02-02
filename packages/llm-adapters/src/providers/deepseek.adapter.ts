import { ChatDeepSeek, ChatDeepSeekInput } from "@langchain/deepseek";
import { LLM } from "@llm/base/llm.base";


export class DeepseekAdapter implements LLM {
  public model: ChatDeepSeek;
  constructor(options: ChatDeepSeekInput) {
    this.model = new ChatDeepSeek(options)
  }
  async *stream(prompt: string): AsyncIterable<string> {
    const response = await this.model.stream(prompt);
    for await (const chunk of response) {
      yield chunk;
    }
  }
}
