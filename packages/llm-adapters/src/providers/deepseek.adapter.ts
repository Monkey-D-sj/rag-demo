import { ChatDeepSeek, ChatDeepSeekInput } from "@langchain/deepseek";
import { LLMBase } from "@llm/base/llm.base";


export class DeepseekAdapter extends LLMBase {
  public model: ChatDeepSeek;
  constructor(options: ChatDeepSeekInput) {
    super()
    this.model = new ChatDeepSeek(options)
  }
}
