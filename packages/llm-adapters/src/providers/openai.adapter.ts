import { ChatOpenAI } from "@langchain/openai";
import { LLMBase } from "@llm/base/llm.base";
import { OpenAIChatInput } from "@langchain/openai/dist/types";

export class OpenaiAdapter extends LLMBase {
  public model: ChatOpenAI;
  constructor(options: OpenAIChatInput) {
    super()
    this.model = new ChatOpenAI(options);
  }
}
