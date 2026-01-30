export const systemPrompt = "你是一个有用的助手";
export function buildUserPrompt(input: string) {
  return `用户输入:${input}`;
}
