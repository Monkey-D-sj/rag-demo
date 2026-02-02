// components/chat/types.ts
export type Role = "user" | "assistant"

export interface Message {
  id: string
  role: Role
  content: string
}
