import { z } from "zod";

export const ChatRequestSchema = z.object({
  message: z.string(),
  sessionId: z.string().optional()
})

export type ChatRequest = z.infer<typeof ChatRequestSchema>
