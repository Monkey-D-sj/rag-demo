import { ChatRequest } from '@ai-rag/contracts'
import { IsOptional, IsString } from 'class-validator'

export class ChatDto implements ChatRequest {
  @IsString()
  message!: string

  @IsOptional()
  sessionId?: string
}
