import { Body, Controller, Post, Res, } from '@nestjs/common'
import { Response } from 'express'
import 'dotenv/config'
import { ChatDto } from "./dto/chat.dto";
import { DeepseekAdapter } from "@ai-rag/llm-adapters";
import { ChatService } from "./chat.service";

@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {
  }
  @Post()
  async chat(@Body() body: ChatDto, @Res() res: Response) {
    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Cache-Control', 'no-cache')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders()

    const llm = new DeepseekAdapter({
      apiKey: process.env.DEEPSEEK_API_KEY,
      model: 'deepseek-chat',
    })
    const stream = llm.stream(body.message)
    for await (const message of stream) {
      res.write(
        `data: ${JSON.stringify({ content: message })}\n\n`
      )
    }

    res.write('data: [DONE]\n\n')
    res.end()
  }
}
