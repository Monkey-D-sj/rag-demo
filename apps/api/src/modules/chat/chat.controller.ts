// chat.controller.ts
import {
  Controller,
  Post,
  Req,
  Res,
} from '@nestjs/common'
import { Request, Response } from 'express'
import {DeepseekAdapter} from "@llm/providers/deepseek.adapter";
import 'dotenv/config'

@Controller('chat')
export class ChatController {
  @Post()
  async chat(@Req() req: Request, @Res() res: Response) {
    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Cache-Control', 'no-cache')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders()
    console.log(req)

    // const llm = new DeepseekAdapter({
    //   apiKey: process.env.DEEPSEEK_API_KEY,
    //   model: 'deepseek-chat',
    // })
    // llm.stream(req.body)
    const chunks = ['你好', '，这是', ' NestJS', ' SSE', ' 流式响应']

    for (const chunk of chunks) {
      res.write(
        `data: ${JSON.stringify({ content: chunk })}\n\n`
      )
      await this.sleep(500)
    }

    res.write('data: [DONE]\n\n')
    res.end()
  }

  private sleep(ms: number) {
    return new Promise((r) => setTimeout(r, ms))
  }
}
