// components/chat/chat-item.tsx
import { Card } from "@/components/ui/card"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Message } from "./types"
import clsx from "clsx"

export function ChatItem({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div
      className={clsx(
        "flex gap-3",
        isUser && "flex-row-reverse"
      )}
    >
      <Avatar>
        <AvatarFallback>
          {isUser ? "你" : "AI"}
        </AvatarFallback>
      </Avatar>

      <Card className="max-w-[70%] p-3 text-sm whitespace-pre-wrap">
        {message.content}
      </Card>
    </div>
  )
}
