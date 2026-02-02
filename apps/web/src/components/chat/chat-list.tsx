// components/chat/chat-list.tsx
"use client"

import { useEffect, useRef } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Message } from "./types"
import { ChatItem } from "./chat-item"
import { Skeleton } from "@/components/ui/skeleton"

interface Props {
  messages: Message[]
  loading: boolean
}

export function ChatList({ messages, loading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="space-y-4">
        {messages.map((msg) => (
          <ChatItem key={msg.id} message={msg} />
        ))}

        {loading && (
          <div className="flex gap-2">
            <Skeleton className="h-4 w-4 rounded-full" />
            <Skeleton className="h-4 w-32" />
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  )
}
