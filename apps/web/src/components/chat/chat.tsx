// components/chat/chat.tsx
"use client"

import { useState, useRef } from "react"
import { Message } from "./types"
import { ChatList } from "./chat-list"
import { ChatInput } from "./chat-input"

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  async function sendMessage(content: string) {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content,
    }

    const aiMsg: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
    }

    setMessages((prev) => [...prev, userMsg, aiMsg])
    setLoading(true)

    controllerRef.current?.abort()
    controllerRef.current = new AbortController()

    const res = await fetch("http://localhost:3001/chat/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [userMsg],
      }),
      signal: controllerRef.current.signal,
    })

    if (!res.body) return

    const reader = res.body.getReader()
    const decoder = new TextDecoder("utf-8")

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split("\n")

      for (const line of lines) {
        if (!line.startsWith("data:")) continue

        const data = line.replace("data:", "").trim()
        if (data === "[DONE]") {
          setLoading(false)
          return
        }

        try {
          const json = JSON.parse(data)
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id
                ? { ...m, content: m.content + (json.content ?? "") }
                : m
            )
          )
        } catch {}
      }
    }

    setLoading(false)
  }

  return (
    <div className="flex h-full flex-col">
      <ChatList messages={messages} loading={loading} />
      <ChatInput onSend={sendMessage} disabled={loading} />
    </div>
  )
}
