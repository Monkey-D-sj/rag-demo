// components/chat/chat-input.tsx
"use client"

import { useState } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"

export function ChatInput({
                            onSend,
                            disabled,
                          }: {
  onSend: (text: string) => void
  disabled?: boolean
}) {
  const [value, setValue] = useState("")

  function submit() {
    if (!value.trim()) return
    onSend(value)
    setValue("")
  }

  return (
    <div className="border-t p-4 flex gap-2">
      <Textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="输入你的问题..."
        rows={2}
        disabled={disabled}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
      />
      <Button onClick={submit} disabled={disabled}>
        发送
      </Button>
    </div>
  )
}
