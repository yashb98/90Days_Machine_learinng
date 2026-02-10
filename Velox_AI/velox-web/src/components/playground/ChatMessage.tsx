// src/components/playground/ChatMessage.tsx

import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Bot, User, Wrench, Copy, Check, Clock, Cpu } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useState } from 'react'

interface ChatMessageProps {
  role: 'user' | 'assistant' | 'tool'
  content: string
  timestamp?: string
  latency?: number
  tokens?: number
}

export function ChatMessage({ role, content, timestamp, latency, tokens }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const isUser = role === 'user'
  const isTool = role === 'tool'

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'flex gap-4 group',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <Avatar className={cn(
        'h-8 w-8 shrink-0 border',
        isTool ? 'bg-purple-100 border-purple-200' : 'border-gray-200'
      )}>
        <AvatarFallback className={isTool ? 'bg-purple-100' : 'bg-gray-100'}>
          {isUser ? (
            <User className="h-4 w-4 text-gray-600" />
          ) : isTool ? (
            <Wrench className="h-4 w-4 text-purple-600" />
          ) : (
            <Bot className="h-4 w-4 text-gray-600" />
          )}
        </AvatarFallback>
      </Avatar>
      
      {/* Message Content */}
      <div className={cn(
        'flex flex-col gap-2 max-w-[80%]',
        isUser ? 'items-end' : 'items-start'
      )}>
        {/* Header with metadata */}
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="font-medium">
            {isUser ? 'You' : isTool ? 'Tool' : 'Assistant'}
          </span>
          {timestamp && <span>·</span>}
          {timestamp && <span>{timestamp}</span>}
          {latency && (
            <>
              <span>·</span>
              <div className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                <span>{latency}ms</span>
              </div>
            </>
          )}
          {tokens && (
            <>
              <span>·</span>
              <div className="flex items-center gap-1">
                <Cpu className="h-3 w-3" />
                <span>{tokens} tokens</span>
              </div>
            </>
          )}
        </div>

        {/* Message Bubble */}
        <div className={cn(
          'rounded-xl px-4 py-3 relative group/message',
          isUser
            ? 'bg-gray-900 text-white'
            : isTool
            ? 'bg-purple-50 border border-purple-200 text-purple-900'
            : 'bg-gray-100 text-gray-900'
        )}>
          {/* Tool Label */}
          {isTool && (
            <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-purple-700">
              <Wrench className="h-3 w-3" />
              Tool Execution
            </div>
          )}

          {/* Content */}
          <div className="text-sm whitespace-pre-wrap break-words">
            {isTool ? (
              <div className="font-mono text-xs">
                {content}
              </div>
            ) : (
              content
            )}
          </div>

          {/* Copy Button */}
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              "absolute top-2 right-2 h-6 w-6 p-0 opacity-0 group-hover/message:opacity-100 transition-opacity",
              isUser ? "text-white hover:bg-gray-800" : "text-gray-600 hover:bg-gray-200"
            )}
            onClick={handleCopy}
          >
            {copied ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  )
}