// src/components/playground/ChatInput.tsx

import { useState, useRef, KeyboardEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Send, Loader2, Lightbulb } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  placeholder?: string
}

const suggestions = [
  "What's the status of order 123?",
  "Do you have gaming mice in stock?",
  "I need to return a damaged product",
  "What are your business hours?",
  "Check inventory for wireless keyboards",
]

export function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSend(message.trim())
      setMessage('')
      setShowSuggestions(false)
      textareaRef.current?.focus()
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion)
    setShowSuggestions(false)
    textareaRef.current?.focus()
  }

  return (
    <div className="border-t bg-background">
      {/* Suggestions */}
      <AnimatePresence>
        {showSuggestions && !message && (
          <motion.div
            id="message-suggestions"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="px-4 pt-3 overflow-hidden"
          >
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-4 w-4 text-yellow-500" />
              <span className="text-xs text-muted-foreground font-medium">
                Try these questions:
              </span>
            </div>
            <div className="flex flex-wrap gap-2 pb-2">
              {suggestions.map((suggestion, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Badge
                    variant="outline"
                    className="cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
                    onClick={() => handleSuggestionClick(suggestion)}
                  >
                    {suggestion}
                  </Badge>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input Area */}
      <div className="flex gap-2 p-4">
        <div className="flex-1 relative">
          <Textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setShowSuggestions(true)}
            placeholder={placeholder || "Type your message... (Shift + Enter for new line)"}
            disabled={disabled}
            className="min-h-[60px] max-h-[200px] resize-none pr-12"
          />
          
          {/* Character count */}
          {message.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute bottom-2 right-2 text-xs text-muted-foreground"
            >
              {message.length}
            </motion.div>
          )}
        </div>

        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            onClick={handleSend}
            disabled={disabled || !message.trim()}
            size="icon"
            className="h-[60px] w-[60px] rounded-xl"
          >
            {disabled ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </Button>
        </motion.div>
      </div>

      {/* Helper Text */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="px-4 pb-3 text-xs text-muted-foreground flex items-center justify-between"
      >
        <span>Press Enter to send, Shift + Enter for new line</span>
        {disabled && (
          <span className="flex items-center gap-2 text-primary">
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="h-2 w-2 rounded-full bg-primary"
            />
            AI is thinking...
          </span>
        )}
      </motion.div>
    </div>
  )
}