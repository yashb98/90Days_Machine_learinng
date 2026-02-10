// src/pages/Playground.tsx

import { useState, useRef, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ChatMessage } from '@/components/playground/ChatMessage'
import { EnterpriseInput } from '@/components/playground/EnterpriseInput'
import { TutorialOverlay } from '@/components/playground/TutorialOverlay'
import { EnterpriseEmptyState } from '@/components/playground/EnterpriseEmptyState'
import { InspectorPanel } from '@/components/playground/InspectorPanel'
import { TypingIndicator } from '@/components/playground/TypingIndicator'
import { Trash2, Download, ArrowLeft, Play, Zap } from 'lucide-react'
import { toast } from 'sonner'
import api from '@/lib/api'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: string
  timestamp: string
  latency?: number
  tokens?: number
}

export default function Playground() {
  const { agentId } = useParams()
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [agent, setAgent] = useState<any>(null)
  const [showTutorial, setShowTutorial] = useState(false)
  const [metrics, setMetrics] = useState({
    totalTokens: 0,
    totalCost: 0,
    avgLatency: 0,
    connected: true,
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const hasSeenTutorial = localStorage.getItem('playground_tutorial_completed')
    if (!hasSeenTutorial) {
      setShowTutorial(true)
    }

    if (agentId) {
      loadAgent()
    }

    // Keyboard shortcuts
    const handleKeyPress = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        handleClearChat()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'e') {
        e.preventDefault()
        handleExportChat()
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [agentId])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const loadAgent = async () => {
    try {
      const response = await api.get(`/api/agents/${agentId}`)
      setAgent(response.data)
      
      if (response.data.greeting) {
        setMessages([{
          id: '1',
          role: 'assistant',
          content: response.data.greeting,
          timestamp: new Date().toLocaleTimeString(),
        }])
      }
    } catch (error) {
      console.error('Failed to load agent:', error)
      toast.error('Failed to load agent')
      setMetrics(prev => ({ ...prev, connected: false }))
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSendMessage = async (content: string) => {
    const startTime = Date.now()
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toLocaleTimeString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const response = await api.post(`/api/playground/${agentId}/message`, {
        message: content,
        conversation_history: messages.map(m => ({
          role: m.role,
          content: m.content,
        })),
      })

      const { response: aiResponse, tool_calls, tokens = 0 } = response.data
      const latency = Date.now() - startTime

      if (tool_calls && tool_calls.length > 0) {
        for (const toolCall of tool_calls) {
          await new Promise(resolve => setTimeout(resolve, 300))
          const toolMessage: Message = {
            id: `tool-${Date.now()}-${Math.random()}`,
            role: 'tool',
            content: `${toolCall.name}(${JSON.stringify(toolCall.args, null, 2)})\n\nResult: ${JSON.stringify(toolCall.result, null, 2)}`,
            timestamp: new Date().toLocaleTimeString(),
          }
          setMessages((prev) => [...prev, toolMessage])
        }
      }

      await new Promise(resolve => setTimeout(resolve, 300))
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: aiResponse,
        timestamp: new Date().toLocaleTimeString(),
        latency,
        tokens,
      }

      setMessages((prev) => [...prev, assistantMessage])
      
      // Update metrics
      setMetrics(prev => ({
        ...prev,
        totalTokens: prev.totalTokens + tokens,
        totalCost: prev.totalCost + (tokens * 0.00002),
        avgLatency: ((prev.avgLatency * messages.length) + latency) / (messages.length + 1),
      }))
    } catch (error: any) {
      console.error('Failed to send message:', error)
      toast.error('Failed to send message')
      
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your message.',
        timestamp: new Date().toLocaleTimeString(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleClearChat = () => {
    if (messages.length === 0) return
    
    if (confirm('Clear conversation? This cannot be undone.')) {
      setMessages([])
      setMetrics({
        totalTokens: 0,
        totalCost: 0,
        avgLatency: 0,
        connected: true,
      })
      toast.success('Conversation cleared')
    }
  }

  const handleExportChat = () => {
    if (messages.length === 0) {
      toast.error('No messages to export')
      return
    }

    const chatData = JSON.stringify(messages, null, 2)
    const blob = new Blob([chatData], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chat-${agentId}-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Exported successfully')
  }

  const handleTemplateSelect = (template: any) => {
    handleSendMessage(template.prompt)
  }

  const handleTutorialComplete = () => {
    localStorage.setItem('playground_tutorial_completed', 'true')
    toast.success('Ready to test!')
  }

  const handleTutorialSkip = () => {
    localStorage.setItem('playground_tutorial_completed', 'true')
  }

  return (
    <>
      {showTutorial && (
        <TutorialOverlay
          onComplete={handleTutorialComplete}
          onSkip={handleTutorialSkip}
        />
      )}

      {/* Enterprise Layout - Unified Light Theme */}
      <div className="h-screen flex flex-col bg-slate-50">
        {/* Compressed Header */}
        <motion.header
          id="playground-header"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="h-14 border-b border-gray-200 bg-white px-6 flex items-center justify-between"
        >
          {/* Left: Navigation */}
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" asChild className="h-8">
              <Link to="/" className="gap-2 text-gray-600 hover:text-gray-900">
                <ArrowLeft className="h-4 w-4" />
                <span className="text-sm font-medium">Back</span>
              </Link>
            </Button>

            <div className="h-4 w-px bg-gray-300" />

            <div className="flex items-center gap-2">
              <Play className="h-4 w-4 text-gray-400" />
              <h1 className="text-sm font-semibold text-gray-900">Playground</h1>
            </div>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-2" id="action-buttons">
            {agent && (
              <Badge variant="outline" className="gap-1 font-normal">
                <Zap className="h-3 w-3" />
                {agent.name}
              </Badge>
            )}

            <Button
              variant="ghost"
              size="sm"
              onClick={handleExportChat}
              disabled={messages.length === 0}
              className="h-8 text-gray-600"
            >
              <Download className="h-4 w-4 mr-1.5" />
              Export
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearChat}
              disabled={messages.length === 0}
              className="h-8 text-gray-600"
            >
              <Trash2 className="h-4 w-4 mr-1.5" />
              Clear
            </Button>
          </div>
        </motion.header>

        {/* Main Workspace */}
        <div className="flex-1 flex overflow-hidden">
          {/* Central Canvas - White Sheet */}
          <div className="flex-1 flex flex-col relative">
            <div className="flex-1 overflow-y-auto px-6">
              <AnimatePresence mode="wait">
                {messages.length === 0 ? (
                  <EnterpriseEmptyState
                    key="empty"
                    onTemplateSelect={handleTemplateSelect}
                  />
                ) : (
                  <motion.div
                    key="messages"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="max-w-4xl mx-auto py-8 space-y-6"
                  >
                    {messages.map((message, index) => (
                      <motion.div
                        key={message.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.03 }}
                      >
                        <ChatMessage
                          role={message.role}
                          content={message.content}
                          timestamp={message.timestamp}
                          latency={message.latency}
                          tokens={message.tokens}
                        />
                      </motion.div>
                    ))}

                    <AnimatePresence>
                      {loading && <TypingIndicator key="typing" />}
                    </AnimatePresence>

                    <div ref={messagesEndRef} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Floating Input Bar */}
            <div className="px-6 pb-8">
              <div className="max-w-4xl mx-auto">
                <EnterpriseInput
                  onSend={handleSendMessage}
                  disabled={loading}
                  showSuggestions={messages.length === 0}
                />
              </div>
            </div>
          </div>

          {/* Inspector Panel (Sidebar) */}
          <InspectorPanel
            agent={agent}
            messages={messages}
            metrics={metrics}
          />
        </div>
      </div>
    </>
  )
}