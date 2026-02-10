// src/components/playground/AgentInfoSidebar.tsx

import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Bot, Settings, Activity, Code } from 'lucide-react'

interface AgentInfoSidebarProps {
  agent: any
  messageCount: number
  userMessageCount: number
  assistantMessageCount: number
  toolCallCount: number
}

export function AgentInfoSidebar({
  agent,
  messageCount,
  userMessageCount,
  assistantMessageCount,
  toolCallCount,
}: AgentInfoSidebarProps) {
  const stats = [
    { label: 'Total Messages', value: messageCount, icon: Activity },
    { label: 'User', value: userMessageCount, icon: Bot },
    { label: 'Assistant', value: assistantMessageCount, icon: Bot },
    { label: 'Tool Calls', value: toolCallCount, icon: Settings },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      className="w-80 border-l bg-muted/30 p-4 overflow-y-auto space-y-4"
      id="sidebar"
    >
      {/* Agent Info Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Bot className="h-4 w-4 text-primary" />
            Agent Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          {agent && (
            <>
              <div>
                <p className="font-medium mb-2 flex items-center gap-2">
                  <Code className="h-3 w-3" />
                  System Prompt
                </p>
                <div className="text-xs text-muted-foreground bg-muted p-3 rounded-md max-h-32 overflow-y-auto">
                  {agent.system_prompt}
                </div>
              </div>

              <Separator />

              <div>
                <p className="font-medium mb-2 flex items-center gap-2">
                  <Settings className="h-3 w-3" />
                  Enabled Tools
                </p>
                <div className="flex flex-wrap gap-1">
                  {agent.tools_enabled && agent.tools_enabled.length > 0 ? (
                    agent.tools_enabled.map((tool: string) => (
                      <Badge key={tool} variant="outline" className="text-xs">
                        🔧 {tool}
                      </Badge>
                    ))
                  ) : (
                    <p className="text-xs text-muted-foreground">No tools enabled</p>
                  )}
                </div>
              </div>

              <Separator />

              <div>
                <p className="font-medium mb-2">LLM Configuration</p>
                <pre className="text-xs bg-muted p-2 rounded overflow-x-auto">
                  {JSON.stringify(agent.llm_config || {}, null, 2)}
                </pre>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Stats Card */}
      <Card id='stats-card'>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            Session Statistics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <stat.icon className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">{stat.label}</span>
              </div>
              <motion.span
                key={stat.value}
                initial={{ scale: 1.5, color: '#3b82f6' }}
                animate={{ scale: 1, color: 'inherit' }}
                className="font-mono font-bold"
              >
                {stat.value}
              </motion.span>
            </motion.div>
          ))}
        </CardContent>
      </Card>

      {/* Tips Card */}
      <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
          <CardTitle className="text-sm">💡 Pro Tips</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground space-y-2">
          <p>• Try edge cases to find weaknesses</p>
          <p>• Test tool calls with specific IDs</p>
          <p>• Export logs to analyze patterns</p>
          <p>• Refine prompts based on results</p>
        </CardContent>
      </Card>
    </motion.div>
  )
}