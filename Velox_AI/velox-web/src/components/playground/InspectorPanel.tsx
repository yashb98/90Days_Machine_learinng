// src/components/playground/InspectorPanel.tsx

import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { useState } from 'react' // ✅ Make sure this is imported
import { 
  Activity, 
  Cpu, 
  DollarSign, 
  Clock, 
  Zap,
  Settings,
  Terminal,
  Circle
} from 'lucide-react'

interface InspectorPanelProps {
  agent: any
  messages: any[]
  metrics: {
    totalTokens: number
    totalCost: number
    avgLatency: number
    connected: boolean
  }
}

export function InspectorPanel({ agent, messages, metrics }: InspectorPanelProps) {
  // ✅ All hooks MUST be at the top, before any conditions
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)

  // ✅ Calculations after hooks
  const userMessages = messages.filter(m => m.role === 'user').length
  const assistantMessages = messages.filter(m => m.role === 'assistant').length
  const toolCalls = messages.filter(m => m.role === 'tool').length

  // Event log (last 5 events)
  const eventLog = messages.slice(-5).map((msg, idx) => ({
    time: msg.timestamp,
    event: msg.role === 'user' 
      ? 'User message received'
      : msg.role === 'tool'
      ? 'Tool executed'
      : 'Assistant response',
    details: msg.role === 'tool' 
      ? msg.content.split('(')[0] 
      : msg.content.substring(0, 40) + '...',
  }))

  return (
    <motion.aside
      id="sidebar"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="w-80 border-l border-gray-200 bg-white overflow-y-auto"
    >
      {/* Rest of the component... */}
      <div className="p-6 space-y-6">
        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Circle 
              className={`h-2 w-2 fill-current ${
                metrics.connected ? 'text-green-500' : 'text-red-500'
              }`} 
            />
          </motion.div>
          <span className="text-xs font-medium text-gray-600">
            {metrics.connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {/* Live Configuration */}
        <Card className="border-gray-200">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold text-gray-900 flex items-center gap-2">
              <Settings className="h-4 w-4 text-gray-400" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {agent && (
              <>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">Model</span>
                    <Badge variant="outline" className="font-mono text-xs">
                      Gemini-2.5
                    </Badge>
                  </div>
                </div>

                <Separator className="bg-gray-200" />

                {/* Temperature Slider */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600">Temperature</span>
                    <span className="text-xs font-mono text-gray-900">
                      {temperature}
                    </span>
                  </div>
                  <Slider
                    value={[temperature]}
                    onValueChange={(v) => setTemperature(v[0])}
                    min={0}
                    max={2}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                {/* Max Tokens Slider */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600">Max Tokens</span>
                    <span className="text-xs font-mono text-gray-900">
                      {maxTokens}
                    </span>
                  </div>
                  <Slider
                    value={[maxTokens]}
                    onValueChange={(v) => setMaxTokens(v[0])}
                    min={256}
                    max={4096}
                    step={256}
                    className="w-full"
                  />
                </div>

                <Separator className="bg-gray-200" />

                <div className="space-y-2">
                  <span className="text-xs text-gray-600">Enabled Tools</span>
                  <div className="flex flex-wrap gap-1">
                    {agent.tools_enabled && agent.tools_enabled.length > 0 ? (
                      agent.tools_enabled.map((tool: string) => (
                        <Badge 
                          key={tool} 
                          variant="secondary" 
                          className="text-xs font-normal"
                        >
                          {tool}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-xs text-gray-500">None</span>
                    )}
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Live Metrics */}
        <Card id="stats-card" className="border-gray-200">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold text-gray-900 flex items-center gap-2">
              <Activity className="h-4 w-4 text-gray-400" />
              Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-gray-400" />
                <span className="text-xs text-gray-600">Messages</span>
              </div>
              <span className="text-sm font-mono font-semibold text-gray-900">
                {messages.length}
              </span>
            </div>

            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-gray-400" />
                <span className="text-xs text-gray-600">Avg Latency</span>
              </div>
              <span className="text-sm font-mono font-semibold text-gray-900">
                {metrics.avgLatency > 0 ? `${metrics.avgLatency.toFixed(0)}ms` : '—'}
              </span>
            </div>

            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-gray-400" />
                <span className="text-xs text-gray-600">Total Tokens</span>
              </div>
              <span className="text-sm font-mono font-semibold text-gray-900">
                {metrics.totalTokens.toLocaleString()}
              </span>
            </div>

            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50">
              <div className="flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-gray-400" />
                <span className="text-xs text-gray-600">Est. Cost</span>
              </div>
              <span className="text-sm font-mono font-semibold text-gray-900">
                ${metrics.totalCost.toFixed(4)}
              </span>
            </div>

            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-gray-400" />
                <span className="text-xs text-gray-600">Tool Calls</span>
              </div>
              <span className="text-sm font-mono font-semibold text-gray-900">
                {toolCalls}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Event Log */}
        <Card className="border-gray-200">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold text-gray-900 flex items-center gap-2">
              <Terminal className="h-4 w-4 text-gray-400" />
              Event Log
            </CardTitle>
          </CardHeader>
          <CardContent>
            {eventLog.length === 0 ? (
              <p className="text-xs text-gray-500 text-center py-4">
                No events yet
              </p>
            ) : (
              <div className="space-y-2">
                {eventLog.reverse().map((event, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="text-xs font-mono"
                  >
                    <div className="flex items-start gap-2">
                      <span className="text-gray-400">[{event.time}]</span>
                      <div>
                        <div className="text-gray-700 font-semibold">
                          {event.event}
                        </div>
                        <div className="text-gray-500 mt-0.5">
                          {event.details}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* System Prompt */}
        {agent && (
          <Card className="border-gray-200">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold text-gray-900">
                System Prompt
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-gray-600 bg-gray-50 p-3 rounded-md font-mono max-h-32 overflow-y-auto border border-gray-100">
                {agent.system_prompt}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </motion.aside>
  )
}