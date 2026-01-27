// src/pages/AgentFlowBuilder.tsx

import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { FlowEditor } from '@/components/flow/FlowEditor'
import { AgentFlow } from '@/types/flow'
import { Button } from '@/components/ui/button'
import { ArrowLeft } from 'lucide-react'
import { toast } from 'sonner'
import  api from '@/lib/api'

export default function AgentFlowBuilder() {
  const { agentId } = useParams()
  const navigate = useNavigate()
  const [initialFlow, setInitialFlow] = useState<AgentFlow>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (agentId) {
      loadAgentFlow()
    } else {
      setLoading(false)
    }
  }, [agentId])

  const loadAgentFlow = async () => {
    try {
      const response = await api.get(`/api/agents/${agentId}`)
      const agent = response.data
      
      // Parse the flow from llm_config.flow
      if (agent.llm_config?.flow) {
        setInitialFlow(agent.llm_config.flow)
      }
    } catch (error) {
      console.error('Failed to load agent flow:', error)
      toast.error('Failed to load agent flow')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async (flow: AgentFlow) => {
    try {
      // Save the flow to the agent's llm_config
      await api.put(`/api/agents/${agentId}`, {
        llm_config: {
          flow,
        },
      })
      
      toast.success('Flow saved successfully!')
    } catch (error) {
      console.error('Failed to save flow:', error)
      toast.error('Failed to save flow')
    }
  }

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <p>Loading flow...</p>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col">
      <div className="border-b p-4 flex items-center gap-4">
        <Button variant="ghost" size="sm" onClick={() => navigate('/agents')}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Agents
        </Button>
        <h1 className="text-xl font-bold">Agent Flow Builder</h1>
      </div>
      <div className="flex-1">
        <FlowEditor initialFlow={initialFlow} onSave={handleSave} />
      </div>
    </div>
  )
}