// src/types/flow.ts

export type NodeType = 'start' | 'prompt' | 'tool' | 'handoff' | 'condition' | 'end'

export interface BaseNodeData {
  label: string
  description?: string
}

export interface StartNodeData extends BaseNodeData {
  greeting?: string
}

export interface PromptNodeData extends BaseNodeData {
  systemPrompt: string
  temperature?: number
  maxTokens?: number
}

export interface ToolNodeData extends BaseNodeData {
  toolName: string
  toolConfig: Record<string, unknown>
}

export interface HandoffNodeData extends BaseNodeData {
  target: string // Phone number or agent ID
  reason?: string
}

export interface ConditionNodeData extends BaseNodeData {
  condition: string
  trueLabel?: string
  falseLabel?: string
}

export interface EndNodeData extends BaseNodeData {
  farewell?: string
}

export type FlowNodeData = 
  | StartNodeData 
  | PromptNodeData 
  | ToolNodeData 
  | HandoffNodeData 
  | ConditionNodeData 
  | EndNodeData

export interface AgentFlow {
  nodes: Array<{
    id: string
    type: string
    position: { x: number; y: number }
    data: Record<string, unknown>
  }>
  edges: Array<{
    id: string
    source: string
    target: string
    label?: string
  }>
}
