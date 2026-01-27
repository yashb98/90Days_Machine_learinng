// src/components/flow/NodePropertiesPanel.tsx

import { useState, useEffect } from 'react'
import { Node } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { X } from 'lucide-react'
import {
  StartNodeData,
  PromptNodeData,
  ToolNodeData,
  HandoffNodeData,
  ConditionNodeData,
  EndNodeData,
} from '@/types/flow'

interface NodePropertiesPanelProps {
  node: Node
  onUpdate: (data: any) => void
  onClose: () => void
}

export function NodePropertiesPanel({ node, onUpdate, onClose }: NodePropertiesPanelProps) {
  const [formData, setFormData] = useState<Record<string, unknown>>(node.data)

  useEffect(() => {
    setFormData(node.data)
  }, [node])

  const handleChange = (field: string, value: any) => {
    const newData = { ...formData, [field]: value }
    setFormData(newData)
    onUpdate(newData)
  }

  const getTypedData = <T,>(): T => formData as T

  const renderFields = () => {
    switch (node.type) {
      case 'start':
        const startData = getTypedData<StartNodeData>()
        return (
          <>
            <div className="space-y-2">
              <Label>Label</Label>
              <Input
                value={startData.label || ''}
                onChange={(e) => handleChange('label', e.target.value)}
                placeholder="Conversation start"
              />
            </div>
            <div className="space-y-2">
              <Label>Greeting Message</Label>
              <Textarea
                value={startData.greeting || ''}
                onChange={(e) => handleChange('greeting', e.target.value)}
                placeholder="Hello! How can I help you today?"
                rows={3}
              />
            </div>
          </>
        )

      case 'prompt':
        const promptData = getTypedData<PromptNodeData>()
        return (
          <>
            <div className="space-y-2">
              <Label>Label</Label>
              <Input
                value={promptData.label || ''}
                onChange={(e) => handleChange('label', e.target.value)}
                placeholder="Customer service prompt"
              />
            </div>
            <div className="space-y-2">
              <Label>System Prompt</Label>
              <Textarea
                value={promptData.systemPrompt || ''}
                onChange={(e) => handleChange('systemPrompt', e.target.value)}
                placeholder="You are a helpful customer service agent..."
                rows={5}
              />
            </div>
            <div className="space-y-2">
              <Label>Temperature ({promptData.temperature || 0.7})</Label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={promptData.temperature || 0.7}
                onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <Label>Max Tokens</Label>
              <Input
                type="number"
                value={promptData.maxTokens || 1000}
                onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
                placeholder="1000"
              />
            </div>
          </>
        )

      case 'tool':
        const toolData = getTypedData<ToolNodeData>()
        return (
          <>
            <div className="space-y-2">
              <Label>Label</Label>
              <Input
                value={toolData.label || ''}
                onChange={(e) => handleChange('label', e.target.value)}
                placeholder="Check order status"
              />
            </div>
            <div className="space-y-2">
              <Label>Tool Name</Label>
              <Select
                value={toolData.toolName || ''}
                onValueChange={(value) => handleChange('toolName', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a tool" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="check_order_status">Check Order Status</SelectItem>
                  <SelectItem value="check_inventory">Check Inventory</SelectItem>
                  <SelectItem value="book_appointment">Book Appointment</SelectItem>
                  <SelectItem value="process_refund">Process Refund</SelectItem>
                  <SelectItem value="update_customer_info">Update Customer Info</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Tool Configuration (JSON)</Label>
              <Textarea
                value={JSON.stringify(toolData.toolConfig || {}, null, 2)}
                onChange={(e) => {
                  try {
                    const config = JSON.parse(e.target.value)
                    handleChange('toolConfig', config)
                  } catch (err) {
                    // Invalid JSON, ignore
                  }
                }}
                placeholder='{"timeout": 5000}'
                rows={4}
                className="font-mono text-xs"
              />
            </div>
          </>
        )

      case 'handoff':
        const handoffData = getTypedData<HandoffNodeData>()
        return (
          <>
            <div className="space-y-2">
              <Label>Label</Label>
              <Input
                value={handoffData.label || ''}
                onChange={(e) => handleChange('label', e.target.value)}
                placeholder="Transfer to human"
              />
            </div>
            <div className="space-y-2">
              <Label>Target (Phone/Agent ID)</Label>
              <Input
                value={handoffData.target || ''}
                onChange={(e) => handleChange('target', e.target.value)}
                placeholder="+1234567890 or agent-id"
              />
            </div>
            <div className="space-y-2">
              <Label>Reason</Label>
              <Textarea
                value={handoffData.reason || ''}
                onChange={(e) => handleChange('reason', e.target.value)}
                placeholder="Customer requested human assistance"
                rows={2}
              />
            </div>
          </>
        )

      case 'condition':
        const conditionData = getTypedData<ConditionNodeData>()
        return (
          <>
            <div className="space-y-2">
              <Label>Label</Label>
              <Input
                value={conditionData.label || ''}
                onChange={(e) => handleChange('label', e.target.value)}
                placeholder="Check user sentiment"
              />
            </div>
            <div className="space-y-2">
              <Label>Condition Expression</Label>
              <Textarea
                value={conditionData.condition || ''}
                onChange={(e) => handleChange('condition', e.target.value)}
                placeholder="sentiment_score > 0.5"
                rows={3}
              />
              <p className="text-xs text-muted-foreground">
                Available variables: sentiment_score, user_message, context
              </p>
            </div>
            <div className="space-y-2">
              <Label>True Path Label</Label>
              <Input
                value={conditionData.trueLabel || 'Yes'}
                onChange={(e) => handleChange('trueLabel', e.target.value)}
                placeholder="Yes"
              />
            </div>
            <div className="space-y-2">
              <Label>False Path Label</Label>
              <Input
                value={conditionData.falseLabel || 'No'}
                onChange={(e) => handleChange('falseLabel', e.target.value)}
                placeholder="No"
              />
            </div>
          </>
        )

      case 'end':
        const endData = getTypedData<EndNodeData>()
        return (
          <>
            <div className="space-y-2">
              <Label>Label</Label>
              <Input
                value={endData.label || ''}
                onChange={(e) => handleChange('label', e.target.value)}
                placeholder="Conversation end"
              />
            </div>
            <div className="space-y-2">
              <Label>Farewell Message</Label>
              <Textarea
                value={endData.farewell || ''}
                onChange={(e) => handleChange('farewell', e.target.value)}
                placeholder="Thank you for calling! Have a great day!"
                rows={3}
              />
            </div>
          </>
        )

      default:
        return null
    }
  }

  const nodeType = node.type || 'Unknown'

  return (
    <Card className="w-80 h-full overflow-auto">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          {nodeType.charAt(0).toUpperCase() + nodeType.slice(1)} Node Properties
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">{renderFields()}</CardContent>
    </Card>
  )
}
