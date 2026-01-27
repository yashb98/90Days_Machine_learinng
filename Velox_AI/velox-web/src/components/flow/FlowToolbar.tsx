// src/components/flow/FlowToolbar.tsx

import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  PlayCircle,
  MessageSquare,
  Wrench,
  PhoneForwarded,
  GitBranch,
  StopCircle,
  Save,
} from 'lucide-react'

interface FlowToolbarProps {
  onAddNode: (type: string) => void
  onSave: () => void
}

export function FlowToolbar({ onAddNode, onSave }: FlowToolbarProps) {
  return (
    <Card className="absolute top-4 left-4 z-10 p-2 flex gap-2">
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAddNode('start')}
        className="gap-2"
      >
        <PlayCircle className="h-4 w-4 text-green-500" />
        Start
      </Button>
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAddNode('prompt')}
        className="gap-2"
      >
        <MessageSquare className="h-4 w-4 text-blue-500" />
        Prompt
      </Button>
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAddNode('tool')}
        className="gap-2"
      >
        <Wrench className="h-4 w-4 text-purple-500" />
        Tool
      </Button>
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAddNode('condition')}
        className="gap-2"
      >
        <GitBranch className="h-4 w-4 text-yellow-500" />
        Condition
      </Button>
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAddNode('handoff')}
        className="gap-2"
      >
        <PhoneForwarded className="h-4 w-4 text-orange-500" />
        Handoff
      </Button>
      <Button
        size="sm"
        variant="outline"
        onClick={() => onAddNode('end')}
        className="gap-2"
      >
        <StopCircle className="h-4 w-4 text-red-500" />
        End
      </Button>
      <div className="ml-auto">
        <Button size="sm" onClick={onSave} className="gap-2">
          <Save className="h-4 w-4" />
          Save Flow
        </Button>
      </div>
    </Card>
  )
}