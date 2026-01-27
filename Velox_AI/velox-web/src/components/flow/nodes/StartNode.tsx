// src/components/flow/nodes/StartNode.tsx

import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { PlayCircle } from 'lucide-react'
import { StartNodeData } from '@/types/flow'

export const StartNode = memo(({ data }: NodeProps) => {
  const nodeData = data as unknown as StartNodeData
  return (
    <Card className="min-w-[200px] border-2 border-green-500">
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center gap-2">
          <PlayCircle className="h-4 w-4 text-green-500" />
          <CardTitle className="text-sm">Start</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xs text-muted-foreground">{nodeData.label}</p>
        {nodeData.greeting && (
          <p className="mt-2 text-xs italic">"{nodeData.greeting}"</p>
        )}
      </CardContent>
      <Handle type="source" position={Position.Bottom} className="!bg-green-500" />
    </Card>
  )
})

StartNode.displayName = 'StartNode'

