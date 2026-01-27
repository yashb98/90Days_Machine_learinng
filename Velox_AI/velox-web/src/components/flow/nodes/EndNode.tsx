// src/components/flow/nodes/EndNode.tsx

import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StopCircle } from 'lucide-react'
import { EndNodeData } from '@/types/flow'

export const EndNode = memo(({ data }: NodeProps) => {
  const nodeData = data as unknown as EndNodeData
  return (
    <Card className="min-w-[200px] border-2 border-red-500">
      <Handle type="target" position={Position.Top} className="!bg-red-500" />
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center gap-2">
          <StopCircle className="h-4 w-4 text-red-500" />
          <CardTitle className="text-sm">End</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xs text-muted-foreground">{nodeData.label}</p>
        {nodeData.farewell && (
          <p className="mt-2 text-xs italic">"{nodeData.farewell}"</p>
        )}
      </CardContent>
    </Card>
  )
})

EndNode.displayName = 'EndNode'
