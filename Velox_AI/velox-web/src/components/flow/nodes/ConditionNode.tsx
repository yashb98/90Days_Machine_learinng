// src/components/flow/nodes/ConditionNode.tsx

import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { GitBranch } from 'lucide-react'
import { ConditionNodeData } from '@/types/flow'

export const ConditionNode = memo(({ data }: NodeProps) => {
  const nodeData = data as unknown as ConditionNodeData
  return (
    <Card className="min-w-[200px] border-2 border-yellow-500">
      <Handle type="target" position={Position.Top} className="!bg-yellow-500" />
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-yellow-500" />
          <CardTitle className="text-sm">Condition</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xs font-medium">{nodeData.label}</p>
        <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
          {nodeData.condition}
        </p>
      </CardContent>
      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        className="!bg-green-500 !left-1/4"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        className="!bg-red-500 !left-3/4"
      />
    </Card>
  )
})

ConditionNode.displayName = 'ConditionNode'
