import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Wrench } from 'lucide-react'
import { ToolNodeData } from '@/types/flow'

export const ToolNode = memo(({ data }: NodeProps) => {
  const nodeData = data as unknown as ToolNodeData
  return (
    <Card className="min-w-[200px] border-2 border-orange-500">
      <Handle type="target" position={Position.Top} className="!bg-orange-500" />
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center gap-2">
          <Wrench className="h-4 w-4 text-orange-500" />
          <CardTitle className="text-sm">Tool</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xs font-medium">{nodeData.label}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          {nodeData.toolName}
        </p>
      </CardContent>
      <Handle type="source" position={Position.Bottom} className="!bg-orange-500" />
    </Card>
  )
})

ToolNode.displayName = 'ToolNode'

