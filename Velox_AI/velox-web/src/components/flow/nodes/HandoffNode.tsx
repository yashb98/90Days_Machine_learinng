import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowRightLeft } from 'lucide-react'
import { HandoffNodeData } from '@/types/flow'

export const HandoffNode = memo(({ data }: NodeProps) => {
  const nodeData = data as unknown as HandoffNodeData
  return (
    <Card className="min-w-[200px] border-2 border-purple-500">
      <Handle type="target" position={Position.Top} className="!bg-purple-500" />
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center gap-2">
          <ArrowRightLeft className="h-4 w-4 text-purple-500" />
          <CardTitle className="text-sm">Handoff</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xs font-medium">{nodeData.label}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Target: {nodeData.target}
        </p>
        {nodeData.reason && (
          <p className="mt-1 text-xs text-muted-foreground italic">
            "{nodeData.reason}"
          </p>
        )}
      </CardContent>
      <Handle type="source" position={Position.Bottom} className="!bg-purple-500" />
    </Card>
  )
})

HandoffNode.displayName = 'HandoffNode'

