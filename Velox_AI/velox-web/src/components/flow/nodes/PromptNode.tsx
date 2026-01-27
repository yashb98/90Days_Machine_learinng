import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { MessageSquare } from 'lucide-react'
import { PromptNodeData } from '@/types/flow'

export const PromptNode = memo(({ data }: NodeProps) => {
  const nodeData = data as unknown as PromptNodeData
  return (
    <Card className="min-w-[200px] border-2 border-blue-500">
      <Handle type="target" position={Position.Top} className="!bg-blue-500" />
      <CardHeader className="p-3 pb-2">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-blue-500" />
          <CardTitle className="text-sm">Prompt</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <p className="text-xs font-medium">{nodeData.label}</p>
        <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
          {nodeData.systemPrompt}
        </p>
        {nodeData.temperature && (
          <p className="mt-1 text-xs text-muted-foreground">
            Temp: {nodeData.temperature}
          </p>
        )}
      </CardContent>
      <Handle type="source" position={Position.Bottom} className="!bg-blue-500" />
    </Card>
  )
})

PromptNode.displayName = 'PromptNode'

