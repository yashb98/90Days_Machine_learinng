// src/components/flow/FlowEditor.tsx

import { useCallback, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Node,
  NodeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { nodeTypes } from './nodes'
import { FlowToolbar } from './FlowToolbar'
import { NodePropertiesPanel } from './NodePropertiesPanel'
import { AgentFlow } from '@/types/flow'

interface FlowEditorProps {
  initialFlow?: AgentFlow
  onSave?: (flow: AgentFlow) => void
}

export function FlowEditor({ initialFlow, onSave }: FlowEditorProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialFlow?.nodes || [])
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialFlow?.edges || [])
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge(connection, eds))
    },
    [setEdges]
  )

  const onNodeClick = useCallback((_: any, node: Node) => {
    setSelectedNode(node)
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedNode(null)
  }, [])

  const addNode = useCallback(
    (type: string) => {
      const newNode = {
        id: `${type}-${Date.now()}`,
        type,
        position: { x: 250, y: 100 + nodes.length * 100 },
        data: {
          label: `New ${type} node`,
        },
      }
      setNodes((nds) => [...nds, newNode])
    },
    [nodes.length, setNodes]
  )

  const handleSave = useCallback(() => {
    const flow: AgentFlow = { 
      nodes: nodes as AgentFlow['nodes'], 
      edges: edges as AgentFlow['edges'] 
    }
    onSave?.(flow)
  }, [nodes, edges, onSave])

  const updateNodeData = useCallback(
    (nodeId: string, data: any) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node
        )
      )
    },
    [setNodes]
  )

  return (
    <div className="flex h-full">
      <div className="flex-1 relative">
        <FlowToolbar onAddNode={addNode} onSave={handleSave} />
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes as unknown as NodeTypes}
          fitView
          className="bg-background"
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>
      {selectedNode && (
        <NodePropertiesPanel
          node={selectedNode}
          onUpdate={(data) => updateNodeData(selectedNode.id, data)}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  )
}
