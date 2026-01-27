// src/components/flow/nodes/index.ts

import { StartNode } from './StartNode'
import { PromptNode } from './PromptNode'
import { ToolNode } from './ToolNode'
import { HandoffNode } from './HandoffNode'
import { ConditionNode } from './ConditionNode'
import { EndNode } from './EndNode'

export const nodeTypes = {
  start: StartNode,
  prompt: PromptNode,
  tool: ToolNode,
  handoff: HandoffNode,
  condition: ConditionNode,
  end: EndNode,
}

export { StartNode, PromptNode, ToolNode, HandoffNode, ConditionNode, EndNode }
