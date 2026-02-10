// src/components/playground/EnterpriseEmptyState.tsx

import { motion } from 'framer-motion'
import { Card } from '@/components/ui/card'
import { 
  FileText, 
  Search, 
  MessageSquare, 
  Zap,
  ArrowRight 
} from 'lucide-react'

interface Template {
  id: string
  title: string
  description: string
  prompt: string
  icon: any
  category: string
}

const templates: Template[] = [
  {
    id: '1',
    title: 'Test Document Retrieval',
    description: 'Query knowledge base with RAG',
    prompt: 'Search our documentation for information about return policies and summarize the key points.',
    icon: Search,
    category: 'RAG',
  },
  {
    id: '2',
    title: 'Simulate Customer Call',
    description: 'Test natural conversation flow',
    prompt: 'Hi, I placed an order last week (order #12345) and I haven\'t received a shipping confirmation yet. Can you help?',
    icon: MessageSquare,
    category: 'Conversation',
  },
  {
    id: '3',
    title: 'Tool Integration Check',
    description: 'Verify external API calls',
    prompt: 'Check the current inventory status for wireless keyboards and let me know which models are in stock.',
    icon: Zap,
    category: 'Tools',
  },
  {
    id: '4',
    title: 'Multi-Step Workflow',
    description: 'Test complex scenarios',
    prompt: 'I want to return a damaged product (order #98765), get a refund, and then place a new order for a replacement item.',
    icon: FileText,
    category: 'Complex',
  },
]

interface EnterpriseEmptyStateProps {
  onTemplateSelect: (template: Template) => void
}

export function EnterpriseEmptyState({ onTemplateSelect }: EnterpriseEmptyStateProps) {
  return (
    <motion.div
      id="chat-area"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex items-center justify-center min-h-full py-12"
    >
      <div className="max-w-4xl w-full px-4">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-semibold text-gray-900 mb-3">
            Start Testing Your Agent
          </h2>
          <p className="text-base text-gray-600 max-w-2xl mx-auto">
            Select a test scenario below or write your own prompt to simulate conversations
          </p>
        </motion.div>

        {/* Template Grid */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={{
            hidden: { opacity: 0 },
            visible: {
              opacity: 1,
              transition: { staggerChildren: 0.08 }
            }
          }}
          className="grid md:grid-cols-2 gap-4"
        >
          {templates.map((template, index) => (
            <motion.div
              key={template.id}
              variants={{
                hidden: { opacity: 0, y: 20 },
                visible: { opacity: 1, y: 0 }
              }}
            >
              <Card
                className="p-5 border border-gray-200 hover:border-gray-300 hover:shadow-md transition-all cursor-pointer group bg-white"
                onClick={() => onTemplateSelect(template)}
              >
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className="h-10 w-10 rounded-lg bg-gray-100 flex items-center justify-center shrink-0 group-hover:bg-gray-200 transition-colors">
                    <template.icon className="h-5 w-5 text-gray-600" />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-gray-900 text-sm">
                        {template.title}
                      </h3>
                      <span className="text-xs text-gray-500 font-medium">
                        {template.category}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-3">
                      {template.description}
                    </p>
                    
                    {/* Preview Prompt */}
                    <div className="bg-gray-50 rounded-md p-3 text-xs text-gray-600 font-mono leading-relaxed border border-gray-100">
                      {template.prompt.length > 100 
                        ? `${template.prompt.substring(0, 100)}...` 
                        : template.prompt}
                    </div>

                    {/* Action Hint */}
                    <div className="mt-3 flex items-center gap-1 text-xs text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span>Click to run</span>
                      <ArrowRight className="h-3 w-3" />
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Footer Tip */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-8 text-center"
        >
          <p className="text-sm text-gray-500">
            💡 Pro tip: Use the templates above to quickly test specific features, or type your own scenario below
          </p>
        </motion.div>
      </div>
    </motion.div>
  )
}