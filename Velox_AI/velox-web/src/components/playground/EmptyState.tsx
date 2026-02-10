// src/components/playground/EmptyState.tsx

import { motion } from 'framer-motion'
import { MessageSquare, Zap, Brain, Wrench } from 'lucide-react'

const features = [
  {
    icon: MessageSquare,
    title: 'Natural Conversations',
    description: 'Test realistic dialogues',
  },
  {
    icon: Brain,
    title: 'RAG Integration',
    description: 'See knowledge retrieval',
  },
  {
    icon: Wrench,
    title: 'Tool Execution',
    description: 'Watch tools in action',
  },
  {
    icon: Zap,
    title: 'Instant Feedback',
    description: 'Real-time responses',
  },
]

export function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full p-8" id="chat-area">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center max-w-2xl"
      >
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            repeatType: 'reverse',
          }}
          className="text-6xl mb-6"
        >
          🤖
        </motion.div>

        <h2 className="text-3xl font-bold mb-3">Ready to Test Your Agent?</h2>
        <p className="text-lg text-muted-foreground mb-8">
          Start a conversation below and see your AI agent in action
        </p>

        <motion.div
          initial="hidden"
          animate="visible"
          variants={{
            hidden: { opacity: 0 },
            visible: {
              opacity: 1,
              transition: { staggerChildren: 0.1 },
            },
          }}
          className="grid grid-cols-2 gap-4 max-w-md mx-auto"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={{
                hidden: { opacity: 0, y: 20 },
                visible: { opacity: 1, y: 0 },
              }}
              className="p-4 rounded-lg border bg-card hover:shadow-md transition-shadow"
            >
              <feature.icon className="h-6 w-6 text-primary mb-2" />
              <h4 className="font-semibold text-sm mb-1">{feature.title}</h4>
              <p className="text-xs text-muted-foreground">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-8 text-sm text-muted-foreground"
        >
          💡 Try asking: "What's the status of order 123?" or "Do you have gaming mice in stock?"
        </motion.div>
      </motion.div>
    </div>
  )
}