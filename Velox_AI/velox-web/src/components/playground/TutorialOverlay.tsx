// src/components/playground/TutorialOverlay.tsx

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { X, ChevronRight, ChevronLeft, Sparkles } from 'lucide-react'

interface TutorialStep {
  title: string
  description: string
  icon: string
  target: string // ID of element to highlight
  position: 'top' | 'bottom' | 'left' | 'right'
}

const tutorialSteps: TutorialStep[] = [
  {
    title: 'Welcome to the Playground! 🎮',
    description: 'This is your safe space to test and refine your AI agent before deploying it to production. Let\'s take a quick tour!',
    icon: '👋',
    target: 'playground-header',
    position: 'bottom',
  },
  {
    title: 'Agent Information 🤖',
    description: 'This shows which agent you\'re testing. Each agent can have different personalities, tools, and capabilities.',
    icon: '🏷️',
    target: 'agent-badge',
    position: 'bottom',
  },
  {
    title: 'Quick Actions ⚡',
    description: 'Export your conversations for analysis, or clear the chat to start fresh. Use keyboard shortcuts for faster workflow!',
    icon: '🎯',
    target: 'action-buttons',
    position: 'bottom',
  },
  {
    title: 'Chat Input Area ✍️',
    description: 'Type your message here. Press Enter to send, or Shift+Enter for a new line. Try the suggested questions below!',
    icon: '💬',
    target: 'chat-input',
    position: 'top',
  },
  {
    title: 'Message Suggestions 💡',
    description: 'Click these example questions to quickly test your agent. They\'re designed to showcase different features.',
    icon: '✨',
    target: 'message-suggestions',
    position: 'top',
  },
  {
    title: 'Agent Configuration 📋',
    description: 'View your agent\'s system prompt, enabled tools, and LLM settings. This helps you understand how your agent behaves.',
    icon: '⚙️',
    target: 'sidebar',
    position: 'left',
  },
  {
    title: 'Live Statistics 📊',
    description: 'Monitor conversation metrics in real-time: message counts, tool usage, and more. Perfect for analyzing agent performance.',
    icon: '📈',
    target: 'stats-card',
    position: 'left',
  },
  {
    title: 'Ready to Test! 🚀',
    description: 'Now you\'re ready to start testing! Try different scenarios, edge cases, and see how your agent responds. Happy testing!',
    icon: '🎉',
    target: 'chat-area',
    position: 'top',
  },
]

interface TutorialOverlayProps {
  onComplete: () => void
  onSkip: () => void
}

export function TutorialOverlay({ onComplete, onSkip }: TutorialOverlayProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [show, setShow] = useState(true)
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 })

  const step = tutorialSteps[currentStep]

  useEffect(() => {
    if (!show) return

    // Calculate tooltip position based on target element
    const targetElement = document.getElementById(step.target)
    if (targetElement) {
      const rect = targetElement.getBoundingClientRect()
      let top = 0
      let left = 0

      switch (step.position) {
        case 'bottom':
          top = rect.bottom + 20
          left = rect.left + rect.width / 2
          break
        case 'top':
          top = rect.top - 20
          left = rect.left + rect.width / 2
          break
        case 'left':
          top = rect.top + rect.height / 2
          left = rect.left - 20
          break
        case 'right':
          top = rect.top + rect.height / 2
          left = rect.right + 20
          break
      }

      setTooltipPosition({ top, left })

      // Scroll target into view
      targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [currentStep, show, step])

  const handleNext = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(prev => prev + 1)
    } else {
      handleComplete()
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1)
    }
  }

  const handleComplete = () => {
    setShow(false)
    onComplete()
  }

  const handleSkipTutorial = () => {
    setShow(false)
    onSkip()
  }

  if (!show) return null

  // Get target element for spotlight
  const targetElement = document.getElementById(step.target)
  const targetRect = targetElement?.getBoundingClientRect()

  return (
    <AnimatePresence>
      {show && (
        <>
          {/* Dark Backdrop with Spotlight Cutout */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 pointer-events-none"
            style={{
              background: targetRect
                ? `radial-gradient(
                    circle at ${targetRect.left + targetRect.width / 2}px ${targetRect.top + targetRect.height / 2}px,
                    transparent ${Math.max(targetRect.width, targetRect.height) / 2 + 10}px,
                    rgba(0, 0, 0, 0.7) ${Math.max(targetRect.width, targetRect.height) / 2 + 50}px
                  )`
                : 'rgba(0, 0, 0, 0.7)',
            }}
          />

          {/* Highlight Border */}
          {targetRect && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="fixed z-50 pointer-events-none"
              style={{
                top: targetRect.top - 4,
                left: targetRect.left - 4,
                width: targetRect.width + 8,
                height: targetRect.height + 8,
                border: '3px solid rgb(59, 130, 246)',
                borderRadius: '12px',
                boxShadow: '0 0 20px rgba(59, 130, 246, 0.5)',
              }}
            />
          )}

          {/* Arrow Pointer */}
          {targetRect && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="fixed z-50 pointer-events-none"
              style={{
                top:
                  step.position === 'bottom'
                    ? targetRect.bottom + 10
                    : step.position === 'top'
                    ? targetRect.top - 40
                    : targetRect.top + targetRect.height / 2 - 15,
                left:
                  step.position === 'left'
                    ? targetRect.left - 40
                    : step.position === 'right'
                    ? targetRect.right + 10
                    : targetRect.left + targetRect.width / 2 - 15,
              }}
            >
              <motion.div
                animate={{
                  y: step.position === 'bottom' ? [0, -10, 0] : step.position === 'top' ? [0, 10, 0] : 0,
                  x: step.position === 'left' ? [0, 10, 0] : step.position === 'right' ? [0, -10, 0] : 0,
                }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="text-4xl"
              >
                {step.position === 'bottom' && '⬆️'}
                {step.position === 'top' && '⬇️'}
                {step.position === 'left' && '➡️'}
                {step.position === 'right' && '⬅️'}
              </motion.div>
            </motion.div>
          )}

          {/* Tutorial Card - Positioned near target */}
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="fixed z-[60] w-full max-w-md px-4 pointer-events-auto"
            style={{
              top:
                step.position === 'bottom'
                  ? Math.min(tooltipPosition.top + 50, window.innerHeight - 400)
                  : step.position === 'top'
                  ? Math.max(tooltipPosition.top - 350, 100)
                  : tooltipPosition.top - 150,
              left: Math.max(Math.min(tooltipPosition.left - 200, window.innerWidth - 450), 20),
              transform: 'none',
            }}
          >
            <Card className="border-2 border-primary shadow-2xl bg-white dark:bg-slate-900">
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="text-5xl">{step.icon}</div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleSkipTutorial}
                    className="h-8 w-8 p-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                <h3 className="text-2xl font-bold mb-3 text-slate-900 dark:text-slate-100">
                  {step.title}
                </h3>
                <p className="text-slate-600 dark:text-slate-400 mb-6">{step.description}</p>

                {/* Progress dots */}
                <div className="flex items-center gap-2 mb-6">
                  {tutorialSteps.map((_, index) => (
                    <div
                      key={index}
                      className={`h-2 rounded-full transition-all ${
                        index === currentStep
                          ? 'w-8 bg-primary'
                          : index < currentStep
                          ? 'w-2 bg-primary/50'
                          : 'w-2 bg-slate-300 dark:bg-slate-700'
                      }`}
                    />
                  ))}
                </div>

                {/* Navigation */}
                <div className="flex items-center justify-between">
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    Step {currentStep + 1} of {tutorialSteps.length}
                  </div>
                  <div className="flex gap-2">
                    {currentStep > 0 && (
                      <Button variant="outline" onClick={handlePrevious}>
                        <ChevronLeft className="h-4 w-4 mr-1" />
                        Back
                      </Button>
                    )}
                    <Button onClick={handleNext}>
                      {currentStep < tutorialSteps.length - 1 ? (
                        <>
                          Next
                          <ChevronRight className="h-4 w-4 ml-1" />
                        </>
                      ) : (
                        <>
                          Start Testing
                          <Sparkles className="h-4 w-4 ml-1" />
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}