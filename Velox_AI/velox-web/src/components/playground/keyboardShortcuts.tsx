// src/components/playground/KeyboardShortcuts.tsx

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Command, X } from 'lucide-react'

const shortcuts = [
  { key: 'Enter', action: 'Send message' },
  { key: 'Shift + Enter', action: 'New line' },
  { key: 'Cmd/Ctrl + K', action: 'Clear chat' },
  { key: 'Cmd/Ctrl + E', action: 'Export chat' },
  { key: 'Esc', action: 'Close shortcuts' },
]

export function KeyboardShortcuts() {
  const [show, setShow] = useState(false)

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setShow(true)}
        className="fixed bottom-4 right-4 z-40"
      >
        <Command className="h-4 w-4 mr-2" />
        Shortcuts
      </Button>

      <AnimatePresence>
        {show && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShow(false)}
              className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md"
            >
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Keyboard Shortcuts</CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShow(false)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {shortcuts.map((shortcut, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="flex items-center justify-between p-2 rounded-lg hover:bg-muted"
                    >
                      <span className="text-sm">{shortcut.action}</span>
                      <kbd className="px-2 py-1 text-xs font-mono bg-muted border rounded">
                        {shortcut.key}
                      </kbd>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}