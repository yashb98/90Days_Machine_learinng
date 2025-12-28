import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, ChevronDown, CheckCircle, Server } from 'lucide-react';

interface ToolCall {
  tool: string;
  args: any;
}

interface ToolLogProps {
  logs: ToolCall[];
}

export function ToolLog({ logs }: ToolLogProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!logs || logs.length === 0) return null;

  return (
    <div className="mt-3 border-t border-dashed border-gray-700 pt-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-mono text-gray-500 hover:text-neon-blue transition-colors w-full text-left"
      >
        <Terminal className="w-3 h-3" />
        <span>SYSTEM_AUDIT_TRACE ({logs.length} OPS)</span>
        <ChevronDown className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-2 bg-black/40 rounded p-3 font-mono text-xs border border-gray-800">
              {logs.map((log, idx) => (
                <div key={idx} className="mb-3 last:mb-0">
                  <div className="flex items-center gap-2 text-neon-blue mb-1">
                    <CheckCircle className="w-3 h-3" />
                    <span className="font-bold uppercase"> EXEC: {log.tool}</span>
                  </div>
                  <div className="pl-5 text-gray-400 break-all">
                    <span className="text-gray-600">ARGS:</span> {JSON.stringify(log.args)}
                  </div>
                </div>
              ))}
              <div className="mt-2 pt-2 border-t border-gray-800 text-neon-green flex items-center gap-2">
                <Server className="w-3 h-3" />
                <span>PROCESS_COMPLETE</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}