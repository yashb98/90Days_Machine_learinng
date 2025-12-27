import { Bot, User, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../../../types';
import { ToolLog } from '../common/ToolLog';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  
  // 🛡️ Defensive Unpacking (Keep this from Day 4)
  let safeContent = "";
  let safeLogs = message.logs;

  if (typeof message.content === 'object' && message.content !== null) {
    // @ts-ignore
    safeContent = message.content.response || JSON.stringify(message.content);
    // @ts-ignore
    if (!safeLogs) safeLogs = message.content.logs;
  } else {
    safeContent = String(message.content);
  }

  // Detect Error Messages for special styling
  const isError = safeContent.includes("System Error") || safeContent.includes("Traceback");

  return (
    <motion.div 
      initial={{ opacity: 0, x: isUser ? 20 : -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start max-w-4xl'}`}
    >
      {/* AI Avatar */}
      {!isUser && (
        <div className={`w-10 h-10 rounded flex items-center justify-center shrink-0 shadow-[0_0_15px_rgba(0,0,0,0.3)] ${
          isError ? 'bg-red-900/20 border border-red-500' : 'bg-surface border border-neon-blue/30'
        }`}>
          {isError ? <AlertTriangle className="w-5 h-5 text-red-500" /> : <Bot className="w-6 h-6 text-neon-blue" />}
        </div>
      )}

      <div className={`space-y-2 ${isUser ? 'max-w-2xl' : 'w-full min-w-0'}`}>
        
        {/* The Chat Bubble */}
        <div className={`p-5 rounded-lg border ${
          isUser 
            ? 'bg-neon-blue/10 border-neon-blue/50 text-blue-100 rounded-tr-none' 
            : 'bg-surface border-surface text-gray-200 rounded-tl-none shadow-lg'
        }`}>
          {/* Markdown Renderer */}
          <div className={`prose prose-invert prose-sm max-w-none ${isUser ? 'prose-p:text-blue-100' : ''}`}>
             <ReactMarkdown 
               remarkPlugins={[remarkGfm]}
               components={{
                 // Custom styling for tables to fit your theme
                 table: ({node, ...props}) => <div className="overflow-x-auto my-4 border border-gray-700 rounded"><table className="min-w-full divide-y divide-gray-700" {...props} /></div>,
                 th: ({node, ...props}) => <th className="bg-gray-800 px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider" {...props} />,
                 td: ({node, ...props}) => <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-400 border-t border-gray-800" {...props} />,
                 // Style bold text to pop
                 strong: ({node, ...props}) => <strong className="text-neon-blue font-bold" {...props} />
               }}
             >
               {safeContent}
             </ReactMarkdown>
          </div>
        </div>

        {/* The Cyberpunk Audit Log */}
        {!isUser && safeLogs && safeLogs.length > 0 && (
          <ToolLog logs={safeLogs} />
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-10 h-10 bg-gray-700 rounded flex items-center justify-center shrink-0">
          <User className="w-6 h-6 text-gray-300" />
        </div>
      )}
    </motion.div>
  );
}