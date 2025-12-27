import { Bot, User, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';
import type { Message } from '../../../types'; // Ensure you use 'import type'
import { ToolLog } from '../common/ToolLog';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  
  // --- 🛡️ DEFENSIVE CODING START ---
  // If 'content' is accidentally an object (the error you are seeing),
  // we extract the string text from it safely.
  let safeContent = "";
  let safeLogs = message.logs;

  if (typeof message.content === 'object' && message.content !== null) {
    // @ts-ignore - We know it might be the backend response object
    safeContent = message.content.response || JSON.stringify(message.content);
    // @ts-ignore
    if (!safeLogs) safeLogs = message.content.logs;
  } else {
    safeContent = String(message.content);
  }
  // --- 🛡️ DEFENSIVE CODING END ---

  return (
    <motion.div 
      initial={{ opacity: 0, x: isUser ? 20 : -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start max-w-4xl'}`}
    >
      {/* AI Avatar */}
      {!isUser && (
        <div className="w-10 h-10 bg-surface border border-neon-blue/30 rounded flex items-center justify-center shrink-0 shadow-[0_0_15px_rgba(59,130,246,0.2)]">
          <Bot className="w-6 h-6 text-neon-blue" />
        </div>
      )}

      <div className={`space-y-3 ${isUser ? 'max-w-2xl' : 'w-full'}`}>
        
        {/* Text Content Bubble */}
        <div className={`p-5 rounded-lg border ${
          isUser 
            ? 'bg-neon-blue/10 border-neon-blue/50 text-blue-100 rounded-tr-none' 
            : 'bg-surface border-surface text-gray-200 rounded-tl-none shadow-lg'
        }`}>
          <p className="leading-relaxed whitespace-pre-wrap text-sm">
            {/* We render the SAFE string here */}
            {safeContent}
          </p>
        </div>

        {/* Render Tool Logs if they exist */}
        {safeLogs && safeLogs.length > 0 && <ToolLog logs={safeLogs} />}
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