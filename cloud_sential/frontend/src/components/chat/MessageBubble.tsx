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
  
  // 🛡️ Defensive Unpacking
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

  // Detect Error Messages
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
                 // 1. HEADINGS (Structure)
                 h1: ({node, ...props}) => <h1 className="text-xl font-bold text-neon-blue mb-4 mt-6 border-b border-gray-700 pb-2" {...props} />,
                 h2: ({node, ...props}) => <h2 className="text-lg font-semibold text-blue-300 mb-3 mt-5" {...props} />,
                 h3: ({node, ...props}) => <h3 className="text-md font-medium text-purple-300 mb-2 mt-4" {...props} />,

                 // 2. PARAGRAPHS (Spacing)
                 // 'last:mb-0' prevents extra space at the bottom of the bubble
                 p: ({node, ...props}) => <p className="mb-4 last:mb-0 leading-relaxed" {...props} />,

                 // 3. LISTS (Bullets & Numbers)
                 ul: ({node, ...props}) => <ul className="list-disc list-outside pl-5 mb-4 space-y-1 text-gray-300" {...props} />,
                 ol: ({node, ...props}) => <ol className="list-decimal list-outside pl-5 mb-4 space-y-1 text-gray-300" {...props} />,
                 li: ({node, ...props}) => <li className="pl-1" {...props} />,

                 // 4. CODE BLOCKS (Cyberpunk styling)
                 code: ({node, ...props}) => {
                    // Check if it's an inline code snippet or a full block
                    // @ts-ignore - The types for react-markdown props are complex, ignore for simplicity
                    const isInline = !props.className; 
                    return isInline 
                      ? <code className="bg-gray-800 px-1.5 py-0.5 rounded text-neon-pink font-mono text-sm border border-gray-700" {...props} />
                      : <code {...props} /> // Let pre handle the block code
                 },
                 pre: ({node, ...props}) => (
                    <div className="relative my-4">
                      <pre className="bg-[#0d1117] p-4 rounded-lg overflow-x-auto border border-gray-700 text-sm font-mono leading-tight scrollbar-thin scrollbar-thumb-gray-600" {...props} />
                    </div>
                 ),

                 // 5. EXISTING CUSTOMIZATIONS
                 table: ({node, ...props}) => <div className="overflow-x-auto my-4 border border-gray-700 rounded"><table className="min-w-full divide-y divide-gray-700" {...props} /></div>,
                 th: ({node, ...props}) => <th className="bg-gray-800 px-3 py-2 text-left text-xs font-medium text-gray-300 uppercase tracking-wider" {...props} />,
                 td: ({node, ...props}) => <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-400 border-t border-gray-800" {...props} />,
                 strong: ({node, ...props}) => <strong className="text-neon-blue font-bold" {...props} />,
                 blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-neon-blue pl-4 italic text-gray-400 my-4" {...props} />,
                 a: ({node, ...props}) => <a className="text-neon-blue hover:underline hover:text-blue-400 transition-colors" target="_blank" rel="noopener noreferrer" {...props} />
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