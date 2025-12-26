import { useEffect, useRef } from 'react';
import { Bot, Cpu } from 'lucide-react';
import { Sidebar } from './components/layout/Sidebar';
import { MessageBubble } from './components/chat/MessageBubble';
import { ChatInput } from './components/chat/ChatInput';
import { useChat } from '../hooks/useChat'; 

function App() {
  const { messages, isLoading, sendMessage } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">
      <Sidebar />
      
      <main className="flex-1 flex flex-col h-screen bg-terminal-bg font-mono relative">
        {/* Background Grid Effect */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(18,22,33,0)_1px,transparent_1px),linear-gradient(90deg,rgba(18,22,33,0)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>

        {/* Messages List */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 z-10 scroll-smooth">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-gray-600 opacity-50">
              <Cpu className="w-16 h-16 mb-4 text-surface" />
              <p>System Initialized. Awaiting Input...</p>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isLoading && (
            <div className="flex gap-4 max-w-4xl">
              <div className="w-10 h-10 bg-surface border border-neon-blue/30 rounded flex items-center justify-center">
                <Bot className="w-6 h-6 text-neon-blue animate-pulse" />
              </div>
              <div className="flex items-center gap-2 text-neon-blue text-sm animate-pulse pt-2">
                Processing security audit...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input Area */}
        <ChatInput onSend={sendMessage} isLoading={isLoading} />
      </main>
    </div>
  );
}

export default App;