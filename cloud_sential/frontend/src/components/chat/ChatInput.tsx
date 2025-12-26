import { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput('');
  };

  return (
    <div className="p-6 bg-terminal-bg border-t border-surface z-20">
      <form onSubmit={handleSubmit} className="max-w-5xl mx-auto relative">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter command or query..."
          className="w-full bg-surface text-white placeholder-gray-600 border border-gray-700 rounded-md py-4 pl-5 pr-14 focus:outline-none focus:border-neon-blue focus:ring-1 focus:ring-neon-blue transition-all shadow-xl font-mono text-sm"
        />
        <button 
          type="submit"
          disabled={isLoading || !input.trim()}
          className="absolute right-2 top-2 p-2.5 bg-neon-blue hover:bg-blue-600 disabled:opacity-50 disabled:hover:bg-neon-blue text-white rounded transition-colors shadow-lg shadow-blue-900/20"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
}