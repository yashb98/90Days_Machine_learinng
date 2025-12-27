import { useState, useRef } from 'react';
import { Send, Paperclip } from 'lucide-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { UploadProgress } from './UploadProgress';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [input, setInput] = useState('');
  const [uploadFile, setUploadFile] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  // --- UPLOAD LOGIC ---
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      setUploadFile(file.name); // Start Animation
      const formData = new FormData();
      formData.append('file', file);
      
      // Artificial delay to let the user see the cool animation (optional)
      await new Promise(r => setTimeout(r, 2000));

      return axios.post('/api/ingest', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['policies'] });
      // Keep "Indexing" message for a moment before clearing
      setTimeout(() => setUploadFile(null), 1000);
    },
    onError: () => {
      alert("Upload Failed!");
      setUploadFile(null);
    }
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadMutation.mutate(e.target.files[0]);
    }
    // Reset input so you can select the same file again if needed
    e.target.value = "";
  };
  // --------------------

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput('');
  };

  return (
    <div className="p-6 bg-terminal-bg border-t border-surface z-20 relative">
      
      {/* The Animated Progress Bar appears here */}
      <UploadProgress 
        isUploading={uploadMutation.isPending} 
        filename={uploadFile || "Unknown Data"} 
      />

      <form onSubmit={handleSubmit} className="max-w-5xl mx-auto relative flex gap-3">
        {/* HIDDEN FILE INPUT */}
        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept="application/pdf" 
          onChange={handleFileSelect} 
        />

        {/* ATTACH BUTTON */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadMutation.isPending || isLoading}
          className="p-4 bg-surface border border-gray-700 rounded-md text-gray-400 hover:text-neon-blue hover:border-neon-blue transition-all disabled:opacity-50 flex items-center justify-center group"
          title="Upload Security Policy (PDF)"
        >
          <Paperclip className="w-5 h-5 group-hover:rotate-45 transition-transform" />
        </button>

        {/* TEXT INPUT */}
        <div className="relative flex-1">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter command or query..."
            className="w-full bg-surface text-white placeholder-gray-600 border border-gray-700 rounded-md py-4 pl-5 pr-14 focus:outline-none focus:border-neon-blue focus:ring-1 focus:ring-neon-blue transition-all shadow-xl font-mono text-sm disabled:opacity-50"
            disabled={isLoading}
          />
          <button 
            type="submit"
            disabled={isLoading || !input.trim()}
            className="absolute right-2 top-2 p-2.5 bg-neon-blue hover:bg-blue-600 disabled:opacity-50 disabled:hover:bg-neon-blue text-white rounded transition-colors shadow-lg shadow-blue-900/20"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
}