import { useState } from 'react';
import { useMutation } from '@tanstack/react-query'; // <--- NEW
import axios from 'axios';
import type { Message } from '../types';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);

  // The Mutation handles the API call + Loading State + Errors automatically
  const mutation = useMutation({
    mutationFn: async (content: string) => {
      return axios.post('/api/chat', { message: content });
    },
    onSuccess: (data) => {
      // Add AI response to state
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.data.response,
        logs: data.data.logs,
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages((prev) => [...prev, aiMsg]);
    },
    onError: () => {
      const errorMsg: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: " Network Error. Please try again.",
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages((prev) => [...prev, errorMsg]);
    }
  });

  const sendMessage = (content: string) => {
    // 1. Optimistically add User Message
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages((prev) => [...prev, userMsg]);

    // 2. Trigger API
    mutation.mutate(content);
  };

  return { 
    messages, 
    isLoading: mutation.isPending, // <--- React Query handles "loading"
    sendMessage 
  };
}