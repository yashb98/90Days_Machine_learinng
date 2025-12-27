import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';
// import { useAuth } from "@clerk/clerk-react"; // Use auth token if needed later
import type { Message } from '../types';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  
  // React Query Mutation
  const mutation = useMutation({
    mutationFn: async (content: string) => {
      // We return the full Axios response here
      return axios.post('/api/chat', { message: content });
    },
    onSuccess: (res) => {
      // 🔍 DEBUGGING: Check exactly what the backend sent
      console.log("Backend Response:", res.data);

      // 1. Extract the data safely
      // The backend returns: { response: "string", logs: [...] }
      const responseText = res.data.response || "No response text found.";
      const toolLogs = res.data.logs || [];

      // 2. Create the AI Message
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responseText, // <--- IMPORTANT: Ensure this is a string!
        logs: toolLogs,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages((prev) => [...prev, aiMsg]);
    },
    onError: (error) => {
      console.error("Chat Error:", error);
      const errorMsg: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: "⚠️ System Error: Unable to reach CloudSentinel Network.",
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages((prev) => [...prev, errorMsg]);
    }
  });

  const sendMessage = (content: string) => {
    // 1. Add User Message immediately (Optimistic UI)
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages((prev) => [...prev, userMsg]);

    // 2. Send to Backend
    mutation.mutate(content);
  };

  return { 
    messages, 
    isLoading: mutation.isPending, 
    sendMessage 
  };
}