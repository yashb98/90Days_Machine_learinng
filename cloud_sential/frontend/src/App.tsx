import { useEffect, useRef, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"; 
import { SignedIn, SignedOut, useUser } from "@clerk/clerk-react"; 
import { Loader2, Cpu } from 'lucide-react';

// Firebase Imports
import { db } from './firebase'; 
import { collection, query, orderBy, onSnapshot } from "firebase/firestore";
import { sendMessageToFirestore, getAIResponse } from '../services/chatService';

// Components
import { Sidebar } from './components/layout/Sidebar';
import { MobileHeader } from './components/layout/MobileHeader'; 
import { MessageBubble } from './components/chat/MessageBubble';
import { ChatInput } from './components/chat/ChatInput';
import { LoginPage } from './components/LoginPage'; 

// --- 1. IMPORT THE TYPE (Don't redefine it locally) ---
import type { Message } from '../types'; // Adjust path if needed, e.g. './types'

function ProtectedDashboard() {
  const { user } = useUser();
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // State
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]); // Uses the imported type
  const [isProcessing, setIsProcessing] = useState(false);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  // --- FIRESTORE LISTENER ---
  useEffect(() => {
    if (!activeChatId) {
      setMessages([]); 
      return;
    }

    const messagesRef = collection(db, "chats", activeChatId, "messages");
    const q = query(messagesRef, orderBy("createdAt", "asc"));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const fetchedMessages = snapshot.docs.map((doc) => {
        const data = doc.data();
        
        // --- 2. VALIDATE THE ROLE ---
        // Ensure the DB role matches one of the allowed string types
        const role = (data.role === "user" || data.role === "assistant" || data.role === "system") 
          ? data.role 
          : "assistant"; // Fallback if DB has bad data

        return {
          id: doc.id,
          content: data.text || "", 
          role: role, 
          timestamp: data.createdAt?.toDate() || new Date(), 
        };
      }) as Message[]; // Now this cast is safe
      
      setMessages(fetchedMessages);
    });

    return () => unsubscribe();
  }, [activeChatId]);

  // --- SEND HANDLER ---
  const handleSendMessage = async (text: string) => {
    if (!user) return;
    
    try {
      setIsProcessing(true);

      const chatId = await sendMessageToFirestore(user.id, activeChatId, text, "user");
      
      if (!activeChatId) setActiveChatId(chatId);

      const aiResponse = await getAIResponse(text);

      await sendMessageToFirestore(user.id, chatId, aiResponse, "assistant");

    } catch (error) {
      console.error("Failed to send", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">
      
      <Sidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
        activeChatId={activeChatId}
        onSelectChat={(id) => {
          setActiveChatId(id);
          setIsSidebarOpen(false);
        }}
        onNewChat={() => {
          setActiveChatId(null);
          setIsSidebarOpen(false);
        }}
      />
      
      <main className="flex-1 flex flex-col h-screen bg-terminal-bg font-mono relative w-full">
        
        <MobileHeader onOpenSidebar={() => setIsSidebarOpen(true)} />

        <div className="absolute inset-0 bg-[linear-gradient(rgba(18,22,33,0)_1px,transparent_1px),linear-gradient(90deg,rgba(18,22,33,0)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 z-10 scroll-smooth custom-scrollbar">
          
          {messages.length === 0 && !activeChatId && (
            <div className="h-full flex flex-col items-center justify-center text-gray-600 opacity-50 px-4 text-center">
              <Cpu className="w-16 h-16 mb-4 text-surface" />
              <p>System Initialized. Awaiting Input...</p>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isProcessing && (
            <div className="flex gap-4 max-w-4xl">
              <div className="w-10 h-10 bg-surface border border-neon-blue/30 rounded flex items-center justify-center shrink-0">
                <Loader2 className="w-6 h-6 text-neon-blue animate-spin" />
              </div>
              <div className="flex items-center gap-2 text-neon-blue text-sm animate-pulse pt-2">
                Processing Data Stream...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <ChatInput onSend={handleSendMessage} isLoading={isProcessing} />
      </main>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/sign-in" element={
          <>
            <SignedIn>
              <Navigate to="/" replace />
            </SignedIn>
            <SignedOut>
              <LoginPage />
            </SignedOut>
          </>
        } />

        <Route path="/" element={
          <>
            <SignedIn>
              <ProtectedDashboard />
            </SignedIn>
            <SignedOut>
              <Navigate to="/sign-in" replace />
            </SignedOut>
          </>
        } />
      </Routes>
    </BrowserRouter>
  );
}

export default App;