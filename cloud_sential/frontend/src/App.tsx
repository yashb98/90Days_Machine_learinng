import { useEffect, useRef, useState } from 'react'; // <--- Import useState
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"; 
import { SignedIn, SignedOut } from "@clerk/clerk-react"; 
import { Bot, Cpu } from 'lucide-react';

import { Sidebar } from './components/layout/Sidebar';
import { MobileHeader } from './components/layout/MobileHeader'; 
import { MessageBubble } from './components/chat/MessageBubble';
import { ChatInput } from './components/chat/ChatInput';
import { LoginPage } from './components/LoginPage'; 
import { useChat } from '../hooks/useChat';

function ProtectedDashboard() {
  const { messages, isLoading, sendMessage } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // 1. Create the state to control the sidebar
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">
      
      {/* 2. Pass the required props to Sidebar */}
      <Sidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
      />
      
      <main className="flex-1 flex flex-col h-screen bg-terminal-bg font-mono relative w-full">
        
        {/* 3. Pass the open function to the Mobile Header */}
        <MobileHeader onOpenSidebar={() => setIsSidebarOpen(true)} />

        <div className="absolute inset-0 bg-[linear-gradient(rgba(18,22,33,0)_1px,transparent_1px),linear-gradient(90deg,rgba(18,22,33,0)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 z-10 scroll-smooth">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-gray-600 opacity-50 px-4 text-center">
              <Cpu className="w-16 h-16 mb-4 text-surface" />
              <p>System Initialized. Awaiting Input...</p>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isLoading && (
            <div className="flex gap-4 max-w-4xl">
              <div className="w-10 h-10 bg-surface border border-neon-blue/30 rounded flex items-center justify-center shrink-0">
                <Bot className="w-6 h-6 text-neon-blue animate-pulse" />
              </div>
              <div className="flex items-center gap-2 text-neon-blue text-sm animate-pulse pt-2">
                Processing...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <ChatInput onSend={sendMessage} isLoading={isLoading} />
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