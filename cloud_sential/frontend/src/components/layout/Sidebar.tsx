import { useRef, useState, useEffect } from 'react';
import { Shield, FileText, Server, X, Loader2, Upload, Plus, MessageSquare, Trash2 } from 'lucide-react';
import { useUser, UserButton } from "@clerk/clerk-react";
import { useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { StatusBadge } from '../common/StatusBadge';
import { usePolicies } from '../../../hooks/usePolicies';

// --- FIREBASE IMPORTS ---
import { db } from '../../firebase'; // Make sure this path is correct!
import { collection, query, where, orderBy, onSnapshot, deleteDoc, doc } from "firebase/firestore";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  // New props for Chat functionality
  activeChatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
}

export function Sidebar({ isOpen, onClose, activeChatId, onSelectChat, onNewChat }: SidebarProps) {
  const { user } = useUser();
  const { data: policies, isLoading: loadingPolicies } = usePolicies();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- 1. CHAT HISTORY STATE ---
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [loadingChats, setLoadingChats] = useState(true);

  // --- 2. FIREBASE LISTENER ---
  useEffect(() => {
    if (!user) return;

    // Query: Get chats for this user, ordered by newest update
    const q = query(
      collection(db, "chats"),
      where("userId", "==", user.id), // Clerk User ID
      orderBy("lastUpdatedAt", "desc")
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const chats = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));
      setChatHistory(chats);
      setLoadingChats(false);
    });

    return () => unsubscribe();
  }, [user]);

  // --- 3. DELETE CHAT HANDLER ---
  const handleDeleteChat = async (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation(); // Stop click from selecting the chat
    if (window.confirm("Purge this operation log?")) {
      await deleteDoc(doc(db, "chats", chatId));
      if (activeChatId === chatId) onNewChat();
    }
  };

  // --- 4. UPLOAD LOGIC  ---
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return axios.post('/api/ingest', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['policies'] });
    },
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadMutation.mutate(e.target.files[0]);
    }
  };

  return (
    <>
      {/* Mobile Overlay */}
      <div className={`fixed inset-0 bg-black/80 z-40 md:hidden ${isOpen ? "block" : "hidden"}`} onClick={onClose} />

      <div className={`
        fixed md:static inset-y-0 left-0 z-50 w-72 bg-terminal-dark border-r border-surface 
        p-5 flex flex-col font-mono text-sm transition-transform duration-300
        ${isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
      `}>
        
        {/* HEADER */}
        <div className="flex items-center gap-3 mb-6 text-neon-blue">
          <Shield className="w-8 h-8" />
          <div>
            <h1 className="text-lg font-bold text-white">CLOUD_SENTINEL</h1>
            <div className="text-[10px] text-neon-blue/60 uppercase">System v1.0.5</div>
          </div>
        </div>

        {/* USER PROFILE */}
        <div className="mb-6 p-3 bg-surface/30 rounded-lg border border-surface flex items-center gap-3">
          <UserButton afterSignOutUrl="/sign-in" />
          <div className="flex flex-col overflow-hidden">
            <span className="text-white text-xs font-bold truncate">{user?.fullName || "Operative"}</span>
            <span className="text-[10px] text-gray-500"> Level 5 Access</span>
          </div>
        </div>

        {/* --- NEW CHAT BUTTON --- */}
        <button 
          onClick={onNewChat}
          className="mb-6 flex items-center justify-center gap-2 w-full py-2 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-blue border border-neon-blue/30 hover:border-neon-blue rounded transition-all uppercase tracking-wider text-xs font-bold"
        >
          <Plus className="w-4 h-4" /> New Operation
        </button>

        {/* SCROLLABLE AREA */}
        <div className="flex-1 overflow-y-auto pr-2 space-y-8 custom-scrollbar">
          
          {/* SECTION 1: CHAT HISTORY */}
          <div>
            <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2">
              <MessageSquare className="w-3 h-3" /> Operation Logs
            </h2>
            
            <div className="space-y-1">
              {loadingChats ? (
                 <div className="text-center py-2 text-gray-600 text-xs">Syncing logs...</div>
              ) : chatHistory.length === 0 ? (
                 <div className="text-gray-600 text-xs italic px-2">No active logs.</div>
              ) : (
                chatHistory.map(chat => (
                  <div 
                    key={chat.id} 
                    onClick={() => onSelectChat(chat.id)}
                    className={`
                      group flex items-center justify-between p-2 rounded cursor-pointer transition-all
                      ${activeChatId === chat.id ? "bg-neon-blue/10 border-l-2 border-neon-blue text-white" : "hover:bg-surface/50 text-gray-400 border-l-2 border-transparent"}
                    `}
                  >
                    <span className="truncate text-xs w-40">{chat.title}</span>
                    <button 
                      onClick={(e) => handleDeleteChat(e, chat.id)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* SECTION 2: KNOWLEDGE BASE  */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                <FileText className="w-3 h-3" /> Knowledge Base
              </h2>
              
              {/* HIDDEN INPUT + TRIGGER BUTTON */}
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept="application/pdf" 
                onChange={handleFileSelect} 
              />
              <button 
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadMutation.isPending}
                className="p-1 hover:bg-white/10 text-gray-500 hover:text-white rounded transition-colors"
                title="Ingest New Data"
              >
                {uploadMutation.isPending ? <Loader2 className="w-3 h-3 animate-spin"/> : <Plus className="w-3 h-3" />}
              </button>
            </div>

            <div className="space-y-2">
              {loadingPolicies ? (
                <div className="text-center py-4"><Loader2 className="w-4 h-4 animate-spin mx-auto text-gray-600"/></div>
              ) : (
                policies?.map(policy => (
                  <div key={policy.id} className="p-2 rounded bg-surface/30 border border-surface/50 flex items-center justify-between">
                    <span className="text-gray-400 text-xs truncate w-40" title={policy.name}>
                      {policy.name}
                    </span>
                    <div className="w-1.5 h-1.5 rounded-full bg-neon-green shadow-[0_0_5px_rgba(16,185,129,0.5)]" />
                  </div>
                ))
              )}
            </div>
          </div>

        </div>

        {/* FOOTER */}
        <div className="pt-4 border-t border-surface mt-4">
          <StatusBadge status="online" label="System Online" />
        </div>
      </div>
    </>
  );
}