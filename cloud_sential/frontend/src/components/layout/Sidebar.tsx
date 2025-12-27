import { useRef, useState } from 'react';
import { Shield, FileText, Server, X, Loader2, Upload, Plus } from 'lucide-react';
import { useUser, UserButton } from "@clerk/clerk-react";
import { useMutation, useQueryClient } from '@tanstack/react-query'; // Import React Query tools
import axios from 'axios';
import { StatusBadge } from '../common/StatusBadge';
import { usePolicies } from '../../../hooks/usePolicies';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const { user } = useUser();
  const { data: policies, isLoading: loadingPolicies } = usePolicies();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- UPLOAD LOGIC ---
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return axios.post('/api/ingest', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
    },
    onSuccess: () => {
      // Refresh the policy list automatically after upload!
      queryClient.invalidateQueries({ queryKey: ['policies'] });
    },
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadMutation.mutate(e.target.files[0]);
    }
  };
  // --------------------

  return (
    <>
      <div className={`fixed inset-0 bg-black/80 z-40 md:hidden ${isOpen ? "block" : "hidden"}`} onClick={onClose} />

      <div className={`
        fixed md:static inset-y-0 left-0 z-50 w-72 bg-terminal-dark border-r border-surface 
        p-5 flex flex-col font-mono text-sm transition-transform duration-300
        ${isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
      `}>
        {/* Header */}
        <div className="flex items-center gap-3 mb-8 text-neon-blue">
          <Shield className="w-8 h-8" />
          <div>
            <h1 className="text-lg font-bold text-white">CLOUD_SENTINEL</h1>
            <div className="text-[10px] text-neon-blue/60 uppercase">System v1.0.5</div>
          </div>
        </div>

        {/* User Profile */}
        <div className="mb-6 p-3 bg-surface/30 rounded-lg border border-surface flex items-center gap-3">
          <UserButton afterSignOutUrl="/sign-in" />
          <div className="flex flex-col overflow-hidden">
            <span className="text-white text-xs font-bold truncate">{user?.fullName || "Operative"}</span>
          </div>
        </div>

        {/* --- KNOWLEDGE BASE SECTION --- */}
        <div className="flex-1 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
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
              className="p-1.5 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-blue rounded transition-colors disabled:opacity-50"
              title="Upload New Policy"
            >
              {uploadMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin"/> : <Plus className="w-4 h-4" />}
            </button>
          </div>

          <div className="space-y-2">
            {loadingPolicies ? (
              <div className="text-center py-4"><Loader2 className="w-5 h-5 animate-spin mx-auto text-gray-600"/></div>
            ) : (
              policies?.map(policy => (
                <div key={policy.id} className="group p-3 rounded bg-surface/50 border border-transparent hover:border-neon-blue/50 hover:bg-surface transition-all cursor-default">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-gray-300 font-medium truncate w-48" title={policy.name}>
                      {policy.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-neon-green shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
                    <span className="text-[10px] text-gray-500 uppercase">{policy.status}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="pt-6 border-t border-surface mt-auto">
          <div className="flex items-center justify-between">
            <StatusBadge status="online" label="System Online" />
          </div>
        </div>
      </div>
    </>
  );
}