import { Shield, FileText, Server, X } from 'lucide-react';
import { useUser, UserButton } from "@clerk/clerk-react";
import type { Policy } from '../../../types';
import { StatusBadge } from '../common/StatusBadge';

const MOCK_POLICIES: Policy[] = [
  { id: '1', name: 'ACME Storage Policy v3', status: 'active', lastUpdated: '10 min ago' },
  { id: '2', name: 'IAM Access Control', status: 'active', lastUpdated: '2 hrs ago' },
  { id: '3', name: 'EC2 Network Boundaries', status: 'inactive', lastUpdated: '1 day ago' },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const { user } = useUser();

  return (
    <>
      {/* 1. Overlay (Darkens the background on mobile when menu is open) */}
      <div 
        className={`fixed inset-0 bg-black/80 z-40 transition-opacity duration-300 md:hidden ${
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
      />

      {/* 2. The Sidebar Panel */}
      <div className={`
        fixed md:static inset-y-0 left-0 z-50
        w-72 bg-terminal-dark border-r border-surface 
        p-5 flex flex-col font-mono text-sm
        transition-transform duration-300 ease-in-out
        ${isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
      `}>
        {/* Header & Close Button */}
        <div className="flex items-center justify-between mb-10 text-neon-blue">
          <div className="flex items-center gap-3">
            <Shield className="w-8 h-8" />
            <div>
              <h1 className="text-lg font-bold tracking-wider text-white">CLOUD_SENTINEL</h1>
              <div className="text-[10px] text-neon-blue/60 uppercase">System v1.0.4</div>
            </div>
          </div>
          {/* Close Button (Mobile Only) */}
          <button onClick={onClose} className="md:hidden text-gray-400 hover:text-white">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* User Profile (The "Burger Menu" content) */}
        <div className="mb-6 p-3 bg-surface/30 rounded-lg border border-surface flex items-center gap-3">
          <UserButton 
            afterSignOutUrl="/sign-in"
            appearance={{
              elements: {
                userButtonAvatarBox: "w-8 h-8 border border-neon-blue/50",
                userButtonPopoverCard: "bg-terminal-dark border border-gray-700 shadow-xl",
                userButtonPopoverFooter: "hidden" // Hides the "Secured by Clerk" footer for cleaner look
              }
            }}
          />
          <div className="flex flex-col overflow-hidden">
            <span className="text-white text-xs font-bold truncate">
              {user?.fullName || user?.username || "Operative"}
            </span>
            <span className="text-gray-500 text-[10px] truncate">
              {user?.primaryEmailAddress?.emailAddress}
            </span>
          </div>
        </div>

        {/* Active Policies */}
        <div className="flex-1">
          <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
            <FileText className="w-3 h-3" /> Knowledge Base
          </h2>
          <div className="space-y-2">
            {MOCK_POLICIES.map(policy => (
              <div key={policy.id} className="group p-3 rounded bg-surface/50 border border-transparent hover:border-neon-blue/50 hover:bg-surface transition-all cursor-default">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-300 font-medium group-hover:text-neon-blue transition-colors">
                    {policy.name}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${policy.status === 'active' ? 'bg-neon-green shadow-[0_0_8px_rgba(16,185,129,0.6)]' : 'bg-gray-600'}`} />
                  <span className="text-[10px] text-gray-500 uppercase">{policy.status}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer Status */}
        <div className="pt-6 border-t border-surface">
          <div className="flex items-center justify-between">
            <StatusBadge status="online" label="System Online" />
            <div className="flex items-center gap-2 text-gray-500 text-xs">
              <Server className="w-3 h-3" />
              <span>eu-west-2</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}