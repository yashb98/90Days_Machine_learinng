import { Shield, FileText, Server } from 'lucide-react';
import type { Policy } from '../../../types';
import { StatusBadge } from '../common/StatusBadge';

const MOCK_POLICIES: Policy[] = [
  { id: '1', name: 'ACME Storage Policy v3', status: 'active', lastUpdated: '10 min ago' },
  { id: '2', name: 'IAM Access Control', status: 'active', lastUpdated: '2 hrs ago' },
  { id: '3', name: 'EC2 Network Boundaries', status: 'inactive', lastUpdated: '1 day ago' },
];

export function Sidebar() {
  return (
    <div className="w-72 bg-terminal-dark border-r border-surface h-screen p-5 flex flex-col font-mono text-sm hidden md:flex">
      {/* Header */}
      <div className="flex items-center gap-3 mb-10 text-neon-blue">
        <Shield className="w-8 h-8" />
        <div>
          <h1 className="text-lg font-bold tracking-wider text-white">CLOUD_SENTINEL</h1>
          <div className="text-[10px] text-neon-blue/60 uppercase">System v1.0.4</div>
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
  );
}