import { Terminal } from 'lucide-react';
import type { ToolLog as ToolLogType } from '../../../types'; // Import from root types

interface ToolLogProps {
  logs: ToolLogType[];
}

export function ToolLog({ logs }: ToolLogProps) {
  if (!logs || logs.length === 0) return null;

  return (
    <div className="rounded border border-gray-700 bg-terminal-dark/80 backdrop-blur overflow-hidden mt-3">
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-800/50 border-b border-gray-700 text-xs text-gray-400">
        <Terminal className="w-3 h-3" />
        <span className="uppercase tracking-wider">Execution Log</span>
      </div>
      <div className="p-3 font-mono text-xs space-y-2">
        {logs.map((log, idx) => (
          <div key={idx} className="flex gap-2">
            <span className="text-gray-500">$</span>
            <span className="text-neon-green">exec</span>
            <span className="text-yellow-400">{log.tool}</span>
            <span className="text-gray-400">{JSON.stringify(log.args)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}