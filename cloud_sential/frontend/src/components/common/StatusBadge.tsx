import { Activity } from 'lucide-react';

interface StatusBadgeProps {
  status: 'online' | 'offline' | 'busy';
  label?: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const colors = {
    online: 'text-neon-green',
    offline: 'text-gray-500',
    busy: 'text-yellow-500'
  };

  return (
    <div className={`flex items-center gap-2 ${colors[status]} text-xs font-mono`}>
      <Activity className="w-3 h-3" />
      <span className="uppercase tracking-wider">{label || status}</span>
    </div>
  );
}