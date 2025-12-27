import { Menu, Shield } from 'lucide-react';

interface MobileHeaderProps {
  onOpenSidebar: () => void;
}

export function MobileHeader({ onOpenSidebar }: MobileHeaderProps) {
  return (
    <div className="md:hidden flex items-center justify-between p-4 bg-terminal-dark border-b border-surface sticky top-0 z-30">
      <div className="flex items-center gap-2 text-neon-blue">
        <Shield className="w-6 h-6" />
        <span className="font-bold tracking-wider text-white text-sm">CLOUD_SENTINEL</span>
      </div>
      
      <button 
        onClick={onOpenSidebar}
        className="p-2 text-gray-400 hover:text-white hover:bg-surface rounded-lg transition-colors"
      >
        <Menu className="w-6 h-6" />
      </button>
    </div>
  );
}