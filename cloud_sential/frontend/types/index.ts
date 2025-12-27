export interface ToolLog {
  tool: string;
  args: Record<string, any>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  logs?: ToolLog[];
  timestamp: string;
}

export interface Policy {
  id: string;
  name: string;
  status: 'active' | 'inactive';
  lastUpdated: string;
}