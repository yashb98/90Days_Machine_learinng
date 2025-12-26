import { SignIn } from "@clerk/clerk-react";
import { Shield } from "lucide-react";

export function LoginPage() {
  return (
    <div className="min-h-screen w-full bg-slate-900 flex items-center justify-center relative overflow-hidden font-mono">
      {/* Background Grid Animation */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(18,22,33,0)_1px,transparent_1px),linear-gradient(90deg,rgba(18,22,33,0)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>

      <div className="z-10 flex flex-col items-center gap-8">
        {/* Logo Header */}
        <div className="flex flex-col items-center gap-4 animate-bounce-slow">
          <div className="p-4 bg-slate-800 rounded-2xl border border-slate-700 shadow-xl shadow-blue-900/20">
            <Shield className="w-12 h-12 text-blue-500" />
          </div>
          <div className="text-center">
            <h1 className="text-3xl font-bold text-white tracking-wider">CLOUD_SENTINEL</h1>
            <p className="text-blue-400 text-sm mt-1">SECURE ACCESS GATEWAY</p>
          </div>
        </div>

        {/* Clerk Login Component */}
        <SignIn 
          appearance={{
            elements: {
              formButtonPrimary: 'bg-blue-600 hover:bg-blue-500 text-sm normal-case',
              card: 'bg-slate-800 border border-slate-700 shadow-2xl',
              headerTitle: 'text-white',
              headerSubtitle: 'text-slate-400',
              socialButtonsBlockButton: 'text-white border-slate-600 hover:bg-slate-700',
              formFieldLabel: 'text-slate-300',
              formFieldInput: 'bg-slate-900 border-slate-700 text-white',
              footerActionLink: 'text-blue-400 hover:text-blue-300'
            }
          }}
        />
      </div>
    </div>
  );
}