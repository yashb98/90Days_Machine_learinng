import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { ClerkProvider, RedirectToSignIn, SignedIn, SignedOut } from "@clerk/clerk-react";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
const queryClient = new QueryClient();

const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={clerkPubKey}>
      <QueryClientProvider client={queryClient}>
      {/* 1. If signed in, show the App */}
      <SignedIn>
        <App />
      </SignedIn>
      
      {/* 2. If signed out, redirect to login */}
      <SignedOut>
        <RedirectToSignIn />
      </SignedOut>
      </QueryClientProvider>
    </ClerkProvider>
  </React.StrictMode>,
)