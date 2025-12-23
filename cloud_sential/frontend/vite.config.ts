import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),          // REQUIRED: Handles React/JSX/TSX files
    tailwindcss(),    // Your new Tailwind v4 plugin
  ],
  resolve: {
    alias: {
      // Allows imports like: import { Button } from "@/components/Button"
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    // Connects Frontend (5173) to Backend (8000) to fix CORS
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // For Server-Sent Events (if you use them for the agent logs)
      '/sse': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})