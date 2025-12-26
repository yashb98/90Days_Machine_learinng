import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({ 
      registerType: 'autoUpdate',
      manifest: {
        name: 'CloudSentinel',
        short_name: 'Sentinel',
        theme_color: '#0f172a',
        icons: [
            {
                src: 'pwa-192x192.png', // You need to add these icons to /public
                sizes: '192x192',
                type: 'image/png'
            }
        ]
      }
    })
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