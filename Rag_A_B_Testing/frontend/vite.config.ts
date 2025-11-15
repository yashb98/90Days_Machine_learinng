import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(), // <-- Added the plugin here
  ],
  server: {
    proxy: {
      // String shorthand: '/api' -> 'http://localhost:5000/api'
      // This tells Vite to forward any request starting with '/api'
      // to your Flask backend running on port 5000.
      '/api': {
        target: 'http://localhost:8000', // <-- This is your Flask backend URL
        changeOrigin: true,
        
      },
    }
  }
})