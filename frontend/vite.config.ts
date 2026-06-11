import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          if (id.includes('@tanstack')) return 'tanstack';
          if (id.includes('lucide-react')) return 'icons';
          if (id.includes('papaparse') || id.includes('react-hook-form') || id.includes('zod')) return 'forms';
          if (id.includes('react') || id.includes('react-dom') || id.includes('react-router-dom')) return 'react';
          return undefined;
        }
      }
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
});
