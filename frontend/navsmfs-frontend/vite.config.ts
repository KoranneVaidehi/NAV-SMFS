import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],

  base: '/static/react/',

  build: {
    outDir: '../../backend/veridex/static/react',
    emptyOutDir: true,
  },
})