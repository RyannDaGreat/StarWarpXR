import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import basicSsl from '@vitejs/plugin-basic-ssl';

/** Query. Selects local HTTPS from Vite's mode; returns build/server configuration. */
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: [svelte(), ...(mode === 'https' ? [basicSsl()] : [])],
  server: { port: 5173, strictPort: true },
}));
