import { createServer } from 'vite';

const server = await createServer({ server: { port: 5173, strictPort: true } });
await server.listen();
server.printUrls();
try {
  await import('./browser-check.mjs');
  await import('./star-gpu-check.mjs');
  await import('./bridge-check.mjs');
  await import('./xr-session-check.mjs');
} finally {
  await server.close();
}
