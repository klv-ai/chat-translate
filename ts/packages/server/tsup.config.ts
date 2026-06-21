import { defineConfig } from 'tsup';

// Bundle the server's own source into a single ESM entrypoint. Dependencies
// (fastify, eld, @chat-translate/sdk) stay external and are resolved from
// node_modules at runtime — keeping eld's bundled language database intact.
export default defineConfig({
  entry: { server: 'src/server.ts' },
  format: ['esm'],
  sourcemap: true,
  clean: true,
  target: 'node20',
});
