import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Resolve the workspace SDK from source so server tests run without a prior
// SDK build.
export default defineConfig({
  resolve: {
    alias: {
      '@chat-translate/sdk': fileURLToPath(new URL('../sdk/src/index.ts', import.meta.url)),
    },
  },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
  },
});
