import { defineConfig } from 'tsup';

// One build entry per public `exports` subpath. esbuild resolves the artifacts'
// extensionless relative imports natively, so the reference files build
// unchanged; `dts` emits the declarations needed for local install into another
// TypeScript project.
export default defineConfig({
  entry: {
    index: 'src/index.ts',
    eval: 'src/eval.ts',
    'eval-source': 'src/eval-source.ts',
    logger: 'src/logger.ts',
    detector: 'src/detector.ts',
  },
  format: ['esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  splitting: false,
  target: 'node20',
});
