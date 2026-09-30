# syntax=docker/dockerfile:1
# Multi-stage build for the thin translation server. Build context = ts/.

FROM node:22-slim AS base
ENV PNPM_HOME=/pnpm
ENV PATH=$PNPM_HOME:$PATH
RUN corepack enable
WORKDIR /app

# --- deps: cacheable dependency fetch keyed on the lockfile ---
FROM base AS deps
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./
COPY packages/sdk/package.json packages/sdk/package.json
COPY packages/server/package.json packages/server/package.json
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm fetch

# --- build: install offline, build the SDK then the server, prune to prod ---
FROM deps AS build
COPY . .
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --offline --frozen-lockfile
RUN pnpm --filter @chat-translate/sdk build
RUN pnpm --filter @chat-translate/server build
RUN pnpm --filter @chat-translate/server deploy --prod /app/out

# --- runtime: non-root, only the deployed server + prod deps ---
FROM base AS runtime
ENV NODE_ENV=production
ENV PORT=8080
COPY --from=build /app/out ./
USER node
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||8080)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "dist/server.js"]
