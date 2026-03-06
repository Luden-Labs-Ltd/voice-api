# ── Build stage ───────────────────────────────────────────────────────────────
# Build context is the repo root (set in docker-compose.yml).
FROM node:20-alpine AS builder

WORKDIR /build

COPY node-api/package*.json ./
RUN npm install

COPY node-api/tsconfig.json ./
COPY node-api/src/ ./src/

RUN npm run build && npm prune --omit=dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

COPY --from=builder /build/dist ./dist
COPY --from=builder /build/node_modules ./node_modules

# Serve the frontend HTML directly from Node
COPY frontend/ ./frontend/

EXPOSE 3000

ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
