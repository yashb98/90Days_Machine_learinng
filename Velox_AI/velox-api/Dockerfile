# ----------------------------
# Stage 1: The Builder
# ----------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (including devDependencies for TypeScript)
COPY package*.json ./
COPY tsconfig.json ./
COPY prisma ./prisma/
RUN npm ci

# Copy source code
COPY src ./src

# Build TypeScript -> JavaScript (dist folder)
RUN npm run build
# Generate Prisma Client (so the artifacts exist)
RUN npx prisma generate

# ----------------------------
# Stage 2: The Production Runner
# ----------------------------
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Copy only package.json to install PROD dependencies
COPY package*.json ./
# Install ONLY production deps (saves space)
RUN npm ci --only=production

# Copy built artifacts from Stage 1
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/prisma ./prisma
# Copy the generated Prisma Client
COPY --from=builder /app/node_modules/@prisma ./node_modules/@prisma

# Don't run as root (Security Best Practice)
USER node

# Expose the port Cloud Run expects
EXPOSE 8080

# Start the server
CMD ["node", "dist/server.js"]