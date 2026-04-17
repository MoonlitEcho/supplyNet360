# Stage 1: Build
FROM node:14 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Production
FROM node:14 AS production
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY package*.json ./
RUN npm install --production

# Create a non-root user to run the application
RUN useradd -m appuser
USER appuser

# Health check
HEALTHCHECK --interval=5m --timeout=3s CMD curl -f http://localhost:3000/ || exit 1

EXPOSE 3000
CMD [ "node", "dist/index.js" ]