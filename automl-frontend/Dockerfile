# --- STAGE 1: BUILD ---
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency configs
COPY package.json package-lock.json ./

# Install packages (using legacy-peer-deps to ignore potential React 19/testing-library strict conflicts)
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Set up build argument for dynamic API URL (defaults to localhost:8000)
ARG REACT_APP_API_URL="http://localhost:8000"
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Build the production package
RUN npm run build

# --- STAGE 2: PRODUCTION SERVE ---
FROM nginx:stable-alpine

# Copy custom SPA Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static assets from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
