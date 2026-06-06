# Stage 1: build the React frontend into static assets
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend that also serves the built frontend (single origin)
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY --from=frontend /app/frontend/dist ./frontend/dist

ENV PORT=5000
EXPOSE 5000
CMD ["sh", "-c", "gunicorn 'api.app:create_app()' --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
