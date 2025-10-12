# Production Deployment Guide

This guide provides comprehensive instructions for deploying the Dementia Care AI Assistant to production environments.

## Table of Contents

- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Environment Configuration](#environment-configuration)
- [Backend Deployment](#backend-deployment)
- [Frontend Deployment](#frontend-deployment)
- [Database Setup](#database-setup)
- [Security Hardening](#security-hardening)
- [Monitoring & Logging](#monitoring--logging)
- [Scaling Strategies](#scaling-strategies)
- [Backup & Recovery](#backup--recovery)

## Pre-Deployment Checklist

- [ ] All environment variables configured
- [ ] HTTPS/TLS certificates obtained
- [ ] Database migrations tested
- [ ] API keys secured in secrets manager
- [ ] CORS origins restricted to production domains
- [ ] Error logging configured
- [ ] Monitoring alerts set up
- [ ] Backup strategy implemented
- [ ] Load testing completed
- [ ] Security audit performed

## Environment Configuration

### Production Environment Variables

Create a `.env` file (never commit to version control):

```bash
# API Keys
OPENAI_API_KEY=your_production_api_key

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=false  # Disable auto-reload in production
WORKERS=4  # Number of Uvicorn workers

# CORS (Restrict to production domains)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ALLOWED_METHODS=GET,POST,OPTIONS
ALLOWED_HEADERS=Content-Type,Authorization

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dementia_care
REDIS_URL=redis://localhost:6379/0

# Session Management
SESSION_TIMEOUT=3600  # 1 hour
MAX_SESSIONS=1000

# Security
SECRET_KEY=your-secret-key-min-32-characters
JWT_ALGORITHM=HS256
JWT_EXPIRATION=86400  # 24 hours

# Logging
LOG_LEVEL=WARNING  # INFO for staging, WARNING for production
LOG_FILE=/var/log/dementia-care/app.log

# Frontend API URL
VITE_API_URL=https://api.yourdomain.com
```

## Backend Deployment

### Option 1: Docker Deployment (Recommended)

1. **Create Dockerfile**:

```dockerfile
# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Build knowledge base (pre-compute for production)
RUN python build_knowledge_base.py

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run with Gunicorn + Uvicorn workers
CMD ["gunicorn", "backend.main:app", \\
     "--workers", "4", \\
     "--worker-class", "uvicorn.workers.UvicornWorker", \\
     "--bind", "0.0.0.0:8000", \\
     "--access-logfile", "-", \\
     "--error-logfile", "-"]
```

2. **Build and Run**:

```bash
# Build image
docker build -t dementia-care-api .

# Run container
docker run -d \\
  --name dementia-api \\
  -p 8000:8000 \\
  --env-file .env \\
  --restart unless-stopped \\
  dementia-care-api
```

3. **Docker Compose (Full Stack)**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/dementia
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: dementia
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
```

### Option 2: Cloud Platform Deployment

#### AWS (Elastic Beanstalk)

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 dementia-care

# Create environment
eb create dementia-care-prod

# Deploy
eb deploy

# Configure environment variables
eb setenv OPENAI_API_KEY=xxx DATABASE_URL=xxx
```

#### Google Cloud Platform (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/dementia-api

# Deploy
gcloud run deploy dementia-api \\
  --image gcr.io/PROJECT_ID/dementia-api \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated \\
  --set-env-vars OPENAI_API_KEY=xxx
```

#### Azure (App Service)

```bash
# Create App Service
az webapp create \\
  --resource-group dementia-rg \\
  --plan dementia-plan \\
  --name dementia-api \\
  --runtime "PYTHON|3.10"

# Deploy code
az webapp up --name dementia-api

# Configure app settings
az webapp config appsettings set \\
  --name dementia-api \\
  --settings OPENAI_API_KEY=xxx
```

## Frontend Deployment

### Build Production Assets

```bash
cd frontend
npm run build
```

### Option 1: Static Hosting

#### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel --prod

# Configure environment variables in Vercel dashboard
```

#### Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
cd frontend
netlify deploy --prod --dir=dist
```

#### AWS S3 + CloudFront

```bash
# Build
npm run build

# Upload to S3
aws s3 sync dist/ s3://dementia-care-frontend --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation \\
  --distribution-id DISTRIBUTION_ID \\
  --paths "/*"
```

### Option 2: Nginx Static Server

```nginx
# /etc/nginx/sites-available/dementia-care
server {
    listen 80;
    listen [::]:80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    root /var/www/dementia-care/frontend/dist;
    index index.html;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Database Setup

### PostgreSQL (Recommended for Production)

```sql
-- Create database
CREATE DATABASE dementia_care;
CREATE USER dementia_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE dementia_care TO dementia_user;

-- Create tables (example)
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    model_type VARCHAR(50),
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    agent VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_session_id ON sessions(session_id);
CREATE INDEX idx_conversations_session_id ON conversations(session_id);
```

### Redis (Session Storage)

```bash
# Install Redis
sudo apt install redis-server

# Configure (production settings)
sudo nano /etc/redis/redis.conf

# Set:
# maxmemory 256mb
# maxmemory-policy allkeys-lru
# requirepass your_secure_password

# Restart
sudo systemctl restart redis
```

## Security Hardening

### 1. HTTPS/TLS Configuration

```bash
# Install Certbot (Let's Encrypt)
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 2. API Rate Limiting

```python
# backend/api/middleware.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # 100 requests per minute per IP
    @limiter.limit("100/minute")
    async def limited_call():
        return await call_next(request)
    return await limited_call()
```

### 3. Input Validation & Sanitization

Already implemented via Pydantic models. Ensure all endpoints use typed models.

### 4. Secrets Management

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \\
  --name dementia-care/openai-key \\
  --secret-string "sk-xxx"

# Retrieve in application
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='dementia-care/openai-key')
OPENAI_API_KEY = response['SecretString']
```

## Monitoring & Logging

### Application Logging

```python
# backend/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    logger = logging.getLogger("dementia_care")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
```

### Prometheus Metrics

```python
# Install: pip install prometheus-fastapi-instrumentator
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Error Tracking (Sentry)

```python
# Install: pip install sentry-sdk
import sentry_sdk
sentry_sdk.init(
    dsn="https://xxx@sentry.io/xxx",
    environment="production"
)
```

## Scaling Strategies

### Horizontal Scaling

```bash
# Run multiple instances behind load balancer
# Instance 1
uvicorn backend.main:app --host 0.0.0.0 --port 8001

# Instance 2
uvicorn backend.main:app --host 0.0.0.0 --port 8002

# Nginx load balancer
upstream backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}
```

### Auto-Scaling (Kubernetes)

```yaml
# kubernetes/deployment.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dementia-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dementia-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Caching Strategy

```python
# Redis caching for LLM responses
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_response(query):
    cache_key = hashlib.md5(query.encode()).hexdigest()
    cached = redis_client.get(f"response:{cache_key}")
    if cached:
        return json.loads(cached)
    return None

def cache_response(query, response, ttl=3600):
    cache_key = hashlib.md5(query.encode()).hexdigest()
    redis_client.setex(
        f"response:{cache_key}",
        ttl,
        json.dumps(response)
    )
```

## Backup & Recovery

### Database Backups

```bash
# Automated PostgreSQL backup script
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

pg_dump -U dementia_user dementia_care > $BACKUP_DIR/backup_$DATE.sql

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE.sql \\
  s3://dementia-care-backups/postgres/
```

### Model & Data Backups

```bash
# Backup trained models and vector stores
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/ data/vector_store/
aws s3 cp models_backup_*.tar.gz s3://dementia-care-backups/models/
```

## Health Checks & Readiness Probes

```python
# backend/api/routes/health.py
@router.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": check_database(),
            "redis": check_redis(),
            "vector_store": check_vector_store(),
            "llm": check_llm()
        }
    }
    return checks

def check_database():
    try:
        # Test database connection
        return {"status": "up"}
    except Exception as e:
        return {"status": "down", "error": str(e)}
```

## Performance Optimization

### 1. Connection Pooling

```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### 2. Async Database Queries

```python
from databases import Database

database = Database(DATABASE_URL)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
```

### 3. CDN for Static Assets

Configure CloudFlare, Cloudfront, or similar for frontend assets.

## Disaster Recovery Plan

1. **Regular Backups**: Automated daily backups to S3/Cloud Storage
2. **Multi-Region Deployment**: Deploy to multiple cloud regions
3. **Database Replication**: Primary-replica setup for PostgreSQL
4. **Monitoring Alerts**: Set up PagerDuty/OpsGenie for critical failures
5. **Incident Response**: Document procedures for common failure scenarios

## Post-Deployment Verification

```bash
# Test API endpoints
curl https://api.yourdomain.com/health

# Load testing
ab -n 1000 -c 10 https://api.yourdomain.com/health

# Security scan
nmap -sV https://yourdomain.com
```

## Support & Maintenance

- Monitor error rates and response times daily
- Review logs weekly for anomalies
- Update dependencies monthly
- Security patches: Apply within 7 days
- Performance tuning: Quarterly review
- Disaster recovery drill: Every 6 months

---

For questions or issues, contact: f20220209@pilani.bits-pilani.ac.in
