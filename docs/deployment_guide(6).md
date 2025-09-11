# YouTube AI CLI - Production Deployment Guide

This guide covers deploying YouTube AI CLI in production environments using various deployment strategies.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Deployment Options](#deployment-options)
  - [Docker Deployment](#docker-deployment)
  - [Kubernetes Deployment](#kubernetes-deployment)
  - [Traditional Server Deployment](#traditional-server-deployment)
- [Configuration Management](#configuration-management)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Scaling and Performance](#scaling-and-performance)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, recommended 4+ cores
- **Memory**: Minimum 4GB RAM, recommended 8GB+ RAM
- **Storage**: Minimum 50GB SSD, recommended 200GB+ SSD
- **Network**: Stable internet connection with sufficient bandwidth
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+), macOS, or Windows Server

### Software Dependencies

- **Python 3.9+** with pip
- **Docker** (for containerized deployment)
- **Kubernetes** (for orchestrated deployment)
- **FFmpeg** (for video processing)
- **PostgreSQL** (optional, for production analytics)
- **Redis** (optional, for caching)

### Required API Keys

- **OpenAI API Key** - For GPT-4 and TTS services
- **YouTube Data API Key** - For video uploads and analytics
- **Anthropic API Key** (Optional) - Alternative to OpenAI
- **ElevenLabs API Key** (Optional) - Premium voice synthesis

## Environment Setup

### 1. Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your actual values
nano .env
```

### 2. Set Required Environment Variables

```bash
# Essential API keys
export OPENAI_API_KEY="your_openai_api_key"
export YOUTUBE_API_KEY="your_youtube_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"  # Optional

# Application settings
export OUTPUT_DIR="/app/output"
export DEBUG="false"
export LOG_LEVEL="INFO"

# Database (for production)
export DATABASE_URL="postgresql://user:pass@localhost:5432/youtube_ai"
```

### 3. Create Directory Structure

```bash
# Create required directories
sudo mkdir -p /opt/youtube-ai-cli/{config,data,logs,backups}
sudo mkdir -p /var/lib/youtube-ai-cli/{output,cache,analytics}

# Set permissions
sudo chown -R youtube-ai:youtube-ai /opt/youtube-ai-cli
sudo chown -R youtube-ai:youtube-ai /var/lib/youtube-ai-cli
```

## Deployment Options

### Docker Deployment

#### Single Container Deployment

```bash
# 1. Build the image
docker build -t youtube-ai-cli:latest .

# 2. Run with environment file
docker run -d \
  --name youtube-ai-cli \
  --env-file .env \
  -v youtube_ai_output:/app/output \
  -v youtube_ai_config:/home/app/.youtube-ai \
  -p 8080:8080 \
  -p 8081:8081 \
  --restart unless-stopped \
  youtube-ai-cli:latest
```

#### Docker Compose Deployment

```bash
# 1. Create docker-compose.yml (see docker-compose.yml file)

# 2. Start all services
docker-compose up -d

# 3. Check status
docker-compose ps

# 4. View logs
docker-compose logs -f youtube-ai-cli
```

#### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  youtube-ai-cli:
    image: yourusername/youtube-ai-cli:v1.0.0
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
      restart_policy:
        condition: on-failure
        max_attempts: 3
    environment:
      - DATABASE_URL=postgresql://youtube_ai:${POSTGRES_PASSWORD}@postgres:5432/youtube_ai
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - youtube-ai-network

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: youtube_ai
      POSTGRES_USER: youtube_ai
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

volumes:
  postgres_data:
  redis_data:

networks:
  youtube-ai-network:
```

### Kubernetes Deployment

#### 1. Create Namespace and Secrets

```bash
# Create namespace
kubectl create namespace youtube-ai

# Create secrets
kubectl create secret generic youtube-ai-secrets \
  --from-literal=openai-api-key="your_openai_key" \
  --from-literal=youtube-api-key="your_youtube_key" \
  --from-literal=anthropic-api-key="your_anthropic_key" \
  --from-literal=database-url="postgresql://..." \
  --namespace youtube-ai

# Create OAuth credentials secret
kubectl create secret generic youtube-ai-credentials \
  --from-file=client_secrets.json \
  --namespace youtube-ai
```

#### 2. Deploy Application

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/ --namespace youtube-ai

# Check deployment status
kubectl get pods -n youtube-ai
kubectl get services -n youtube-ai

# Check logs
kubectl logs -f deployment/youtube-ai-cli -n youtube-ai
```

#### 3. Configure Ingress (Optional)

```yaml
# k8s/ingress.yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: youtube-ai-ingress
  namespace: youtube-ai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.youtube-ai-cli.com
    secretName: youtube-ai-tls
  rules:
  - host: api.youtube-ai-cli.com
    http:
      paths:
      - path: /health
        pathType: Prefix
        backend:
          service:
            name: youtube-ai-cli-service
            port:
              number: 8081
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: youtube-ai-cli-service
            port:
              number: 8080
```

### Traditional Server Deployment

#### 1. System Setup

```bash
# Install Python and dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-pip python3.11-venv ffmpeg

# Create user for the application
sudo useradd -r -s /bin/false youtube-ai
sudo mkdir -p /opt/youtube-ai-cli
sudo chown youtube-ai:youtube-ai /opt/youtube-ai-cli
```

#### 2. Application Installation

```bash
# Clone repository
cd /opt/youtube-ai-cli
sudo -u youtube-ai git clone https://github.com/yourusername/youtube-ai-cli.git .

# Create virtual environment
sudo -u youtube-ai python3.11 -m venv venv
sudo -u youtube-ai ./venv/bin/pip install -e .

# Install production dependencies
sudo -u youtube-ai ./venv/bin/pip install gunicorn supervisor
```

#### 3. Configure Systemd Service

```ini
# /etc/systemd/system/youtube-ai-cli.service
[Unit]
Description=YouTube AI CLI Application
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=exec
User=youtube-ai
Group=youtube-ai
WorkingDirectory=/opt/youtube-ai-cli
Environment=PATH=/opt/youtube-ai-cli/venv/bin
EnvironmentFile=/opt/youtube-ai-cli/.env
ExecStart=/opt/youtube-ai-cli/venv/bin/python -m youtube_ai.api.health
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable youtube-ai-cli
sudo systemctl start youtube-ai-cli

# Check status
sudo systemctl status youtube-ai-cli
```

#### 4. Configure Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/youtube-ai-cli
server {
    listen 80;
    server_name api.youtube-ai-cli.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.youtube-ai-cli.com;

    ssl_certificate /etc/letsencrypt/live/api.youtube-ai-cli.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.youtube-ai-cli.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location /health {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /metrics {
        # Restrict metrics endpoint
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Configuration Management

### 1. Environment-specific Configurations

```bash
# Development
cp config/templates/development.yml ~/.youtube-ai/config.yml

# Staging
cp config/templates/staging.yml ~/.youtube-ai/config.yml

# Production
cp config/templates/production.yml ~/.youtube-ai/config.yml
```

### 2. Secret Management

#### Using Kubernetes Secrets

```bash
# Rotate API keys
kubectl patch secret youtube-ai-secrets \
  -p='{"data":{"openai-api-key":"'$(echo -n "new_key" | base64)'"}}' \
  -n youtube-ai

# Restart deployment to pick up new secrets
kubectl rollout restart deployment/youtube-ai-cli -n youtube-ai
```

#### Using HashiCorp Vault (Advanced)

```bash
# Install Vault agent
vault write secret/youtube-ai-cli \
  openai_api_key="your_key" \
  youtube_api_key="your_key"

# Configure Vault agent for secret injection
vault auth -method=kubernetes
```

### 3. Configuration Validation

```bash
# Validate configuration
youtube-ai config validate

# Test API connectivity
youtube-ai system health --check apis

# Verify permissions
youtube-ai system health --check storage
```

## Monitoring and Observability

### 1. Prometheus Setup

```yaml
# monitoring/prometheus.yml (see monitoring/prometheus.yml file)
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "YouTube AI CLI Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(youtube_ai_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(youtube_ai_errors_total[5m]) / rate(youtube_ai_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(youtube_ai_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### 3. Alerting Rules

```yaml
# See monitoring/rules/youtube-ai-alerts.yml for complete alerting setup
```

### 4. Log Aggregation

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/youtube-ai-cli/*.log
  fields:
    service: youtube-ai-cli
    environment: production

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "youtube-ai-cli-%{+yyyy.MM.dd}"
```

## Security Considerations

### 1. API Key Security

```bash
# Use environment variables, never hardcode
export OPENAI_API_KEY="$(vault kv get -field=api_key secret/openai)"

# Rotate keys regularly
youtube-ai config set ai.openai_api_key "new_key"

# Monitor API usage
youtube-ai analytics summary --days 7
```

### 2. Network Security

```bash
# Firewall rules (UFW example)
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8080/tcp  # Metrics port - internal only
sudo ufw deny 8081/tcp  # Health port - internal only
sudo ufw enable
```

### 3. Container Security

```dockerfile
# Use non-root user
USER app

# Remove unnecessary packages
RUN apt-get remove -y wget curl && apt-get autoremove -y

# Use specific image tags
FROM python:3.11-slim@sha256:...
```

### 4. Access Control

```yaml
# Kubernetes RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: youtube-ai
  name: youtube-ai-cli-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
```

## Scaling and Performance

### 1. Horizontal Scaling

```yaml
# Kubernetes HPA (see k8s/deployment.yml for complete configuration)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: youtube-ai-cli-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: youtube-ai-cli
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

### 2. Performance Optimization

```bash
# Optimize batch processing
youtube-ai config set batch.max_concurrent 5
youtube-ai config set batch.retry_failed true

# Enable caching
youtube-ai config set cache.enabled true
youtube-ai config set cache.ttl 3600

# Adjust worker processes
export WORKERS=4
export MAX_CONCURRENT=3
```

### 3. Resource Limits

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2"
```

## Backup and Recovery

### 1. Data Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily backup script

DATE=$(date +%Y%m%d)
BACKUP_DIR="/opt/backups/youtube-ai-cli"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup configuration
cp -r ~/.youtube-ai $BACKUP_DIR/$DATE/config

# Backup database
pg_dump youtube_ai > $BACKUP_DIR/$DATE/database.sql

# Backup analytics
youtube-ai analytics export --format json --output $BACKUP_DIR/$DATE/analytics.json

# Backup workflows
cp -r ~/.youtube-ai/workflows $BACKUP_DIR/$DATE/

# Create archive
tar -czf $BACKUP_DIR/youtube-ai-backup-$DATE.tar.gz -C $BACKUP_DIR $DATE

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_DIR/youtube-ai-backup-$DATE.tar.gz s3://your-backup-bucket/

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### 2. Automated Backup (Cron)

```bash
# Add to crontab
0 2 * * * /opt/youtube-ai-cli/scripts/backup.sh
```

### 3. Recovery Procedures

```bash
# Restore from backup
DATE=20240101
BACKUP_FILE="/opt/backups/youtube-ai-cli/youtube-ai-backup-$DATE.tar.gz"

# Extract backup
cd /tmp
tar -xzf $BACKUP_FILE

# Restore configuration
cp -r $DATE/config/* ~/.youtube-ai/

# Restore database
dropdb youtube_ai
createdb youtube_ai
psql youtube_ai < $DATE/database.sql

# Restart services
sudo systemctl restart youtube-ai-cli
```

## Troubleshooting

### 1. Common Issues

#### Application Won't Start

```bash
# Check logs
sudo journalctl -u youtube-ai-cli -f

# Check configuration
youtube-ai config validate

# Check dependencies
youtube-ai system health --check dependencies
```

#### High Memory Usage

```bash
# Monitor memory usage
youtube-ai system monitor --interval 5

# Check for memory leaks
youtube-ai analytics trends --days 7

# Restart if necessary
sudo systemctl restart youtube-ai-cli
```

#### API Rate Limiting

```bash
# Check API usage
youtube-ai analytics summary --days 1

# Monitor costs
youtube-ai analytics optimize

# Implement rate limiting
youtube-ai config set api.rate_limit_rpm 30
```

### 2. Performance Issues

```bash
# Check system resources
youtube-ai system status --detailed

# Monitor batch jobs
youtube-ai batch list
youtube-ai batch status <job_id>

# Optimize configuration
youtube-ai config set video.resolution 720p
youtube-ai config set batch.max_concurrent 2
```

### 3. Database Issues

```bash
# Check database connectivity
youtube-ai system health --check database

# Backup before maintenance
pg_dump youtube_ai > backup.sql

# Vacuum and analyze
psql youtube_ai -c "VACUUM ANALYZE;"
```

### 4. Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
youtube-ai --debug generate script --topic "test"

# Check debug logs
tail -f ~/.youtube-ai/logs/debug.log
```

## Support and Maintenance

### 1. Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Update system packages
sudo apt update && sudo apt upgrade -y

# Clean up temporary files
youtube-ai system cleanup --days 7

# Rotate logs
sudo logrotate /etc/logrotate.d/youtube-ai-cli

# Check disk space
df -h

# Restart services if needed
sudo systemctl restart youtube-ai-cli
```

### 2. Updates and Upgrades

```bash
# Update application
cd /opt/youtube-ai-cli
sudo -u youtube-ai git pull
sudo -u youtube-ai ./venv/bin/pip install -e .

# Restart service
sudo systemctl restart youtube-ai-cli

# Verify update
youtube-ai --version
youtube-ai system status
```

### 3. Support Contacts

- **GitHub Issues**: https://github.com/yourusername/youtube-ai-cli/issues
- **Documentation**: https://docs.youtube-ai-cli.com
- **Community**: https://discord.gg/youtube-ai-cli

---

This deployment guide provides comprehensive instructions for production deployment. For additional help, consult the [documentation](https://docs.youtube-ai-cli.com) or [open an issue](https://github.com/yourusername/youtube-ai-cli/issues).
