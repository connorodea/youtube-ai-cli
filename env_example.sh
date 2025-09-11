# YouTube AI CLI Environment Configuration
# Copy this file to .env and fill in your actual values

# =============================================================================
# API Keys (Required)
# =============================================================================

# OpenAI API Key (required for GPT-4 and TTS)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, alternative to OpenAI)
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# YouTube Data API Key (required for video uploads)
# Get from: https://console.developers.google.com/
YOUTUBE_API_KEY=your_youtube_api_key_here

# ElevenLabs API Key (optional, for premium voice synthesis)
# Get from: https://elevenlabs.io/
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# =============================================================================
# YouTube Authentication
# =============================================================================

# YouTube Channel ID (optional, for analytics)
YOUTUBE_CHANNEL_ID=your_youtube_channel_id

# Path to YouTube OAuth client secrets file
# Download from Google Cloud Console
YOUTUBE_CLIENT_SECRETS_FILE=./credentials/client_secrets.json

# =============================================================================
# Application Settings
# =============================================================================

# Output directory for generated content
OUTPUT_DIR=./output

# Enable debug mode (true/false)
DEBUG=false

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# =============================================================================
# AI Provider Preferences
# =============================================================================

# Default LLM provider (openai, anthropic)
DEFAULT_LLM=openai

# Default TTS provider (openai, elevenlabs)
DEFAULT_TTS=openai

# =============================================================================
# Video Generation Settings
# =============================================================================

# Default video resolution (720p, 1080p, 1440p, 4k)
VIDEO_RESOLUTION=1080p

# Default video FPS
VIDEO_FPS=30

# Default video format
VIDEO_FORMAT=mp4

# =============================================================================
# Audio Settings
# =============================================================================

# Default voice for TTS (alloy, echo, fable, onyx, nova, shimmer)
DEFAULT_VOICE=alloy

# Default speech speed (0.5 to 2.0)
DEFAULT_SPEECH_SPEED=1.0

# Enable background music (true/false)
BACKGROUND_MUSIC=false

# Background music volume (0.0 to 1.0)
MUSIC_VOLUME=0.1

# =============================================================================
# Content Settings
# =============================================================================

# Default content style (educational, entertaining, informative, tutorial, review)
DEFAULT_CONTENT_STYLE=educational

# Default target audience
DEFAULT_AUDIENCE=general

# Default content language
DEFAULT_LANGUAGE=en

# =============================================================================
# Performance Settings
# =============================================================================

# Maximum concurrent operations
MAX_CONCURRENT=3

# Number of worker threads
WORKERS=3

# Request timeout in seconds
REQUEST_TIMEOUT=300

# =============================================================================
# Cache and Storage
# =============================================================================

# Cache directory
CACHE_DIR=./cache

# Enable caching (true/false)
ENABLE_CACHE=true

# Cache TTL in seconds (3600 = 1 hour)
CACHE_TTL=3600

# Maximum cache size in MB
MAX_CACHE_SIZE=1024

# =============================================================================
# Analytics and Monitoring
# =============================================================================

# Enable analytics tracking (true/false)
ENABLE_ANALYTICS=true

# Analytics database path
ANALYTICS_DB=~/.youtube-ai/analytics/analytics.db

# Export analytics data automatically (true/false)
AUTO_EXPORT_ANALYTICS=false

# =============================================================================
# Security Settings
# =============================================================================

# Encrypt stored credentials (true/false)
ENCRYPT_CREDENTIALS=true

# Auto-backup configuration (true/false)
AUTO_BACKUP_CONFIG=true

# Backup retention days
BACKUP_RETENTION_DAYS=30

# =============================================================================
# Development Settings
# =============================================================================

# Enable development mode (true/false)
DEV_MODE=false

# Enable hot reload for config changes (true/false)
HOT_RELOAD=false

# Mock API responses for testing (true/false)
MOCK_API_RESPONSES=false

# =============================================================================
# Notification Settings
# =============================================================================

# Email notifications (optional)
NOTIFICATION_EMAIL=your_email@example.com

# Slack webhook for notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Discord webhook for notifications (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK

# =============================================================================
# Advanced Configuration
# =============================================================================

# Custom config directory
CONFIG_DIR=~/.youtube-ai

# Custom templates directory
TEMPLATES_DIR=./config/templates

# Enable experimental features (true/false)
ENABLE_EXPERIMENTAL=false

# Rate limiting (requests per minute)
RATE_LIMIT_RPM=60

# =============================================================================
# Database Settings (for advanced usage)
# =============================================================================

# Database type (sqlite, postgresql)
DATABASE_TYPE=sqlite

# PostgreSQL connection (if using PostgreSQL)
# DATABASE_URL=postgresql://user:password@localhost:5432/youtube_ai

# =============================================================================
# Cloud Storage (optional)
# =============================================================================

# AWS S3 settings (for cloud storage)
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AWS_BUCKET_NAME=your_s3_bucket
# AWS_REGION=us-east-1

# Google Cloud Storage settings
# GOOGLE_CLOUD_PROJECT=your_project_id
# GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcs-key.json

# =============================================================================
# Monitoring and Observability
# =============================================================================

# Prometheus metrics endpoint (true/false)
ENABLE_METRICS=false

# Metrics port
METRICS_PORT=8080

# Health check endpoint (true/false)
ENABLE_HEALTH_CHECK=true

# Health check port
HEALTH_CHECK_PORT=8081

# =============================================================================
# Tips and Notes
# =============================================================================

# 1. Never commit this file with real API keys to version control
# 2. Use quotes for values with spaces: SOME_VAR="value with spaces"
# 3. Boolean values should be lowercase: true/false
# 4. Paths can be absolute or relative to the project root
# 5. For production, consider using a secret management system
# 6. Test your configuration with: youtube-ai config validate