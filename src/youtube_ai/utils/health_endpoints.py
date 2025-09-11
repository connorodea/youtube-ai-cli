"""Health check and metrics API endpoints."""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

from ..core.config import config_manager
from ..core.logger import get_logger
from ..modules.analytics.tracker import analytics_tracker
from ..modules.batch.processor import batch_processor

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="YouTube AI CLI Health API",
    description="Health checks and metrics for YouTube AI CLI",
    version="0.1.0"
)

# Global state
_startup_time = time.time()
_health_cache = {}
_cache_ttl = 30  # Cache health checks for 30 seconds


class HealthChecker:
    """Health check implementation."""
    
    @staticmethod
    async def check_basic() -> Dict[str, Any]:
        """Basic health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - _startup_time,
            "version": "0.1.0"
        }
    
    @staticmethod
    async def check_detailed() -> Dict[str, Any]:
        """Detailed health check."""
        # Use cache if available and fresh
        cache_key = "detailed_health"
        now = time.time()
        
        if (cache_key in _health_cache and 
            now - _health_cache[cache_key]["timestamp"] < _cache_ttl):
            return _health_cache[cache_key]["data"]
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - _startup_time,
            "checks": {}
        }
        
        # Configuration check
        try:
            is_valid, issues = config_manager.validate_config()
            health_data["checks"]["configuration"] = {
                "status": "pass" if is_valid else "fail",
                "issues": issues
            }
        except Exception as e:
            health_data["checks"]["configuration"] = {
                "status": "error",
                "error": str(e)
            }
        
        # System resources check
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Determine status based on thresholds
            resource_status = "pass"
            if memory.percent > 90 or (disk.used / disk.total) * 100 > 90 or cpu_percent > 90:
                resource_status = "warn"
            
            health_data["checks"]["resources"] = {
                "status": resource_status,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "cpu_percent": cpu_percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            health_data["checks"]["resources"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Storage check
        try:
            config = config_manager.load_config()
            output_dir = Path(config.output_dir)
            
            storage_status = "pass"
            storage_info = {"writable": False, "exists": False}
            
            if output_dir.exists():
                storage_info["exists"] = True
                # Test write access
                try:
                    test_file = output_dir / ".health_check"
                    test_file.write_text("test")
                    test_file.unlink()
                    storage_info["writable"] = True
                except Exception:
                    storage_status = "fail"
            else:
                storage_status = "fail"
            
            health_data["checks"]["storage"] = {
                "status": storage_status,
                **storage_info
            }
        except Exception as e:
            health_data["checks"]["storage"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Database check
        try:
            # Test analytics database
            summary = analytics_tracker.get_performance_summary(days=1)
            health_data["checks"]["database"] = {
                "status": "pass",
                "events_24h": sum(stats['count'] for stats in summary['event_statistics'].values())
            }
        except Exception as e:
            health_data["checks"]["database"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Active jobs check
        try:
            jobs = batch_processor.list_batch_jobs()
            running_jobs = [job for job in jobs if job['status'] == 'running']
            stuck_jobs = [job for job in jobs if job['status'] == 'running' and 
                         job.get('total_duration', 0) > 3600]  # Over 1 hour
            
            jobs_status = "warn" if stuck_jobs else "pass"
            
            health_data["checks"]["jobs"] = {
                "status": jobs_status,
                "total_jobs": len(jobs),
                "running_jobs": len(running_jobs),
                "stuck_jobs": len(stuck_jobs)
            }
        except Exception as e:
            health_data["checks"]["jobs"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall status
        failed_checks = [name for name, check in health_data["checks"].items() 
                        if check["status"] in ["fail", "error"]]
        warn_checks = [name for name, check in health_data["checks"].items() 
                      if check["status"] == "warn"]
        
        if failed_checks:
            health_data["status"] = "unhealthy"
            health_data["failed_checks"] = failed_checks
        elif warn_checks:
            health_data["status"] = "degraded"
            health_data["warning_checks"] = warn_checks
        
        # Cache result
        _health_cache[cache_key] = {
            "timestamp": now,
            "data": health_data
        }
        
        return health_data


# Health check endpoints
@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        health_data = await HealthChecker.check_basic()
        status_code = 200
        return JSONResponse(content=health_data, status_code=status_code)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/health/detailed", response_class=JSONResponse)
async def detailed_health_check():
    """Detailed health check endpoint."""
    try:
        health_data = await HealthChecker.check_detailed()
        
        # Set appropriate status code
        status_code = 200
        if health_data["status"] == "unhealthy":
            status_code = 503
        elif health_data["status"] == "degraded":
            status_code = 200  # Still healthy, but with warnings
        
        return JSONResponse(content=health_data, status_code=status_code)
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/ready", response_class=JSONResponse)
async def readiness_check():
    """Readiness check for Kubernetes."""
    try:
        # Check if application is ready to serve requests
        config = config_manager.load_config()
        is_valid, issues = config_manager.validate_config()
        
        if not is_valid:
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "Configuration invalid",
                    "issues": issues
                },
                status_code=503
            )
        
        # Check if output directory is accessible
        output_dir = Path(config.output_dir)
        if not output_dir.exists() or not output_dir.is_dir():
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "Output directory not accessible"
                },
                status_code=503
            )
        
        return JSONResponse(
            content={
                "status": "ready",
                "timestamp": datetime.now().isoformat()
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        # Generate Prometheus-format metrics
        metrics_lines = []
        
        # Basic application metrics
        metrics_lines.append(f"# HELP youtube_ai_uptime_seconds Application uptime in seconds")
        metrics_lines.append(f"# TYPE youtube_ai_uptime_seconds gauge")
        metrics_lines.append(f"youtube_ai_uptime_seconds {time.time() - _startup_time}")
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        disk = psutil.disk_usage('/')
        
        metrics_lines.extend([
            f"# HELP youtube_ai_memory_usage_percent Memory usage percentage",
            f"# TYPE youtube_ai_memory_usage_percent gauge",
            f"youtube_ai_memory_usage_percent {memory.percent}",
            
            f"# HELP youtube_ai_cpu_usage_percent CPU usage percentage",
            f"# TYPE youtube_ai_cpu_usage_percent gauge",
            f"youtube_ai_cpu_usage_percent {cpu_percent}",
            
            f"# HELP youtube_ai_disk_usage_percent Disk usage percentage",
            f"# TYPE youtube_ai_disk_usage_percent gauge",
            f"youtube_ai_disk_usage_percent {(disk.used / disk.total) * 100}",
        ])
        
        # Application-specific metrics
        try:
            summary = analytics_tracker.get_performance_summary(days=1)
            total_events = sum(stats['count'] for stats in summary['event_statistics'].values())
            total_cost = summary['cost_analysis']['total_cost']
            avg_success_rate = summary['session_statistics']['average_success_rate']
            
            metrics_lines.extend([
                f"# HELP youtube_ai_events_total_24h Total events in last 24 hours",
                f"# TYPE youtube_ai_events_total_24h counter",
                f"youtube_ai_events_total_24h {total_events}",
                
                f"# HELP youtube_ai_cost_usd_24h Total cost in USD for last 24 hours",
                f"# TYPE youtube_ai_cost_usd_24h counter", 
                f"youtube_ai_cost_usd_24h {total_cost}",
                
                f"# HELP youtube_ai_success_rate_percent Average success rate percentage",
                f"# TYPE youtube_ai_success_rate_percent gauge",
                f"youtube_ai_success_rate_percent {avg_success_rate}",
            ])
        except Exception as e:
            logger.warning(f"Could not get analytics metrics: {e}")
        
        # Batch job metrics
        try:
            jobs = batch_processor.list_batch_jobs()
            running_jobs = len([job for job in jobs if job['status'] == 'running'])
            completed_jobs = len([job for job in jobs if job['status'] == 'completed'])
            failed_jobs = len([job for job in jobs if job['status'] == 'failed'])
            
            metrics_lines.extend([
                f"# HELP youtube_ai_batch_jobs_running Currently running batch jobs",
                f"# TYPE youtube_ai_batch_jobs_running gauge",
                f"youtube_ai_batch_jobs_running {running_jobs}",
                
                f"# HELP youtube_ai_batch_jobs_completed_total Total completed batch jobs",
                f"# TYPE youtube_ai_batch_jobs_completed_total counter",
                f"youtube_ai_batch_jobs_completed_total {completed_jobs}",
                
                f"# HELP youtube_ai_batch_jobs_failed_total Total failed batch jobs",
                f"# TYPE youtube_ai_batch_jobs_failed_total counter",
                f"youtube_ai_batch_jobs_failed_total {failed_jobs}",
            ])
        except Exception as e:
            logger.warning(f"Could not get batch job metrics: {e}")
        
        return PlainTextResponse("\n".join(metrics_lines))
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", response_class=JSONResponse)
async def info():
    """Application information endpoint."""
    try:
        config = config_manager.load_config()
        
        info_data = {
            "application": {
                "name": "YouTube AI CLI",
                "version": "0.1.0",
                "description": "AI-powered YouTube automation CLI"
            },
            "system": {
                "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
                "python_version": f"{psutil.version_info.major}.{psutil.version_info.minor}",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "configuration": {
                "output_dir": config.output_dir,
                "debug_mode": config.debug,
                "ai_providers": {
                    "openai_configured": bool(config.ai.openai_api_key),
                    "anthropic_configured": bool(config.ai.anthropic_api_key),
                    "elevenlabs_configured": bool(config.ai.elevenlabs_api_key)
                },
                "youtube_configured": bool(config.youtube.api_key)
            },
            "features": {
                "script_generation": True,
                "video_creation": True,
                "audio_synthesis": True,
                "seo_optimization": True,
                "batch_processing": True,
                "workflow_automation": True,
                "analytics_tracking": True
            }
        }
        
        return JSONResponse(content=info_data)
    except Exception as e:
        logger.error(f"Info endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("YouTube AI CLI Health API starting up")
    global _startup_time
    _startup_time = time.time()


# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("YouTube AI CLI Health API shutting down")


def run_health_server(host: str = "0.0.0.0", port: int = 8081):
    """Run the health check server."""
    logger.info(f"Starting health check server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False  # Reduce noise from health checks
    )


if __name__ == "__main__":
    run_health_server()