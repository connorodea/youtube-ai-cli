"""Analytics and performance tracking for YouTube AI CLI operations."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import statistics

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    YOUTUBE_ANALYTICS_AVAILABLE = True
except ImportError:
    YOUTUBE_ANALYTICS_AVAILABLE = False

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    EXECUTION_TIME = "execution_time"
    TOKEN_USAGE = "token_usage"
    FILE_SIZE = "file_size"
    API_CALLS = "api_calls"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    COST = "cost"
    QUALITY_SCORE = "quality_score"


class EventType(Enum):
    SCRIPT_GENERATED = "script_generated"
    AUDIO_CREATED = "audio_created"
    VIDEO_CREATED = "video_created"
    THUMBNAIL_GENERATED = "thumbnail_generated"
    VIDEO_UPLOADED = "video_uploaded"
    WORKFLOW_COMPLETED = "workflow_completed"
    ERROR_OCCURRED = "error_occurred"
    API_CALL_MADE = "api_call_made"


@dataclass
class PerformanceMetric:
    """A single performance metric measurement."""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    context: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class AnalyticsEvent:
    """An analytics event record."""
    timestamp: datetime
    event_type: EventType
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SessionSummary:
    """Summary of a user session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: Optional[float]
    events_count: int
    errors_count: int
    success_rate: float
    operations: List[str]
    total_cost: float = 0.0


@dataclass
class YouTubeVideoAnalytics:
    """YouTube video performance analytics."""
    video_id: str
    title: str
    views: int
    likes: int
    comments: int
    watch_time_minutes: float
    ctr: float  # Click-through rate
    retention_rate: float
    subscribers_gained: int
    revenue: float = 0.0
    updated_at: datetime = None

    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now()


class AnalyticsTracker:
    """Tracks and analyzes performance metrics and events."""

    def __init__(self):
        self.config = config_manager.load_config()
        self.analytics_dir = Path.home() / ".youtube-ai" / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.analytics_dir / "analytics.db"
        self.current_session_id = None
        self.session_start_time = None
        
        # Initialize database
        self._init_database()
        
        # Cost tracking (rough estimates)
        self.cost_estimates = {
            "openai_gpt4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "openai_tts": 0.015,  # per 1K characters
            "anthropic_claude": {"input": 0.015, "output": 0.075},  # per 1K tokens
            "elevenlabs_tts": 0.30,  # per 1K characters
        }

    def _init_database(self):
        """Initialize SQLite database for analytics storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    context TEXT,
                    session_id TEXT
                );
                
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    duration REAL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    metadata TEXT,
                    session_id TEXT
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_duration REAL,
                    events_count INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    operations TEXT,
                    total_cost REAL DEFAULT 0.0
                );
                
                CREATE TABLE IF NOT EXISTS youtube_analytics (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    watch_time_minutes REAL DEFAULT 0.0,
                    ctr REAL DEFAULT 0.0,
                    retention_rate REAL DEFAULT 0.0,
                    subscribers_gained INTEGER DEFAULT 0,
                    revenue REAL DEFAULT 0.0,
                    updated_at TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
            """)

    def start_session(self) -> str:
        """Start a new analytics session."""
        self.current_session_id = f"session_{int(time.time())}_{hash(str(datetime.now()))}"
        self.session_start_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, start_time, operations)
                VALUES (?, ?, ?)
            """, (self.current_session_id, self.session_start_time.isoformat(), "[]"))
        
        logger.debug(f"Started analytics session: {self.current_session_id}")
        return self.current_session_id

    def end_session(self):
        """End the current analytics session."""
        if not self.current_session_id:
            return
        
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds()
        
        # Calculate session statistics
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count events and errors
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)
                FROM events WHERE session_id = ?
            """, (self.current_session_id,))
            events_count, errors_count = cursor.fetchone()
            
            success_rate = ((events_count - errors_count) / events_count * 100) if events_count > 0 else 0
            
            # Get unique operations
            cursor.execute("""
                SELECT DISTINCT event_type FROM events WHERE session_id = ?
            """, (self.current_session_id,))
            operations = [row[0] for row in cursor.fetchall()]
            
            # Update session record
            conn.execute("""
                UPDATE sessions 
                SET end_time = ?, total_duration = ?, events_count = ?, 
                    errors_count = ?, success_rate = ?, operations = ?
                WHERE session_id = ?
            """, (end_time.isoformat(), duration, events_count, errors_count, 
                  success_rate, json.dumps(operations), self.current_session_id))
        
        logger.info(f"Ended session {self.current_session_id}: {duration:.1f}s, {events_count} events, {success_rate:.1f}% success")
        self.current_session_id = None
        self.session_start_time = None

    def track_metric(
        self,
        metric_type: MetricType,
        value: float,
        unit: str,
        context: Dict[str, Any] = None
    ):
        """Track a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            unit=unit,
            context=context or {},
            session_id=self.current_session_id
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (timestamp, metric_type, value, unit, context, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric.timestamp.isoformat(), metric.metric_type.value, 
                  metric.value, metric.unit, json.dumps(metric.context), metric.session_id))
        
        logger.debug(f"Tracked metric: {metric_type.value} = {value} {unit}")

    def track_event(
        self,
        event_type: EventType,
        duration: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Track an analytics event."""
        event = AnalyticsEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            duration=duration,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            session_id=self.current_session_id
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events (timestamp, event_type, duration, success, error_message, metadata, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event.timestamp.isoformat(), event.event_type.value, event.duration,
                  event.success, event.error_message, json.dumps(event.metadata), event.session_id))
        
        logger.debug(f"Tracked event: {event_type.value} (success: {success})")

    def track_cost(
        self,
        provider: str,
        operation_type: str,
        units: int,
        cost: float
    ):
        """Track API usage costs."""
        self.track_metric(
            MetricType.COST,
            cost,
            "USD",
            {
                "provider": provider,
                "operation": operation_type,
                "units": units
            }
        )
        
        # Update session cost
        if self.current_session_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE sessions 
                    SET total_cost = total_cost + ?
                    WHERE session_id = ?
                """, (cost, self.current_session_id))

    def estimate_ai_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        characters: int = 0
    ) -> float:
        """Estimate cost for AI operations."""
        provider_key = f"{provider}_{model}"
        
        if provider_key in self.cost_estimates:
            rates = self.cost_estimates[provider_key]
            
            if isinstance(rates, dict):
                # Token-based pricing
                input_cost = (input_tokens / 1000) * rates["input"]
                output_cost = (output_tokens / 1000) * rates["output"]
                total_cost = input_cost + output_cost
            else:
                # Character-based pricing
                total_cost = (characters / 1000) * rates
        else:
            # Unknown provider, use default estimate
            total_cost = ((input_tokens + output_tokens) / 1000) * 0.02
        
        self.track_cost(provider, model, input_tokens + output_tokens + characters, total_cost)
        return total_cost

    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Event statistics
            cursor.execute("""
                SELECT event_type, COUNT(*), AVG(duration), 
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                FROM events 
                WHERE timestamp >= ?
                GROUP BY event_type
            """, (cutoff_date.isoformat(),))
            
            event_stats = {}
            for event_type, count, avg_duration, successes in cursor.fetchall():
                event_stats[event_type] = {
                    "count": count,
                    "avg_duration": avg_duration,
                    "success_rate": (successes / count * 100) if count > 0 else 0
                }
            
            # Cost statistics
            cursor.execute("""
                SELECT SUM(value) as total_cost, AVG(value) as avg_cost
                FROM metrics 
                WHERE metric_type = 'cost' AND timestamp >= ?
            """, (cutoff_date.isoformat(),))
            
            cost_data = cursor.fetchone()
            total_cost = cost_data[0] or 0
            avg_cost = cost_data[1] or 0
            
            # Session statistics
            cursor.execute("""
                SELECT COUNT(*), AVG(total_duration), AVG(success_rate)
                FROM sessions 
                WHERE start_time >= ?
            """, (cutoff_date.isoformat(),))
            
            session_data = cursor.fetchone()
            session_count = session_data[0] or 0
            avg_session_duration = session_data[1] or 0
            avg_success_rate = session_data[2] or 0
            
        return {
            "period_days": days,
            "event_statistics": event_stats,
            "cost_analysis": {
                "total_cost": total_cost,
                "average_cost_per_operation": avg_cost
            },
            "session_statistics": {
                "total_sessions": session_count,
                "average_duration": avg_session_duration,
                "average_success_rate": avg_success_rate
            },
            "generated_at": datetime.now().isoformat()
        }

    def get_usage_trends(self, days: int = 30) -> Dict[str, List[Tuple[str, float]]]:
        """Get usage trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Daily event counts
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM events 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (cutoff_date.isoformat(),))
            
            daily_events = [(row[0], row[1]) for row in cursor.fetchall()]
            
            # Daily costs
            cursor.execute("""
                SELECT DATE(timestamp) as date, SUM(value) as cost
                FROM metrics 
                WHERE metric_type = 'cost' AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (cutoff_date.isoformat(),))
            
            daily_costs = [(row[0], row[1]) for row in cursor.fetchall()]
            
            # Error rates
            cursor.execute("""
                SELECT DATE(timestamp) as date, 
                       COUNT(*) as total,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors
                FROM events 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (cutoff_date.isoformat(),))
            
            daily_error_rates = []
            for row in cursor.fetchall():
                date, total, errors = row
                error_rate = (errors / total * 100) if total > 0 else 0
                daily_error_rates.append((date, error_rate))
        
        return {
            "daily_events": daily_events,
            "daily_costs": daily_costs,
            "daily_error_rates": daily_error_rates
        }

    async def sync_youtube_analytics(self, video_id: str) -> Optional[YouTubeVideoAnalytics]:
        """Sync analytics data from YouTube."""
        if not YOUTUBE_ANALYTICS_AVAILABLE:
            logger.warning("YouTube Analytics API not available")
            return None
        
        try:
            # This would require YouTube Analytics API setup
            # For now, return placeholder data
            analytics = YouTubeVideoAnalytics(
                video_id=video_id,
                title="Sample Video",
                views=0,
                likes=0,
                comments=0,
                watch_time_minutes=0.0,
                ctr=0.0,
                retention_rate=0.0,
                subscribers_gained=0
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO youtube_analytics 
                    (video_id, title, views, likes, comments, watch_time_minutes, 
                     ctr, retention_rate, subscribers_gained, revenue, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (analytics.video_id, analytics.title, analytics.views,
                      analytics.likes, analytics.comments, analytics.watch_time_minutes,
                      analytics.ctr, analytics.retention_rate, analytics.subscribers_gained,
                      analytics.revenue, analytics.updated_at.isoformat()))
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error syncing YouTube analytics: {e}")
            return None

    def export_analytics(self, format: str = "json", output_file: Optional[Path] = None) -> Path:
        """Export analytics data to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.analytics_dir / f"analytics_export_{timestamp}.{format}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all data
            cursor.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 1000")
            metrics_data = [dict(zip([col[0] for col in cursor.description], row)) 
                           for row in cursor.fetchall()]
            
            cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT 1000")
            events_data = [dict(zip([col[0] for col in cursor.description], row)) 
                          for row in cursor.fetchall()]
            
            cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 100")
            sessions_data = [dict(zip([col[0] for col in cursor.description], row)) 
                            for row in cursor.fetchall()]
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics": metrics_data,
            "events": events_data,
            "sessions": sessions_data
        }
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
            import csv
            
            # Export each table to separate CSV files
            base_path = output_file.with_suffix('')
            
            for table_name, data in [("metrics", metrics_data), ("events", events_data), ("sessions", sessions_data)]:
                if data:
                    csv_file = f"{base_path}_{table_name}.csv"
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
        
        logger.info(f"Exported analytics data to: {output_file}")
        return output_file

    def get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """Get optimization suggestions based on analytics data."""
        suggestions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check error rates
            cursor.execute("""
                SELECT event_type, COUNT(*) as total,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors
                FROM events 
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY event_type
                HAVING errors > 0
            """, )
            
            for event_type, total, errors in cursor.fetchall():
                error_rate = (errors / total) * 100
                if error_rate > 10:  # More than 10% error rate
                    suggestions.append({
                        "type": "error_rate",
                        "priority": "high" if error_rate > 25 else "medium",
                        "message": f"High error rate ({error_rate:.1f}%) for {event_type}. Consider reviewing error logs.",
                        "action": f"Check logs for {event_type} operations"
                    })
            
            # Check costs
            cursor.execute("""
                SELECT SUM(value) as total_cost
                FROM metrics 
                WHERE metric_type = 'cost' AND timestamp >= datetime('now', '-7 days')
            """)
            
            weekly_cost = cursor.fetchone()[0] or 0
            if weekly_cost > 50:  # $50 per week threshold
                suggestions.append({
                    "type": "cost",
                    "priority": "medium",
                    "message": f"Weekly costs are ${weekly_cost:.2f}. Consider optimizing API usage.",
                    "action": "Review API call patterns and optimize token usage"
                })
            
            # Check performance
            cursor.execute("""
                SELECT AVG(duration) as avg_duration
                FROM events 
                WHERE event_type = 'script_generated' AND timestamp >= datetime('now', '-7 days')
            """)
            
            avg_script_time = cursor.fetchone()[0] or 0
            if avg_script_time > 60:  # More than 1 minute
                suggestions.append({
                    "type": "performance",
                    "priority": "low",
                    "message": f"Script generation taking {avg_script_time:.1f}s on average. Consider shorter prompts.",
                    "action": "Optimize prompts for faster generation"
                })
        
        return suggestions


# Global analytics tracker instance
analytics_tracker = AnalyticsTracker()