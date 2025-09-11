import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

console = Console()


class YouTubeAILogger:
    """Custom logger for YouTube AI CLI."""
    
    def __init__(self, name: str, level: str = "INFO", log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[Path] = None):
        """Setup logging handlers."""
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        console_handler.setLevel(logging.INFO)
        
        # Custom formatter for console
        console_formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        
        # File handler if log file is specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Detailed formatter for file
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    def setLevel(self, level: str):
        """Set logging level."""
        self.logger.setLevel(getattr(logging, level.upper()))
        for handler in self.logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(getattr(logging, level.upper()))


def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> YouTubeAILogger:
    """Get a logger instance for the given name."""
    log_path = Path(log_file) if log_file else None
    return YouTubeAILogger(name, level, log_path)


def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Setup global logging configuration."""
    level = "DEBUG" if debug else "INFO"
    
    # Suppress some noisy third-party loggers
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("google_auth_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return get_logger("youtube_ai", level, log_file)