"""YouTube AI CLI - Automate YouTube content creation with AI."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.config import config_manager
from .core.logger import get_logger
from .content.script_generator import ScriptGenerator
from .content.seo_optimizer import SEOOptimizer

__all__ = [
    "config_manager",
    "get_logger", 
    "ScriptGenerator",
    "SEOOptimizer",
]