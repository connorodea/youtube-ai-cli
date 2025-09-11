"""
YouTube AI CLI - AI-powered YouTube automation library.

A comprehensive CLI tool for automating YouTube content creation using AI.
Features include script generation, video creation, SEO optimization, and more.
"""

__version__ = "0.1.0"
__author__ = "YouTube AI CLI Team"
__email__ = "contact@youtube-ai-cli.com"
__license__ = "MIT"

from .core.config import config_manager, AppConfig
from .core.logger import get_logger, setup_logging
from .modules.ai.llm_client import llm_manager, LLMManager
from .modules.content.script_generator import ScriptGenerator
from .modules.content.seo_optimizer import SEOOptimizer

__all__ = [
    "__version__",
    "config_manager",
    "AppConfig", 
    "get_logger",
    "setup_logging",
    "llm_manager",
    "LLMManager",
    "ScriptGenerator",
    "SEOOptimizer",
]