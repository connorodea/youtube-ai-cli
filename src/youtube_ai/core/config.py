import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class YouTubeConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, description="YouTube Data API key")
    channel_id: Optional[str] = Field(default=None, description="YouTube channel ID")
    default_privacy: str = Field(default="private", description="Default video privacy setting")
    client_secrets_file: Optional[str] = Field(default=None, description="OAuth client secrets file path")


class AIConfig(BaseModel):
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    elevenlabs_api_key: Optional[str] = Field(default=None, description="ElevenLabs API key")
    deepgram_api_key: Optional[str] = Field(default=None, description="Deepgram API key")
    stability_api_key: Optional[str] = Field(default=None, description="Stability AI API key")
    default_llm: str = Field(default="openai", description="Default LLM provider")
    default_tts: str = Field(default="openai", description="Default TTS provider")
    default_image_provider: str = Field(default="openai", description="Default image generation provider")


class VideoConfig(BaseModel):
    resolution: str = Field(default="1080p", description="Video resolution")
    fps: int = Field(default=30, description="Frames per second")
    format: str = Field(default="mp4", description="Video format")
    quality: str = Field(default="high", description="Video quality")
    max_duration: int = Field(default=600, description="Maximum video duration in seconds")


class AudioConfig(BaseModel):
    voice: str = Field(default="alloy", description="Default voice for TTS")
    speed: float = Field(default=1.0, description="Speech speed multiplier")
    background_music: bool = Field(default=False, description="Include background music")
    music_volume: float = Field(default=0.1, description="Background music volume")


class ContentConfig(BaseModel):
    language: str = Field(default="en", description="Content language")
    target_audience: str = Field(default="general", description="Target audience")
    default_style: str = Field(default="educational", description="Default content style")
    include_intro: bool = Field(default=True, description="Include video intro")
    include_outro: bool = Field(default=True, description="Include video outro")


class AppConfig(BaseModel):
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    debug: bool = Field(default=False, description="Enable debug mode")
    output_dir: str = Field(default="./output", description="Output directory for generated content")


class ConfigManager:
    """Manages application configuration from multiple sources."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".youtube-ai"
        self.config_file = self.config_dir / "config.yml"
        self.ensure_config_dir()
        load_dotenv()  # Load environment variables
        
    def ensure_config_dir(self):
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        config_data = {}
        
        # Load from config file if it exists
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        
        # Override with environment variables
        env_config = self._load_from_env()
        config_data = self._deep_merge(config_data, env_config)
        
        return AppConfig(**config_data)
    
    def save_config(self, config: AppConfig):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config.model_dump(exclude_none=True), f, default_flow_style=False)
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "youtube": {
                "api_key": os.getenv("YOUTUBE_API_KEY"),
                "channel_id": os.getenv("YOUTUBE_CHANNEL_ID"),
                "client_secrets_file": os.getenv("YOUTUBE_CLIENT_SECRETS_FILE"),
            },
            "ai": {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
            },
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "output_dir": os.getenv("OUTPUT_DIR", "./output"),
        }
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif value is not None:  # Only override with non-None values
                result[key] = value
        return result
    
    def set_value(self, key_path: str, value: Any):
        """Set a configuration value using dot notation (e.g., 'ai.openai_api_key')."""
        config = self.load_config()
        keys = key_path.split('.')
        
        # Navigate to the parent object
        obj = config
        for key in keys[:-1]:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                raise ValueError(f"Invalid configuration path: {key_path}")
        
        # Set the final value
        final_key = keys[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
            self.save_config(config)
        else:
            raise ValueError(f"Invalid configuration key: {final_key}")
    
    def get_value(self, key_path: str) -> Any:
        """Get a configuration value using dot notation."""
        config = self.load_config()
        keys = key_path.split('.')
        
        obj = config
        for key in keys:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                return None
        return obj
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate the current configuration and return issues."""
        config = self.load_config()
        issues = []
        
        # Check required API keys
        if not config.ai.openai_api_key and not config.ai.anthropic_api_key:
            issues.append("No AI API key configured (OpenAI or Anthropic required)")
        
        if not config.youtube.api_key and not config.youtube.client_secrets_file:
            issues.append("No YouTube API credentials configured")
        
        # Check output directory
        output_path = Path(config.output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")
        
        return len(issues) == 0, issues


# Global config manager instance
config_manager = ConfigManager()