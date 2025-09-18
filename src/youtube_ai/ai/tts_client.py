import asyncio
import io
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, BinaryIO
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import openai
from openai import AsyncOpenAI
try:
    import elevenlabs
    from elevenlabs.client import AsyncElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

try:
    # Temporarily disabled due to Python version compatibility
    # from deepgram import DeepgramClient, SpeakOptions
    DEEPGRAM_AVAILABLE = False
except ImportError:
    DEEPGRAM_AVAILABLE = False

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

logger = get_logger(__name__)


class TTSProvider(Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    DEEPGRAM = "deepgram"
    LOCAL = "local"


@dataclass
class Voice:
    """Represents a TTS voice."""
    id: str
    name: str
    provider: str
    language: str = "en"
    gender: str = "neutral"
    style: str = "neutral"
    sample_rate: int = 22050


@dataclass
class TTSRequest:
    """TTS generation request."""
    text: str
    voice: str
    speed: float = 1.0
    pitch: float = 1.0
    output_format: str = "mp3"
    sample_rate: int = 22050


@dataclass
class TTSResponse:
    """TTS generation response."""
    audio_data: bytes
    provider: str
    voice: str
    format: str
    sample_rate: int
    duration: Optional[float] = None
    metadata: Optional[Dict] = None


class BaseTTSClient(ABC):
    """Base class for TTS clients."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.provider_name = self.__class__.__name__.lower().replace('client', '')
    
    @abstractmethod
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech from text."""
        pass
    
    @abstractmethod
    async def get_available_voices(self) -> List[Voice]:
        """Get list of available voices."""
        pass
    
    @abstractmethod
    def validate_voice(self, voice_id: str) -> bool:
        """Validate if voice ID is available."""
        pass


class OpenAITTSClient(BaseTTSClient):
    """OpenAI TTS client."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncOpenAI(api_key=api_key)
        self.voices = {
            "alloy": Voice("alloy", "Alloy", "openai", "en", "neutral", "neutral"),
            "echo": Voice("echo", "Echo", "openai", "en", "male", "neutral"),
            "fable": Voice("fable", "Fable", "openai", "en", "neutral", "expressive"),
            "onyx": Voice("onyx", "Onyx", "openai", "en", "male", "deep"),
            "nova": Voice("nova", "Nova", "openai", "en", "female", "energetic"),
            "shimmer": Voice("shimmer", "Shimmer", "openai", "en", "female", "gentle"),
        }
    
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using OpenAI TTS."""
        try:
            logger.debug(f"Synthesizing speech with OpenAI: {len(request.text)} characters")
            
            response = await self.client.audio.speech.create(
                model="tts-1-hd",  # Use HD model for better quality
                voice=request.voice,
                input=request.text,
                speed=request.speed,
                response_format=request.output_format
            )
            
            # Get audio data
            audio_data = response.content
            
            return TTSResponse(
                audio_data=audio_data,
                provider="openai",
                voice=request.voice,
                format=request.output_format,
                sample_rate=request.sample_rate,
                metadata={
                    "model": "tts-1-hd",
                    "input_length": len(request.text)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            raise
    
    async def get_available_voices(self) -> List[Voice]:
        """Get available OpenAI voices."""
        return list(self.voices.values())
    
    def validate_voice(self, voice_id: str) -> bool:
        """Validate OpenAI voice ID."""
        return voice_id in self.voices


class ElevenLabsTTSClient(BaseTTSClient):
    """ElevenLabs TTS client."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("ElevenLabs package not installed. Install with: pip install elevenlabs")
        
        self.client = AsyncElevenLabs(api_key=api_key)
        self._voices_cache = None
    
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech using ElevenLabs."""
        try:
            logger.debug(f"Synthesizing speech with ElevenLabs: {len(request.text)} characters")
            
            # ElevenLabs API call
            audio_generator = await self.client.generate(
                text=request.text,
                voice=request.voice,
                model="eleven_multilingual_v2",
                stream=False
            )
            
            # Collect audio data
            audio_data = b""
            async for chunk in audio_generator:
                audio_data += chunk
            
            return TTSResponse(
                audio_data=audio_data,
                provider="elevenlabs",
                voice=request.voice,
                format="mp3",
                sample_rate=request.sample_rate,
                metadata={
                    "model": "eleven_multilingual_v2",
                    "input_length": len(request.text)
                }
            )
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise
    
    async def get_available_voices(self) -> List[Voice]:
        """Get available ElevenLabs voices."""
        try:
            if self._voices_cache is None:
                response = await self.client.voices.get_all()
                self._voices_cache = [
                    Voice(
                        id=voice.voice_id,
                        name=voice.name,
                        provider="elevenlabs",
                        language="en",  # ElevenLabs supports multiple languages
                        gender="neutral",  # Would need additional API call to determine
                        style="neutral"
                    )
                    for voice in response.voices
                ]
            return self._voices_cache
        except Exception as e:
            logger.error(f"Error getting ElevenLabs voices: {e}")
            return []
    
    def validate_voice(self, voice_id: str) -> bool:
        """Validate ElevenLabs voice ID."""
        # This would require an API call, so we'll assume it's valid for now
        # In production, you might want to cache voice validation
        return True


class DeepgramTTSClient(BaseTTSClient):
    """Deepgram TTS client implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = DeepgramClient(api_key)
        self.provider_name = "deepgram"
        self._voices_cache = None
    
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Deepgram TTS."""
        try:
            # Set up Deepgram TTS options
            options = SpeakOptions(
                model="aura-asteria-en",  # High-quality voice model
                encoding="mp3",
                sample_rate=22050,
            )
            
            # Override voice model if specified
            voice_mapping = {
                "asteria": "aura-asteria-en",
                "luna": "aura-luna-en", 
                "stella": "aura-stella-en",
                "athena": "aura-athena-en",
                "hera": "aura-hera-en",
                "orion": "aura-orion-en",
                "arcas": "aura-arcas-en",
                "perseus": "aura-perseus-en",
                "angus": "aura-angus-en",
            }
            
            if request.voice in voice_mapping:
                options.model = voice_mapping[request.voice]
            
            # Generate speech
            response = self.client.speak.v("1").stream(
                source={"text": request.text},
                options=options
            )
            
            # Get audio bytes
            audio_data = b"".join(response)
            
            return TTSResponse(
                audio_data=audio_data,
                provider="deepgram",
                voice=request.voice,
                format=request.output_format,
                sample_rate=request.sample_rate,
                duration=len(request.text.split()) * 0.5,  # Rough estimate
                metadata={
                    "model": options.model,
                    "input_length": len(request.text)
                }
            )
            
        except Exception as e:
            logger.error(f"Deepgram TTS error: {e}")
            raise
    
    async def get_available_voices(self) -> List[Voice]:
        """Get available Deepgram voices."""
        if self._voices_cache is None:
            self._voices_cache = [
                Voice(id="asteria", name="Asteria", provider="deepgram", gender="female", style="conversational"),
                Voice(id="luna", name="Luna", provider="deepgram", gender="female", style="expressive"),
                Voice(id="stella", name="Stella", provider="deepgram", gender="female", style="friendly"),
                Voice(id="athena", name="Athena", provider="deepgram", gender="female", style="authoritative"),
                Voice(id="hera", name="Hera", provider="deepgram", gender="female", style="warm"),
                Voice(id="orion", name="Orion", provider="deepgram", gender="male", style="deep"),
                Voice(id="arcas", name="Arcas", provider="deepgram", gender="male", style="calm"),
                Voice(id="perseus", name="Perseus", provider="deepgram", gender="male", style="confident"),
                Voice(id="angus", name="Angus", provider="deepgram", gender="male", style="narrative"),
            ]
        return self._voices_cache
    
    def validate_voice(self, voice_id: str) -> bool:
        """Validate Deepgram voice ID."""
        valid_voices = ["asteria", "luna", "stella", "athena", "hera", "orion", "arcas", "perseus", "angus"]
        return voice_id in valid_voices


class TTSManager:
    """Manages multiple TTS providers with fallback support."""
    
    def __init__(self):
        self.clients: Dict[str, BaseTTSClient] = {}
        self.config = config_manager.load_config()
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available TTS clients based on configuration."""
        # OpenAI TTS
        if self.config.ai.openai_api_key:
            self.clients["openai"] = OpenAITTSClient(
                api_key=self.config.ai.openai_api_key
            )
            logger.debug("Initialized OpenAI TTS client")
        
        # ElevenLabs TTS
        if self.config.ai.elevenlabs_api_key and ELEVENLABS_AVAILABLE:
            try:
                self.clients["elevenlabs"] = ElevenLabsTTSClient(
                    api_key=self.config.ai.elevenlabs_api_key
                )
                logger.debug("Initialized ElevenLabs TTS client")
            except ImportError as e:
                logger.warning(f"ElevenLabs not available: {e}")
        
        # Deepgram TTS
        if hasattr(self.config.ai, 'deepgram_api_key') and self.config.ai.deepgram_api_key and DEEPGRAM_AVAILABLE:
            try:
                self.clients["deepgram"] = DeepgramTTSClient(
                    api_key=self.config.ai.deepgram_api_key
                )
                logger.debug("Initialized Deepgram TTS client")
            except Exception as e:
                logger.warning(f"Deepgram not available: {e}")
        
        if not self.clients:
            logger.warning("No TTS providers configured")
    
    def get_client(self, provider: Optional[str] = None) -> BaseTTSClient:
        """Get a TTS client for the specified provider."""
        if provider and provider in self.clients:
            return self.clients[provider]
        
        # Use default provider
        default_provider = self.config.ai.default_tts
        if default_provider in self.clients:
            return self.clients[default_provider]
        
        # Fallback to any available provider
        if self.clients:
            provider_name = next(iter(self.clients))
            logger.warning(f"Using fallback TTS provider: {provider_name}")
            return self.clients[provider_name]
        
        raise ValueError("No TTS providers available")
    
    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "mp3",
        output_file: Optional[Path] = None
    ) -> TTSResponse:
        """Synthesize speech using the specified or default provider."""
        client = self.get_client(provider)
        
        # Use default voice if not specified
        if not voice:
            if client.provider_name == "openai":
                voice = self.config.audio.voice if hasattr(self.config.audio, 'voice') else "alloy"
            else:
                # Get first available voice
                voices = await client.get_available_voices()
                voice = voices[0].id if voices else "default"
        
        # Validate voice
        if not client.validate_voice(voice):
            logger.warning(f"Invalid voice '{voice}' for provider '{client.provider_name}', using default")
            voices = await client.get_available_voices()
            voice = voices[0].id if voices else "default"
        
        request = TTSRequest(
            text=text,
            voice=voice,
            speed=speed,
            output_format=output_format
        )
        
        try:
            response = await client.synthesize_speech(request)
            
            # Save to file if specified
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'wb') as f:
                    f.write(response.audio_data)
                logger.info(f"Audio saved to: {output_file}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error with TTS provider {client.provider_name}: {e}")
            
            # Try fallback if not already using fallback
            if provider and provider != client.provider_name:
                logger.info("Attempting fallback TTS provider...")
                fallback_client = self.get_client(None)
                request.voice = "alloy" if fallback_client.provider_name == "openai" else voice
                return await fallback_client.synthesize_speech(request)
            raise
    
    async def get_available_voices(self, provider: Optional[str] = None) -> Dict[str, List[Voice]]:
        """Get available voices from all or specified providers."""
        if provider:
            if provider in self.clients:
                voices = await self.clients[provider].get_available_voices()
                return {provider: voices}
            else:
                return {}
        
        all_voices = {}
        for provider_name, client in self.clients.items():
            try:
                voices = await client.get_available_voices()
                all_voices[provider_name] = voices
            except Exception as e:
                logger.error(f"Error getting voices from {provider_name}: {e}")
                all_voices[provider_name] = []
        
        return all_voices
    
    def list_available_providers(self) -> List[str]:
        """List all available TTS providers."""
        return list(self.clients.keys())
    
    def estimate_audio_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate audio duration in seconds based on text length."""
        # Average speaking rate is about 150-160 words per minute
        # We'll use 150 WPM as base rate
        words = len(text.split())
        base_duration = (words / 150) * 60  # Convert to seconds
        return base_duration / speed


# Global TTS manager instance
tts_manager = TTSManager()