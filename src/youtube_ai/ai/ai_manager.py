import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import openai
import anthropic
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

logger = get_logger(__name__)


class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class AIMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class AIResponse:
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider_name = self.__class__.__name__.lower().replace('client', '')
    
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[AIMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """Generate a completion from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[AIMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming completion from the LLM."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate_completion(
        self,
        messages: List[AIMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """Generate completion using OpenAI API."""
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[AIMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate streaming completion using OpenAI API."""
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key, model)
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate_completion(
        self,
        messages: List[AIMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """Generate completion using Anthropic API."""
        try:
            # Anthropic expects system message separately
            system_message = None
            claude_messages = []
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    claude_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            response = await self.client.messages.create(
                model=self.model,
                messages=claude_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                **kwargs
            )
            
            return AIResponse(
                content=response.content[0].text,
                provider="anthropic",
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                } if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[AIMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate streaming completion using Anthropic API."""
        try:
            # Anthropic expects system message separately
            system_message = None
            claude_messages = []
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    claude_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            async with self.client.messages.stream(
                model=self.model,
                messages=claude_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise


class LLMManager:
    """Manages multiple LLM providers with fallback support."""
    
    def __init__(self):
        self.clients: Dict[str, BaseLLMClient] = {}
        self.config = config_manager.load_config()
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available LLM clients based on configuration."""
        # OpenAI
        if self.config.ai.openai_api_key:
            self.clients["openai"] = OpenAIClient(
                api_key=self.config.ai.openai_api_key,
                model="gpt-4"
            )
            logger.debug("Initialized OpenAI client")
        
        # Anthropic
        if self.config.ai.anthropic_api_key:
            self.clients["anthropic"] = AnthropicClient(
                api_key=self.config.ai.anthropic_api_key,
                model="claude-3-5-sonnet-20241022"
            )
            logger.debug("Initialized Anthropic client")
        
        if not self.clients:
            raise ValueError("No AI providers configured. Please set API keys.")
    
    def get_client(self, provider: Optional[str] = None) -> BaseLLMClient:
        """Get an LLM client for the specified provider."""
        if provider and provider in self.clients:
            return self.clients[provider]
        
        # Use default provider
        default_provider = self.config.ai.default_llm
        if default_provider in self.clients:
            return self.clients[default_provider]
        
        # Fallback to any available provider
        if self.clients:
            provider_name = next(iter(self.clients))
            logger.warning(f"Using fallback provider: {provider_name}")
            return self.clients[provider_name]
        
        raise ValueError("No LLM providers available")
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AIResponse:
        """Generate a completion using the specified or default provider."""
        messages = []
        
        if system_prompt:
            messages.append(AIMessage(role="system", content=system_prompt))
        
        messages.append(AIMessage(role="user", content=prompt))
        
        client = self.get_client(provider)
        
        try:
            return await client.generate_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error with provider {client.provider_name}: {e}")
            
            # Try fallback if not already using fallback
            if provider and provider != client.provider_name:
                logger.info("Attempting fallback provider...")
                fallback_client = self.get_client(None)
                return await fallback_client.generate_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate a streaming completion."""
        messages = []
        
        if system_prompt:
            messages.append(AIMessage(role="system", content=system_prompt))
        
        messages.append(AIMessage(role="user", content=prompt))
        
        client = self.get_client(provider)
        
        async for chunk in client.generate_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk
    
    def list_available_providers(self) -> List[str]:
        """List all available providers."""
        return list(self.clients.keys())


# Global LLM manager instance
llm_manager = LLMManager()