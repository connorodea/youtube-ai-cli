"""
AI-powered image generation for contextually relevant video visuals.
Analyzes script content and generates images that perfectly match the narrative.
"""

import asyncio
import io
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import openai
from openai import AsyncOpenAI
from PIL import Image
import requests

try:
    import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
    from stability_sdk import client
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

logger = get_logger(__name__)


class ImageProvider(Enum):
    """AI image generation providers."""
    OPENAI = "openai"
    STABILITY = "stability"
    FALLBACK = "fallback"


class ImageStyle(Enum):
    """Visual styles for generated images."""
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    TECH = "tech"
    DOCUMENTARY = "documentary"
    MODERN = "modern"
    MINIMALIST = "minimalist"


@dataclass
class ImagePrompt:
    """A prompt for AI image generation."""
    segment_text: str
    prompt: str
    style: str
    scene_number: int
    duration: float
    keywords: List[str]


@dataclass
class GeneratedImage:
    """A generated image with metadata."""
    file_path: Path
    prompt: str
    segment_text: str
    scene_number: int
    style: str
    provider: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = None


class ScriptAnalyzer:
    """Analyzes script content to extract visual concepts."""
    
    def __init__(self):
        self.config = config_manager.load_config()
    
    def parse_script_segments(self, script: str, num_segments: int = 5) -> List[str]:
        """Parse script into logical segments for image generation."""
        # Remove markdown headers and cleanup
        cleaned_script = re.sub(r'^#.*$', '', script, flags=re.MULTILINE)
        cleaned_script = re.sub(r'\[.*?\]', '', cleaned_script)  # Remove tone directions
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in cleaned_script.split('\n\n') if p.strip()]
        
        # If we have fewer paragraphs than desired segments, split longer ones
        if len(paragraphs) < num_segments:
            # Split longer paragraphs by sentences
            sentences = []
            for para in paragraphs:
                sentences.extend([s.strip() + '.' for s in para.split('.') if s.strip()])
            
            # Group sentences into segments
            segment_size = max(1, len(sentences) // num_segments)
            segments = []
            for i in range(0, len(sentences), segment_size):
                segment = ' '.join(sentences[i:i + segment_size])
                if segment:
                    segments.append(segment)
        else:
            segments = paragraphs[:num_segments]
        
        # Ensure we have exactly num_segments
        while len(segments) < num_segments:
            segments.append(segments[-1] if segments else "Technology and programming concepts")
        
        return segments[:num_segments]
    
    def extract_visual_keywords(self, text: str) -> List[str]:
        """Extract visual keywords from text segment."""
        # Common visual concepts and their mappings
        visual_mappings = {
            'python': ['python programming', 'code editor', 'programming'],
            'programming': ['coding', 'developer workspace', 'computer screen'],
            'artificial intelligence': ['AI technology', 'neural networks', 'machine learning'],
            'machine learning': ['data visualization', 'algorithms', 'AI'],
            'code': ['programming', 'syntax highlighting', 'developer'],
            'data': ['charts', 'graphs', 'analytics'],
            'web': ['websites', 'browser', 'internet'],
            'app': ['mobile interface', 'application', 'smartphone'],
            'game': ['gaming', 'video games', 'controller'],
            'instagram': ['social media', 'smartphone', 'social network'],
            'minecraft': ['gaming', 'blocks', 'creativity'],
            'variable': ['programming concepts', 'code storage'],
            'function': ['programming logic', 'code structure'],
            'loop': ['programming patterns', 'repetition concepts'],
        }
        
        keywords = []
        text_lower = text.lower()
        
        for key, visuals in visual_mappings.items():
            if key in text_lower:
                keywords.extend(visuals)
        
        # Add general concepts based on content type
        if any(word in text_lower for word in ['learn', 'tutorial', 'guide', 'how to']):
            keywords.extend(['education', 'learning', 'knowledge'])
        
        if any(word in text_lower for word in ['technology', 'digital', 'computer']):
            keywords.extend(['technology', 'digital workspace', 'modern tech'])
        
        return keywords[:5]  # Limit to top 5 keywords


class ImageGenerator:
    """Generates contextually relevant images using AI."""
    
    def __init__(self):
        self.config = config_manager.load_config()
        self.openai_client = None
        self.stability_client = None
        
        # Initialize OpenAI client
        if self.config.ai.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.config.ai.openai_api_key)
        
        # Initialize Stability AI client
        if self.config.ai.stability_api_key and STABILITY_AVAILABLE:
            self.stability_client = client.StabilityInference(
                key=self.config.ai.stability_api_key,
                verbose=True,
                engine="stable-diffusion-xl-1024-v1-0"
            )
        
        self.analyzer = ScriptAnalyzer()
    
    def create_image_prompt(
        self, 
        segment_text: str, 
        style: ImageStyle, 
        scene_number: int
    ) -> ImagePrompt:
        """Create an optimized prompt for image generation."""
        keywords = self.analyzer.extract_visual_keywords(segment_text)
        
        # Style-specific prompt modifiers
        style_modifiers = {
            ImageStyle.EDUCATIONAL: "clean, modern educational illustration, professional learning environment",
            ImageStyle.PROFESSIONAL: "sleek corporate design, professional workspace, business atmosphere",
            ImageStyle.TECH: "futuristic technology, digital interface, modern tech workspace",
            ImageStyle.DOCUMENTARY: "realistic, informative, documentary-style visual",
            ImageStyle.MODERN: "contemporary design, minimalist, clean lines",
            ImageStyle.MINIMALIST: "simple, clean, minimal design, subtle colors"
        }
        
        # Create main subject from keywords
        main_subject = ', '.join(keywords[:3]) if keywords else "abstract technology concept"
        
        # Build comprehensive prompt
        base_prompt = f"{main_subject}, {style_modifiers[style]}"
        
        # Add quality and format specifications
        quality_specs = "high quality, professional, 16:9 aspect ratio, youtube thumbnail style"
        
        # Add color scheme based on style
        color_specs = {
            ImageStyle.EDUCATIONAL: "blue and white color scheme, bright and engaging",
            ImageStyle.PROFESSIONAL: "dark blue and silver, sophisticated",
            ImageStyle.TECH: "blue and purple gradients, futuristic",
            ImageStyle.DOCUMENTARY: "natural colors, realistic lighting",
            ImageStyle.MODERN: "clean color palette, modern aesthetics",
            ImageStyle.MINIMALIST: "monochromatic, subtle colors"
        }
        
        full_prompt = f"{base_prompt}, {color_specs[style]}, {quality_specs}"
        
        return ImagePrompt(
            segment_text=segment_text,
            prompt=full_prompt,
            style=style.value,
            scene_number=scene_number,
            duration=5.0,  # Default duration per image
            keywords=keywords
        )
    
    async def generate_image(
        self, 
        prompt: ImagePrompt, 
        output_path: Path, 
        provider: ImageProvider = ImageProvider.OPENAI
    ) -> GeneratedImage:
        """Generate a single image using the specified AI provider."""
        if provider == ImageProvider.OPENAI:
            return await self._generate_openai_image(prompt, output_path)
        elif provider == ImageProvider.STABILITY:
            return await self._generate_stability_image(prompt, output_path)
        else:
            return self._create_placeholder_image(prompt, output_path)
    
    async def _generate_openai_image(self, prompt: ImagePrompt, output_path: Path) -> GeneratedImage:
        """Generate image using OpenAI DALL-E."""
        if not self.openai_client:
            logger.warning("OpenAI client not available, creating placeholder")
            return self._create_placeholder_image(prompt, output_path)
        
        try:
            logger.info(f"Generating DALL-E image for scene {prompt.scene_number}: {prompt.keywords[:2]}")
            
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt.prompt,
                size="1792x1024",  # 16:9 aspect ratio
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            
            # Download and save image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            # Save image
            with open(output_path, 'wb') as f:
                f.write(image_response.content)
            
            logger.info(f"✅ Generated DALL-E image: {output_path.name}")
            
            return GeneratedImage(
                file_path=output_path,
                prompt=prompt.prompt,
                segment_text=prompt.segment_text,
                scene_number=prompt.scene_number,
                style=prompt.style,
                provider="openai",
                url=image_url,
                metadata={
                    "model": "dall-e-3",
                    "keywords": prompt.keywords,
                    "generated_at": str(output_path.stat().st_mtime)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating image with DALL-E: {e}")
            logger.info("Creating fallback gradient image")
            return self._create_placeholder_image(prompt, output_path)
    
    async def _generate_stability_image(self, prompt: ImagePrompt, output_path: Path) -> GeneratedImage:
        """Generate image using Stability AI."""
        if not self.stability_client:
            logger.warning("Stability AI client not available, creating placeholder")
            return self._create_placeholder_image(prompt, output_path)
        
        try:
            logger.info(f"Generating Stability AI image for scene {prompt.scene_number}: {prompt.keywords[:2]}")
            
            # Generate image with Stability AI
            answers = self.stability_client.generate(
                prompt=prompt.prompt,
                seed=123463,  # Fixed seed for reproducibility 
                steps=30,  # Generation steps
                cfg_scale=8.0,  # Prompt adherence
                width=1024,
                height=1024,
                samples=1,
                sampler=generation.SAMPLER_K_DPMPP_2M
            )
            
            # Save the generated image
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        logger.warning("Image was filtered by safety filter")
                        return self._create_placeholder_image(prompt, output_path)
                    
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        
                        # Resize to 16:9 aspect ratio for YouTube
                        img = img.resize((1792, 1024), Image.Resampling.LANCZOS)
                        img.save(output_path, 'PNG', quality=95)
                        
                        logger.info(f"✅ Generated Stability AI image: {output_path.name}")
                        
                        return GeneratedImage(
                            file_path=output_path,
                            prompt=prompt.prompt,
                            segment_text=prompt.segment_text,
                            scene_number=prompt.scene_number,
                            style=prompt.style,
                            provider="stability",
                            metadata={
                                "model": "stable-diffusion-xl-1024-v1-0",
                                "keywords": prompt.keywords,
                                "steps": 30,
                                "cfg_scale": 8.0,
                                "generated_at": str(output_path.stat().st_mtime)
                            }
                        )
            
            # If no valid image was generated, create placeholder
            return self._create_placeholder_image(prompt, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image with Stability AI: {e}")
            logger.info("Creating fallback gradient image")
            return self._create_placeholder_image(prompt, output_path)
    
    def _create_placeholder_image(self, prompt: ImagePrompt, output_path: Path) -> GeneratedImage:
        """Create a stylized placeholder image when AI generation fails."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create base image
        width, height = 1792, 1024
        img = Image.new('RGB', (width, height), '#1e293b')
        draw = ImageDraw.Draw(img)
        
        # Create gradient background
        for y in range(height):
            color_ratio = y / height
            r = int(30 + color_ratio * 50)
            g = int(41 + color_ratio * 60)
            b = int(59 + color_ratio * 80)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add visual elements based on keywords
        if prompt.keywords:
            self._add_visual_elements(draw, prompt.keywords, width, height)
        
        # Save image
        img.save(output_path, 'PNG', quality=95)
        
        return GeneratedImage(
            file_path=output_path,
            prompt=prompt.prompt,
            segment_text=prompt.segment_text,
            scene_number=prompt.scene_number,
            style=prompt.style,
            provider="fallback",
            metadata={
                "model": "placeholder",
                "keywords": prompt.keywords,
                "type": "gradient_with_elements"
            }
        )
    
    def _add_visual_elements(self, draw, keywords: List[str], width: int, height: int):
        """Add simple visual elements based on keywords."""
        import random
        
        # Add geometric shapes representing concepts
        for i, keyword in enumerate(keywords[:3]):
            x = width // 4 + i * (width // 4)
            y = height // 2
            
            if 'programming' in keyword or 'code' in keyword:
                # Draw code-like rectangles
                for j in range(3):
                    draw.rectangle(
                        [x - 50 + j * 20, y - 30 + j * 15, x + 50 + j * 20, y - 15 + j * 15],
                        fill=(70, 130, 180, 100)
                    )
            elif 'technology' in keyword or 'digital' in keyword:
                # Draw tech circles
                draw.ellipse([x - 40, y - 40, x + 40, y + 40], fill=(100, 150, 200, 120))
            else:
                # Draw abstract shapes
                draw.polygon(
                    [(x - 30, y + 20), (x, y - 30), (x + 30, y + 20)],
                    fill=(80, 120, 160, 100)
                )
    
    async def generate_script_visuals(
        self,
        script: str,
        output_dir: Path,
        style: ImageStyle = ImageStyle.EDUCATIONAL,
        num_images: int = 5,
        provider: ImageProvider = ImageProvider.OPENAI
    ) -> List[GeneratedImage]:
        """Generate a complete set of visuals for a script."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse script into segments
        segments = self.analyzer.parse_script_segments(script, num_images)
        
        logger.info(f"🎨 Generating {num_images} AI-powered visuals...")
        logger.info(f"Style: {style.value}")
        logger.info(f"Provider: {provider.value}")
        
        # Generate images for each segment
        generated_images = []
        
        for i, segment in enumerate(segments):
            # Create prompt
            prompt = self.create_image_prompt(segment, style, i + 1)
            
            # Generate image
            output_path = output_dir / f"ai_visual_{i+1:02d}.png"
            
            image = await self.generate_image(prompt, output_path, provider)
            generated_images.append(image)
        
        logger.info(f"✅ Generated {len(generated_images)} contextual visuals")
        return generated_images


# Create global instance
image_generator = ImageGenerator()