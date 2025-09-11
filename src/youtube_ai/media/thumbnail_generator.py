import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import io
import base64

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import requests

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.ai.ai_manager import llm_manager

logger = get_logger(__name__)


class ThumbnailStyle(Enum):
    MINIMALIST = "minimalist"
    BOLD = "bold"
    TECH = "tech"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    GAMING = "gaming"
    BUSINESS = "business"
    TUTORIAL = "tutorial"


class ThumbnailLayout(Enum):
    LEFT_TEXT = "left_text"
    RIGHT_TEXT = "right_text"
    CENTER_TEXT = "center_text"
    TOP_TEXT = "top_text"
    BOTTOM_TEXT = "bottom_text"
    SPLIT = "split"


@dataclass
class ThumbnailConfig:
    """Configuration for thumbnail generation."""
    width: int = 1280
    height: int = 720
    style: ThumbnailStyle = ThumbnailStyle.MINIMALIST
    layout: ThumbnailLayout = ThumbnailLayout.CENTER_TEXT
    background_color: str = "#1a1a1a"
    text_color: str = "#ffffff"
    accent_color: str = "#ff6b35"
    font_size: int = 64
    title_font_size: int = 72
    subtitle_font_size: int = 48
    padding: int = 40
    border_radius: int = 20
    use_gradient: bool = True
    add_glow: bool = False
    add_shadow: bool = True


@dataclass
class ThumbnailElements:
    """Elements to include in thumbnail."""
    title: str
    subtitle: Optional[str] = None
    background_image: Optional[Path] = None
    overlay_image: Optional[Path] = None
    icon: Optional[str] = None
    number: Optional[str] = None
    emoji: Optional[str] = None


@dataclass
class ThumbnailVariant:
    """A generated thumbnail variant."""
    image: Image.Image
    config: ThumbnailConfig
    elements: ThumbnailElements
    style_description: str
    file_path: Optional[Path] = None


class ThumbnailGenerator:
    """Generates YouTube thumbnails with various styles and layouts."""
    
    def __init__(self):
        self.config = config_manager.load_config()
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_ai_thumbnails"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Style configurations
        self.style_configs = {
            ThumbnailStyle.MINIMALIST: {
                "background_color": "#ffffff",
                "text_color": "#2c3e50",
                "accent_color": "#3498db",
                "use_gradient": False,
                "add_glow": False,
                "add_shadow": False,
                "font_weight": "normal"
            },
            ThumbnailStyle.BOLD: {
                "background_color": "#000000",
                "text_color": "#ffffff",
                "accent_color": "#ff3333",
                "use_gradient": True,
                "add_glow": True,
                "add_shadow": True,
                "font_weight": "bold"
            },
            ThumbnailStyle.TECH: {
                "background_color": "#0f1419",
                "text_color": "#00d4ff",
                "accent_color": "#ff6b35",
                "use_gradient": True,
                "add_glow": True,
                "add_shadow": False,
                "font_weight": "bold"
            },
            ThumbnailStyle.EDUCATIONAL: {
                "background_color": "#f8f9fa",
                "text_color": "#212529",
                "accent_color": "#007bff",
                "use_gradient": False,
                "add_glow": False,
                "add_shadow": True,
                "font_weight": "normal"
            },
            ThumbnailStyle.ENTERTAINMENT: {
                "background_color": "#ff1744",
                "text_color": "#ffffff",
                "accent_color": "#ffeb3b",
                "use_gradient": True,
                "add_glow": True,
                "add_shadow": True,
                "font_weight": "bold"
            },
            ThumbnailStyle.GAMING: {
                "background_color": "#9c27b0",
                "text_color": "#ffffff",
                "accent_color": "#00ff41",
                "use_gradient": True,
                "add_glow": True,
                "add_shadow": True,
                "font_weight": "bold"
            }
        }
    
    async def generate_thumbnail(
        self,
        title: str,
        style: ThumbnailStyle = ThumbnailStyle.MINIMALIST,
        layout: ThumbnailLayout = ThumbnailLayout.CENTER_TEXT,
        subtitle: Optional[str] = None,
        background_image: Optional[Path] = None,
        custom_config: Optional[ThumbnailConfig] = None,
        output_file: Optional[Path] = None
    ) -> ThumbnailVariant:
        """Generate a single thumbnail."""
        # Create config
        config = custom_config or ThumbnailConfig()
        config.style = style
        config.layout = layout
        
        # Apply style-specific settings
        style_settings = self.style_configs.get(style, {})
        for key, value in style_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create elements
        elements = ThumbnailElements(
            title=title,
            subtitle=subtitle,
            background_image=background_image
        )
        
        # Generate thumbnail
        image = await self._create_thumbnail_image(config, elements)
        
        # Save if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(output_file), "PNG", quality=95)
        
        return ThumbnailVariant(
            image=image,
            config=config,
            elements=elements,
            style_description=f"{style.value} style with {layout.value} layout",
            file_path=output_file
        )
    
    async def generate_multiple_variants(
        self,
        title: str,
        styles: List[ThumbnailStyle] = None,
        layouts: List[ThumbnailLayout] = None,
        subtitle: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> List[ThumbnailVariant]:
        """Generate multiple thumbnail variants."""
        if styles is None:
            styles = [ThumbnailStyle.MINIMALIST, ThumbnailStyle.BOLD, ThumbnailStyle.TECH]
        
        if layouts is None:
            layouts = [ThumbnailLayout.CENTER_TEXT, ThumbnailLayout.LEFT_TEXT]
        
        variants = []
        
        for style in styles:
            for layout in layouts:
                output_file = None
                if output_dir:
                    filename = f"thumbnail_{style.value}_{layout.value}.png"
                    output_file = output_dir / filename
                
                variant = await self.generate_thumbnail(
                    title=title,
                    style=style,
                    layout=layout,
                    subtitle=subtitle,
                    output_file=output_file
                )
                
                variants.append(variant)
        
        logger.info(f"Generated {len(variants)} thumbnail variants")
        return variants
    
    async def generate_ai_optimized_thumbnail(
        self,
        title: str,
        content_description: str,
        target_audience: str = "general",
        provider: Optional[str] = None
    ) -> ThumbnailVariant:
        """Generate a thumbnail optimized using AI suggestions."""
        # Get AI recommendations for thumbnail design
        recommendations = await self._get_ai_thumbnail_recommendations(
            title, content_description, target_audience, provider
        )
        
        # Parse recommendations and create config
        config = self._parse_ai_recommendations(recommendations)
        
        # Generate thumbnail
        elements = ThumbnailElements(
            title=title,
            subtitle=recommendations.get('subtitle'),
            emoji=recommendations.get('emoji')
        )
        
        image = await self._create_thumbnail_image(config, elements)
        
        return ThumbnailVariant(
            image=image,
            config=config,
            elements=elements,
            style_description="AI-optimized design"
        )
    
    async def _create_thumbnail_image(
        self,
        config: ThumbnailConfig,
        elements: ThumbnailElements
    ) -> Image.Image:
        """Create the actual thumbnail image."""
        # Create base image
        image = Image.new('RGB', (config.width, config.height), config.background_color)
        draw = ImageDraw.Draw(image)
        
        # Add background if specified
        if elements.background_image and elements.background_image.exists():
            bg_img = await self._prepare_background_image(
                elements.background_image, config.width, config.height
            )
            image.paste(bg_img, (0, 0))
        
        # Add gradient overlay if enabled
        if config.use_gradient:
            gradient = self._create_gradient_overlay(config)
            image = Image.alpha_composite(image.convert('RGBA'), gradient).convert('RGB')
        
        # Add text elements
        await self._add_text_elements(image, config, elements)
        
        # Add decorative elements
        await self._add_decorative_elements(image, config, elements)
        
        # Apply effects
        if config.add_glow or config.add_shadow:
            image = await self._apply_effects(image, config)
        
        return image
    
    async def _prepare_background_image(
        self,
        bg_path: Path,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """Prepare background image with proper sizing and effects."""
        bg_img = Image.open(bg_path).convert('RGB')
        
        # Resize to cover the entire thumbnail
        bg_ratio = bg_img.width / bg_img.height
        target_ratio = target_width / target_height
        
        if bg_ratio > target_ratio:
            # Background is wider, scale by height
            new_height = target_height
            new_width = int(target_height * bg_ratio)
        else:
            # Background is taller, scale by width
            new_width = target_width
            new_height = int(target_width / bg_ratio)
        
        bg_img = bg_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crop to center
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        bg_img = bg_img.crop((left, top, left + target_width, top + target_height))
        
        # Apply subtle blur and darkening for text readability
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=1))
        enhancer = ImageEnhance.Brightness(bg_img)
        bg_img = enhancer.enhance(0.7)  # Darken to 70%
        
        return bg_img
    
    def _create_gradient_overlay(self, config: ThumbnailConfig) -> Image.Image:
        """Create a gradient overlay."""
        gradient = Image.new('RGBA', (config.width, config.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)
        
        # Create vertical gradient
        for y in range(config.height):
            alpha = int(255 * (y / config.height) * 0.5)  # Max 50% opacity
            color = (*self._hex_to_rgb(config.background_color), alpha)
            draw.line([(0, y), (config.width, y)], fill=color)
        
        return gradient
    
    async def _add_text_elements(
        self,
        image: Image.Image,
        config: ThumbnailConfig,
        elements: ThumbnailElements
    ) -> None:
        """Add text elements to the thumbnail."""
        draw = ImageDraw.Draw(image)
        
        # Load fonts (fallback to default if custom fonts not available)
        try:
            title_font = ImageFont.truetype("arial.ttf", config.title_font_size)
            subtitle_font = ImageFont.truetype("arial.ttf", config.subtitle_font_size)
        except OSError:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Prepare title text
        title_text = elements.title.upper() if config.style in [ThumbnailStyle.BOLD, ThumbnailStyle.GAMING] else elements.title
        
        # Word wrap title if needed
        wrapped_title = self._wrap_text(title_text, title_font, config.width - 2 * config.padding)
        
        # Calculate text positions based on layout
        title_pos, subtitle_pos = self._calculate_text_positions(
            config, wrapped_title, elements.subtitle, title_font, subtitle_font
        )
        
        # Add text shadow if enabled
        if config.add_shadow:
            shadow_offset = 3
            self._draw_text_with_outline(
                draw, title_pos, wrapped_title, title_font,
                (0, 0, 0), shadow_offset
            )
            if elements.subtitle:
                self._draw_text_with_outline(
                    draw, (subtitle_pos[0] + shadow_offset, subtitle_pos[1] + shadow_offset),
                    elements.subtitle, subtitle_font, (0, 0, 0)
                )
        
        # Add main text
        self._draw_text_with_outline(
            draw, title_pos, wrapped_title, title_font,
            config.text_color, 2 if config.style == ThumbnailStyle.BOLD else 1
        )
        
        if elements.subtitle:
            self._draw_text_with_outline(
                draw, subtitle_pos, elements.subtitle, subtitle_font,
                config.accent_color, 1
            )
    
    async def _add_decorative_elements(
        self,
        image: Image.Image,
        config: ThumbnailConfig,
        elements: ThumbnailElements
    ) -> None:
        """Add decorative elements like borders, shapes, icons."""
        draw = ImageDraw.Draw(image)
        
        # Add accent border or shapes based on style
        if config.style == ThumbnailStyle.TECH:
            # Add tech-style corner elements
            corner_size = 40
            accent_color = self._hex_to_rgb(config.accent_color)
            
            # Top-left corner
            draw.polygon([
                (0, 0), (corner_size, 0), (0, corner_size)
            ], fill=accent_color)
            
            # Bottom-right corner
            draw.polygon([
                (config.width, config.height),
                (config.width - corner_size, config.height),
                (config.width, config.height - corner_size)
            ], fill=accent_color)
        
        elif config.style == ThumbnailStyle.GAMING:
            # Add gaming-style elements
            self._add_gaming_elements(draw, config)
        
        # Add emoji if specified
        if elements.emoji:
            await self._add_emoji(image, elements.emoji, config)
        
        # Add number if specified
        if elements.number:
            await self._add_number_badge(image, elements.number, config)
    
    def _add_gaming_elements(self, draw: ImageDraw.Draw, config: ThumbnailConfig):
        """Add gaming-style decorative elements."""
        accent_color = self._hex_to_rgb(config.accent_color)
        
        # Add diagonal stripes
        stripe_width = 10
        for i in range(0, config.width + config.height, stripe_width * 3):
            draw.polygon([
                (i, 0), (i + stripe_width, 0),
                (i + stripe_width - config.height, config.height),
                (i - config.height, config.height)
            ], fill=(*accent_color, 50))  # Semi-transparent
    
    async def _add_emoji(self, image: Image.Image, emoji: str, config: ThumbnailConfig):
        """Add emoji to thumbnail."""
        # For now, we'll use text rendering for emoji
        # In a production environment, you might want to use proper emoji fonts
        draw = ImageDraw.Draw(image)
        
        try:
            emoji_font = ImageFont.truetype("seguiemj.ttf", 80)  # Windows emoji font
        except OSError:
            emoji_font = ImageFont.load_default()
        
        # Position emoji in top-right corner
        emoji_pos = (config.width - 120, 20)
        draw.text(emoji_pos, emoji, font=emoji_font, fill=config.text_color)
    
    async def _add_number_badge(self, image: Image.Image, number: str, config: ThumbnailConfig):
        """Add a number badge to the thumbnail."""
        draw = ImageDraw.Draw(image)
        
        # Create circular badge
        badge_size = 80
        badge_pos = (config.width - badge_size - 20, 20)
        
        # Draw circle
        draw.ellipse([
            badge_pos,
            (badge_pos[0] + badge_size, badge_pos[1] + badge_size)
        ], fill=config.accent_color, outline=config.text_color, width=3)
        
        # Add number text
        try:
            number_font = ImageFont.truetype("arial.ttf", 36)
        except OSError:
            number_font = ImageFont.load_default()
        
        # Center text in circle
        text_bbox = draw.textbbox((0, 0), number, font=number_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_pos = (
            badge_pos[0] + (badge_size - text_width) // 2,
            badge_pos[1] + (badge_size - text_height) // 2
        )
        
        draw.text(text_pos, number, font=number_font, fill=config.text_color)
    
    async def _apply_effects(self, image: Image.Image, config: ThumbnailConfig) -> Image.Image:
        """Apply visual effects to the thumbnail."""
        if config.add_glow:
            # Create a simple glow effect by applying blur to a brightened version
            glow = image.copy()
            enhancer = ImageEnhance.Brightness(glow)
            glow = enhancer.enhance(1.5)
            glow = glow.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Blend with original
            image = Image.blend(image, glow, 0.3)
        
        return image
    
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        """Wrap text to fit within specified width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, break it
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _calculate_text_positions(
        self,
        config: ThumbnailConfig,
        title_lines: List[str],
        subtitle: Optional[str],
        title_font: ImageFont.ImageFont,
        subtitle_font: ImageFont.ImageFont
    ) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Calculate optimal text positions based on layout."""
        # Calculate total text height
        title_height = len(title_lines) * (title_font.getbbox('A')[3] - title_font.getbbox('A')[1])
        subtitle_height = 0
        if subtitle:
            subtitle_height = subtitle_font.getbbox(subtitle)[3] - subtitle_font.getbbox(subtitle)[1]
        
        total_height = title_height + subtitle_height + (20 if subtitle else 0)
        
        # Base positions
        if config.layout == ThumbnailLayout.CENTER_TEXT:
            title_y = (config.height - total_height) // 2
            title_x = config.padding
        elif config.layout == ThumbnailLayout.TOP_TEXT:
            title_y = config.padding
            title_x = config.padding
        elif config.layout == ThumbnailLayout.BOTTOM_TEXT:
            title_y = config.height - total_height - config.padding
            title_x = config.padding
        elif config.layout == ThumbnailLayout.LEFT_TEXT:
            title_y = (config.height - total_height) // 2
            title_x = config.padding
        elif config.layout == ThumbnailLayout.RIGHT_TEXT:
            title_y = (config.height - total_height) // 2
            title_x = config.width // 2
        else:
            title_y = (config.height - total_height) // 2
            title_x = config.padding
        
        title_pos = (title_x, title_y)
        
        subtitle_pos = None
        if subtitle:
            subtitle_pos = (title_x, title_y + title_height + 20)
        
        return title_pos, subtitle_pos
    
    def _draw_text_with_outline(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[int, int],
        text: Union[str, List[str]],
        font: ImageFont.ImageFont,
        color: Union[str, Tuple[int, int, int]],
        outline_width: int = 1
    ):
        """Draw text with outline for better visibility."""
        x, y = position
        
        if isinstance(text, list):
            # Multi-line text
            line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
            for i, line in enumerate(text):
                line_y = y + i * (line_height + 5)
                self._draw_single_line_with_outline(
                    draw, (x, line_y), line, font, color, outline_width
                )
        else:
            self._draw_single_line_with_outline(
                draw, position, text, font, color, outline_width
            )
    
    def _draw_single_line_with_outline(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.ImageFont,
        color: Union[str, Tuple[int, int, int]],
        outline_width: int
    ):
        """Draw a single line of text with outline."""
        x, y = position
        
        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        
        # Draw main text
        text_color = self._hex_to_rgb(color) if isinstance(color, str) else color
        draw.text(position, text, font=font, fill=text_color)
    
    async def _get_ai_thumbnail_recommendations(
        self,
        title: str,
        content_description: str,
        target_audience: str,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI recommendations for thumbnail design."""
        prompt = f"""
        Create thumbnail design recommendations for a YouTube video with the following details:
        
        Title: {title}
        Content: {content_description}
        Target Audience: {target_audience}
        
        Please provide recommendations in JSON format with the following fields:
        - style: one of "minimalist", "bold", "tech", "educational", "entertainment", "gaming"
        - layout: one of "center_text", "left_text", "right_text", "top_text", "bottom_text"
        - background_color: hex color code
        - text_color: hex color code
        - accent_color: hex color code
        - subtitle: optional subtitle text
        - emoji: optional emoji to include
        - design_reasoning: explanation of design choices
        
        Focus on creating a thumbnail that will attract clicks while accurately representing the content.
        """
        
        try:
            response = await llm_manager.generate_completion(
                prompt=prompt,
                provider=provider,
                temperature=0.7
            )
            
            # Try to parse JSON response
            import json
            recommendations = json.loads(response.content)
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to get AI recommendations: {e}")
            # Return default recommendations
            return {
                "style": "minimalist",
                "layout": "center_text",
                "background_color": "#1a1a1a",
                "text_color": "#ffffff",
                "accent_color": "#ff6b35",
                "design_reasoning": "Default design due to AI recommendation failure"
            }
    
    def _parse_ai_recommendations(self, recommendations: Dict[str, Any]) -> ThumbnailConfig:
        """Parse AI recommendations into ThumbnailConfig."""
        config = ThumbnailConfig()
        
        # Map style
        style_map = {
            "minimalist": ThumbnailStyle.MINIMALIST,
            "bold": ThumbnailStyle.BOLD,
            "tech": ThumbnailStyle.TECH,
            "educational": ThumbnailStyle.EDUCATIONAL,
            "entertainment": ThumbnailStyle.ENTERTAINMENT,
            "gaming": ThumbnailStyle.GAMING
        }
        
        config.style = style_map.get(recommendations.get('style', 'minimalist'), ThumbnailStyle.MINIMALIST)
        
        # Map layout
        layout_map = {
            "center_text": ThumbnailLayout.CENTER_TEXT,
            "left_text": ThumbnailLayout.LEFT_TEXT,
            "right_text": ThumbnailLayout.RIGHT_TEXT,
            "top_text": ThumbnailLayout.TOP_TEXT,
            "bottom_text": ThumbnailLayout.BOTTOM_TEXT
        }
        
        config.layout = layout_map.get(recommendations.get('layout', 'center_text'), ThumbnailLayout.CENTER_TEXT)
        
        # Apply colors
        config.background_color = recommendations.get('background_color', '#1a1a1a')
        config.text_color = recommendations.get('text_color', '#ffffff')
        config.accent_color = recommendations.get('accent_color', '#ff6b35')
        
        return config
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# Global thumbnail generator instance
thumbnail_generator = ThumbnailGenerator()