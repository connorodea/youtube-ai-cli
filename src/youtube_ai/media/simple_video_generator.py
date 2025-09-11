"""
Simple video generation without MoviePy dependency.
Creates video assets and provides instructions for final assembly.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.ai.tts_client import tts_manager

logger = get_logger(__name__)


@dataclass
class VideoAsset:
    """A video asset created for later assembly."""
    file_path: Path
    asset_type: str  # "audio", "image", "subtitle", "script"
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoProject:
    """Complete video project ready for assembly."""
    title: str
    script: str
    audio_file: Path
    background_images: List[Path]
    subtitle_file: Optional[Path] = None
    duration: float = 0.0
    resolution: Tuple[int, int] = (1920, 1080)
    style: str = "educational"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleVideoGenerator:
    """Simple video generator that creates assets for manual or external assembly."""
    
    def __init__(self):
        self.config = config_manager.load_config()
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_ai_simple"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.temp_dir / "audio").mkdir(exist_ok=True)
        (self.temp_dir / "images").mkdir(exist_ok=True)
        (self.temp_dir / "subtitles").mkdir(exist_ok=True)
        (self.temp_dir / "scripts").mkdir(exist_ok=True)
    
    async def create_video_project(
        self,
        script: str,
        output_dir: Path,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
        style: str = "educational",
        num_backgrounds: int = 5
    ) -> VideoProject:
        """Create a complete video project with all assets."""
        try:
            logger.info(f"Creating video project in: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Generate audio
            logger.info("Generating professional voiceover...")
            audio_file = await self._generate_audio(script, output_dir, voice, provider)
            
            # Step 2: Create background images
            logger.info("Creating background images...")
            background_images = await self._create_background_images(
                script, output_dir, style, num_backgrounds
            )
            
            # Step 3: Generate subtitle file
            logger.info("Creating subtitle file...")
            subtitle_file = await self._create_subtitle_file(script, output_dir, audio_file)
            
            # Step 4: Save script
            script_file = output_dir / "script.txt"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script)
            
            # Step 5: Create assembly instructions
            instructions_file = await self._create_assembly_instructions(
                output_dir, audio_file, background_images, subtitle_file
            )
            
            # Get audio duration (estimate)
            duration = await self._estimate_audio_duration(script)
            
            project = VideoProject(
                title="Generated Video Project",
                script=script,
                audio_file=audio_file,
                background_images=background_images,
                subtitle_file=subtitle_file,
                duration=duration,
                style=style,
                metadata={
                    "created_at": str(Path().absolute()),
                    "voice": voice or "alloy",
                    "provider": provider or "openai",
                    "instructions_file": str(instructions_file)
                }
            )
            
            logger.info(f"Video project created successfully!")
            logger.info(f"📁 Project directory: {output_dir}")
            logger.info(f"🎵 Audio: {audio_file.name}")
            logger.info(f"🖼️  Images: {len(background_images)} backgrounds created")
            logger.info(f"📝 Subtitles: {subtitle_file.name if subtitle_file else 'None'}")
            logger.info(f"📋 Instructions: {instructions_file.name}")
            
            return project
            
        except Exception as e:
            logger.error(f"Error creating video project: {e}")
            raise
    
    async def _generate_audio(
        self,
        script: str,
        output_dir: Path,
        voice: Optional[str],
        provider: Optional[str]
    ) -> Path:
        """Generate high-quality audio narration."""
        audio_file = output_dir / "narration.mp3"
        
        # Generate audio using TTS
        audio_response = await tts_manager.synthesize_speech(
            text=script,
            voice=voice or "alloy",
            provider=provider,
            speed=1.0,
            output_file=audio_file
        )
        
        logger.info(f"✅ Audio generated: {audio_file}")
        return audio_file
    
    async def _create_background_images(
        self,
        script: str,
        output_dir: Path,
        style: str,
        num_images: int
    ) -> List[Path]:
        """Create professional background images."""
        images_dir = output_dir / "backgrounds"
        images_dir.mkdir(exist_ok=True)
        
        # Define style color schemes
        color_schemes = {
            "educational": [
                ["#1e3a8a", "#3b82f6"],  # Blue gradient
                ["#059669", "#10b981"],  # Green gradient
                ["#7c3aed", "#a855f7"],  # Purple gradient
                ["#dc2626", "#ef4444"],  # Red gradient
                ["#ea580c", "#f97316"],  # Orange gradient
            ],
            "cinematic": [
                ["#1a1a1a", "#404040"],  # Dark grays
                ["#2d1b69", "#5b21b6"],  # Deep purple
                ["#7c2d12", "#dc2626"],  # Dark red
                ["#14532d", "#166534"],  # Dark green
                ["#78350f", "#a16207"],  # Dark amber
            ],
            "professional": [
                ["#1e293b", "#334155"],  # Slate
                ["#1f2937", "#374151"],  # Gray
                ["#1e40af", "#2563eb"],  # Blue
                ["#7c3aed", "#8b5cf6"],  # Violet
                ["#059669", "#10b981"],  # Emerald
            ]
        }
        
        schemes = color_schemes.get(style, color_schemes["educational"])
        background_images = []
        
        for i in range(min(num_images, len(schemes))):
            image_file = images_dir / f"background_{i+1:02d}.png"
            colors = schemes[i % len(schemes)]
            
            # Create gradient background
            self._create_gradient_background(
                image_file, colors[0], colors[1], (1920, 1080)
            )
            
            background_images.append(image_file)
        
        logger.info(f"✅ Created {len(background_images)} background images")
        return background_images
    
    def _create_gradient_background(
        self,
        output_path: Path,
        color1: str,
        color2: str,
        size: Tuple[int, int]
    ) -> None:
        """Create a professional gradient background."""
        width, height = size
        
        # Convert hex to RGB
        rgb1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        rgb2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Create gradient image
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Vertical gradient
        for y in range(height):
            ratio = y / height
            r = int(rgb1[0] * (1 - ratio) + rgb2[0] * ratio)
            g = int(rgb1[1] * (1 - ratio) + rgb2[1] * ratio)
            b = int(rgb1[2] * (1 - ratio) + rgb2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add subtle texture
        pixels = np.array(image)
        noise = np.random.randint(-8, 8, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add subtle vignette
        Y, X = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        vignette = 1 - (dist_from_center / max_dist) * 0.15
        vignette = np.clip(vignette, 0.85, 1.0)
        
        pixels = pixels * vignette[:, :, np.newaxis]
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        
        # Save final image
        final_image = Image.fromarray(pixels)
        final_image.save(output_path, "PNG", quality=95)
    
    async def _create_subtitle_file(
        self,
        script: str,
        output_dir: Path,
        audio_file: Path
    ) -> Optional[Path]:
        """Create subtitle file in SRT format."""
        try:
            subtitle_file = output_dir / "subtitles.srt"
            
            # Split script into segments
            sentences = self._split_into_sentences(script)
            if not sentences:
                return None
            
            # Estimate timing (rough approximation)
            duration = await self._estimate_audio_duration(script)
            segment_duration = duration / len(sentences)
            
            # Create SRT content
            srt_content = []
            current_time = 0.0
            
            for i, sentence in enumerate(sentences):
                start_time = current_time
                end_time = current_time + segment_duration
                
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)
                
                srt_content.append(f"{i + 1}")
                srt_content.append(f"{start_srt} --> {end_srt}")
                srt_content.append(sentence.strip())
                srt_content.append("")  # Empty line
                
                current_time = end_time
            
            # Write SRT file
            with open(subtitle_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            logger.info(f"✅ Subtitle file created: {subtitle_file}")
            return subtitle_file
            
        except Exception as e:
            logger.warning(f"Could not create subtitle file: {e}")
            return None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    async def _estimate_audio_duration(self, script: str) -> float:
        """Estimate audio duration based on text length."""
        # Average reading speed: ~150 words per minute
        words = len(script.split())
        duration = (words / 150) * 60  # Convert to seconds
        return max(duration, 10.0)  # Minimum 10 seconds
    
    async def _create_assembly_instructions(
        self,
        output_dir: Path,
        audio_file: Path,
        background_images: List[Path],
        subtitle_file: Optional[Path]
    ) -> Path:
        """Create detailed instructions for video assembly."""
        instructions_file = output_dir / "VIDEO_ASSEMBLY_INSTRUCTIONS.md"
        
        duration = await self._estimate_audio_duration(open(output_dir / "script.txt").read())
        image_duration = duration / len(background_images) if background_images else 5.0
        
        instructions = f"""# 🎬 Professional Video Assembly Instructions

## 📁 Project Overview
- **Audio**: `{audio_file.name}` (Professional voiceover)
- **Background Images**: {len(background_images)} high-quality backgrounds
- **Subtitles**: `{subtitle_file.name if subtitle_file else 'Not created'}`
- **Estimated Duration**: {duration:.1f} seconds
- **Resolution**: 1920x1080 (Full HD)

## 🛠️ Assembly Options

### Option 1: Using FFmpeg (Command Line)
```bash
# Create image list for slideshow
echo "file 'backgrounds/background_01.png'
duration {image_duration:.2f}
file 'backgrounds/background_02.png'
duration {image_duration:.2f}
file 'backgrounds/background_03.png'
duration {image_duration:.2f}
file 'backgrounds/background_04.png'
duration {image_duration:.2f}
file 'backgrounds/background_05.png'
duration {image_duration:.2f}" > image_list.txt

# Create video from images
ffmpeg -f concat -safe 0 -i image_list.txt -vf "scale=1920:1080" -pix_fmt yuv420p temp_video.mp4

# Add audio
ffmpeg -i temp_video.mp4 -i {audio_file.name} -c:v copy -c:a aac -shortest final_video.mp4
```

### Option 2: Using DaVinci Resolve (Free Professional Editor)
1. Create new project (1920x1080, 30fps)
2. Import all background images and audio file
3. Drag images to timeline, each for {image_duration:.1f} seconds
4. Add audio track
5. Import subtitle file if available
6. Add transitions between images (crossfade recommended)
7. Export as MP4

### Option 3: Using Adobe Premiere Pro
1. New sequence: 1920x1080, 29.97fps
2. Import all assets
3. Create slideshow with background images
4. Add voiceover track
5. Import SRT subtitles
6. Add Ken Burns effect to images (subtle zoom/pan)
7. Export with H.264 codec

### Option 4: Using Canva Pro (Online)
1. Create video project (1920x1080)
2. Upload background images and audio
3. Create slideshow timing each image to match audio segments
4. Add text overlays using subtitle content
5. Download as MP4

## 🎨 Professional Enhancement Tips

### Visual Effects:
- **Ken Burns Effect**: Subtle zoom (1.0x to 1.1x scale)
- **Transitions**: 0.5-1.0 second crossfades between images
- **Color Grading**: Slightly increase contrast and saturation
- **Vignette**: Subtle edge darkening for focus

### Audio Enhancement:
- **EQ**: Slight high-frequency boost for clarity
- **Compression**: Light compression for consistent levels
- **Background Music**: Optional at 10-15% volume

### Subtitle Styling:
- **Font**: Arial Bold or Roboto
- **Size**: 40-48px
- **Color**: White with black outline (2px)
- **Position**: Bottom center
- **Animation**: Fade in/out

## 📊 Quality Settings

### Export Settings:
- **Format**: MP4 (H.264)
- **Resolution**: 1920x1080
- **Frame Rate**: 30fps
- **Bitrate**: 8-12 Mbps for high quality
- **Audio**: AAC, 128-192 kbps

## 🚀 Advanced Features Available

When MoviePy is properly installed, this system supports:
- ✅ Automatic Ken Burns effects
- ✅ Professional color grading
- ✅ Film grain and texture overlays
- ✅ Automatic subtitle synchronization
- ✅ Multiple video styles (cinematic, educational, etc.)
- ✅ Transition effects and compositing

## 🔧 MoviePy Installation (For Automated Assembly)
```bash
# Install FFmpeg first
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu/Debian

# Then install MoviePy
pip install moviepy

# Test installation
python -c "from moviepy.editor import VideoFileClip; print('MoviePy ready!')"
```

Once MoviePy is working, use:
```bash
youtube-ai create professional-video --script script.txt --style cinematic
```

## 📞 Support
- For MoviePy issues: Check FFmpeg installation
- For quality questions: Adjust export bitrate
- For timing issues: Edit subtitle timestamps manually

---
*Generated by YouTube AI CLI - Professional Video Generation System*
"""
        
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.info(f"✅ Assembly instructions created: {instructions_file}")
        return instructions_file


# Global instance
simple_video_generator = SimpleVideoGenerator()