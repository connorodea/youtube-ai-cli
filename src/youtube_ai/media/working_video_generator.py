"""
Working video generator with proper MoviePy integration and FFmpeg support.
"""

import asyncio
import tempfile
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Configure FFmpeg path
try:
    import imageio_ffmpeg
    os.environ['FFMPEG_BINARY'] = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"FFmpeg configured: {os.environ['FFMPEG_BINARY']}")
except ImportError:
    print("Warning: imageio_ffmpeg not available")

# Import MoviePy components directly
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip, ColorClip, TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.audio.AudioClip import CompositeAudioClip
    from moviepy.video.fx import FadeIn, FadeOut
    MOVIEPY_AVAILABLE = True
    print("MoviePy components imported successfully")
except ImportError as e:
    print(f"MoviePy import error: {e}")
    MOVIEPY_AVAILABLE = False

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.ai.tts_client import tts_manager

logger = get_logger(__name__)


class ImageClip(VideoClip):
    """Custom ImageClip implementation."""
    
    def __init__(self, img_path, duration=None):
        if isinstance(img_path, (str, Path)):
            img = np.array(Image.open(img_path))
        else:
            img = img_path
        
        def make_frame(t):
            return img
        
        super().__init__(make_frame, duration=duration)
        self.size = (img.shape[1], img.shape[0])


def concatenate_videoclips(clips, method="chain"):
    """Simple concatenation of video clips."""
    if not clips:
        return None
    
    if len(clips) == 1:
        return clips[0]
    
    # Calculate total duration
    total_duration = sum(clip.duration for clip in clips if clip.duration)
    
    def make_frame(t):
        # Find which clip this time belongs to
        current_time = 0
        for clip in clips:
            if current_time + clip.duration > t:
                return clip.get_frame(t - current_time)
            current_time += clip.duration
        # If beyond all clips, return last frame
        return clips[-1].get_frame(clips[-1].duration - 0.01)
    
    final_clip = VideoClip(make_frame, duration=total_duration)
    final_clip.size = clips[0].size
    final_clip.fps = clips[0].fps if hasattr(clips[0], 'fps') else 30
    
    # Combine audio if available
    audio_clips = [clip.audio for clip in clips if hasattr(clip, 'audio') and clip.audio]
    if audio_clips:
        combined_audio = CompositeAudioClip(audio_clips)
        final_clip = final_clip.set_audio(combined_audio)
    
    return final_clip


@dataclass
class SimpleVideoProject:
    """Simplified video project for automated generation."""
    script: str
    audio_file: Path
    images: List[Path]
    output_file: Path
    duration: float
    style: str = "professional"


class WorkingVideoGenerator:
    """Working automated video generator with MoviePy and FFmpeg."""
    
    def __init__(self):
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy not available")
        
        self.config = config_manager.load_config()
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_ai_working"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def create_automated_video(
        self,
        script: str,
        output_file: Path,
        voice: str = "alloy",
        provider: Optional[str] = None,
        style: str = "professional"
    ) -> Dict[str, Any]:
        """Create a complete automated video with voiceover, images, and effects."""
        try:
            logger.info(f"🎬 Creating automated video: {output_file}")
            
            # Step 1: Generate audio
            logger.info("🎵 Generating professional voiceover...")
            audio_file = await self._generate_audio(script, voice, provider)
            
            # Step 2: Create background images
            logger.info("🖼️  Creating background images...")
            background_images = await self._create_backgrounds(script, style)
            
            # Step 3: Get audio duration for timing
            logger.info("⏱️  Analyzing audio timing...")
            audio_clip = AudioFileClip(str(audio_file))
            duration = audio_clip.duration
            logger.info(f"Audio duration: {duration:.1f} seconds")
            
            # Step 4: Create video clips with Ken Burns effects
            logger.info("🎞️  Creating video clips with Ken Burns effects...")
            video_clips = await self._create_video_clips(background_images, duration)
            
            # Step 5: Add subtle effects
            logger.info("✨ Applying professional effects...")
            video_clips = await self._apply_effects(video_clips, style)
            
            # Step 6: Concatenate video clips
            logger.info("🔗 Combining video clips...")
            final_video = concatenate_videoclips(video_clips)
            
            # Step 7: Add audio
            logger.info("🔊 Adding voiceover audio...")
            final_video = final_video.set_audio(audio_clip)
            
            # Step 8: Add subtitles
            logger.info("📝 Adding subtitles...")
            final_video = await self._add_subtitles(final_video, script, duration)
            
            # Step 9: Export video
            logger.info(f"💾 Exporting video to: {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            final_video.write_videofile(
                str(output_file),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / "temp_audio.m4a"),
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            final_video.close()
            audio_clip.close()
            for clip in video_clips:
                clip.close()
            
            file_size = output_file.stat().st_size
            
            logger.info(f"🎉 Video created successfully!")
            
            return {
                "video_file": output_file,
                "duration": duration,
                "file_size": file_size,
                "resolution": "1920x1080",
                "fps": 30,
                "audio_file": audio_file,
                "num_images": len(background_images),
                "style": style
            }
            
        except Exception as e:
            logger.error(f"Error creating automated video: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _generate_audio(self, script: str, voice: str, provider: Optional[str]) -> Path:
        """Generate voiceover audio."""
        audio_file = self.temp_dir / "voiceover.mp3"
        
        await tts_manager.synthesize_speech(
            text=script,
            voice=voice,
            provider=provider,
            speed=1.0,
            output_file=audio_file
        )
        
        return audio_file
    
    async def _create_backgrounds(self, script: str, style: str) -> List[Path]:
        """Create professional background images."""
        # Create different backgrounds based on script length
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
        num_images = max(len(paragraphs), 3)
        
        style_colors = {
            "professional": [
                ("#1e293b", "#334155"),  # Slate
                ("#1f2937", "#374151"),  # Gray
                ("#1e40af", "#2563eb"),  # Blue
            ],
            "cinematic": [
                ("#1a1a1a", "#404040"),  # Dark
                ("#2d1b69", "#5b21b6"),  # Purple
                ("#7c2d12", "#dc2626"),  # Red
            ],
            "educational": [
                ("#1e3a8a", "#3b82f6"),  # Blue
                ("#059669", "#10b981"),  # Green
                ("#7c3aed", "#a855f7"),  # Purple
            ]
        }
        
        colors = style_colors.get(style, style_colors["professional"])
        background_files = []
        
        for i in range(num_images):
            image_file = self.temp_dir / f"bg_{i}.png"
            color1, color2 = colors[i % len(colors)]
            self._create_gradient_image(image_file, color1, color2)
            background_files.append(image_file)
        
        return background_files
    
    def _create_gradient_image(self, output_path: Path, color1: str, color2: str):
        """Create professional gradient background."""
        width, height = 1920, 1080
        
        # Convert hex to RGB
        rgb1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        rgb2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Create gradient
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        for y in range(height):
            ratio = y / height
            r = int(rgb1[0] * (1 - ratio) + rgb2[0] * ratio)
            g = int(rgb1[1] * (1 - ratio) + rgb2[1] * ratio)
            b = int(rgb1[2] * (1 - ratio) + rgb2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add subtle texture
        pixels = np.array(image)
        noise = np.random.randint(-10, 10, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save
        final_image = Image.fromarray(pixels)
        final_image.save(output_path, "PNG")
    
    async def _create_video_clips(self, image_files: List[Path], total_duration: float) -> List:
        """Create video clips with Ken Burns effects."""
        clips = []
        clip_duration = total_duration / len(image_files)
        
        for i, image_file in enumerate(image_files):
            # Create base clip with Ken Burns effect via resize transform
            clip = ImageClip(str(image_file), duration=clip_duration)
            
            # Add subtle zoom effect (Ken Burns style)
            zoom_factor = 1.05 + (i % 3) * 0.01
            if zoom_factor > 1.0:
                try:
                    from moviepy.video.fx import Resize
                    clip = clip.fx(Resize, zoom_factor)
                except:
                    # Fallback: use basic clip without zoom
                    pass
            
            clips.append(clip)
        
        return clips
    
    async def _apply_effects(self, clips: List, style: str) -> List:
        """Apply professional effects to clips."""
        enhanced_clips = []
        
        for clip in clips:
            # Add subtle transitions
            if len(enhanced_clips) == 0:
                # First clip - fade in
                try:
                    clip = FadeIn(clip, 0.5)
                except:
                    pass  # Skip if fade doesn't work
            
            # Last clip will get fade out later
            enhanced_clips.append(clip)
        
        # Add fade out to last clip
        if enhanced_clips:
            try:
                enhanced_clips[-1] = FadeOut(enhanced_clips[-1], 0.5)
            except:
                pass  # Skip if fade doesn't work
        
        return enhanced_clips
    
    async def _add_subtitles(self, video_clip, script: str, duration: float):
        """Add simple text overlays as subtitles."""
        # Split script into segments
        sentences = [s.strip() + '.' for s in script.split('.') if s.strip()]
        if not sentences:
            return video_clip
        
        segment_duration = duration / len(sentences)
        subtitle_clips = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence) > 100:  # Skip very long sentences
                continue
                
            start_time = i * segment_duration
            
            try:
                # Create text clip
                txt_clip = TextClip(
                    sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    fontsize=36,
                    color='white',
                    font='Arial',
                    size=(1600, None)  # Width constraint
                )
                
                txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(start_time).set_duration(min(segment_duration, 5.0))
                subtitle_clips.append(txt_clip)
                
            except Exception as e:
                logger.warning(f"Could not create subtitle for: {sentence[:50]}... Error: {e}")
        
        if subtitle_clips:
            # Composite subtitles with video
            video_clip = CompositeVideoClip([video_clip] + subtitle_clips)
        
        return video_clip


# Global instance
working_video_generator = WorkingVideoGenerator() if MOVIEPY_AVAILABLE else None

# Test the generator
if __name__ == "__main__":
    print(f"Working Video Generator available: {working_video_generator is not None}")
    if working_video_generator:
        print("✅ Ready for automated video generation!")