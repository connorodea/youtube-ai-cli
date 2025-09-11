import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
        concatenate_videoclips, TextClip, ColorClip
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.ai.tts_client import tts_manager

logger = get_logger(__name__)


class VideoStyle(Enum):
    SLIDESHOW = "slideshow"
    TALKING_HEAD = "talking_head"
    SCREEN_RECORDING = "screen_recording"
    ANIMATED = "animated"


@dataclass
class VideoSegment:
    """Represents a segment of video content."""
    text: str
    duration: float
    background_color: str = "#1a1a1a"
    text_color: str = "#ffffff"
    font_size: int = 48
    audio_file: Optional[Path] = None
    image_file: Optional[Path] = None


@dataclass
class VideoProject:
    """Complete video project configuration."""
    title: str
    segments: List[VideoSegment]
    style: VideoStyle
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    background_music: Optional[Path] = None
    outro_text: Optional[str] = None
    intro_text: Optional[str] = None


@dataclass
class VideoOutput:
    """Video generation output."""
    video_file: Path
    audio_file: Optional[Path]
    metadata: Dict
    duration: float
    file_size: int


class VideoGenerator:
    """Generates videos from scripts and audio."""
    
    def __init__(self):
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy not installed. Install with: pip install moviepy")
        
        self.config = config_manager.load_config()
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_ai"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def create_video_from_script(
        self,
        script: str,
        output_file: Path,
        style: VideoStyle = VideoStyle.SLIDESHOW,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
        background_color: str = "#1a1a1a",
        text_color: str = "#ffffff"
    ) -> VideoOutput:
        """Create a video from a script text."""
        try:
            logger.info(f"Creating video from script: {len(script)} characters")
            
            # Generate audio from script
            logger.info("Generating audio...")
            audio_response = await tts_manager.synthesize_speech(
                text=script,
                voice=voice,
                provider=provider,
                speed=self.config.audio.speed,
                output_file=self.temp_dir / "script_audio.mp3"
            )
            
            audio_file = self.temp_dir / "script_audio.mp3"
            
            # Create video segments based on script structure
            segments = self._split_script_into_segments(script, audio_response.duration or 60)
            
            # Create video project
            project = VideoProject(
                title="Generated Video",
                segments=segments,
                style=style,
                resolution=self._get_resolution(),
                fps=self.config.video.fps
            )
            
            # Generate the video
            video_output = await self._generate_video(project, output_file, audio_file)
            
            logger.info(f"Video created successfully: {output_file}")
            return video_output
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            raise
    
    async def create_video_from_segments(
        self,
        segments: List[VideoSegment],
        output_file: Path,
        style: VideoStyle = VideoStyle.SLIDESHOW,
        title: str = "Generated Video"
    ) -> VideoOutput:
        """Create a video from pre-defined segments."""
        try:
            logger.info(f"Creating video from {len(segments)} segments")
            
            project = VideoProject(
                title=title,
                segments=segments,
                style=style,
                resolution=self._get_resolution(),
                fps=self.config.video.fps
            )
            
            return await self._generate_video(project, output_file)
            
        except Exception as e:
            logger.error(f"Error creating video from segments: {e}")
            raise
    
    async def _generate_video(
        self,
        project: VideoProject,
        output_file: Path,
        audio_file: Optional[Path] = None
    ) -> VideoOutput:
        """Generate video from project configuration."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        clips = []
        total_duration = 0
        
        if project.style == VideoStyle.SLIDESHOW:
            clips, total_duration = await self._create_slideshow_video(project, audio_file)
        else:
            raise NotImplementedError(f"Video style {project.style} not yet implemented")
        
        # Combine all clips
        logger.info("Combining video clips...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add background music if specified
        if project.background_music and project.background_music.exists():
            logger.info("Adding background music...")
            background_audio = AudioFileClip(str(project.background_music))
            
            # Loop background music if shorter than video
            if background_audio.duration < final_video.duration:
                background_audio = background_audio.loop(duration=final_video.duration)
            else:
                background_audio = background_audio.subclip(0, final_video.duration)
            
            # Lower volume for background music
            background_audio = background_audio.volumex(self.config.audio.music_volume)
            
            # Composite with existing audio
            if final_video.audio:
                final_audio = CompositeAudioClip([final_video.audio, background_audio])
                final_video = final_video.set_audio(final_audio)
            else:
                final_video = final_video.set_audio(background_audio)
        
        # Write video file
        logger.info(f"Writing video to: {output_file}")
        final_video.write_videofile(
            str(output_file),
            fps=project.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(self.temp_dir / "temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None  # Suppress moviepy logging
        )
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()
        
        # Get file info
        file_size = output_file.stat().st_size
        
        metadata = {
            "title": project.title,
            "style": project.style.value,
            "resolution": project.resolution,
            "fps": project.fps,
            "segments": len(project.segments),
            "total_duration": total_duration
        }
        
        return VideoOutput(
            video_file=output_file,
            audio_file=audio_file,
            metadata=metadata,
            duration=total_duration,
            file_size=file_size
        )
    
    async def _create_slideshow_video(
        self,
        project: VideoProject,
        audio_file: Optional[Path] = None
    ) -> Tuple[List, float]:
        """Create a slideshow-style video."""
        clips = []
        total_duration = 0
        
        # Add intro if specified
        if project.intro_text:
            intro_clip = self._create_text_slide(
                project.intro_text,
                duration=3.0,
                resolution=project.resolution,
                background_color="#000000",
                text_color="#ffffff",
                font_size=60
            )
            clips.append(intro_clip)
            total_duration += 3.0
        
        # Create slides for each segment
        for i, segment in enumerate(project.segments):
            logger.debug(f"Creating slide {i+1}/{len(project.segments)}")
            
            slide_clip = self._create_text_slide(
                segment.text,
                duration=segment.duration,
                resolution=project.resolution,
                background_color=segment.background_color,
                text_color=segment.text_color,
                font_size=segment.font_size,
                image_file=segment.image_file
            )
            
            clips.append(slide_clip)
            total_duration += segment.duration
        
        # Add outro if specified
        if project.outro_text:
            outro_clip = self._create_text_slide(
                project.outro_text,
                duration=3.0,
                resolution=project.resolution,
                background_color="#000000",
                text_color="#ffffff",
                font_size=48
            )
            clips.append(outro_clip)
            total_duration += 3.0
        
        # Add audio if provided
        if audio_file and audio_file.exists():
            logger.info("Adding audio track...")
            audio_clip = AudioFileClip(str(audio_file))
            
            # Adjust video duration to match audio if needed
            if abs(audio_clip.duration - total_duration) > 1.0:  # More than 1 second difference
                logger.info(f"Adjusting video duration from {total_duration:.1f}s to {audio_clip.duration:.1f}s")
                
                # Redistribute segment durations proportionally
                scale_factor = audio_clip.duration / total_duration
                new_clips = []
                
                for clip in clips:
                    new_duration = clip.duration * scale_factor
                    new_clip = clip.set_duration(new_duration)
                    new_clips.append(new_clip)
                
                clips = new_clips
                total_duration = audio_clip.duration
            
            # Set audio for the entire video
            combined_video = concatenate_videoclips(clips, method="compose")
            combined_video = combined_video.set_audio(audio_clip)
            clips = [combined_video]
        
        return clips, total_duration
    
    def _create_text_slide(
        self,
        text: str,
        duration: float,
        resolution: Tuple[int, int],
        background_color: str = "#1a1a1a",
        text_color: str = "#ffffff",
        font_size: int = 48,
        image_file: Optional[Path] = None
    ) -> VideoFileClip:
        """Create a text slide with optional background image."""
        width, height = resolution
        
        if image_file and image_file.exists():
            # Use image as background
            background_clip = ImageClip(str(image_file)).set_duration(duration)
            background_clip = background_clip.resize(resolution)
        else:
            # Use solid color background
            background_clip = ColorClip(
                size=resolution,
                color=self._hex_to_rgb(background_color)
            ).set_duration(duration)
        
        # Create text overlay
        if text.strip():
            try:
                text_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color=text_color,
                    font='Arial-Bold',  # Default font
                    size=(width * 0.8, None),  # 80% of video width
                    method='caption'
                ).set_duration(duration)
                
                # Center the text
                text_clip = text_clip.set_position('center')
                
                # Composite text over background
                slide_clip = CompositeVideoClip([background_clip, text_clip])
            except Exception as e:
                logger.warning(f"Error creating text clip: {e}, using background only")
                slide_clip = background_clip
        else:
            slide_clip = background_clip
        
        return slide_clip.set_fps(self.config.video.fps)
    
    def _split_script_into_segments(self, script: str, total_duration: float) -> List[VideoSegment]:
        """Split script into segments for video creation."""
        # Split by paragraphs or sentences
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback: split by sentences
            sentences = [s.strip() + '.' for s in script.split('.') if s.strip()]
            paragraphs = sentences
        
        # Calculate duration per segment
        segment_duration = total_duration / len(paragraphs) if paragraphs else total_duration
        
        segments = []
        for paragraph in paragraphs:
            segments.append(VideoSegment(
                text=paragraph,
                duration=segment_duration,
                background_color="#1a1a1a",
                text_color="#ffffff",
                font_size=48
            ))
        
        return segments
    
    def _get_resolution(self) -> Tuple[int, int]:
        """Get video resolution from config."""
        resolution_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "4k": (3840, 2160)
        }
        
        resolution_str = self.config.video.resolution
        return resolution_map.get(resolution_str, (1920, 1080))
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def estimate_video_size(self, duration: float, resolution: Tuple[int, int]) -> int:
        """Estimate video file size in bytes."""
        # Rough estimation based on typical bitrates
        width, height = resolution
        pixels = width * height
        
        # Estimate bitrate based on resolution (bits per second)
        if pixels <= 1280 * 720:  # 720p
            bitrate = 2500000  # 2.5 Mbps
        elif pixels <= 1920 * 1080:  # 1080p
            bitrate = 5000000  # 5 Mbps
        else:  # Higher resolutions
            bitrate = 10000000  # 10 Mbps
        
        # Calculate file size (include audio overhead)
        video_size = (bitrate * duration) / 8  # Convert to bytes
        audio_size = (128000 * duration) / 8  # 128 kbps audio
        
        return int(video_size + audio_size)


# Global video generator instance
video_generator = VideoGenerator() if MOVIEPY_AVAILABLE else None