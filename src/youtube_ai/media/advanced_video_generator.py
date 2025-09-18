"""
Advanced video generation with professional features including:
- Voiceover synchronization with subtitles
- Ken Burns effects on images
- Film grain and scratch overlays
- Video/image concatenation
- Professional transitions and effects
"""

import asyncio
import json
import tempfile
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Configure FFmpeg before importing MoviePy
import os
try:
    import imageio_ffmpeg
    os.environ['FFMPEG_BINARY'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

try:
    # MoviePy 2.x imports
    from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip
    from moviepy import concatenate_videoclips, TextClip, ColorClip, CompositeAudioClip, VideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        # MoviePy 1.x fallback imports  
        from moviepy.editor import (
            VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
            concatenate_videoclips, TextClip, ColorClip, CompositeAudioClip, VideoClip
        )
        MOVIEPY_AVAILABLE = True
    except ImportError:
        try:
            # Direct imports for older versions
            from moviepy.video.io.VideoFileClip import VideoFileClip
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import VideoClip, ColorClip, TextClip
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
            from moviepy.audio.AudioClip import CompositeAudioClip
            from moviepy.video.tools.cuts import concatenate_videoclips
            MOVIEPY_AVAILABLE = True
        except ImportError as e:
            print(f"MoviePy import error: {e}")
            MOVIEPY_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError:
    AUDIO_ANALYSIS_AVAILABLE = False

# Create ImageClip if not available
if MOVIEPY_AVAILABLE:
    try:
        from moviepy.video.io.VideoFileClip import ImageClip
    except ImportError:
        # Create our own ImageClip
        class ImageClip(VideoClip):
            def __init__(self, img, duration=None):
                if isinstance(img, str):
                    from PIL import Image
                    img = np.array(Image.open(img))
                elif isinstance(img, Image.Image):
                    img = np.array(img)
                
                def make_frame(t):
                    return img
                
                super().__init__(make_frame, duration=duration)
                self.size = (img.shape[1], img.shape[0])

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.media.custom_assets import custom_assets_manager, AssetType
from youtube_ai.ai.tts_client import tts_manager

logger = get_logger(__name__)


class AdvancedVideoStyle(Enum):
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    STORYTELLING = "storytelling"
    SOCIAL_MEDIA = "social_media"


class TransitionType(Enum):
    FADE = "fade"
    CROSSFADE = "crossfade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    NONE = "none"


class EffectType(Enum):
    FILM_GRAIN = "film_grain"
    LIGHT_LEAKS = "light_leaks"
    SCRATCHES = "scratches"
    VIGNETTE = "vignette"
    COLOR_GRADE = "color_grade"
    CHROMATIC_ABERRATION = "chromatic_aberration"


@dataclass
class SubtitleSegment:
    """Individual subtitle segment with timing."""
    text: str
    start_time: float
    end_time: float
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaAsset:
    """Media asset for video composition."""
    file_path: Path
    asset_type: str  # "image", "video", "audio"
    duration: Optional[float] = None
    start_time: float = 0.0
    effects: List[str] = field(default_factory=list)
    ken_burns: bool = False
    zoom_start: float = 1.0
    zoom_end: float = 1.2
    pan_start: Tuple[float, float] = (0.5, 0.5)
    pan_end: Tuple[float, float] = (0.5, 0.5)


@dataclass
class VideoScene:
    """A scene in the video with multiple assets and timing."""
    duration: float
    media_assets: List[MediaAsset]
    text_overlay: Optional[str] = None
    voiceover_text: Optional[str] = None
    background_music: Optional[Path] = None
    transition_in: TransitionType = TransitionType.FADE
    transition_out: TransitionType = TransitionType.FADE
    effects: List[EffectType] = field(default_factory=list)


@dataclass
class AdvancedVideoProject:
    """Advanced video project with professional features."""
    title: str
    scenes: List[VideoScene]
    style: AdvancedVideoStyle
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    background_music: Optional[Path] = None
    intro_scene: Optional[VideoScene] = None
    outro_scene: Optional[VideoScene] = None
    subtitle_style: Dict[str, Any] = field(default_factory=dict)
    global_effects: List[EffectType] = field(default_factory=list)
    custom_overlay: Optional[Path] = None
    custom_transition: Optional[Path] = None
    transition_sound: Optional[Path] = None
    overlay_opacity: float = 0.3
    music_volume: float = 0.2
    use_custom_assets: bool = True


class AdvancedVideoGenerator:
    """Professional video generator with advanced features."""
    
    def __init__(self):
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy not installed. Install with: pip install moviepy")
        
        self.config = config_manager.load_config()
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_ai_advanced"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different assets
        (self.temp_dir / "audio").mkdir(exist_ok=True)
        (self.temp_dir / "video").mkdir(exist_ok=True)
        (self.temp_dir / "images").mkdir(exist_ok=True)
        (self.temp_dir / "effects").mkdir(exist_ok=True)
        (self.temp_dir / "subtitles").mkdir(exist_ok=True)
        
        self.style_presets = self._load_style_presets()
    
    def _load_style_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined style presets for different video types."""
        return {
            "cinematic": {
                "color_grade": {"contrast": 1.2, "brightness": 0.9, "saturation": 1.1},
                "effects": [EffectType.FILM_GRAIN, EffectType.VIGNETTE],
                "transition_duration": 1.0,
                "subtitle_style": {
                    "fontsize": 40,
                    "color": "white",
                    "font": "Arial-Bold",
                    "stroke_color": "black",
                    "stroke_width": 2
                }
            },
            "documentary": {
                "color_grade": {"contrast": 1.1, "brightness": 1.0, "saturation": 0.9},
                "effects": [EffectType.LIGHT_LEAKS],
                "transition_duration": 0.5,
                "subtitle_style": {
                    "fontsize": 36,
                    "color": "white",
                    "font": "Arial",
                    "stroke_color": "black",
                    "stroke_width": 1
                }
            },
            "educational": {
                "color_grade": {"contrast": 1.15, "brightness": 1.05, "saturation": 1.1, "warmth": 1.02},
                "effects": [EffectType.COLOR_GRADE],
                "transition_duration": 0.8,
                "subtitle_style": {
                    "fontsize": 44,
                    "color": "white",
                    "font": "Arial-Bold",
                    "stroke_color": "navy",
                    "stroke_width": 2
                }
            },
            "social_media": {
                "color_grade": {"contrast": 1.3, "brightness": 1.1, "saturation": 1.2},
                "effects": [EffectType.CHROMATIC_ABERRATION],
                "transition_duration": 0.2,
                "subtitle_style": {
                    "fontsize": 48,
                    "color": "yellow",
                    "font": "Arial-Bold",
                    "stroke_color": "black",
                    "stroke_width": 3
                }
            }
        }
    
    async def create_professional_video(
        self,
        script: str,
        output_file: Path,
        style: AdvancedVideoStyle = AdvancedVideoStyle.EDUCATIONAL,
        media_assets: Optional[List[MediaAsset]] = None,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
        include_subtitles: bool = True,
        include_effects: bool = True
    ) -> Dict[str, Any]:
        """Create a professional video with all advanced features."""
        try:
            logger.info(f"Creating professional video: {style.value} style")
            
            # Step 1: Generate voiceover with timing analysis
            logger.info("Generating voiceover with timing analysis...")
            audio_data = await self._generate_voiceover_with_timing(
                script, voice, provider
            )
            
            # Step 2: Create subtitle segments
            logger.info("Generating subtitle segments...")
            subtitle_segments = await self._generate_subtitle_segments(
                script, audio_data['duration'], audio_data.get('word_timings', [])
            )
            
            # Step 3: Prepare media assets and scenes
            logger.info("Preparing video scenes...")
            if media_assets is None:
                media_assets = await self._generate_default_media_assets(script)
            
            scenes = await self._create_scenes_from_script(
                script, audio_data['duration'], media_assets
            )
            
            # Step 4: Create video project
            project = AdvancedVideoProject(
                title=f"Generated Video - {style.value}",
                scenes=scenes,
                style=style,
                subtitle_style=self.style_presets[style.value]["subtitle_style"],
                global_effects=self.style_presets[style.value]["effects"] if include_effects else []
            )
            
            # Step 5: Render the video
            logger.info("Rendering professional video...")
            video_output = await self._render_professional_video(
                project, audio_data, subtitle_segments if include_subtitles else None, output_file
            )
            
            logger.info(f"Professional video created: {output_file}")
            return video_output
            
        except Exception as e:
            logger.error(f"Error creating professional video: {e}")
            raise
    
    async def _generate_voiceover_with_timing(
        self,
        script: str,
        voice: Optional[str],
        provider: Optional[str]
    ) -> Dict[str, Any]:
        """Generate voiceover and analyze timing for synchronization."""
        # Generate audio
        audio_response = await tts_manager.synthesize_speech(
            text=script,
            voice=voice,
            provider=provider,
            speed=self.config.audio.speed,
            output_file=self.temp_dir / "audio" / "voiceover.mp3"
        )
        
        audio_file = self.temp_dir / "audio" / "voiceover.mp3"
        
        # Analyze audio for word timing if librosa is available
        word_timings = []
        if AUDIO_ANALYSIS_AVAILABLE:
            try:
                word_timings = await self._analyze_audio_timing(audio_file, script)
            except Exception as e:
                logger.warning(f"Could not analyze audio timing: {e}")
        
        return {
            "file_path": audio_file,
            "duration": audio_response.duration or 60.0,
            "word_timings": word_timings,
            "response": audio_response
        }
    
    async def _analyze_audio_timing(self, audio_file: Path, script: str) -> List[Dict[str, Any]]:
        """Analyze audio to extract word timing information."""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_file))
            
            # Detect speech segments using energy-based approach
            frame_length = 2048
            hop_length = 512
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert to time
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            
            # Simple word timing estimation (split script evenly across speech segments)
            words = script.split()
            speech_threshold = np.mean(rms) * 0.3
            speech_segments = []
            
            in_speech = False
            start_time = 0
            
            for i, (time, energy) in enumerate(zip(times, rms)):
                if energy > speech_threshold and not in_speech:
                    start_time = time
                    in_speech = True
                elif energy <= speech_threshold and in_speech:
                    speech_segments.append((start_time, time))
                    in_speech = False
            
            if in_speech:
                speech_segments.append((start_time, times[-1]))
            
            # Distribute words across speech segments
            word_timings = []
            total_speech_time = sum(end - start for start, end in speech_segments)
            
            if total_speech_time > 0 and words:
                words_per_second = len(words) / total_speech_time
                current_word_idx = 0
                
                for segment_start, segment_end in speech_segments:
                    segment_duration = segment_end - segment_start
                    words_in_segment = max(1, int(segment_duration * words_per_second))
                    
                    for i in range(words_in_segment):
                        if current_word_idx < len(words):
                            word_start = segment_start + (i / words_in_segment) * segment_duration
                            word_end = segment_start + ((i + 1) / words_in_segment) * segment_duration
                            
                            word_timings.append({
                                "word": words[current_word_idx],
                                "start": word_start,
                                "end": word_end
                            })
                            current_word_idx += 1
            
            return word_timings
            
        except Exception as e:
            logger.warning(f"Audio timing analysis failed: {e}")
            return []
    
    async def _generate_subtitle_segments(
        self,
        script: str,
        total_duration: float,
        word_timings: List[Dict[str, Any]]
    ) -> List[SubtitleSegment]:
        """Generate subtitle segments with proper timing."""
        segments = []
        
        if word_timings:
            # Use word timings to create subtitle segments
            current_segment_words = []
            current_start = 0
            max_words_per_segment = 8
            
            for i, word_timing in enumerate(word_timings):
                current_segment_words.append(word_timing["word"])
                
                # Create segment if we hit max words or if there's a significant pause
                should_end_segment = (
                    len(current_segment_words) >= max_words_per_segment or
                    i == len(word_timings) - 1 or
                    (i < len(word_timings) - 1 and 
                     word_timings[i + 1]["start"] - word_timing["end"] > 0.5)
                )
                
                if should_end_segment:
                    segment_text = " ".join(current_segment_words)
                    segments.append(SubtitleSegment(
                        text=segment_text,
                        start_time=current_start,
                        end_time=word_timing["end"]
                    ))
                    
                    if i < len(word_timings) - 1:
                        current_start = word_timings[i + 1]["start"]
                    current_segment_words = []
        else:
            # Fallback: split script into segments based on sentences/time
            sentences = self._split_into_sentences(script)
            segment_duration = total_duration / len(sentences) if sentences else total_duration
            
            for i, sentence in enumerate(sentences):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, total_duration)
                
                segments.append(SubtitleSegment(
                    text=sentence.strip(),
                    start_time=start_time,
                    end_time=end_time
                ))
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for subtitle segmentation."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _generate_default_media_assets(self, script: str) -> List[MediaAsset]:
        """Generate default media assets when none are provided."""
        # Create simple background images with different colors/gradients
        assets = []
        colors = [
            "#1a1a2e", "#16213e", "#0f3460", "#533483", "#7b68ee",
            "#2c3e50", "#34495e", "#2980b9", "#27ae60", "#f39c12"
        ]
        
        for i, color in enumerate(colors[:5]):  # Create 5 background images
            image_path = self.temp_dir / "images" / f"background_{i}.png"
            self._create_gradient_background(image_path, color)
            
            assets.append(MediaAsset(
                file_path=image_path,
                asset_type="image",
                duration=4.0,
                ken_burns=True,
                zoom_start=1.0,
                zoom_end=1.1,
                pan_start=(0.4, 0.4),
                pan_end=(0.6, 0.6)
            ))
        
        return assets
    
    def _create_gradient_background(self, output_path: Path, base_color: str) -> None:
        """Create a gradient background image."""
        width, height = 1920, 1080
        
        # Convert hex to RGB
        base_rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Create gradient
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        for y in range(height):
            # Create gradient from base color to darker version
            factor = y / height
            darker_rgb = tuple(int(c * (1 - factor * 0.3)) for c in base_rgb)
            draw.line([(0, y), (width, y)], fill=darker_rgb)
        
        # Add some noise for texture
        pixels = np.array(image)
        noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(pixels)
        image.save(output_path)
    
    async def _create_scenes_from_script(
        self,
        script: str,
        total_duration: float,
        media_assets: List[MediaAsset]
    ) -> List[VideoScene]:
        """Create video scenes from script and media assets."""
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
        scenes = []
        
        if not paragraphs:
            paragraphs = [script]
        
        scene_duration = total_duration / len(paragraphs)
        
        for i, paragraph in enumerate(paragraphs):
            # Cycle through available media assets
            asset = media_assets[i % len(media_assets)] if media_assets else None
            scene_assets = [asset] if asset else []
            
            # Adjust asset duration to match scene
            if asset:
                asset.duration = scene_duration
            
            scene = VideoScene(
                duration=scene_duration,
                media_assets=scene_assets,
                text_overlay=paragraph if len(paragraph) < 100 else None,
                voiceover_text=paragraph,
                transition_in=TransitionType.FADE,
                transition_out=TransitionType.FADE,
                effects=[EffectType.FILM_GRAIN] if i % 2 == 0 else []
            )
            
            scenes.append(scene)
        
        return scenes
    
    async def _render_professional_video(
        self,
        project: AdvancedVideoProject,
        audio_data: Dict[str, Any],
        subtitle_segments: Optional[List[SubtitleSegment]],
        output_file: Path
    ) -> Dict[str, Any]:
        """Render the complete professional video."""
        clips = []
        current_time = 0.0
        
        style_preset = self.style_presets.get(project.style.value, {})
        
        for scene in project.scenes:
            scene_clips = await self._render_scene(
                scene, current_time, style_preset, project.resolution, project.fps
            )
            clips.extend(scene_clips)
            current_time += scene.duration
        
        # Combine all video clips
        logger.info("Combining video clips...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Add voiceover audio
        logger.info("Adding voiceover...")
        voiceover = AudioFileClip(str(audio_data["file_path"]))
        final_video = final_video.set_audio(voiceover)
        
        # Add subtitles if provided
        if subtitle_segments:
            logger.info("Adding subtitles...")
            final_video = self._add_subtitles(final_video, subtitle_segments, project.subtitle_style)
        
        # Apply global effects
        if project.global_effects:
            logger.info("Applying global effects...")
            final_video = await self._apply_global_effects(final_video, project.global_effects, style_preset)
        
        # Add background music if specified
        if project.background_music and project.background_music.exists():
            logger.info("Adding background music...")
            final_video = self._add_background_music(final_video, project.background_music)
        
        # Write final video
        logger.info(f"Writing final video to: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        final_video.write_videofile(
            str(output_file),
            fps=project.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(self.temp_dir / "temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()
        
        return {
            "video_file": output_file,
            "duration": audio_data["duration"],
            "file_size": output_file.stat().st_size,
            "resolution": project.resolution,
            "fps": project.fps,
            "style": project.style.value,
            "has_subtitles": subtitle_segments is not None,
            "effects_applied": len(project.global_effects)
        }
    
    async def _render_scene(
        self,
        scene: VideoScene,
        start_time: float,
        style_preset: Dict[str, Any],
        resolution: Tuple[int, int],
        fps: int
    ) -> List:
        """Render a single scene with all its components."""
        scene_clips = []
        
        for asset in scene.media_assets:
            if asset.asset_type == "image":
                clip = await self._create_image_clip(asset, scene.duration, resolution)
            elif asset.asset_type == "video":
                clip = await self._create_video_clip(asset, scene.duration)
            else:
                continue
            
            # Apply Ken Burns effect if enabled
            if asset.ken_burns:
                clip = self._apply_ken_burns_effect(
                    clip, asset.zoom_start, asset.zoom_end,
                    asset.pan_start, asset.pan_end
                )
            
            # Apply scene effects
            clip = await self._apply_scene_effects(clip, scene.effects, style_preset)
            
            # Apply transitions
            clip = self._apply_transitions(clip, scene.transition_in, scene.transition_out)
            
            scene_clips.append(clip)
        
        # If no media assets, create a colored background
        if not scene_clips:
            background = ColorClip(
                size=resolution,
                color=(26, 26, 46),  # Dark blue
                duration=scene.duration
            )
            scene_clips.append(background)
        
        return scene_clips
    
    async def _create_image_clip(self, asset: MediaAsset, duration: float, resolution: Tuple[int, int]):
        """Create a video clip from an image with proper scaling."""
        clip = ImageClip(str(asset.file_path))
        # MoviePy 2.x compatibility: use duration parameter in resize call
        clip = clip.resized(resolution)
        clip = clip.with_duration(duration) if hasattr(clip, 'with_duration') else clip.set_duration(duration)
        return clip
    
    async def _create_video_clip(self, asset: MediaAsset, max_duration: float):
        """Create a video clip with proper duration handling."""
        clip = VideoFileClip(str(asset.file_path))
        
        if clip.duration > max_duration:
            clip = clip.subclip(0, max_duration)
        elif clip.duration < max_duration:
            # Loop the video if it's shorter than needed
            loops_needed = math.ceil(max_duration / clip.duration)
            clip = concatenate_videoclips([clip] * loops_needed).subclip(0, max_duration)
        
        return clip
    
    def _apply_ken_burns_effect(
        self,
        clip,
        zoom_start: float,
        zoom_end: float,
        pan_start: Tuple[float, float],
        pan_end: Tuple[float, float]
    ):
        """Apply Ken Burns effect (slow zoom and pan) to a clip."""
        def make_frame(get_frame, t):
            # Calculate zoom and pan at time t
            progress = t / clip.duration
            current_zoom = zoom_start + (zoom_end - zoom_start) * progress
            current_pan_x = pan_start[0] + (pan_end[0] - pan_start[0]) * progress
            current_pan_y = pan_start[1] + (pan_end[1] - pan_start[1]) * progress
            
            frame = get_frame(t)
            h, w = frame.shape[:2]
            
            # Apply zoom
            new_h, new_w = int(h * current_zoom), int(w * current_zoom)
            if new_h > h or new_w > w:
                # Calculate crop coordinates
                crop_x = int((new_w - w) * current_pan_x)
                crop_y = int((new_h - h) * current_pan_y)
                
                # Simple crop (moviepy will handle the resize internally)
                pass
            
            return frame
        
        return clip.resized(lambda t: zoom_start + (zoom_end - zoom_start) * t / clip.duration)
    
    async def _apply_scene_effects(self, clip, effects: List[EffectType], style_preset: Dict[str, Any]):
        """Apply visual effects to a scene clip."""
        for effect in effects:
            if effect == EffectType.FILM_GRAIN:
                clip = self._add_film_grain(clip)
            elif effect == EffectType.VIGNETTE:
                clip = self._add_vignette(clip)
            elif effect == EffectType.COLOR_GRADE:
                color_grade = style_preset.get("color_grade", {})
                clip = self._apply_color_grade(clip, color_grade)
        
        return clip
    
    def _add_film_grain(self, clip):
        """Add film grain effect to a clip."""
        def make_frame(frame):
            # Add random noise for film grain effect
            noise = np.random.randint(-15, 15, frame.shape, dtype=np.int16)
            noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return noisy_frame
        
        return clip.fl_image(make_frame)
    
    def _add_vignette(self, clip):
        """Add vignette effect to darken edges."""
        def make_frame(frame):
            h, w = frame.shape[:2]
            
            # Create vignette mask
            Y, X = np.ogrid[:h, :w]
            center_x, center_y = w // 2, h // 2
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            vignette = 1 - (dist_from_center / max_dist) * 0.4
            vignette = np.clip(vignette, 0.6, 1.0)
            
            # Apply vignette
            vignetted_frame = frame * vignette[:, :, np.newaxis]
            return np.clip(vignetted_frame, 0, 255).astype(np.uint8)
        
        return clip.fl_image(make_frame)
    
    def _apply_color_grade(self, clip, color_grade: Dict[str, float]):
        """Apply professional color grading to a clip."""
        contrast = color_grade.get("contrast", 1.0)
        brightness = color_grade.get("brightness", 1.0)
        saturation = color_grade.get("saturation", 1.0)
        warmth = color_grade.get("warmth", 1.0)
        
        def make_frame(get_frame, t):
            frame = get_frame(t)
            frame = frame.astype(np.float32)
            
            # Apply brightness
            frame = frame * brightness
            
            # Apply contrast
            frame = (frame - 128) * contrast + 128
            
            # Apply saturation
            if saturation != 1.0:
                gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
                frame = gray[..., np.newaxis] * (1 - saturation) + frame * saturation
            
            # Apply warmth (color temperature adjustment)
            if warmth != 1.0:
                # Increase red/yellow tones for warmth > 1.0, blue tones for < 1.0
                if warmth > 1.0:
                    frame[..., 0] = frame[..., 0] * warmth  # Red channel
                    frame[..., 1] = frame[..., 1] * (1 + (warmth - 1) * 0.5)  # Green channel
                else:
                    frame[..., 2] = frame[..., 2] * (2 - warmth)  # Blue channel
            
            # Professional color enhancement
            # Slight increase in mid-tone contrast
            frame = self._enhance_midtones(frame)
            
            return np.clip(frame, 0, 255).astype(np.uint8)
        
        return clip.fl_image(make_frame)
    
    def _enhance_midtones(self, frame):
        """Enhance midtones for a more professional look."""
        # Apply subtle S-curve for better contrast in midtones
        normalized = frame / 255.0
        enhanced = np.where(
            normalized < 0.5,
            2 * normalized ** 2,
            1 - 2 * (1 - normalized) ** 2
        )
        return enhanced * 255
    
    def _apply_transitions(self, clip, transition_in: TransitionType, transition_out: TransitionType):
        """Apply professional transition effects to a clip."""
        transition_duration = 0.8  # Longer, smoother transitions
        
        if transition_in == TransitionType.FADE:
            clip = clip.fadein(transition_duration)
        elif transition_in == TransitionType.CROSSFADE:
            clip = clip.crossfadein(transition_duration)
        elif transition_in == TransitionType.SLIDE_LEFT:
            clip = self._apply_slide_transition(clip, 'in', 'left', transition_duration)
        elif transition_in == TransitionType.ZOOM_IN:
            clip = self._apply_zoom_transition(clip, 'in', transition_duration)
        
        if transition_out == TransitionType.FADE:
            clip = clip.fadeout(transition_duration)
        elif transition_out == TransitionType.CROSSFADE:
            clip = clip.crossfadeout(transition_duration)
        elif transition_out == TransitionType.SLIDE_RIGHT:
            clip = self._apply_slide_transition(clip, 'out', 'right', transition_duration)
        elif transition_out == TransitionType.ZOOM_OUT:
            clip = self._apply_zoom_transition(clip, 'out', transition_duration)
        
        return clip
    
    def _apply_slide_transition(self, clip, direction: str, side: str, duration: float):
        """Apply smooth slide transition effect."""
        # Simplified slide transition - just return clip for now
        # TODO: Implement proper slide transition with newer MoviePy API
        return clip
    
    def _apply_zoom_transition(self, clip, direction: str, duration: float):
        """Apply smooth zoom transition effect."""
        # Simplified zoom transition using .resized()
        
        if direction == 'in' and clip.duration > duration:
            return clip.resized(lambda t: 0.5 + 0.5 * min(t / duration, 1.0))
        elif direction == 'out' and clip.duration > duration:
            return clip.resized(lambda t: 0.5 + 0.5 * min((clip.duration - t) / duration, 1.0))
        return clip
    
    def _add_subtitles(self, video_clip, subtitle_segments: List[SubtitleSegment], style: Dict[str, Any]):
        """Add subtitles to the video clip."""
        subtitle_clips = []
        
        for segment in subtitle_segments:
            # Create subtitle clip
            txt_clip = TextClip(
                segment.text,
                fontsize=style.get("fontsize", 40),
                color=style.get("color", "white"),
                font=style.get("font", "Arial-Bold"),
                stroke_color=style.get("stroke_color", "black"),
                stroke_width=style.get("stroke_width", 2)
            )
            
            # Position subtitle at bottom center
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(
                segment.end_time - segment.start_time
            ).set_start(segment.start_time)
            
            subtitle_clips.append(txt_clip)
        
        # Composite subtitles with video
        if subtitle_clips:
            video_clip = CompositeVideoClip([video_clip] + subtitle_clips)
        
        return video_clip
    
    async def _apply_global_effects(self, clip, effects: List[EffectType], style_preset: Dict[str, Any]):
        """Apply global effects to the entire video."""
        for effect in effects:
            if effect == EffectType.FILM_GRAIN:
                clip = self._add_film_grain(clip)
            elif effect == EffectType.VIGNETTE:
                clip = self._add_vignette(clip)
            elif effect == EffectType.COLOR_GRADE:
                color_grade = style_preset.get("color_grade", {})
                clip = self._apply_color_grade(clip, color_grade)
        
        return clip
    
    def _add_background_music(self, video_clip, music_file: Path):
        """Add background music to the video."""
        music = AudioFileClip(str(music_file))
        
        # Adjust music duration to match video
        if music.duration < video_clip.duration:
            # Loop music if it's shorter
            loops_needed = math.ceil(video_clip.duration / music.duration)
            music = concatenate_audioclips([music] * loops_needed)
        
        music = music.subclip(0, video_clip.duration)
        music = music.volumex(0.1)  # Lower volume for background
        
        # Mix with existing audio
        if video_clip.audio:
            final_audio = CompositeAudioClip([video_clip.audio, music])
        else:
            final_audio = music
        
        return video_clip.set_audio(final_audio)
    
    def _load_custom_assets(self, project: AdvancedVideoProject) -> Dict[str, Any]:
        """Load and validate custom assets for the project."""
        custom_assets = {}
        
        if not project.use_custom_assets:
            return custom_assets
        
        # Scan available assets
        available_assets = custom_assets_manager.scan_assets()
        style_name = project.style.value if hasattr(project.style, 'value') else str(project.style)
        
        # Load custom overlay
        if project.custom_overlay:
            overlay_asset = custom_assets_manager.get_asset_by_name(
                project.custom_overlay.stem, AssetType.OVERLAY
            )
            if overlay_asset and custom_assets_manager.validate_asset(overlay_asset):
                custom_assets['overlay'] = overlay_asset
        else:
            # Try to get a random overlay matching the style
            overlay_asset = custom_assets_manager.get_random_asset(AssetType.OVERLAY, style_name)
            if overlay_asset:
                custom_assets['overlay'] = overlay_asset
        
        # Load custom transition
        if project.custom_transition:
            transition_asset = custom_assets_manager.get_asset_by_name(
                project.custom_transition.stem, AssetType.TRANSITION
            )
            if transition_asset and custom_assets_manager.validate_asset(transition_asset):
                custom_assets['transition'] = transition_asset
        else:
            # Try to get a random transition matching the style
            transition_asset = custom_assets_manager.get_random_asset(AssetType.TRANSITION, style_name)
            if transition_asset:
                custom_assets['transition'] = transition_asset
        
        # Load transition sound
        if project.transition_sound:
            sound_asset = custom_assets_manager.get_asset_by_name(
                project.transition_sound.stem, AssetType.SOUND_EFFECT
            )
            if sound_asset and custom_assets_manager.validate_asset(sound_asset):
                custom_assets['transition_sound'] = sound_asset
        else:
            # Try to get a random sound effect matching the style
            sound_asset = custom_assets_manager.get_random_asset(AssetType.SOUND_EFFECT, style_name)
            if sound_asset:
                custom_assets['transition_sound'] = sound_asset
        
        # Load background music if not specified
        if not project.background_music:
            music_asset = custom_assets_manager.get_random_asset(AssetType.BACKGROUND_MUSIC, style_name)
            if music_asset:
                custom_assets['background_music'] = music_asset
                project.background_music = music_asset.file_path
        
        logger.info(f"Loaded {len(custom_assets)} custom assets for {style_name} style")
        return custom_assets
    
    def _apply_custom_overlay(self, video_clip, overlay_asset, opacity: float = 0.3):
        """Apply custom overlay to video clip."""
        try:
            overlay_path = str(overlay_asset.file_path)
            
            if overlay_asset.file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                # Static image overlay
                overlay = ImageClip(overlay_path)
                overlay = overlay.resize(video_clip.size)
                overlay = overlay.with_duration(video_clip.duration) if hasattr(overlay, 'with_duration') else overlay.set_duration(video_clip.duration)
                overlay = overlay.set_opacity(opacity)
                
                return CompositeVideoClip([video_clip, overlay])
            
            elif overlay_asset.file_path.suffix.lower() in ['.mp4', '.mov', '.webm']:
                # Video overlay
                overlay = VideoFileClip(overlay_path)
                overlay = overlay.resize(video_clip.size)
                
                # Loop overlay if shorter than main video
                if overlay.duration < video_clip.duration:
                    loops_needed = math.ceil(video_clip.duration / overlay.duration)
                    overlay = concatenate_videoclips([overlay] * loops_needed)
                
                overlay = overlay.subclip(0, video_clip.duration)
                overlay = overlay.set_opacity(opacity)
                
                return CompositeVideoClip([video_clip, overlay])
            
        except Exception as e:
            logger.warning(f"Failed to apply custom overlay: {e}")
            return video_clip
        
        return video_clip
    
    def _apply_custom_transition(self, clip1, clip2, transition_asset, duration: float = 1.0):
        """Apply custom transition between two clips."""
        try:
            transition_path = str(transition_asset.file_path)
            
            if transition_asset.file_path.suffix.lower() in ['.mp4', '.mov', '.webm']:
                # Video transition
                transition = VideoFileClip(transition_path)
                transition = transition.resize(clip1.size)
                
                # Adjust transition duration
                if transition.duration > duration:
                    transition = transition.subclip(0, duration)
                else:
                    # If transition is shorter, adjust the requested duration
                    duration = min(duration, transition.duration)
                
                # Create transition effect
                # Fade out first clip
                clip1_end = clip1.subclip(clip1.duration - duration/2, clip1.duration)
                clip1_end = clip1_end.fadeout(duration/2)
                
                # Fade in second clip
                clip2_start = clip2.subclip(0, duration/2)
                clip2_start = clip2_start.fadein(duration/2)
                
                # Composite with transition
                transition_composite = CompositeVideoClip([
                    clip1_end,
                    transition.set_start(0),
                    clip2_start.set_start(duration/2)
                ], size=clip1.size)
                
                # Combine: main clip1, transition, main clip2
                final_clip1 = clip1.subclip(0, clip1.duration - duration/2)
                final_clip2 = clip2.subclip(duration/2, clip2.duration)
                
                return concatenate_videoclips([
                    final_clip1,
                    transition_composite,
                    final_clip2
                ])
            
        except Exception as e:
            logger.warning(f"Failed to apply custom transition: {e}")
            # Fallback to simple crossfade
            return concatenate_videoclips([
                clip1.fadeout(duration/2),
                clip2.fadein(duration/2).set_start(clip1.duration - duration/2)
            ])
        
        # Fallback to simple concatenation
        return concatenate_videoclips([clip1, clip2])
    
    def _add_transition_sound(self, video_clip, transition_sound_asset, transition_times: List[float]):
        """Add transition sound effects at specified times."""
        try:
            sound_path = str(transition_sound_asset.file_path)
            sound_effect = AudioFileClip(sound_path)
            
            # Create sound effects for each transition
            sound_clips = []
            for transition_time in transition_times:
                sound_clip = sound_effect.set_start(transition_time)
                sound_clips.append(sound_clip)
            
            # Combine all audio
            if video_clip.audio:
                all_audio = [video_clip.audio] + sound_clips
            else:
                all_audio = sound_clips
            
            final_audio = CompositeAudioClip(all_audio)
            return video_clip.set_audio(final_audio)
            
        except Exception as e:
            logger.warning(f"Failed to add transition sounds: {e}")
            return video_clip
    
    def _apply_custom_background_music(self, video_clip, music_asset, volume: float = 0.2):
        """Apply custom background music with specified volume."""
        try:
            music_path = str(music_asset.file_path)
            music = AudioFileClip(music_path)
            
            # Adjust music duration to match video
            if music.duration < video_clip.duration:
                # Loop music if it's shorter
                loops_needed = math.ceil(video_clip.duration / music.duration)
                music = concatenate_audioclips([music] * loops_needed)
            
            music = music.subclip(0, video_clip.duration)
            music = music.volumex(volume)  # Use custom volume
            
            # Mix with existing audio
            if video_clip.audio:
                final_audio = CompositeAudioClip([video_clip.audio, music])
            else:
                final_audio = music
            
            return video_clip.set_audio(final_audio)
            
        except Exception as e:
            logger.warning(f"Failed to apply custom background music: {e}")
            return video_clip


# Global instance
advanced_video_generator = AdvancedVideoGenerator() if MOVIEPY_AVAILABLE else None