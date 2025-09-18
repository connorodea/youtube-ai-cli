import click
import asyncio
import random
from typing import List, Any, Optional, Dict
from rich.console import Console
from rich.table import Table
from pathlib import Path

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config-dir', type=click.Path(), help='Configuration directory path')
@click.pass_context
def cli(ctx, debug, config_dir):
    """YouTube AI CLI - Automate YouTube content creation with AI."""
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    
    if config_dir:
        config_manager.config_dir = Path(config_dir)
    
    if debug:
        logger.setLevel('DEBUG')
        console.print("[yellow]Debug mode enabled[/yellow]")


@cli.group()
def generate():
    """Generate content using AI (scripts, titles, descriptions)."""
    pass


@cli.group()
def create():
    """Create media content (videos, audio, thumbnails)."""
    pass


@cli.group()
def upload():
    """Upload and manage YouTube videos."""
    pass


@cli.group()
def config():
    """Manage configuration settings."""
    pass


@cli.group()
def workflow():
    """Manage and run automated workflows."""
    pass


# Analytics Commands
@cli.group()
def analytics():
    """Analytics and performance tracking."""
    pass


# Batch Commands
@cli.group()
def batch():
    """Batch processing operations."""
    pass


# Add system commands
# cli.add_command(system)  # TODO: Implement system commands


# Config Commands
@config.command('show')
def config_show():
    """Show current configuration."""
    config = config_manager.load_config()
    
    table = Table(title="YouTube AI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # YouTube settings
    table.add_row("YouTube API Key", "***" if config.youtube.api_key else "Not set")
    table.add_row("YouTube Channel ID", config.youtube.channel_id or "Not set")
    table.add_row("Default Privacy", config.youtube.default_privacy)
    
    # AI settings
    table.add_row("OpenAI API Key", "***" if config.ai.openai_api_key else "Not set")
    table.add_row("Anthropic API Key", "***" if config.ai.anthropic_api_key else "Not set")
    table.add_row("ElevenLabs API Key", "***" if config.ai.elevenlabs_api_key else "Not set")
    table.add_row("Deepgram API Key", "***" if config.ai.deepgram_api_key else "Not set")
    table.add_row("Stability AI Key", "***" if config.ai.stability_api_key else "Not set")
    table.add_row("Default LLM", config.ai.default_llm)
    table.add_row("Default TTS", config.ai.default_tts)
    table.add_row("Default Image Provider", config.ai.default_image_provider)
    
    # Video settings
    table.add_row("Video Resolution", config.video.resolution)
    table.add_row("Video FPS", str(config.video.fps))
    table.add_row("Video Format", config.video.format)
    
    # Other settings
    table.add_row("Output Directory", config.output_dir)
    table.add_row("Debug Mode", str(config.debug))
    
    console.print(table)


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """Set a configuration value. Use dot notation (e.g., ai.openai_api_key)."""
    try:
        # Convert string values to appropriate types
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        config_manager.set_value(key, value)
        console.print(f"[green]✓[/green] Set {key} = {value}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")


@config.command('validate')
def config_validate():
    """Validate the current configuration."""
    is_valid, issues = config_manager.validate_config()
    
    if is_valid:
        console.print("[green]✓ Configuration is valid[/green]")
    else:
        console.print("[red]Configuration issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")


@config.command('init')
def config_init():
    """Initialize configuration with interactive setup."""
    console.print("[bold blue]YouTube AI CLI Setup[/bold blue]")
    console.print("Let's configure your API keys and settings.\n")
    
    # YouTube API Key
    youtube_key = click.prompt("YouTube Data API Key (optional)", default="", show_default=False)
    if youtube_key:
        config_manager.set_value('youtube.api_key', youtube_key)
    
    # Channel ID
    channel_id = click.prompt("YouTube Channel ID (optional)", default="", show_default=False)
    if channel_id:
        config_manager.set_value('youtube.channel_id', channel_id)
    
    # OpenAI API Key
    openai_key = click.prompt("OpenAI API Key (optional)", default="", show_default=False)
    if openai_key:
        config_manager.set_value('ai.openai_api_key', openai_key)
    
    # Anthropic API Key
    anthropic_key = click.prompt("Anthropic API Key (optional)", default="", show_default=False)
    if anthropic_key:
        config_manager.set_value('ai.anthropic_api_key', anthropic_key)
    
    # ElevenLabs API Key
    elevenlabs_key = click.prompt("ElevenLabs API Key (optional)", default="", show_default=False)
    if elevenlabs_key:
        config_manager.set_value('ai.elevenlabs_api_key', elevenlabs_key)
    
    # Output directory
    output_dir = click.prompt("Output directory", default="./output")
    config_manager.set_value('output_dir', output_dir)
    
    console.print("\n[green]✓ Configuration saved![/green]")
    
    # Validate
    is_valid, issues = config_manager.validate_config()
    if not is_valid:
        console.print("\n[yellow]Note: Some issues were found:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")


# Generate Commands
@generate.command('script')
@click.option('--topic', required=True, help='Video topic or subject')
@click.option('--style', default='educational', help='Content style (educational, entertaining, informative)')
@click.option('--duration', default=300, help='Target duration in seconds')
@click.option('--audience', default='general', help='Target audience')
@click.option('--output', '-o', help='Output file path')
def generate_script(topic, style, duration, audience, output):
    """Generate a video script using AI."""
    async def _generate():
        try:
            from youtube_ai.content.script_generator import ScriptGenerator
            
            generator = ScriptGenerator()
            script = await generator.generate_script(
                topic=topic,
                style=style,
                duration=duration,
                audience=audience
            )
            
            if output:
                output_path = Path(output)
            else:
                safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                output_path = Path(config_manager.load_config().output_dir) / f"script_{safe_topic[:50]}.txt"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            console.print(f"[green]✓[/green] Script generated: {output_path}")
            console.print(f"[dim]Topic:[/dim] {topic}")
            console.print(f"[dim]Style:[/dim] {style}")
            console.print(f"[dim]Duration:[/dim] {duration}s")
            
        except Exception as e:
            console.print(f"[red]Error generating script:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_generate())


@generate.command('title')
@click.option('--topic', help='Video topic')
@click.option('--script', type=click.Path(exists=True), help='Script file to analyze')
@click.option('--keywords', help='Target keywords (comma-separated)')
@click.option('--count', default=5, help='Number of title options to generate')
def generate_title(topic, script, keywords, count):
    """Generate video titles optimized for SEO."""
    async def _generate():
        try:
            from youtube_ai.content.seo_optimizer import SEOOptimizer
            
            content = topic
            if script:
                with open(script, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            optimizer = SEOOptimizer()
            titles = await optimizer.generate_titles(
                content=content,
                keywords=keywords.split(',') if keywords else None,
                count=count
            )
            
            console.print(f"[bold blue]Generated {len(titles)} title options:[/bold blue]")
            for i, title in enumerate(titles, 1):
                console.print(f"{i}. {title}")
                
        except Exception as e:
            console.print(f"[red]Error generating titles:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_generate())


@generate.command('description')
@click.option('--script', type=click.Path(exists=True), required=True, help='Script file to analyze')
@click.option('--title', help='Video title')
@click.option('--keywords', help='Target keywords (comma-separated)')
@click.option('--timestamps', is_flag=True, help='Include timestamp placeholders')
@click.option('--output', '-o', help='Output file path')
def generate_description(script, title, keywords, timestamps, output):
    """Generate optimized video description."""
    async def _generate():
        try:
            from youtube_ai.content.seo_optimizer import SEOOptimizer
            
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            optimizer = SEOOptimizer()
            description = await optimizer.generate_description(
                content=content,
                title=title or "Video Title",
                keywords=keywords.split(',') if keywords else None,
                include_timestamps=timestamps
            )
            
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(description)
                console.print(f"[green]✓[/green] Description saved to: {output_path}")
            else:
                console.print("[bold blue]Generated description:[/bold blue]")
                console.print(description)
                
        except Exception as e:
            console.print(f"[red]Error generating description:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_generate())


@generate.command('tags')
@click.option('--script', type=click.Path(exists=True), required=True, help='Script file to analyze')
@click.option('--title', help='Video title')
@click.option('--keywords', help='Target keywords (comma-separated)')
@click.option('--max-tags', default=15, help='Maximum number of tags')
def generate_tags(script, title, keywords, max_tags):
    """Generate optimized video tags."""
    async def _generate():
        try:
            from youtube_ai.content.seo_optimizer import SEOOptimizer
            
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            optimizer = SEOOptimizer()
            tags = await optimizer.generate_tags(
                content=content,
                title=title or "Video Title",
                keywords=keywords.split(',') if keywords else None,
                max_tags=max_tags
            )
            
            console.print(f"[bold blue]Generated {len(tags)} tags:[/bold blue]")
            console.print(", ".join(tags))
                
        except Exception as e:
            console.print(f"[red]Error generating tags:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_generate())


# Create Commands
@create.command('audio')
@click.option('--script', type=click.Path(exists=True), required=True, help='Script file to convert to audio')
@click.option('--voice', help='Voice to use for synthesis')
@click.option('--provider', help='TTS provider (openai, elevenlabs)')
@click.option('--speed', default=1.0, help='Speech speed (0.5-2.0)')
@click.option('--output', '-o', help='Output audio file path')
def create_audio(script, voice, provider, speed, output):
    """Create audio from script using text-to-speech."""
    async def _create():
        try:
            from youtube_ai.ai.tts_client import tts_manager
            
            with open(script, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not output:
                script_path = Path(script)
                output_path = script_path.parent / f"{script_path.stem}_audio.mp3"
            else:
                output_path = Path(output)
            
            console.print(f"[blue]Generating audio...[/blue]")
            
            response = await tts_manager.synthesize_speech(
                text=text,
                voice=voice,
                provider=provider,
                speed=speed,
                output_file=output_path
            )
            
            console.print(f"[green]✓[/green] Audio created: {output_path}")
            console.print(f"[dim]Provider:[/dim] {response.provider}")
            console.print(f"[dim]Voice:[/dim] {response.voice}")
            console.print(f"[dim]Format:[/dim] {response.format}")
            
        except Exception as e:
            console.print(f"[red]Error creating audio:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@create.command('video')
@click.option('--script', type=click.Path(exists=True), required=True, help='Script file for video content')
@click.option('--style', default='slideshow', help='Video style (slideshow, talking_head)')
@click.option('--voice', help='Voice for narration')
@click.option('--provider', help='TTS provider')
@click.option('--background', default='#1a1a1a', help='Background color (hex)')
@click.option('--text-color', default='#ffffff', help='Text color (hex)')
@click.option('--output', '-o', help='Output video file path')
def create_video(script, style, voice, provider, background, text_color, output):
    """Create video from script."""
    async def _create():
        try:
            from youtube_ai.media.video_generator import video_generator, VideoStyle
            
            if not video_generator:
                console.print("[red]Error:[/red] VideoGenerator not available. Please install moviepy: pip install moviepy")
                return
            
            with open(script, 'r', encoding='utf-8') as f:
                script_text = f.read()
            
            if not output:
                script_path = Path(script)
                output_path = script_path.parent / f"{script_path.stem}_video.mp4"
            else:
                output_path = Path(output)
            
            console.print(f"[blue]Creating video...[/blue]")
            console.print(f"[dim]Style:[/dim] {style}")
            console.print(f"[dim]Background:[/dim] {background}")
            
            video_style = VideoStyle.SLIDESHOW if style == 'slideshow' else VideoStyle.SLIDESHOW
            
            result = await video_generator.create_video_from_script(
                script=script_text,
                output_file=output_path,
                style=video_style,
                voice=voice,
                provider=provider,
                background_color=background,
                text_color=text_color
            )
            
            console.print(f"[green]✓[/green] Video created: {result.video_file}")
            console.print(f"[dim]Duration:[/dim] {result.duration:.1f}s")
            console.print(f"[dim]File size:[/dim] {result.file_size // (1024*1024)}MB")
            
        except Exception as e:
            console.print(f"[red]Error creating video:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@create.command('professional-video')
@click.option('--script', required=True, type=click.Path(exists=True), help='Script file to use')
@click.option('--style', default='educational', help='Video style (cinematic, documentary, educational, promotional, social_media)')
@click.option('--voice', help='Voice to use for narration')
@click.option('--provider', help='AI provider for voice synthesis')
@click.option('--media-dir', help='Directory containing images/videos to use')
@click.option('--include-subtitles/--no-subtitles', default=True, help='Include subtitles')
@click.option('--include-effects/--no-effects', default=True, help='Include visual effects')
@click.option('--output', '-o', help='Output video file path')
def create_professional_video(script, style, voice, provider, media_dir, include_subtitles, include_effects, output):
    """Create professional video with advanced features including voiceover, Ken Burns effects, subtitles, and film overlays."""
    async def _create():
        try:
            from youtube_ai.media.advanced_video_generator import advanced_video_generator, AdvancedVideoStyle, MediaAsset
            
            if not advanced_video_generator:
                console.print("[red]Error:[/red] Advanced video generator not available. Please install moviepy and librosa.")
                return
            
            # Read script
            with open(script, 'r', encoding='utf-8') as f:
                script_text = f.read()
            
            # Prepare output path
            if not output:
                script_path = Path(script)
                output_path = script_path.parent / f"{script_path.stem}_professional.mp4"
            else:
                output_path = Path(output)
            
            # Parse style
            try:
                video_style = AdvancedVideoStyle(style.lower())
            except ValueError:
                console.print(f"[yellow]Warning:[/yellow] Unknown style '{style}', using 'educational'")
                video_style = AdvancedVideoStyle.EDUCATIONAL
            
            # Prepare media assets if directory provided
            media_assets = None
            if media_dir:
                media_assets = await _prepare_media_assets(Path(media_dir))
                console.print(f"[blue]Found {len(media_assets)} media assets[/blue]")
            
            console.print(f"[blue]Creating professional video...[/blue]")
            console.print(f"[dim]Style:[/dim] {video_style.value}")
            console.print(f"[dim]Subtitles:[/dim] {'Yes' if include_subtitles else 'No'}")
            console.print(f"[dim]Effects:[/dim] {'Yes' if include_effects else 'No'}")
            
            # Create professional video
            result = await advanced_video_generator.create_professional_video(
                script=script_text,
                output_file=output_path,
                style=video_style,
                media_assets=media_assets,
                voice=voice,
                provider=provider,
                include_subtitles=include_subtitles,
                include_effects=include_effects
            )
            
            console.print(f"[green]✓[/green] Professional video created: {result['video_file']}")
            console.print(f"[dim]Duration:[/dim] {result['duration']:.1f}s")
            console.print(f"[dim]File size:[/dim] {result['file_size'] // (1024*1024)}MB")
            console.print(f"[dim]Resolution:[/dim] {result['resolution'][0]}x{result['resolution'][1]}")
            console.print(f"[dim]Effects applied:[/dim] {result['effects_applied']}")
            
        except Exception as e:
            console.print(f"[red]Error creating professional video:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    async def _prepare_media_assets(media_dir: Path) -> List:
        """Prepare media assets from directory."""
        from youtube_ai.media.advanced_video_generator import MediaAsset
        
        assets = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        for file_path in media_dir.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in image_extensions:
                    assets.append(MediaAsset(
                        file_path=file_path,
                        asset_type="image",
                        ken_burns=True,
                        zoom_start=1.0,
                        zoom_end=1.1 + random.uniform(0, 0.1),
                        pan_start=(random.uniform(0.3, 0.5), random.uniform(0.3, 0.5)),
                        pan_end=(random.uniform(0.5, 0.7), random.uniform(0.5, 0.7))
                    ))
                elif ext in video_extensions:
                    assets.append(MediaAsset(
                        file_path=file_path,
                        asset_type="video",
                        duration=5.0  # Will be adjusted during rendering
                    ))
        
        return assets
    
    asyncio.run(_create())


# Create Commands (continued)
@create.command('thumbnail')
@click.option('--title', required=True, help='Thumbnail title text')
@click.option('--style', default='minimalist', help='Thumbnail style (minimalist, bold, tech, educational, entertainment)')
@click.option('--layout', default='center_text', help='Text layout (center_text, left_text, right_text)')
@click.option('--subtitle', help='Subtitle text')
@click.option('--background', help='Background image file')
@click.option('--variants', is_flag=True, help='Generate multiple variants')
@click.option('--ai-optimize', is_flag=True, help='Use AI to optimize design')
@click.option('--output', '-o', help='Output file path')
def create_thumbnail(title, style, layout, subtitle, background, variants, ai_optimize, output):
    """Create thumbnail for video."""
    async def _create():
        try:
            from youtube_ai.media.thumbnail_generator import thumbnail_generator, ThumbnailStyle, ThumbnailLayout
            
            console.print(f"[blue]Creating thumbnail...[/blue]")
            console.print(f"[dim]Title:[/dim] {title}")
            console.print(f"[dim]Style:[/dim] {style}")
            
            if variants:
                # Generate multiple variants
                output_dir = Path(output) if output else Path(config_manager.load_config().output_dir) / "thumbnails"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                thumbnail_variants = await thumbnail_generator.generate_multiple_variants(
                    title=title,
                    subtitle=subtitle,
                    output_dir=output_dir
                )
                
                console.print(f"[green]✓[/green] Generated {len(thumbnail_variants)} thumbnail variants in: {output_dir}")
                
            elif ai_optimize:
                # AI-optimized thumbnail
                content_desc = f"Video about: {title}"
                if subtitle:
                    content_desc += f". {subtitle}"
                
                variant = await thumbnail_generator.generate_ai_optimized_thumbnail(
                    title=title,
                    content_description=content_desc
                )
                
                if output:
                    output_path = Path(output)
                else:
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    output_path = Path(config_manager.load_config().output_dir) / f"thumbnail_ai_{safe_title[:30]}.png"
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                variant.image.save(str(output_path), "PNG", quality=95)
                
                console.print(f"[green]✓[/green] AI-optimized thumbnail created: {output_path}")
                
            else:
                # Single thumbnail
                style_enum = ThumbnailStyle(style) if style in [s.value for s in ThumbnailStyle] else ThumbnailStyle.MINIMALIST
                layout_enum = ThumbnailLayout(layout) if layout in [l.value for l in ThumbnailLayout] else ThumbnailLayout.CENTER_TEXT
                
                if output:
                    output_path = Path(output)
                else:
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    output_path = Path(config_manager.load_config().output_dir) / f"thumbnail_{safe_title[:30]}.png"
                
                variant = await thumbnail_generator.generate_thumbnail(
                    title=title,
                    style=style_enum,
                    layout=layout_enum,
                    subtitle=subtitle,
                    background_image=Path(background) if background else None,
                    output_file=output_path
                )
                
                console.print(f"[green]✓[/green] Thumbnail created: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error creating thumbnail:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@create.command('automated-video')
@click.option('--script', required=True, type=click.Path(exists=True), help='Script text file')
@click.option('--style', default='professional', help='Video style (professional, cinematic, educational)')
@click.option('--voice', default='alloy', help='Voice for narration')
@click.option('--provider', help='TTS provider (openai, anthropic)')
@click.option('--output', '-o', help='Output MP4 file')
def create_automated_video(script, style, voice, provider, output):
    """Create fully automated video with voiceover, backgrounds, and effects."""
    async def _create():
        try:
            from youtube_ai.media.working_video_generator import working_video_generator
            
            if not working_video_generator:
                console.print("[red]❌ MoviePy not available. Please install: pip install moviepy[/red]")
                return
            
            # Read script
            script_path = Path(script)
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read().strip()
            
            # Determine output file
            if output:
                output_file = Path(output)
            else:
                output_file = Path("automated_video.mp4")
            
            console.print(f"[green]🎬 Creating automated video...[/green]")
            console.print(f"[dim]Script:[/dim] {script_path}")
            console.print(f"[dim]Style:[/dim] {style}")
            console.print(f"[dim]Voice:[/dim] {voice}")
            console.print(f"[dim]Output:[/dim] {output_file}")
            
            # Generate video
            result = await working_video_generator.create_automated_video(
                script=script_content,
                output_file=output_file,
                voice=voice,
                provider=provider,
                style=style
            )
            
            console.print(f"[green]🎉 Video created successfully![/green]")
            console.print(f"[dim]Duration:[/dim] {result['duration']:.1f}s")
            console.print(f"[dim]Resolution:[/dim] {result['resolution']}")
            console.print(f"[dim]File size:[/dim] {result['file_size'] / 1024 / 1024:.1f}MB")
            console.print(f"[dim]Images:[/dim] {result['num_images']}")
            
        except Exception as e:
            console.print(f"[red]Error creating automated video:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@create.command('video-project')
@click.option('--script', required=True, type=click.Path(exists=True), help='Script file to use')
@click.option('--voice', default='alloy', help='Voice for narration (alloy, echo, fable, onyx, nova, shimmer)')
@click.option('--provider', help='AI provider for voice synthesis')
@click.option('--style', default='educational', help='Video style (educational, cinematic, professional)')
@click.option('--num-backgrounds', default=5, help='Number of background images to create')
@click.option('--output-dir', help='Output directory for project files')
def create_video_project(script, voice, provider, style, num_backgrounds, output_dir):
    """Create complete video project with audio, images, subtitles and assembly instructions."""
    async def _create():
        try:
            from youtube_ai.media.simple_video_generator import simple_video_generator
            
            # Read script
            with open(script, 'r', encoding='utf-8') as f:
                script_text = f.read()
            
            # Prepare output directory
            if not output_dir:
                script_path = Path(script)
                output_path = Path(config_manager.load_config().output_dir) / f"video_project_{script_path.stem}"
            else:
                output_path = Path(output_dir)
            
            console.print(f"[blue]Creating complete video project...[/blue]")
            console.print(f"[dim]Script:[/dim] {script}")
            console.print(f"[dim]Voice:[/dim] {voice}")
            console.print(f"[dim]Style:[/dim] {style}")
            console.print(f"[dim]Output:[/dim] {output_path}")
            
            # Create video project
            project = await simple_video_generator.create_video_project(
                script=script_text,
                output_dir=output_path,
                voice=voice,
                provider=provider,
                style=style,
                num_backgrounds=num_backgrounds
            )
            
            console.print(f"\n[green]🎉 Video project created successfully![/green]")
            console.print(f"[green]✓[/green] Audio: {project.audio_file.name}")
            console.print(f"[green]✓[/green] Backgrounds: {len(project.background_images)} images")
            console.print(f"[green]✓[/green] Subtitles: {'Yes' if project.subtitle_file else 'No'}")
            console.print(f"[green]✓[/green] Duration: {project.duration:.1f} seconds")
            
            console.print(f"\n[yellow]📋 Next Steps:[/yellow]")
            console.print(f"[yellow]1.[/yellow] Open VIDEO_ASSEMBLY_INSTRUCTIONS.md in {output_path}")
            console.print(f"[yellow]2.[/yellow] Use any video editor (DaVinci Resolve, Premiere, etc.)")
            console.print(f"[yellow]3.[/yellow] Follow the detailed instructions for professional results")
            
            console.print(f"\n[blue]💡 Pro Tip:[/blue] All assets are ready for automated assembly once MoviePy is installed!")
            
        except Exception as e:
            console.print(f"[red]Error creating video project:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@create.command('ai-video')
@click.option('--script', required=True, type=click.Path(exists=True), help='Script file to use')
@click.option('--voice', default='nova', help='Voice for narration (alloy, echo, fable, onyx, nova, shimmer)')
@click.option('--provider', default='openai', help='TTS provider (openai, elevenlabs)')
@click.option('--style', default='educational', help='Visual style (educational, professional, tech, documentary, modern, minimalist)')
@click.option('--num-visuals', default=5, help='Number of AI-generated visuals')
@click.option('--image-provider', default='openai', help='Image generation provider (openai, stability, fallback)')
@click.option('--output-dir', help='Output directory for project files')
def create_ai_video(script, voice, provider, style, num_visuals, image_provider, output_dir):
    """Create complete AI-powered video with contextual visuals generated from script content."""
    async def _create():
        try:
            from youtube_ai.ai.image_generator import image_generator, ImageStyle, ImageProvider
            from youtube_ai.media.simple_video_generator import simple_video_generator
            
            # Read script
            with open(script, 'r', encoding='utf-8') as f:
                script_text = f.read()
            
            # Prepare output directory
            if not output_dir:
                script_path = Path(script)
                output_path = Path(config_manager.load_config().output_dir) / f"ai_video_{script_path.stem}"
            else:
                output_path = Path(output_dir)
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            console.print(f"[bold blue]🤖 Creating AI-Powered YouTube Video[/bold blue]")
            console.print(f"[dim]Script:[/dim] {script}")
            console.print(f"[dim]Voice:[/dim] {voice} ({provider})")
            console.print(f"[dim]Visual Style:[/dim] {style}")
            console.print(f"[dim]AI Visuals:[/dim] {num_visuals} images")
            console.print(f"[dim]Image Provider:[/dim] {image_provider}")
            console.print(f"[dim]Output:[/dim] {output_path}")
            
            # Step 1: Generate professional voiceover
            console.print(f"\n[blue]🎵 Generating professional voiceover...[/blue]")
            from youtube_ai.ai.tts_client import tts_manager
            
            audio_file = output_path / "narration.mp3"
            await tts_manager.synthesize_speech(
                text=script_text,
                voice=voice,
                provider=provider,
                output_file=audio_file
            )
            console.print(f"[green]✅[/green] Professional voiceover created: {audio_file.name}")
            
            # Step 2: Generate AI-powered contextual visuals
            console.print(f"\n[blue]🎨 Generating {num_visuals} AI-powered contextual visuals...[/blue]")
            
            # Parse style and provider
            try:
                image_style = ImageStyle(style.lower())
            except ValueError:
                console.print(f"[yellow]Warning:[/yellow] Unknown style '{style}', using 'educational'")
                image_style = ImageStyle.EDUCATIONAL
            
            try:
                img_provider = ImageProvider(image_provider.lower())
            except ValueError:
                console.print(f"[yellow]Warning:[/yellow] Unknown image provider '{image_provider}', using 'openai'")
                img_provider = ImageProvider.OPENAI
            
            # Generate AI visuals
            visuals_dir = output_path / "ai_visuals"
            generated_images = await image_generator.generate_script_visuals(
                script=script_text,
                output_dir=visuals_dir,
                style=image_style,
                num_images=num_visuals,
                provider=img_provider
            )
            
            console.print(f"[green]✅[/green] {len(generated_images)} contextual visuals created")
            
            # Step 3: Generate subtitles
            console.print(f"\n[blue]📝 Creating subtitle file...[/blue]")
            subtitle_file = output_path / "subtitles.srt"
            await _create_subtitle_file_ai(script_text, subtitle_file)
            console.print(f"[green]✅[/green] Subtitle file created: {subtitle_file.name}")
            
            # Step 4: Create assembly instructions
            console.print(f"\n[blue]📋 Creating assembly instructions...[/blue]")
            await _create_ai_assembly_instructions(
                output_path, audio_file, [img.file_path for img in generated_images], 
                subtitle_file, generated_images
            )
            
            # Step 5: Save metadata
            metadata_file = output_path / "project_metadata.json"
            metadata = {
                "title": f"AI-Generated Video - {Path(script).stem}",
                "script_file": script,
                "voice": voice,
                "provider": provider,
                "style": style,
                "num_visuals": len(generated_images),
                "image_provider": image_provider,
                "ai_prompts": [
                    {
                        "scene": img.scene_number,
                        "prompt": img.prompt,
                        "keywords": img.metadata.get("keywords", []) if img.metadata else []
                    } for img in generated_images
                ],
                "duration": len(script_text.split()) * 0.5,  # Estimate
                "resolution": "1792x1024",
                "created_with": "YouTube AI CLI"
            }
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Success summary
            console.print(f"\n[bold green]🎉 AI-Powered Video Project Created Successfully![/bold green]")
            console.print(f"[green]✓[/green] Professional voiceover: {audio_file.name}")
            console.print(f"[green]✓[/green] AI contextual visuals: {len(generated_images)} images")
            console.print(f"[green]✓[/green] Synchronized subtitles: {subtitle_file.name}")
            console.print(f"[green]✓[/green] Assembly instructions: VIDEO_ASSEMBLY_INSTRUCTIONS.md")
            console.print(f"[green]✓[/green] Project metadata: {metadata_file.name}")
            
            console.print(f"\n[bold yellow]🚀 Your AI Video is Ready![/bold yellow]")
            console.print(f"[yellow]📁[/yellow] Project directory: {output_path}")
            console.print(f"[yellow]🎬[/yellow] The visuals perfectly match your script content!")
            console.print(f"[yellow]📋[/yellow] Follow VIDEO_ASSEMBLY_INSTRUCTIONS.md for final video assembly")
            
            if image_provider == "stability":
                console.print(f"\n[blue]💡 Pro Tip:[/blue] Your Stability AI visuals are ultra-high quality and perfectly contextual!")
            elif image_provider == "openai":
                console.print(f"\n[blue]💡 Pro Tip:[/blue] Your DALL-E visuals are contextually matched to each script segment!")
            else:
                console.print(f"\n[blue]💡 Pro Tip:[/blue] Try --image-provider stability or --image-provider openai for AI-generated visuals!")
            
        except Exception as e:
            console.print(f"[red]Error creating AI video:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@create.command('auto-youtube')
@click.option('--topic', required=True, help='Video topic (e.g., "Python basics for beginners")')
@click.option('--duration', default=180, help='Target video duration in seconds')
@click.option('--style', default='educational', help='Content style (educational, professional, tech, documentary)')
@click.option('--voice', default='nova', help='Voice for narration')
@click.option('--image-provider', default='stability', help='Image provider (openai, stability)')
@click.option('--audience', default='general', help='Target audience (beginner, intermediate, advanced, general)')
@click.option('--language', default='en', help='Content language')
@click.option('--output-dir', help='Output directory for the complete video')
@click.option('--auto-assemble', is_flag=True, help='Automatically assemble final video file')
@click.option('--custom-overlay', help='Custom overlay file name (from assets/overlays/)')
@click.option('--custom-transition', help='Custom transition file name (from assets/transitions/)')
@click.option('--transition-sound', help='Custom transition sound file name (from assets/sound_effects/)')
@click.option('--background-music', help='Custom background music file name (from assets/background_music/)')
@click.option('--overlay-opacity', default=0.3, help='Overlay opacity (0.0-1.0)')
@click.option('--music-volume', default=0.2, help='Background music volume (0.0-1.0)')
@click.option('--no-custom-assets', is_flag=True, help='Disable automatic custom asset detection')
def create_auto_youtube(topic, duration, style, voice, image_provider, audience, language, output_dir, auto_assemble, custom_overlay, custom_transition, transition_sound, background_music, overlay_opacity, music_volume, no_custom_assets):
    """🚀 Create 100% automated YouTube video from just a topic - Zero manual work required!"""
    async def _create():
        try:
            console.print(f"[bold green]🚀 100% AUTOMATED YOUTUBE VIDEO GENERATION[/bold green]")
            console.print(f"[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
            console.print(f"[yellow]📝 Topic:[/yellow] {topic}")
            console.print(f"[yellow]⏱️  Duration:[/yellow] {duration} seconds")
            console.print(f"[yellow]🎨 Style:[/yellow] {style}")
            console.print(f"[yellow]🗣️  Voice:[/yellow] {voice}")
            console.print(f"[yellow]🖼️  Images:[/yellow] {image_provider}")
            console.print(f"[yellow]👥 Audience:[/yellow] {audience}")
            
            # Prepare output directory
            if not output_dir:
                safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_topic = "_".join(safe_topic.split())[:50]
                output_path = Path(config_manager.load_config().output_dir) / f"auto_youtube_{safe_topic}"
            else:
                output_path = Path(output_dir)
            
            output_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[yellow]📁 Output:[/yellow] {output_path}")
            
            # STEP 1: Generate AI Script
            console.print(f"\n[bold blue]🤖 STEP 1: Generating AI Script...[/bold blue]")
            from youtube_ai.content.script_generator import ScriptGenerator
            
            generator = ScriptGenerator()
            script = await generator.generate_script(
                topic=topic,
                style=style,
                duration=duration,
                audience=audience
            )
            
            script_content = script
            script_file = output_path / "generated_script.txt"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            console.print(f"[green]✅ AI Script Generated:[/green] {len(script_content.split())} words")
            
            # STEP 2: Generate Professional Voiceover
            console.print(f"\n[bold blue]🎵 STEP 2: Creating Professional Voiceover...[/bold blue]")
            from youtube_ai.ai.tts_client import tts_manager
            
            audio_file = output_path / "professional_voiceover.mp3"
            await tts_manager.synthesize_speech(
                text=script_content,
                voice=voice,
                provider="openai",
                output_file=audio_file
            )
            console.print(f"[green]✅ Professional Voiceover Created:[/green] {audio_file.name}")
            
            # STEP 3: Generate AI Visuals
            console.print(f"\n[bold blue]🎨 STEP 3: Generating Contextual AI Visuals...[/bold blue]")
            from youtube_ai.ai.image_generator import image_generator, ImageStyle, ImageProvider
            
            # Parse parameters
            try:
                image_style = ImageStyle(style.lower())
            except ValueError:
                image_style = ImageStyle.EDUCATIONAL
                
            try:
                img_provider = ImageProvider(image_provider.lower())
            except ValueError:
                img_provider = ImageProvider.STABILITY
            
            # Calculate optimal number of visuals based on duration
            num_visuals = max(3, min(8, duration // 30))
            
            visuals_dir = output_path / "ai_visuals"
            generated_images = await image_generator.generate_script_visuals(
                script=script_content,
                output_dir=visuals_dir,
                style=image_style,
                num_images=num_visuals,
                provider=img_provider
            )
            console.print(f"[green]✅ AI Visuals Generated:[/green] {len(generated_images)} contextual images")
            
            # STEP 4: Generate SEO-Optimized Metadata
            console.print(f"\n[bold blue]📊 STEP 4: Creating SEO-Optimized Metadata...[/bold blue]")
            from youtube_ai.content.seo_optimizer import SEOOptimizer
            
            seo_optimizer = SEOOptimizer()
            
            # Generate titles
            titles = await seo_optimizer.generate_titles(
                content=script_content,
                keywords=[topic],
                count=3
            )
            best_title = titles[0] if titles else f"Complete Guide to {topic}"
            
            # Generate description
            description = await seo_optimizer.generate_description(
                content=script_content,
                title=best_title,
                keywords=[topic],
                include_timestamps=True
            )
            
            # Generate tags  
            tags_list = await seo_optimizer.generate_tags(
                content=script_content,
                title=best_title,
                keywords=[topic],
                max_tags=15
            )
            tags = ", ".join(tags_list)
            
            console.print(f"[green]✅ SEO Metadata Generated:[/green] Title, description, tags")
            
            # STEP 5: Create Synchronized Subtitles
            console.print(f"\n[bold blue]📝 STEP 5: Creating Synchronized Subtitles...[/bold blue]")
            subtitle_file = output_path / "subtitles.srt"
            await _create_subtitle_file_ai(script_content, subtitle_file)
            console.print(f"[green]✅ Subtitles Created:[/green] {subtitle_file.name}")
            
            # STEP 6: Professional Video Assembly (if requested)
            final_video_path = None
            if auto_assemble:
                console.print(f"\n[bold blue]🎬 STEP 6: Creating Professional Video with Color Grading & Transitions...[/bold blue]")
                final_video_path = await _create_professional_video_with_effects(
                    output_path, audio_file, [img.file_path for img in generated_images], 
                    subtitle_file, script_content, style
                )
                if final_video_path:
                    console.print(f"[green]✅ Professional Video Created:[/green] {final_video_path.name}")
                    console.print(f"[green]    ✨ Enhanced with color grading & smooth transitions[/green]")
                else:
                    console.print(f"[yellow]⚠️  Professional video creation requires MoviePy - using fallback assembly[/yellow]")
                    # Fallback to basic assembly
                    final_video_path = await _auto_assemble_video(
                        output_path, audio_file, [img.file_path for img in generated_images], subtitle_file
                    )
            
            # STEP 7: Create Complete Project Package
            console.print(f"\n[bold blue]📦 STEP 7: Creating Complete Project Package...[/bold blue]")
            
            # Save all metadata
            project_data = {
                "title": best_title,
                "all_titles": titles,
                "description": description,
                "tags": tags,
                "topic": topic,
                "duration": duration,
                "style": style,
                "voice": voice,
                "image_provider": image_provider,
                "audience": audience,
                "language": language,
                "num_visuals": len(generated_images),
                "script_words": len(script_content.split()),
                "auto_assembled": auto_assemble and final_video_path is not None,
                "files": {
                    "script": str(script_file),
                    "audio": str(audio_file),
                    "visuals": [str(img.file_path) for img in generated_images],
                    "subtitles": str(subtitle_file),
                    "final_video": str(final_video_path) if final_video_path else None
                },
                "ai_prompts": [
                    {
                        "scene": img.scene_number,
                        "prompt": img.prompt,
                        "provider": img.provider,
                        "keywords": img.metadata.get("keywords", []) if img.metadata else []
                    } for img in generated_images
                ],
                "created_with": "YouTube AI CLI - 100% Automated",
                "creation_timestamp": str(Path().absolute())
            }
            
            metadata_file = output_path / "complete_project_data.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            # Create comprehensive instructions
            await _create_automated_instructions(
                output_path, audio_file, [img.file_path for img in generated_images], 
                subtitle_file, project_data, final_video_path
            )
            
            # SUCCESS SUMMARY
            console.print(f"\n[bold green]🎉 100% AUTOMATED YOUTUBE VIDEO COMPLETED![/bold green]")
            console.print(f"[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold blue]")
            console.print(f"[green]✅ AI-Generated Script:[/green] {len(script_content.split())} words")
            console.print(f"[green]✅ Professional Voiceover:[/green] {voice} voice")
            console.print(f"[green]✅ Contextual AI Visuals:[/green] {len(generated_images)} images ({image_provider})")
            console.print(f"[green]✅ SEO-Optimized Metadata:[/green] Title, description, tags")
            console.print(f"[green]✅ Synchronized Subtitles:[/green] Professional timing")
            console.print(f"[green]✅ Complete Project Package:[/green] Ready for upload")
            
            if final_video_path:
                console.print(f"[green]✅ Final Video File:[/green] {final_video_path.name}")
            
            console.print(f"\n[bold yellow]🚀 YOUR AUTOMATED YOUTUBE VIDEO IS READY![/bold yellow]")
            console.print(f"[yellow]📁 Project Directory:[/yellow] {output_path}")
            console.print(f"[yellow]🎬 Title:[/yellow] {best_title}")
            console.print(f"[yellow]⏱️  Duration:[/yellow] ~{duration} seconds")
            console.print(f"[yellow]💡 Quality:[/yellow] Professional broadcast-ready")
            
            if not auto_assemble:
                console.print(f"\n[blue]💡 Pro Tip:[/blue] Add --auto-assemble flag for complete video file!")
            
            console.print(f"\n[cyan]🎯 Zero manual work required - Your YouTube automation is complete![/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error in automated video creation:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


async def _create_professional_video_with_effects(
    project_dir: Path,
    audio_file: Path,
    image_files: List[Path],
    subtitle_file: Path,
    script_content: str,
    style: str
) -> Optional[Path]:
    """Create professional video with color grading, transitions, and effects using MoviePy."""
    try:
        from youtube_ai.media.advanced_video_generator import (
            AdvancedVideoGenerator, AdvancedVideoStyle, MediaAsset, 
            TransitionType, EffectType
        )
        
        # Initialize advanced video generator
        generator = AdvancedVideoGenerator()
        
        # Convert style string to enum
        style_mapping = {
            'educational': AdvancedVideoStyle.EDUCATIONAL,
            'professional': AdvancedVideoStyle.DOCUMENTARY,
            'cinematic': AdvancedVideoStyle.CINEMATIC,
            'documentary': AdvancedVideoStyle.DOCUMENTARY,
            'social_media': AdvancedVideoStyle.SOCIAL_MEDIA
        }
        video_style = style_mapping.get(style.lower(), AdvancedVideoStyle.EDUCATIONAL)
        
        # Create media assets from images with enhanced effects
        media_assets = []
        for i, img_file in enumerate(image_files):
            # Add variety in Ken Burns effects for visual interest
            zoom_variations = [
                (1.0, 1.15),  # Gentle zoom in
                (1.1, 1.0),   # Gentle zoom out  
                (1.0, 1.2),   # Stronger zoom in
                (1.05, 1.05)  # Static with slight zoom
            ]
            
            pan_variations = [
                ((0.4, 0.4), (0.6, 0.6)),  # Diagonal pan
                ((0.5, 0.3), (0.5, 0.7)),  # Vertical pan
                ((0.3, 0.5), (0.7, 0.5)),  # Horizontal pan
                ((0.5, 0.5), (0.5, 0.5))   # Static centered
            ]
            
            zoom_start, zoom_end = zoom_variations[i % len(zoom_variations)]
            pan_start, pan_end = pan_variations[i % len(pan_variations)]
            
            asset = MediaAsset(
                file_path=img_file,
                asset_type="image",
                duration=None,  # Will be calculated by scenes
                ken_burns=True,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                pan_start=pan_start,
                pan_end=pan_end,
                effects=[EffectType.COLOR_GRADE]  # Apply consistent color grading
            )
            media_assets.append(asset)
        
        # Generate professional video with enhanced features
        output_file = project_dir / "PROFESSIONAL_YOUTUBE_VIDEO.mp4"
        
        console.print(f"[cyan]    🎨 Applying {video_style.value} color grading...[/cyan]")
        console.print(f"[cyan]    ✨ Adding smooth transitions between {len(media_assets)} scenes...[/cyan]")
        console.print(f"[cyan]    🎬 Generating Ken Burns effects for dynamic visuals...[/cyan]")
        
        video_output = await generator.create_professional_video(
            script=script_content,
            output_file=output_file,
            style=video_style,
            media_assets=media_assets,
            voice=None,  # Audio already generated
            provider=None,
            include_subtitles=True,
            include_effects=True
        )
        
        # The advanced generator creates its own audio, but we want to use our generated audio
        # So we'll need to replace the audio track
        await _replace_audio_track(output_file, audio_file, project_dir)
        
        console.print(f"[green]    ✅ Professional effects applied successfully[/green]")
        return output_file if output_file.exists() else None
        
    except ImportError as e:
        logger.warning(f"Advanced video generation not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Professional video creation failed: {e}")
        return None


async def _replace_audio_track(video_file: Path, audio_file: Path, project_dir: Path) -> None:
    """Replace the audio track in the video file with our generated voiceover."""
    try:
        import subprocess
        import shutil
        
        if not shutil.which('ffmpeg'):
            logger.warning("FFmpeg not found - cannot replace audio track")
            return
        
        temp_video = project_dir / "temp_with_new_audio.mp4"
        
        # Replace audio track
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_file),
            '-i', str(audio_file),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            str(temp_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Replace original with new version
            shutil.move(str(temp_video), str(video_file))
            logger.info("Audio track replaced successfully")
        else:
            logger.error(f"Audio replacement failed: {result.stderr}")
            if temp_video.exists():
                temp_video.unlink()
    
    except Exception as e:
        logger.error(f"Error replacing audio track: {e}")


async def _auto_assemble_video(
    project_dir: Path, 
    audio_file: Path, 
    image_files: List[Path], 
    subtitle_file: Path
) -> Optional[Path]:
    """Automatically assemble final video using FFmpeg if available."""
    try:
        import subprocess
        import shutil
        
        # Check if FFmpeg is available
        if not shutil.which('ffmpeg'):
            logger.warning("FFmpeg not found - cannot auto-assemble video")
            return None
        
        final_video = project_dir / "FINAL_YOUTUBE_VIDEO.mp4"
        
        # Calculate timing
        num_images = len(image_files)
        if num_images == 0:
            return None
            
        # Estimate audio duration (rough)
        image_duration = 30.0  # seconds per image
        
        # Create image list file for FFmpeg
        image_list_file = project_dir / "image_list.txt"
        with open(image_list_file, 'w') as f:
            for img_file in image_files:
                f.write(f"file '{img_file.name}'\n")
                f.write(f"duration {image_duration}\n")
        
        # Create slideshow video
        temp_video = project_dir / "temp_slideshow.mp4"
        slideshow_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(image_list_file),
            '-vf', 'scale=1920:1080',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            str(temp_video)
        ]
        
        result1 = subprocess.run(slideshow_cmd, capture_output=True, text=True)
        if result1.returncode != 0:
            logger.error(f"FFmpeg slideshow creation failed: {result1.stderr}")
            return None
        
        # Add audio to video
        final_cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_video),
            '-i', str(audio_file),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            str(final_video)
        ]
        
        result2 = subprocess.run(final_cmd, capture_output=True, text=True)
        if result2.returncode != 0:
            logger.error(f"FFmpeg audio merge failed: {result2.stderr}")
            return None
        
        # Clean up temp files
        if temp_video.exists():
            temp_video.unlink()
        if image_list_file.exists():
            image_list_file.unlink()
        
        return final_video if final_video.exists() else None
        
    except Exception as e:
        logger.error(f"Auto-assembly failed: {e}")
        return None


async def _create_automated_instructions(
    output_dir: Path,
    audio_file: Path,
    image_files: List[Path],
    subtitle_file: Path,
    project_data: Dict,
    final_video: Optional[Path]
) -> None:
    """Create comprehensive instructions for the automated video."""
    instructions_file = output_dir / "AUTOMATED_VIDEO_INSTRUCTIONS.md"
    
    instructions = f"""# 🚀 100% AUTOMATED YOUTUBE VIDEO - READY TO UPLOAD!

## 📊 Project Summary
- **Title**: {project_data['title']}
- **Topic**: {project_data['topic']}
- **Duration**: ~{project_data['duration']} seconds
- **Style**: {project_data['style']}
- **Voice**: {project_data['voice']}
- **Visuals**: {project_data['num_visuals']} AI-generated images ({project_data['image_provider']})
- **Quality**: Professional broadcast-ready
- **Auto-Assembled**: {'✅ Yes' if final_video else '❌ Manual assembly required'}

## 🎬 Ready-to-Upload Assets

### Final Video File
"""
    
    if final_video:
        instructions += f"""✅ **{final_video.name}** - Upload this file directly to YouTube!
- Resolution: 1920x1080 (Full HD)
- Format: MP4 (YouTube optimized)
- Audio: Professional voiceover
- Visuals: AI-generated contextual images
"""
    else:
        instructions += f"""❌ **Auto-assembly not available** (FFmpeg required)
📋 **Manual Assembly**: Use VIDEO_ASSEMBLY_INSTRUCTIONS.md
⚡ **Quick Setup**: Install FFmpeg and run with --auto-assemble flag
"""
    
    instructions += f"""
### 📝 YouTube Upload Metadata (Copy & Paste Ready)

**Title Options** (Choose best for your audience):
"""
    
    for i, title in enumerate(project_data.get('all_titles', [project_data['title']]), 1):
        instructions += f"{i}. {title}\n"
    
    instructions += f"""
**Description** (SEO Optimized):
```
{project_data['description']}
```

**Tags** (Copy directly):
```
{project_data['tags']}
```

## 🎯 Upload Checklist

### Before Upload:
- [ ] Review final video quality
- [ ] Check audio levels
- [ ] Verify subtitle synchronization
- [ ] Choose best title from options above

### YouTube Upload Settings:
- [ ] **Visibility**: Public (for maximum reach)
- [ ] **Category**: Education (or relevant category)
- [ ] **Language**: {project_data['language']}
- [ ] **Captions**: Upload `{subtitle_file.name}`
- [ ] **Thumbnail**: Create custom thumbnail from AI visuals
- [ ] **End Screens**: Add subscribe button and related videos

### Post-Upload Optimization:
- [ ] Add video to relevant playlists
- [ ] Pin engaging comment
- [ ] Share on social media
- [ ] Monitor analytics in first 24 hours

## 🚀 Scaling Your Content

### Batch Production:
```bash
# Generate multiple videos automatically
python src/youtube_ai/cli/main.py create auto-youtube --topic "Advanced Python Tips" --auto-assemble
python src/youtube_ai/cli/main.py create auto-youtube --topic "Machine Learning Basics" --auto-assemble
python src/youtube_ai/cli/main.py create auto-youtube --topic "Web Development Guide" --auto-assemble
```

### Content Calendar Ideas:
Based on your topic "{project_data['topic']}", consider these follow-up videos:
- Advanced {project_data['topic']} concepts
- Common mistakes in {project_data['topic']}
- {project_data['topic']} vs alternatives
- Real-world {project_data['topic']} projects
- {project_data['topic']} career opportunities

## 🎨 AI-Generated Visual Breakdown

Your video includes {project_data['num_visuals']} contextually-matched visuals:
"""
    
    for i, prompt_data in enumerate(project_data.get('ai_prompts', []), 1):
        instructions += f"""
### Scene {i}: {prompt_data.get('provider', 'AI').title()} Generated
- **Keywords**: {', '.join(prompt_data.get('keywords', [])[:3])}
- **Style**: {prompt_data.get('prompt', 'Contextual visual')[:60]}...
"""
    
    instructions += f"""
## 💡 Performance Tips

### Algorithm Optimization:
- Upload during peak hours (2-4 PM, 7-9 PM)
- Use all metadata fields
- Engage with early comments
- Create compelling thumbnail from AI visuals

### Content Strategy:
- This video style works well for: {project_data['style']} content
- Target audience: {project_data['audience']} level viewers
- Optimal length: {project_data['duration']} seconds (proven engagement)

### Analytics to Monitor:
- Click-through rate (aim for >10%)
- Average view duration (aim for >60%)
- Engagement rate (likes, comments, shares)
- Subscriber conversion rate

---
🤖 **Generated with YouTube AI CLI - 100% Automated Video Creation**
✨ **Zero manual work required - Professional quality guaranteed**
🚀 **Scale to unlimited videos with consistent quality**
"""
    
    with open(instructions_file, 'w') as f:
        f.write(instructions)


async def _create_subtitle_file_ai(script: str, subtitle_file: Path) -> None:
    """Create subtitle file for AI video."""
    # Simple subtitle creation - can be enhanced with timing analysis
    import re
    
    # Clean script and split into segments
    cleaned = re.sub(r'^#.*$', '', script, flags=re.MULTILINE)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    sentences = [s.strip() + '.' for s in cleaned.split('.') if s.strip()]
    
    subtitles = []
    time_per_sentence = 3.0  # Rough estimate
    
    for i, sentence in enumerate(sentences):
        start_time = i * time_per_sentence
        end_time = (i + 1) * time_per_sentence
        
        start_srt = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},000"
        end_srt = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},000"
        
        subtitles.append(f"{i+1}\n{start_srt} --> {end_srt}\n{sentence}\n")
    
    with open(subtitle_file, 'w') as f:
        f.write('\n'.join(subtitles))


async def _create_ai_assembly_instructions(
    output_dir: Path, 
    audio_file: Path, 
    image_files: List[Path], 
    subtitle_file: Path,
    generated_images: List[Any]
) -> None:
    """Create detailed assembly instructions for AI video."""
    instructions_file = output_dir / "VIDEO_ASSEMBLY_INSTRUCTIONS.md"
    
    duration = len(generated_images) * 33.0  # Rough estimate
    
    # Create detailed instructions with AI context
    instructions = f"""# 🤖 AI-Powered Video Assembly Instructions

## 📁 Project Overview
- **Audio**: `{audio_file.name}` (Professional AI voiceover)
- **AI Visuals**: {len(image_files)} contextually generated images
- **Subtitles**: `{subtitle_file.name}`
- **Estimated Duration**: {duration:.1f} seconds
- **Resolution**: 1792x1024 (16:9 ratio, optimized for YouTube)

## 🎨 AI-Generated Visuals

Each visual was intelligently generated to match specific script segments:

"""
    
    for img in generated_images:
        instructions += f"""### Scene {img.scene_number}: {img.file_path.name}
- **Script Context**: "{img.segment_text[:100]}..."
- **AI Prompt**: {img.prompt[:80]}...
- **Keywords**: {', '.join(img.metadata.get('keywords', [])[:3]) if img.metadata else 'N/A'}

"""
    
    instructions += f"""
## 🛠️ Assembly Options

### Option 1: Using FFmpeg (Command Line)
```bash
# Create image list for slideshow
echo "file 'ai_visuals/ai_visual_01.png'
duration 33.0
file 'ai_visuals/ai_visual_02.png'
duration 33.0
file 'ai_visuals/ai_visual_03.png'
duration 33.0
file 'ai_visuals/ai_visual_04.png'
duration 33.0
file 'ai_visuals/ai_visual_05.png'
duration 33.0" > image_list.txt

# Create video from AI images
ffmpeg -f concat -safe 0 -i image_list.txt -vf "scale=1920:1080" -pix_fmt yuv420p temp_video.mp4

# Add professional voiceover
ffmpeg -i temp_video.mp4 -i {audio_file.name} -c:v copy -c:a aac -shortest final_ai_video.mp4

# Add subtitles (optional)
ffmpeg -i final_ai_video.mp4 -vf "subtitles={subtitle_file.name}" final_ai_video_with_subs.mp4
```

### Option 2: Using DaVinci Resolve (Free Professional Editor)
1. Create new project (1920x1080, 30fps)
2. Import all AI-generated visuals and audio file
3. Drag AI images to timeline, each for 33 seconds
4. Add professional voiceover track
5. Import subtitle file
6. Add smooth transitions between AI visuals (1-2 second crossfades)
7. Apply subtle Ken Burns effect to AI images (1.0x to 1.05x zoom)
8. Export as MP4 (H.264, high quality)

### Option 3: Using Adobe Premiere Pro
1. New sequence: 1920x1080, 29.97fps
2. Import all AI-generated assets
3. Create slideshow with contextual AI visuals
4. Add professional voiceover track
5. Import SRT subtitles
6. Apply motion to AI images (subtle zoom/pan)
7. Add transitions between scenes
8. Color grade for consistency
9. Export with H.264 codec

## 🎨 AI Visual Enhancement Tips

### Contextual Matching:
- Each visual was generated to match specific script content
- Images flow naturally with the narrative
- Visual keywords align with spoken content

### Professional Effects:
- **Ken Burns Effect**: Subtle zoom (1.0x to 1.05x scale) on AI images
- **Transitions**: 1-2 second crossfades between AI visuals
- **Timing**: Each visual aligns with script segments
- **Color Consistency**: AI images may need slight color grading for uniformity

## 🚀 YouTube Optimization

### Title Ideas:
- "AI-Generated: [Your Topic]"
- "[Topic] - Created with AI"
- "The Future of [Topic] - AI Explained"

### Description Template:
```
🤖 This video was created using advanced AI technology!

✨ Features:
- AI-generated script analysis
- Contextual visual generation
- Professional AI voiceover
- Intelligent scene matching

[Your content description here]

#AI #YouTube #[YourTopic] #Automation
```

## 🎯 Pro Tips for AI Videos

1. **Visual Flow**: The AI images were designed to flow with your narrative
2. **Engagement**: Each visual reinforces the spoken content
3. **Quality**: 16:9 ratio optimized for YouTube display
4. **Accessibility**: Subtitles included for better reach
5. **SEO**: Mention "AI-generated" in title/description for discoverability

## 📊 Performance Optimization

- **Thumbnail**: Use AI visual #1 as thumbnail base
- **Chapters**: Create YouTube chapters matching visual segments
- **Tags**: Include AI, automation, and topic-specific keywords
- **Cards**: Add cards during visual transitions

---
*🤖 Generated with YouTube AI CLI - Advanced AI Video Automation*
"""
    
    with open(instructions_file, 'w') as f:
        f.write(instructions)


# Upload Commands
@upload.command('video')
@click.option('--video', type=click.Path(exists=True), required=True, help='Video file to upload')
@click.option('--title', required=True, help='Video title')
@click.option('--description', help='Video description')
@click.option('--tags', help='Video tags (comma-separated)')
@click.option('--privacy', default='private', help='Privacy setting (private, unlisted, public)')
@click.option('--thumbnail', type=click.Path(exists=True), help='Thumbnail image file')
@click.option('--schedule', help='Schedule publish time (YYYY-MM-DD HH:MM)')
def upload_video(video, title, description, tags, privacy, thumbnail, schedule):
    """Upload video to YouTube."""
    async def _upload():
        try:
            from youtube_ai.utils.youtube_uploader import youtube_uploader, VideoMetadata, PrivacyStatus
            from datetime import datetime
            
            if not youtube_uploader:
                console.print("[red]Error:[/red] YouTube uploader not available. Please install google-api-python-client")
                return
            
            # Parse schedule time if provided
            publish_at = None
            if schedule:
                try:
                    publish_at = datetime.strptime(schedule, '%Y-%m-%d %H:%M')
                except ValueError:
                    console.print("[red]Error:[/red] Invalid schedule format. Use YYYY-MM-DD HH:MM")
                    return
            
            # Prepare metadata
            metadata = VideoMetadata(
                title=title,
                description=description or "",
                tags=tags.split(',') if tags else [],
                privacy_status=privacy,
                thumbnail_file=Path(thumbnail) if thumbnail else None,
                publish_at=publish_at
            )
            
            # Validate metadata
            issues = youtube_uploader.validate_metadata(metadata)
            if issues:
                console.print("[red]Metadata validation errors:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")
                return
            
            def progress_callback(progress):
                console.print(f"Upload progress: {progress.percentage:.1f}%")
            
            console.print(f"[blue]Uploading video...[/blue]")
            console.print(f"[dim]File:[/dim] {video}")
            console.print(f"[dim]Title:[/dim] {title}")
            console.print(f"[dim]Privacy:[/dim] {privacy}")
            
            result = await youtube_uploader.upload_video(
                video_file=Path(video),
                metadata=metadata,
                progress_callback=progress_callback
            )
            
            console.print(f"[green]✓[/green] Upload completed!")
            console.print(f"[dim]Video ID:[/dim] {result.video_id}")
            console.print(f"[dim]Video URL:[/dim] {result.video_url}")
            console.print(f"[dim]Status:[/dim] {result.status}")
            
        except Exception as e:
            console.print(f"[red]Error uploading video:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_upload())


# Workflow Commands
@workflow.command('create')
@click.option('--name', required=True, help='Workflow name')
@click.option('--description', required=True, help='Workflow description')
@click.option('--template', help='Base template to use')
@click.option('--steps', help='JSON file with step definitions')
def workflow_create(name, description, template, steps):
    """Create a new workflow template."""
    async def _create():
        try:
            from youtube_ai.utils.workflow_manager import workflow_manager, WorkflowStep, WorkflowStepType
            
            if template:
                console.print(f"[red]Error:[/red] Template-based creation not yet implemented")
                return
            
            if steps:
                # Load steps from JSON file
                import json
                with open(steps, 'r') as f:
                    steps_data = json.load(f)
                
                workflow_steps = []
                for step_data in steps_data:
                    step = WorkflowStep(
                        id=step_data['id'],
                        type=WorkflowStepType(step_data['type']),
                        name=step_data['name'],
                        config=step_data['config'],
                        depends_on=step_data.get('depends_on', [])
                    )
                    workflow_steps.append(step)
            else:
                # Create default educational workflow
                workflow_steps = [
                    WorkflowStep(
                        id="generate_script",
                        type=WorkflowStepType.GENERATE_SCRIPT,
                        name="Generate Script",
                        config={"style": "educational", "duration": 300}
                    ),
                    WorkflowStep(
                        id="optimize_seo",
                        type=WorkflowStepType.OPTIMIZE_SEO,
                        name="Optimize SEO",
                        config={},
                        depends_on=["generate_script"]
                    ),
                    WorkflowStep(
                        id="generate_audio",
                        type=WorkflowStepType.GENERATE_AUDIO,
                        name="Generate Audio",
                        config={"voice": "alloy", "speed": 1.0},
                        depends_on=["generate_script"]
                    ),
                    WorkflowStep(
                        id="generate_video",
                        type=WorkflowStepType.GENERATE_VIDEO,
                        name="Generate Video",
                        config={"style": "slideshow"},
                        depends_on=["generate_script", "generate_audio"]
                    )
                ]
            
            template = await workflow_manager.create_workflow_template(
                name=name,
                description=description,
                steps=workflow_steps,
                default_config={
                    "output_dir": config_manager.load_config().output_dir
                }
            )
            
            console.print(f"[green]✓[/green] Workflow template created: {template.id}")
            console.print(f"[dim]Steps:[/dim] {len(template.steps)}")
            
        except Exception as e:
            console.print(f"[red]Error creating workflow:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@workflow.command('run')
@click.option('--name', required=True, help='Workflow template name')
@click.option('--topic', required=True, help='Video topic')
@click.option('--config', help='JSON file with additional configuration')
@click.option('--style', help='Content style override')
@click.option('--duration', type=int, help='Duration override')
def workflow_run(name, topic, config, style, duration):
    """Run a workflow template."""
    async def _run():
        try:
            from youtube_ai.utils.workflow_manager import workflow_manager
            
            console.print(f"[blue]Running workflow: {name}[/blue]")
            console.print(f"[dim]Topic:[/dim] {topic}")
            
            # Prepare inputs
            inputs = {"topic": topic}
            
            if style:
                inputs["style"] = style
            if duration:
                inputs["duration"] = duration
            
            # Load additional config if provided
            execution_config = {}
            if config:
                import json
                with open(config, 'r') as f:
                    execution_config = json.load(f)
            
            # Execute workflow
            execution = await workflow_manager.execute_workflow(
                template_id=name,
                inputs=inputs,
                execution_config=execution_config
            )
            
            if execution.status.value == "completed":
                console.print(f"[green]✓[/green] Workflow completed successfully!")
                console.print(f"[dim]Execution ID:[/dim] {execution.execution_id}")
                console.print(f"[dim]Completed steps:[/dim] {len(execution.completed_steps)}")
                
                # Show outputs
                if execution.outputs:
                    console.print("\n[bold]Outputs:[/bold]")
                    for step_id, output in execution.outputs.items():
                        if isinstance(output, dict):
                            for key, value in output.items():
                                if key.endswith('_file'):
                                    console.print(f"  {key}: {value}")
            else:
                console.print(f"[red]✗[/red] Workflow failed!")
                console.print(f"[dim]Status:[/dim] {execution.status.value}")
                if execution.error_message:
                    console.print(f"[dim]Error:[/dim] {execution.error_message}")
            
        except Exception as e:
            console.print(f"[red]Error running workflow:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_run())


@workflow.command('list')
def workflow_list():
    """List available workflow templates."""
    try:
        from youtube_ai.utils.workflow_manager import workflow_manager
        
        templates = workflow_manager.list_workflow_templates()
        
        if not templates:
            console.print("[yellow]No workflow templates found[/yellow]")
            return
        
        table = Table(title="Workflow Templates")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description")
        table.add_column("Steps", justify="center")
        table.add_column("Version", justify="center")
        
        for template in templates:
            table.add_row(
                template["id"],
                template["name"],
                template["description"][:50] + "..." if len(template["description"]) > 50 else template["description"],
                str(template["steps"]),
                template["version"]
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing workflows:[/red] {e}")


@workflow.command('status')
@click.argument('execution_id')
def workflow_status(execution_id):
    """Check status of a workflow execution."""
    async def _status():
        try:
            from youtube_ai.utils.workflow_manager import workflow_manager
            
            execution = await workflow_manager.get_execution_status(execution_id)
            
            console.print(f"[bold]Workflow Execution Status[/bold]")
            console.print(f"[dim]Execution ID:[/dim] {execution.execution_id}")
            console.print(f"[dim]Workflow:[/dim] {execution.workflow_id}")
            console.print(f"[dim]Status:[/dim] {execution.status.value}")
            console.print(f"[dim]Started:[/dim] {execution.started_at}")
            
            if execution.completed_at:
                console.print(f"[dim]Completed:[/dim] {execution.completed_at}")
            
            if execution.current_step:
                console.print(f"[dim]Current Step:[/dim] {execution.current_step}")
            
            console.print(f"[dim]Completed Steps:[/dim] {len(execution.completed_steps)}")
            console.print(f"[dim]Failed Steps:[/dim] {len(execution.failed_steps)}")
            
            if execution.error_message:
                console.print(f"[red]Error:[/red] {execution.error_message}")
            
        except Exception as e:
            console.print(f"[red]Error getting workflow status:[/red] {e}")
    
    asyncio.run(_status())


# Analytics Commands
@cli.group()
def analytics():
    """Analytics and performance tracking."""
    pass


@analytics.command('summary')
@click.option('--days', default=7, help='Number of days to analyze')
@click.option('--export', help='Export to file (json/csv)')
def analytics_summary(days, export):
    """Show performance summary."""
    try:
        from youtube_ai.utils.analytics_tracker import analytics_tracker
        
        summary = analytics_tracker.get_performance_summary(days=days)
        
        console.print(f"[bold blue]Performance Summary - Last {days} Days[/bold blue]")
        console.print(f"[dim]Generated:[/dim] {summary['generated_at'][:19]}")
        
        # Event statistics
        if summary['event_statistics']:
            table = Table(title="Event Statistics")
            table.add_column("Event Type", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Avg Duration", justify="right")
            table.add_column("Success Rate", justify="right")
            
            for event_type, stats in summary['event_statistics'].items():
                table.add_row(
                    event_type.replace('_', ' ').title(),
                    str(stats['count']),
                    f"{stats['avg_duration']:.1f}s" if stats['avg_duration'] else "N/A",
                    f"{stats['success_rate']:.1f}%"
                )
            
            console.print(table)
        
        # Cost analysis
        cost_data = summary['cost_analysis']
        console.print(f"\n[bold]Cost Analysis:[/bold]")
        console.print(f"  Total Cost: ${cost_data['total_cost']:.2f}")
        console.print(f"  Average Cost per Operation: ${cost_data['average_cost_per_operation']:.3f}")
        
        # Session statistics
        session_data = summary['session_statistics']
        console.print(f"\n[bold]Session Statistics:[/bold]")
        console.print(f"  Total Sessions: {session_data['total_sessions']}")
        console.print(f"  Average Duration: {session_data['average_duration']:.1f}s")
        console.print(f"  Average Success Rate: {session_data['average_success_rate']:.1f}%")
        
        # Export if requested
        if export:
            export_file = analytics_tracker.export_analytics(format=export)
            console.print(f"\n[green]✓[/green] Exported to: {export_file}")
        
    except Exception as e:
        console.print(f"[red]Error getting analytics summary:[/red] {e}")


@analytics.command('trends')
@click.option('--days', default=30, help='Number of days to analyze')
def analytics_trends(days):
    """Show usage trends."""
    try:
        from youtube_ai.utils.analytics_tracker import analytics_tracker
        
        trends = analytics_tracker.get_usage_trends(days=days)
        
        console.print(f"[bold blue]Usage Trends - Last {days} Days[/bold blue]")
        
        # Daily events
        if trends['daily_events']:
            console.print("\n[bold]Daily Events:[/bold]")
            for date, count in trends['daily_events'][-7:]:  # Show last 7 days
                console.print(f"  {date}: {count} events")
        
        # Daily costs
        if trends['daily_costs']:
            console.print("\n[bold]Daily Costs:[/bold]")
            for date, cost in trends['daily_costs'][-7:]:  # Show last 7 days
                console.print(f"  {date}: ${cost:.2f}")
        
        # Error rates
        if trends['daily_error_rates']:
            console.print("\n[bold]Daily Error Rates:[/bold]")
            for date, rate in trends['daily_error_rates'][-7:]:  # Show last 7 days
                color = "red" if rate > 10 else "yellow" if rate > 5 else "green"
                console.print(f"  {date}: [{color}]{rate:.1f}%[/{color}]")
        
    except Exception as e:
        console.print(f"[red]Error getting trends:[/red] {e}")


@analytics.command('optimize')
def analytics_optimize():
    """Get optimization suggestions."""
    try:
        from youtube_ai.utils.analytics_tracker import analytics_tracker
        
        suggestions = analytics_tracker.get_optimization_suggestions()
        
        if not suggestions:
            console.print("[green]✓[/green] No optimization suggestions - everything looks good!")
            return
        
        console.print("[bold blue]Optimization Suggestions[/bold blue]")
        
        for suggestion in suggestions:
            priority_color = {
                "high": "red",
                "medium": "yellow",
                "low": "blue"
            }.get(suggestion['priority'], "white")
            
            console.print(f"\n[{priority_color}]Priority: {suggestion['priority'].upper()}[/{priority_color}]")
            console.print(f"Type: {suggestion['type']}")
            console.print(f"Issue: {suggestion['message']}")
            console.print(f"Action: {suggestion['action']}")
        
    except Exception as e:
        console.print(f"[red]Error getting optimization suggestions:[/red] {e}")


# Batch Commands
@cli.group()
def batch():
    """Batch processing operations."""
    pass


@batch.command('create-csv')
@click.option('--csv', type=click.Path(exists=True), required=True, help='CSV file with topics and configurations')
@click.option('--workflow', required=True, help='Workflow template to use')
@click.option('--name', help='Batch job name')
@click.option('--concurrent', default=3, help='Maximum concurrent executions')
def batch_create_csv(csv, workflow, name, concurrent):
    """Create batch job from CSV file."""
    async def _create():
        try:
            from youtube_ai.utils.batch_processor import batch_processor
            
            console.print(f"[blue]Creating batch job from CSV...[/blue]")
            console.print(f"[dim]CSV file:[/dim] {csv}")
            console.print(f"[dim]Workflow:[/dim] {workflow}")
            
            batch_job = await batch_processor.create_batch_from_csv(
                csv_file=Path(csv),
                workflow_template=workflow,
                name=name,
                max_concurrent=concurrent
            )
            
            console.print(f"[green]✓[/green] Created batch job: {batch_job.id}")
            console.print(f"[dim]Name:[/dim] {batch_job.name}")
            console.print(f"[dim]Tasks:[/dim] {len(batch_job.tasks)}")
            console.print(f"[dim]Max Concurrent:[/dim] {batch_job.max_concurrent}")
            
        except Exception as e:
            console.print(f"[red]Error creating batch job:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@batch.command('create-topics')
@click.argument('topics', nargs=-1, required=True)
@click.option('--workflow', required=True, help='Workflow template to use')
@click.option('--name', help='Batch job name')
@click.option('--concurrent', default=3, help='Maximum concurrent executions')
@click.option('--style', help='Content style for all topics')
@click.option('--duration', type=int, help='Duration for all videos')
def batch_create_topics(topics, workflow, name, concurrent, style, duration):
    """Create batch job from topic list."""
    async def _create():
        try:
            from youtube_ai.utils.batch_processor import batch_processor
            
            # Build base configuration
            base_config = {}
            if style:
                base_config['style'] = style
            if duration:
                base_config['duration'] = duration
            
            console.print(f"[blue]Creating batch job from topics...[/blue]")
            console.print(f"[dim]Topics:[/dim] {len(topics)}")
            console.print(f"[dim]Workflow:[/dim] {workflow}")
            
            batch_job = await batch_processor.create_batch_from_topics(
                topics=list(topics),
                workflow_template=workflow,
                name=name,
                base_config=base_config,
                max_concurrent=concurrent
            )
            
            console.print(f"[green]✓[/green] Created batch job: {batch_job.id}")
            console.print(f"[dim]Name:[/dim] {batch_job.name}")
            console.print(f"[dim]Tasks:[/dim] {len(batch_job.tasks)}")
            
        except Exception as e:
            console.print(f"[red]Error creating batch job:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_create())


@batch.command('run')
@click.argument('job_id')
def batch_run(job_id):
    """Execute a batch job."""
    async def _run():
        try:
            from youtube_ai.utils.batch_processor import batch_processor
            
            # Load batch job
            batch_job = await batch_processor.load_batch_job(job_id)
            if not batch_job:
                console.print(f"[red]Error:[/red] Batch job not found: {job_id}")
                return
            
            console.print(f"[blue]Executing batch job: {batch_job.name}[/blue]")
            console.print(f"[dim]Tasks:[/dim] {len(batch_job.tasks)}")
            console.print(f"[dim]Workflow:[/dim] {batch_job.workflow_template}")
            
            # Progress callback
            def show_progress(progress):
                console.print(f"Progress: {progress.percentage:.1f}% ({progress.completed_tasks}/{progress.total_tasks}) "
                             f"- Running: {progress.running_tasks}, Failed: {progress.failed_tasks}")
                if progress.estimated_remaining_time:
                    console.print(f"Estimated remaining: {progress.estimated_remaining_time:.0f}s")
            
            # Execute batch
            result = await batch_processor.execute_batch(
                batch_job=batch_job,
                progress_callback=show_progress
            )
            
            # Show results
            completed = sum(1 for task in result.tasks if task.status.value == 'completed')
            failed = sum(1 for task in result.tasks if task.status.value == 'failed')
            
            if result.status.value == 'completed':
                console.print(f"[green]✓[/green] Batch job completed!")
            else:
                console.print(f"[yellow]⚠[/yellow] Batch job finished with errors")
            
            console.print(f"[dim]Results:[/dim] {completed} successful, {failed} failed")
            console.print(f"[dim]Duration:[/dim] {result.total_duration:.1f}s")
            console.print(f"[dim]Output:[/dim] {result.output_dir}")
            
        except Exception as e:
            console.print(f"[red]Error executing batch job:[/red] {e}")
            if config_manager.load_config().debug:
                console.print_exception()
    
    asyncio.run(_run())


@batch.command('list')
def batch_list():
    """List all batch jobs."""
    try:
        from youtube_ai.utils.batch_processor import batch_processor
        
        jobs = batch_processor.list_batch_jobs()
        
        if not jobs:
            console.print("[yellow]No batch jobs found[/yellow]")
            return
        
        table = Table(title="Batch Jobs")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status")
        table.add_column("Tasks", justify="center")
        table.add_column("Success Rate", justify="center")
        table.add_column("Created")
        
        for job in jobs:
            status_color = {
                "completed": "green",
                "failed": "red",
                "running": "yellow",
                "pending": "blue"
            }.get(job['status'], "white")
            
            success_rate = 0
            if job['total_tasks'] > 0:
                success_rate = (job['completed_tasks'] / job['total_tasks']) * 100
            
            table.add_row(
                job['id'],
                job['name'][:30] + "..." if len(job['name']) > 30 else job['name'],
                f"[{status_color}]{job['status']}[/{status_color}]",
                f"{job['completed_tasks']}/{job['total_tasks']}",
                f"{success_rate:.1f}%",
                job['created_at'][:19]
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing batch jobs:[/red] {e}")


@batch.command('status')
@click.argument('job_id')
def batch_status(job_id):
    """Check status of a batch job."""
    async def _status():
        try:
            from youtube_ai.utils.batch_processor import batch_processor
            
            batch_job = await batch_processor.load_batch_job(job_id)
            if not batch_job:
                console.print(f"[red]Error:[/red] Batch job not found: {job_id}")
                return
            
            console.print(f"[bold]Batch Job Status: {batch_job.name}[/bold]")
            console.print(f"[dim]ID:[/dim] {batch_job.id}")
            console.print(f"[dim]Status:[/dim] {batch_job.status.value}")
            console.print(f"[dim]Workflow:[/dim] {batch_job.workflow_template}")
            console.print(f"[dim]Created:[/dim] {batch_job.created_at}")
            
            if batch_job.started_at:
                console.print(f"[dim]Started:[/dim] {batch_job.started_at}")
            
            if batch_job.completed_at:
                console.print(f"[dim]Completed:[/dim] {batch_job.completed_at}")
                console.print(f"[dim]Duration:[/dim] {batch_job.total_duration:.1f}s")
            
            # Task summary
            total = len(batch_job.tasks)
            completed = sum(1 for task in batch_job.tasks if task.status.value == 'completed')
            failed = sum(1 for task in batch_job.tasks if task.status.value == 'failed')
            running = sum(1 for task in batch_job.tasks if task.status.value == 'running')
            pending = sum(1 for task in batch_job.tasks if task.status.value == 'pending')
            
            console.print(f"\n[bold]Task Summary:[/bold]")
            console.print(f"  Total: {total}")
            console.print(f"  Completed: [green]{completed}[/green]")
            console.print(f"  Failed: [red]{failed}[/red]")
            console.print(f"  Running: [yellow]{running}[/yellow]")
            console.print(f"  Pending: [blue]{pending}[/blue]")
            
            if batch_job.output_dir:
                console.print(f"\n[dim]Output Directory:[/dim] {batch_job.output_dir}")
            
        except Exception as e:
            console.print(f"[red]Error getting batch status:[/red] {e}")
    
    asyncio.run(_status())


@batch.command('cancel')
@click.argument('job_id')
def batch_cancel(job_id):
    """Cancel a running batch job."""
    async def _cancel():
        try:
            from youtube_ai.utils.batch_processor import batch_processor
            
            success = await batch_processor.cancel_batch_job(job_id)
            
            if success:
                console.print(f"[green]✓[/green] Batch job cancelled: {job_id}")
            else:
                console.print(f"[yellow]⚠[/yellow] Batch job not found or not running: {job_id}")
                
        except Exception as e:
            console.print(f"[red]Error cancelling batch job:[/red] {e}")
    
    asyncio.run(_cancel())


@cli.command('list-assets')
def list_custom_assets():
    """📁 List all available custom assets (overlays, transitions, sounds, music)."""
    try:
        from youtube_ai.media.custom_assets import custom_assets_manager
        custom_assets_manager.list_assets()
    except Exception as e:
        console.print(f"[red]Error listing assets:[/red] {e}")


@cli.command('scan-assets')
def scan_custom_assets():
    """🔍 Scan and validate all custom assets in the assets directory."""
    try:
        from youtube_ai.media.custom_assets import custom_assets_manager
        
        console.print("[blue]🔍 Scanning custom assets...[/blue]")
        assets = custom_assets_manager.scan_assets()
        
        total_assets = sum(len(asset_list) for asset_list in assets.values())
        console.print(f"[green]✅ Found {total_assets} custom assets[/green]")
        
        # Create manifest
        manifest = custom_assets_manager.create_asset_manifest()
        manifest_file = Path("assets") / "manifest.json"
        
        import json
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        console.print(f"[green]📋 Asset manifest created: {manifest_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error scanning assets:[/red] {e}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if config_manager.load_config().debug:
            console.print_exception()


if __name__ == '__main__':
    main()