"""
Custom Assets Manager for YouTube AI CLI
Handles custom overlays, transitions, sound effects, and background music
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

logger = get_logger(__name__)


class AssetType(Enum):
    OVERLAY = "overlay"
    TRANSITION = "transition"
    SOUND_EFFECT = "sound_effect"
    BACKGROUND_MUSIC = "background_music"
    FONT = "font"


@dataclass
class CustomAsset:
    """Represents a custom asset."""
    name: str
    file_path: Path
    asset_type: AssetType
    metadata: Dict[str, Any]
    
    @property
    def exists(self) -> bool:
        return self.file_path.exists()
    
    @property
    def size_mb(self) -> float:
        if self.exists:
            return self.file_path.stat().st_size / (1024 * 1024)
        return 0.0


@dataclass 
class AssetPack:
    """A collection of themed assets."""
    name: str
    style: str
    overlays: List[CustomAsset]
    transitions: List[CustomAsset] 
    sound_effects: List[CustomAsset]
    background_music: List[CustomAsset]
    fonts: List[CustomAsset]
    metadata: Dict[str, Any]


class CustomAssetsManager:
    """Manages custom assets for video generation."""
    
    def __init__(self):
        self.config = config_manager.load_config()
        self.assets_dir = Path("assets")
        self.assets_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.overlays_dir = self.assets_dir / "overlays"
        self.transitions_dir = self.assets_dir / "transitions"
        self.sound_effects_dir = self.assets_dir / "sound_effects"
        self.background_music_dir = self.assets_dir / "background_music"
        self.fonts_dir = self.assets_dir / "fonts"
        
        for dir_path in [self.overlays_dir, self.transitions_dir, 
                        self.sound_effects_dir, self.background_music_dir, self.fonts_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self._asset_cache = {}
        self._supported_formats = {
            AssetType.OVERLAY: {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov', '.webm'},
            AssetType.TRANSITION: {'.mp4', '.mov', '.webm', '.gif'},
            AssetType.SOUND_EFFECT: {'.wav', '.mp3', '.aac', '.ogg'},
            AssetType.BACKGROUND_MUSIC: {'.mp3', '.wav', '.aac', '.ogg'},
            AssetType.FONT: {'.ttf', '.otf', '.woff', '.woff2'}
        }
    
    def scan_assets(self) -> Dict[AssetType, List[CustomAsset]]:
        """Scan and catalog all available custom assets."""
        logger.info("Scanning custom assets...")
        
        assets = {asset_type: [] for asset_type in AssetType}
        
        # Scan each directory
        directory_mapping = {
            AssetType.OVERLAY: self.overlays_dir,
            AssetType.TRANSITION: self.transitions_dir,
            AssetType.SOUND_EFFECT: self.sound_effects_dir,
            AssetType.BACKGROUND_MUSIC: self.background_music_dir,
            AssetType.FONT: self.fonts_dir
        }
        
        for asset_type, directory in directory_mapping.items():
            if directory.exists():
                for file_path in directory.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self._supported_formats[asset_type]:
                        metadata = self._extract_metadata(file_path, asset_type)
                        asset = CustomAsset(
                            name=file_path.stem,
                            file_path=file_path,
                            asset_type=asset_type,
                            metadata=metadata
                        )
                        assets[asset_type].append(asset)
        
        # Cache the results
        self._asset_cache = assets
        
        # Log findings
        for asset_type, asset_list in assets.items():
            if asset_list:
                logger.info(f"Found {len(asset_list)} {asset_type.value}(s)")
        
        return assets
    
    def _extract_metadata(self, file_path: Path, asset_type: AssetType) -> Dict[str, Any]:
        """Extract metadata from asset file."""
        metadata = {
            "file_size": file_path.stat().st_size,
            "format": file_path.suffix.lower(),
            "created": file_path.stat().st_ctime
        }
        
        # Add type-specific metadata
        if asset_type in [AssetType.OVERLAY, AssetType.TRANSITION]:
            metadata.update(self._get_video_metadata(file_path))
        elif asset_type in [AssetType.SOUND_EFFECT, AssetType.BACKGROUND_MUSIC]:
            metadata.update(self._get_audio_metadata(file_path))
        
        return metadata
    
    def _get_video_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get video/image metadata."""
        metadata = {}
        
        try:
            if file_path.suffix.lower() in ['.mp4', '.mov', '.webm']:
                # Try to get video metadata with moviepy
                try:
                    from moviepy import VideoFileClip
                    with VideoFileClip(str(file_path)) as clip:
                        metadata.update({
                            "duration": clip.duration,
                            "fps": clip.fps,
                            "resolution": (clip.w, clip.h)
                        })
                except ImportError:
                    pass
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                # Get image metadata with PIL
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        metadata.update({
                            "resolution": img.size,
                            "mode": img.mode,
                            "has_transparency": img.mode in ['RGBA', 'LA'] or 'transparency' in img.info
                        })
                except ImportError:
                    pass
        except Exception as e:
            logger.debug(f"Could not extract metadata from {file_path}: {e}")
        
        return metadata
    
    def _get_audio_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get audio metadata."""
        metadata = {}
        
        try:
            # Try to get audio metadata with moviepy
            try:
                from moviepy import AudioFileClip
                with AudioFileClip(str(file_path)) as clip:
                    metadata.update({
                        "duration": clip.duration,
                        "fps": getattr(clip, 'fps', None)
                    })
            except ImportError:
                pass
        except Exception as e:
            logger.debug(f"Could not extract audio metadata from {file_path}: {e}")
        
        return metadata
    
    def get_assets_by_type(self, asset_type: AssetType) -> List[CustomAsset]:
        """Get all assets of a specific type."""
        if not self._asset_cache:
            self.scan_assets()
        return self._asset_cache.get(asset_type, [])
    
    def get_asset_by_name(self, name: str, asset_type: AssetType) -> Optional[CustomAsset]:
        """Get a specific asset by name and type."""
        assets = self.get_assets_by_type(asset_type)
        for asset in assets:
            if asset.name == name or asset.file_path.name == name:
                return asset
        return None
    
    def get_random_asset(self, asset_type: AssetType, style_filter: Optional[str] = None) -> Optional[CustomAsset]:
        """Get a random asset of the specified type, optionally filtered by style."""
        assets = self.get_assets_by_type(asset_type)
        
        # Filter by style if specified
        if style_filter:
            style_keywords = {
                'educational': ['clean', 'minimal', 'simple', 'professional'],
                'professional': ['corporate', 'business', 'formal', 'clean'],
                'tech': ['tech', 'digital', 'glitch', 'electronic', 'modern'],
                'creative': ['artistic', 'creative', 'colorful', 'unique']
            }
            
            keywords = style_keywords.get(style_filter.lower(), [])
            if keywords:
                filtered_assets = []
                for asset in assets:
                    asset_name_lower = asset.name.lower()
                    if any(keyword in asset_name_lower for keyword in keywords):
                        filtered_assets.append(asset)
                if filtered_assets:
                    assets = filtered_assets
        
        return random.choice(assets) if assets else None
    
    def get_asset_pack(self, style: str) -> AssetPack:
        """Get a curated pack of assets for a specific style."""
        overlays = self.get_assets_by_type(AssetType.OVERLAY)
        transitions = self.get_assets_by_type(AssetType.TRANSITION)
        sound_effects = self.get_assets_by_type(AssetType.SOUND_EFFECT)
        background_music = self.get_assets_by_type(AssetType.BACKGROUND_MUSIC)
        fonts = self.get_assets_by_type(AssetType.FONT)
        
        # Filter assets by style
        def filter_by_style(assets: List[CustomAsset]) -> List[CustomAsset]:
            style_keywords = {
                'educational': ['clean', 'minimal', 'simple', 'soft'],
                'professional': ['corporate', 'business', 'formal', 'clean'],
                'tech': ['tech', 'digital', 'glitch', 'electronic', 'modern'],
                'creative': ['artistic', 'creative', 'colorful', 'unique']
            }
            
            keywords = style_keywords.get(style.lower(), [])
            if not keywords:
                return assets
            
            filtered = []
            for asset in assets:
                asset_name_lower = asset.name.lower()
                if any(keyword in asset_name_lower for keyword in keywords):
                    filtered.append(asset)
            
            # If no matches, return all assets
            return filtered if filtered else assets
        
        return AssetPack(
            name=f"{style.title()} Pack",
            style=style,
            overlays=filter_by_style(overlays),
            transitions=filter_by_style(transitions),
            sound_effects=filter_by_style(sound_effects),
            background_music=filter_by_style(background_music),
            fonts=filter_by_style(fonts),
            metadata={
                "total_assets": len(overlays) + len(transitions) + len(sound_effects) + len(background_music) + len(fonts),
                "style": style
            }
        )
    
    def validate_asset(self, asset: CustomAsset) -> bool:
        """Validate that an asset is usable."""
        if not asset.exists:
            logger.warning(f"Asset file not found: {asset.file_path}")
            return False
        
        # Check file format
        if asset.file_path.suffix.lower() not in self._supported_formats[asset.asset_type]:
            logger.warning(f"Unsupported format for {asset.asset_type.value}: {asset.file_path.suffix}")
            return False
        
        # Check file size (warn if too large)
        if asset.size_mb > 100:
            logger.warning(f"Large asset file ({asset.size_mb:.1f}MB): {asset.file_path.name}")
        
        return True
    
    def create_asset_manifest(self) -> Dict[str, Any]:
        """Create a manifest of all available assets."""
        assets = self.scan_assets()
        
        manifest = {
            "version": "1.0",
            "generated": str(Path.cwd()),
            "assets": {}
        }
        
        for asset_type, asset_list in assets.items():
            manifest["assets"][asset_type.value] = []
            for asset in asset_list:
                manifest["assets"][asset_type.value].append({
                    "name": asset.name,
                    "file": str(asset.file_path.relative_to(Path.cwd())),
                    "size_mb": asset.size_mb,
                    "metadata": asset.metadata
                })
        
        return manifest
    
    def list_assets(self) -> None:
        """Print a formatted list of all available assets."""
        assets = self.scan_assets()
        
        print("\n🎨 Custom Assets Available:")
        print("=" * 50)
        
        for asset_type, asset_list in assets.items():
            if asset_list:
                print(f"\n📁 {asset_type.value.replace('_', ' ').title()} ({len(asset_list)})")
                for asset in asset_list:
                    size_str = f"({asset.size_mb:.1f}MB)" if asset.size_mb > 0 else ""
                    print(f"  • {asset.name} {size_str}")
        
        if not any(assets.values()):
            print("\n📂 No custom assets found.")
            print("   Add assets to the 'assets/' directory to get started!")


# Global instance
custom_assets_manager = CustomAssetsManager()