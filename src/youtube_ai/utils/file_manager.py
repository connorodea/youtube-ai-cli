"""File management utilities for YouTube AI CLI."""

import os
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import json
import yaml

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FileInfo:
    """Information about a file."""
    path: Path
    size: int
    created: datetime
    modified: datetime
    file_type: str
    mime_type: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class ProjectStructure:
    """Structure of a video project."""
    project_id: str
    base_dir: Path
    script_file: Optional[Path] = None
    audio_file: Optional[Path] = None
    video_file: Optional[Path] = None
    thumbnail_files: List[Path] = None
    metadata_file: Optional[Path] = None
    
    def __post_init__(self):
        if self.thumbnail_files is None:
            self.thumbnail_files = []


class FileManager:
    """Manages files and directories for YouTube AI CLI operations."""
    
    def __init__(self, base_output_dir: Optional[Path] = None):
        self.base_output_dir = base_output_dir or Path("./output")
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_ai"
        self.archive_dir = self.base_output_dir / "archive"
        
        # Ensure directories exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project_structure(self, project_name: str, timestamp: bool = True) -> ProjectStructure:
        """Create a project directory structure."""
        # Sanitize project name
        safe_name = self.sanitize_filename(project_name)
        
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_id = f"{safe_name}_{timestamp_str}"
        else:
            project_id = safe_name
        
        project_dir = self.base_output_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_dir / "scripts").mkdir(exist_ok=True)
        (project_dir / "audio").mkdir(exist_ok=True)
        (project_dir / "video").mkdir(exist_ok=True)
        (project_dir / "thumbnails").mkdir(exist_ok=True)
        (project_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Created project structure: {project_dir}")
        
        return ProjectStructure(
            project_id=project_id,
            base_dir=project_dir
        )
    
    def get_project_files(self, project_dir: Path) -> ProjectStructure:
        """Analyze a project directory and return its structure."""
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        
        structure = ProjectStructure(
            project_id=project_dir.name,
            base_dir=project_dir
        )
        
        # Find script files
        script_patterns = ["*.txt", "script.*", "*script*"]
        for pattern in script_patterns:
            for script_file in project_dir.glob(f"**/{pattern}"):
                if script_file.is_file() and "script" in script_file.name.lower():
                    structure.script_file = script_file
                    break
            if structure.script_file:
                break
        
        # Find audio files
        audio_patterns = ["*.mp3", "*.wav", "*.m4a", "*.aac"]
        for pattern in audio_patterns:
            for audio_file in project_dir.glob(f"**/{pattern}"):
                if audio_file.is_file():
                    structure.audio_file = audio_file
                    break
            if structure.audio_file:
                break
        
        # Find video files
        video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        for pattern in video_patterns:
            for video_file in project_dir.glob(f"**/{pattern}"):
                if video_file.is_file():
                    structure.video_file = video_file
                    break
            if structure.video_file:
                break
        
        # Find thumbnail files
        thumbnail_patterns = ["*.png", "*.jpg", "*.jpeg"]
        for pattern in thumbnail_patterns:
            for thumb_file in (project_dir / "thumbnails").glob(pattern):
                if thumb_file.is_file():
                    structure.thumbnail_files.append(thumb_file)
        
        # Find metadata file
        for metadata_file in project_dir.glob("**/metadata.*"):
            if metadata_file.is_file():
                structure.metadata_file = metadata_file
                break
        
        return structure
    
    def save_project_metadata(self, structure: ProjectStructure, metadata: Dict[str, Any]) -> Path:
        """Save project metadata to a JSON file."""
        metadata_file = structure.base_dir / "metadata" / "project_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file paths to metadata
        metadata.update({
            "project_id": structure.project_id,
            "created_at": datetime.now().isoformat(),
            "files": {
                "script": str(structure.script_file) if structure.script_file else None,
                "audio": str(structure.audio_file) if structure.audio_file else None,
                "video": str(structure.video_file) if structure.video_file else None,
                "thumbnails": [str(f) for f in structure.thumbnail_files],
                "base_dir": str(structure.base_dir)
            }
        })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        structure.metadata_file = metadata_file
        logger.info(f"Saved project metadata: {metadata_file}")
        return metadata_file
    
    def load_project_metadata(self, project_dir: Path) -> Optional[Dict[str, Any]]:
        """Load project metadata from JSON file."""
        metadata_file = project_dir / "metadata" / "project_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading project metadata: {e}")
            return None
    
    def get_file_info(self, file_path: Path) -> FileInfo:
        """Get detailed information about a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return FileInfo(
            path=file_path,
            size=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            file_type=file_path.suffix.lower(),
            checksum=self.calculate_checksum(file_path)
        )
    
    def calculate_checksum(self, file_path: Path, algorithm: str = "md5") -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def organize_files(self, source_dir: Path, target_structure: ProjectStructure) -> None:
        """Organize loose files into a proper project structure."""
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Define file type mappings
        file_mappings = {
            'scripts': ['.txt', '.md', '.script'],
            'audio': ['.mp3', '.wav', '.m4a', '.aac', '.flac'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            'thumbnails': ['.png', '.jpg', '.jpeg', '.webp'],
            'metadata': ['.json', '.yml', '.yaml', '.xml']
        }
        
        for file_path in source_dir.rglob('*'):
            if not file_path.is_file():
                continue
            
            file_ext = file_path.suffix.lower()
            moved = False
            
            # Determine target directory based on file extension
            for category, extensions in file_mappings.items():
                if file_ext in extensions:
                    target_dir = target_structure.base_dir / category
                    target_dir.mkdir(exist_ok=True)
                    
                    # Generate unique filename if needed
                    target_path = target_dir / file_path.name
                    counter = 1
                    while target_path.exists():
                        stem = file_path.stem
                        target_path = target_dir / f"{stem}_{counter}{file_ext}"
                        counter += 1
                    
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Moved {file_path} to {target_path}")
                    moved = True
                    break
            
            if not moved:
                # Unknown file type, move to misc folder
                misc_dir = target_structure.base_dir / "misc"
                misc_dir.mkdir(exist_ok=True)
                target_path = misc_dir / file_path.name
                shutil.copy2(file_path, target_path)
                logger.info(f"Moved unknown file {file_path} to {target_path}")
    
    def archive_project(self, project_dir: Path, archive_name: Optional[str] = None) -> Path:
        """Archive a completed project."""
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        
        if not archive_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{project_dir.name}_{timestamp}"
        
        archive_path = self.archive_dir / archive_name
        
        # Create archive using shutil
        shutil.make_archive(
            base_name=str(archive_path),
            format='zip',
            root_dir=project_dir.parent,
            base_dir=project_dir.name
        )
        
        archive_file = Path(f"{archive_path}.zip")
        logger.info(f"Archived project to: {archive_file}")
        return archive_file
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified age."""
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        for temp_file in self.temp_dir.rglob('*'):
            if temp_file.is_file():
                try:
                    if temp_file.stat().st_mtime < cutoff_time:
                        temp_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} temporary files")
        return deleted_count
    
    def sanitize_filename(self, filename: str, max_length: int = 255) -> str:
        """Sanitize filename for cross-platform compatibility."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        sanitized = filename
        
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "untitled"
        
        return sanitized
    
    def get_disk_usage(self, directory: Path) -> Dict[str, int]:
        """Get disk usage statistics for a directory."""
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
            elif item.is_dir():
                dir_count += 1
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "file_count": file_count,
            "directory_count": dir_count
        }
    
    def find_duplicates(self, directory: Path) -> Dict[str, List[Path]]:
        """Find duplicate files based on checksum."""
        checksums = {}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                checksum = self.calculate_checksum(file_path)
                if checksum:
                    if checksum not in checksums:
                        checksums[checksum] = []
                    checksums[checksum].append(file_path)
        
        # Return only duplicates (groups with more than one file)
        duplicates = {k: v for k, v in checksums.items() if len(v) > 1}
        return duplicates
    
    def create_temp_file(self, suffix: str = "", prefix: str = "youtube_ai_") -> Path:
        """Create a temporary file and return its path."""
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix=prefix,
            dir=self.temp_dir
        )
        temp_file.close()
        return Path(temp_file.name)
    
    def safe_delete(self, file_path: Path, backup: bool = False) -> bool:
        """Safely delete a file with optional backup."""
        if not file_path.exists():
            return False
        
        try:
            if backup:
                backup_dir = file_path.parent / "backup"
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"{file_path.name}.backup"
                shutil.copy2(file_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the output directory."""
        projects = []
        
        for project_dir in self.base_output_dir.iterdir():
            if project_dir.is_dir() and project_dir.name != "archive":
                metadata = self.load_project_metadata(project_dir)
                structure = self.get_project_files(project_dir)
                
                project_info = {
                    "id": structure.project_id,
                    "path": str(structure.base_dir),
                    "created": metadata.get("created_at") if metadata else None,
                    "has_script": structure.script_file is not None,
                    "has_audio": structure.audio_file is not None,
                    "has_video": structure.video_file is not None,
                    "thumbnail_count": len(structure.thumbnail_files),
                    "metadata": metadata
                }
                
                projects.append(project_info)
        
        return sorted(projects, key=lambda x: x["created"] or "", reverse=True)


# Global file manager instance
file_manager = FileManager()