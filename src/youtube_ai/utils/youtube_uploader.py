import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger

logger = get_logger(__name__)


class PrivacyStatus(Enum):
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"


class VideoCategory(Enum):
    AUTOS_VEHICLES = "2"
    COMEDY = "23"
    EDUCATION = "27"
    ENTERTAINMENT = "24"
    FILM_ANIMATION = "1"
    GAMING = "20"
    HOWTO_STYLE = "26"
    MUSIC = "10"
    NEWS_POLITICS = "25"
    NONPROFITS_ACTIVISM = "29"
    PEOPLE_BLOGS = "22"
    PETS_ANIMALS = "15"
    SCIENCE_TECHNOLOGY = "28"
    SPORTS = "17"
    TRAVEL_EVENTS = "19"


@dataclass
class VideoMetadata:
    """Video metadata for YouTube upload."""
    title: str
    description: str
    tags: List[str]
    category_id: str = VideoCategory.EDUCATION.value
    privacy_status: str = PrivacyStatus.PRIVATE.value
    made_for_kids: bool = False
    language: str = "en"
    default_language: str = "en"
    thumbnail_file: Optional[Path] = None
    publish_at: Optional[datetime] = None


@dataclass
class UploadProgress:
    """Upload progress information."""
    bytes_uploaded: int
    total_bytes: int
    percentage: float
    status: str
    resumable_uri: Optional[str] = None


@dataclass
class UploadResult:
    """Upload result information."""
    video_id: str
    video_url: str
    status: str
    upload_status: str
    privacy_status: str
    published_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class YouTubeUploader:
    """Handles YouTube video uploads and management."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/youtube'
    ]
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'
    
    def __init__(self):
        if not GOOGLE_APIS_AVAILABLE:
            raise ImportError("Google API client not installed. Install with: pip install google-api-python-client google-auth-oauthlib")
        
        self.config = config_manager.load_config()
        self.youtube = None
        self.credentials = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize YouTube API client."""
        try:
            self.credentials = self._get_credentials()
            if self.credentials:
                self.youtube = build(
                    self.API_SERVICE_NAME,
                    self.API_VERSION,
                    credentials=self.credentials
                )
                logger.info("YouTube API client initialized successfully")
            else:
                logger.warning("No valid YouTube credentials found")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube client: {e}")
    
    def _get_credentials(self) -> Optional[Credentials]:
        """Get YouTube API credentials."""
        creds = None
        token_file = Path.home() / ".youtube-ai" / "token.json"
        
        # Load existing credentials
        if token_file.exists():
            creds = Credentials.from_authorized_user_file(str(token_file), self.SCOPES)
        
        # If no valid credentials, authorize user
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    creds = None
            
            if not creds:
                if not self.config.youtube.client_secrets_file:
                    logger.error("No client secrets file configured. Please set youtube.client_secrets_file in config")
                    return None
                
                secrets_file = Path(self.config.youtube.client_secrets_file)
                if not secrets_file.exists():
                    logger.error(f"Client secrets file not found: {secrets_file}")
                    return None
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(secrets_file), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            token_file.parent.mkdir(exist_ok=True)
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        return creds
    
    async def upload_video(
        self,
        video_file: Path,
        metadata: VideoMetadata,
        progress_callback: Optional[callable] = None
    ) -> UploadResult:
        """Upload a video to YouTube."""
        if not self.youtube:
            raise ValueError("YouTube client not initialized. Check your credentials.")
        
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        try:
            logger.info(f"Starting upload: {video_file.name}")
            
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': metadata.title,
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'categoryId': metadata.category_id,
                    'defaultLanguage': metadata.default_language,
                    'defaultAudioLanguage': metadata.language
                },
                'status': {
                    'privacyStatus': metadata.privacy_status,
                    'madeForKids': metadata.made_for_kids,
                    'selfDeclaredMadeForKids': metadata.made_for_kids
                }
            }
            
            # Add scheduled publish time if specified
            if metadata.publish_at:
                body['status']['publishAt'] = metadata.publish_at.isoformat()
            
            # Prepare media upload
            media = MediaFileUpload(
                str(video_file),
                chunksize=-1,  # Upload in a single chunk
                resumable=True
            )
            
            # Execute upload
            insert_request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Upload with progress tracking
            response = await self._execute_upload_with_progress(
                insert_request, progress_callback
            )
            
            video_id = response['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            logger.info(f"Upload completed. Video ID: {video_id}")
            
            # Upload thumbnail if provided
            if metadata.thumbnail_file and metadata.thumbnail_file.exists():
                await self._upload_thumbnail(video_id, metadata.thumbnail_file)
            
            return UploadResult(
                video_id=video_id,
                video_url=video_url,
                status=response.get('status', {}).get('uploadStatus', 'unknown'),
                upload_status=response.get('status', {}).get('uploadStatus', 'unknown'),
                privacy_status=response.get('status', {}).get('privacyStatus', 'unknown'),
                published_at=self._parse_datetime(response.get('snippet', {}).get('publishedAt')),
                metadata=response
            )
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise
    
    async def _execute_upload_with_progress(
        self,
        insert_request,
        progress_callback: Optional[callable] = None
    ):
        """Execute upload with progress tracking."""
        response = None
        error = None
        retry = 0
        
        while response is None:
            try:
                status, response = insert_request.next_chunk()
                
                if status and progress_callback:
                    progress = UploadProgress(
                        bytes_uploaded=status.resumable_progress,
                        total_bytes=status.total_size,
                        percentage=(status.resumable_progress / status.total_size) * 100,
                        status="uploading"
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, progress_callback, progress
                    )
                    
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retriable error
                    error = f"Retriable error: {e}"
                    retry += 1
                    if retry > 5:
                        raise Exception("Maximum retries exceeded")
                    
                    wait_time = 2 ** retry
                    logger.warning(f"Upload error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        return response
    
    async def _upload_thumbnail(self, video_id: str, thumbnail_file: Path):
        """Upload thumbnail for a video."""
        try:
            logger.info(f"Uploading thumbnail: {thumbnail_file.name}")
            
            media = MediaFileUpload(str(thumbnail_file))
            
            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            
            logger.info("Thumbnail uploaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to upload thumbnail: {e}")
    
    async def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get the status of an uploaded video."""
        try:
            response = self.youtube.videos().list(
                part='status,snippet,statistics',
                id=video_id
            ).execute()
            
            if response['items']:
                return response['items'][0]
            else:
                raise ValueError(f"Video not found: {video_id}")
                
        except Exception as e:
            logger.error(f"Error getting video status: {e}")
            raise
    
    async def update_video_metadata(
        self,
        video_id: str,
        metadata: VideoMetadata
    ) -> Dict[str, Any]:
        """Update metadata for an existing video."""
        try:
            logger.info(f"Updating metadata for video: {video_id}")
            
            body = {
                'id': video_id,
                'snippet': {
                    'title': metadata.title,
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'categoryId': metadata.category_id
                },
                'status': {
                    'privacyStatus': metadata.privacy_status
                }
            }
            
            response = self.youtube.videos().update(
                part='snippet,status',
                body=body
            ).execute()
            
            logger.info("Video metadata updated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error updating video metadata: {e}")
            raise
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete a video from YouTube."""
        try:
            logger.info(f"Deleting video: {video_id}")
            
            self.youtube.videos().delete(id=video_id).execute()
            
            logger.info("Video deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting video: {e}")
            return False
    
    async def list_channel_videos(
        self,
        max_results: int = 50,
        order: str = "date"
    ) -> List[Dict[str, Any]]:
        """List videos from the authenticated channel."""
        try:
            # Get channel ID
            channels_response = self.youtube.channels().list(
                part='contentDetails',
                mine=True
            ).execute()
            
            if not channels_response['items']:
                raise ValueError("No channel found for authenticated user")
            
            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from uploads playlist
            playlist_response = self.youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=max_results
            ).execute()
            
            videos = []
            for item in playlist_response['items']:
                video_info = {
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                }
                videos.append(video_info)
            
            return videos
            
        except Exception as e:
            logger.error(f"Error listing channel videos: {e}")
            raise
    
    def _parse_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not datetime_str:
            return None
        
        try:
            # Remove 'Z' and parse
            dt_str = datetime_str.replace('Z', '+00:00')
            return datetime.fromisoformat(dt_str)
        except Exception:
            return None
    
    def validate_metadata(self, metadata: VideoMetadata) -> List[str]:
        """Validate video metadata and return any issues."""
        issues = []
        
        # Title validation
        if not metadata.title:
            issues.append("Title is required")
        elif len(metadata.title) > 100:
            issues.append("Title must be 100 characters or less")
        
        # Description validation
        if len(metadata.description) > 5000:
            issues.append("Description must be 5000 characters or less")
        
        # Tags validation
        if len(metadata.tags) > 500:
            issues.append("Too many tags (maximum 500)")
        
        # Check individual tag length
        for tag in metadata.tags:
            if len(tag) > 30:
                issues.append(f"Tag too long: '{tag}' (maximum 30 characters)")
        
        return issues


# Global YouTube uploader instance
youtube_uploader = YouTubeUploader() if GOOGLE_APIS_AVAILABLE else None