"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import os
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_script_content():
    """Sample script content for testing."""
    return """
    # Complete Guide to Python Programming

    Welcome to this comprehensive guide about Python programming. Today we'll explore the fundamentals of Python and how you can start building amazing applications.

    ## What is Python?

    Python is a high-level, interpreted programming language known for its simplicity and readability. It's perfect for beginners and powerful enough for experts.

    ## Getting Started

    To begin your Python journey, you'll need to install Python on your computer. Visit python.org and download the latest version.

    ## Basic Syntax

    Python uses indentation to define code blocks. This makes the code clean and easy to read. Let's look at a simple example:

    ```python
    print("Hello, World!")
    ```

    ## Conclusion

    Python is an excellent choice for your first programming language. Its simplicity and versatility make it perfect for web development, data science, and automation.

    Thanks for watching! Don't forget to like and subscribe for more programming tutorials.
    """


@pytest.fixture
def sample_video_metadata():
    """Sample video metadata for testing."""
    return {
        "title": "Complete Python Programming Guide",
        "description": "Learn Python programming from scratch with this comprehensive tutorial.",
        "tags": ["python", "programming", "tutorial", "beginner", "coding"],
        "category": "education",
        "privacy": "private"
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    from youtube_ai.core.config import AppConfig, YouTubeConfig, AIConfig, VideoConfig, AudioConfig, ContentConfig
    
    return AppConfig(
        youtube=YouTubeConfig(
            api_key="test_youtube_key",
            channel_id="test_channel_id",
            default_privacy="private"
        ),
        ai=AIConfig(
            openai_api_key="test_openai_key",
            anthropic_api_key="test_anthropic_key",
            elevenlabs_api_key="test_elevenlabs_key",
            default_llm="openai",
            default_tts="openai"
        ),
        video=VideoConfig(
            resolution="1080p",
            fps=30,
            format="mp4",
            quality="high"
        ),
        audio=AudioConfig(
            voice="alloy",
            speed=1.0,
            background_music=False
        ),
        content=ContentConfig(
            language="en",
            target_audience="general",
            default_style="educational"
        ),
        debug=True,
        output_dir="./test_output"
    )


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for testing."""
    from youtube_ai.ai.ai_manager import AIResponse
    
    manager = Mock()
    manager.generate_completion = AsyncMock()
    manager.generate_stream = AsyncMock()
    manager.list_available_providers = Mock(return_value=["openai", "anthropic"])
    
    # Default response
    manager.generate_completion.return_value = AIResponse(
        content="This is a sample AI response for testing purposes.",
        provider="test",
        model="test-model"
    )
    
    return manager


@pytest.fixture
def mock_tts_manager():
    """Mock TTS manager for testing."""
    from youtube_ai.ai.tts_client import TTSResponse
    
    manager = Mock()
    manager.synthesize_speech = AsyncMock()
    manager.get_available_voices = AsyncMock()
    manager.list_available_providers = Mock(return_value=["openai", "elevenlabs"])
    
    # Default response
    manager.synthesize_speech.return_value = TTSResponse(
        audio_data=b"fake_audio_data",
        provider="test",
        voice="test_voice",
        format="mp3",
        sample_rate=22050,
        duration=30.0
    )
    
    return manager


@pytest.fixture
def mock_youtube_uploader():
    """Mock YouTube uploader for testing."""
    from youtube_ai.utils.youtube_uploader import UploadResult
    
    uploader = Mock()
    uploader.upload_video = AsyncMock()
    uploader.get_video_status = AsyncMock()
    uploader.update_video_metadata = AsyncMock()
    uploader.delete_video = AsyncMock()
    uploader.validate_metadata = Mock(return_value=[])
    
    # Default upload result
    uploader.upload_video.return_value = UploadResult(
        video_id="test_video_id",
        video_url="https://youtube.com/watch?v=test_video_id",
        status="uploaded",
        upload_status="processed",
        privacy_status="private"
    )
    
    return uploader


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    audio_file = temp_dir / "test_audio.mp3"
    # Create a minimal MP3-like file (not actual audio, just for testing)
    audio_file.write_bytes(b"fake_mp3_data" * 100)
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing."""
    video_file = temp_dir / "test_video.mp4"
    # Create a minimal MP4-like file (not actual video, just for testing)
    video_file.write_bytes(b"fake_mp4_data" * 1000)
    return video_file


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Set up test environment variables."""
    # Set up test environment variables
    monkeypatch.setenv("YOUTUBE_AI_TEST_MODE", "true")
    monkeypatch.setenv("YOUTUBE_AI_OUTPUT_DIR", str(temp_dir))
    
    # Mock API keys for testing
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("YOUTUBE_API_KEY", "test_youtube_key")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.nodeid or "upload" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark tests requiring API access
        if any(keyword in item.nodeid for keyword in ["api", "upload", "generate"]):
            item.add_marker(pytest.mark.requires_api)