"""Tests for the script generator module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import os

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from youtube_ai.content.script_generator import ScriptGenerator, VideoScript, ScriptSection
from youtube_ai.ai.ai_manager import AIResponse


class TestScriptGenerator:
    """Test cases for ScriptGenerator."""
    
    @pytest.fixture
    def script_generator(self):
        """Create a ScriptGenerator instance for testing."""
        return ScriptGenerator()
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing."""
        return AIResponse(
            content="""
            Welcome to this comprehensive guide about artificial intelligence and its impact on our daily lives.
            
            Artificial intelligence has revolutionized how we interact with technology. From smart assistants to recommendation systems, AI is everywhere around us.
            
            In this video, we'll explore the key concepts that make AI so powerful and discuss what the future holds for this exciting technology.
            
            Thanks for watching! Don't forget to subscribe for more content about technology and innovation.
            """.strip(),
            provider="mock",
            model="mock-model"
        )
    
    @pytest.mark.asyncio
    async def test_generate_script_basic(self, script_generator, mock_llm_response):
        """Test basic script generation."""
        with patch('youtube_ai.content.script_generator.llm_manager') as mock_llm:
            mock_llm.generate_completion = AsyncMock(return_value=mock_llm_response)
            
            script = await script_generator.generate_script(
                topic="Artificial Intelligence Basics",
                style="educational",
                duration=180,
                audience="beginners"
            )
            
            assert isinstance(script, str)
            assert "artificial intelligence" in script.lower()
            assert len(script) > 100  # Should be substantial content
    
    @pytest.mark.asyncio
    async def test_generate_script_with_parameters(self, script_generator, mock_llm_response):
        """Test script generation with specific parameters."""
        with patch('youtube_ai.content.script_generator.llm_manager') as mock_llm:
            mock_llm.generate_completion = AsyncMock(return_value=mock_llm_response)
            
            script = await script_generator.generate_script(
                topic="Machine Learning for Beginners",
                style="tutorial",
                duration=300,
                audience="students",
                include_intro=True,
                include_outro=True,
                provider="openai"
            )
            
            assert isinstance(script, str)
            assert "machine learning" in script.lower() or "artificial intelligence" in script.lower()
            
            # Verify LLM was called with correct parameters
            mock_llm.generate_completion.assert_called_once()
            call_args = mock_llm.generate_completion.call_args
            assert call_args[1]['provider'] == "openai"
    
    def test_parse_script(self, script_generator):
        """Test script parsing functionality."""
        script_text = """
        Welcome to this comprehensive guide about artificial intelligence.
        
        Artificial intelligence has revolutionized how we interact with technology.
        From smart assistants to recommendation systems, AI is everywhere.
        
        Thanks for watching! Don't forget to subscribe for more content.
        """
        
        video_script = script_generator._parse_script(
            script_text=script_text,
            topic="AI Basics",
            style="educational",
            audience="general",
            duration=180
        )
        
        assert isinstance(video_script, VideoScript)
        assert video_script.topic == "AI Basics"
        assert video_script.style == "educational"
        assert video_script.total_duration == 180
        assert len(video_script.sections) > 0
        assert video_script.word_count > 0
    
    def test_estimate_duration(self, script_generator):
        """Test duration estimation."""
        text = "This is a test sentence. " * 150  # Approximately 150 words
        duration = script_generator._estimate_duration(text)
        
        # Should be around 60 seconds (150 words at 150 WPM)
        assert 50 < duration < 70
    
    def test_generate_title(self, script_generator):
        """Test title generation."""
        title = script_generator._generate_title("Machine Learning", "educational")
        assert "Machine Learning" in title
        assert len(title) > 0
    
    @pytest.mark.asyncio
    async def test_improve_script(self, script_generator):
        """Test script improvement functionality."""
        original_script = "This is a basic script about AI."
        feedback = "Add more technical details and examples."
        
        improved_response = AIResponse(
            content="This is an improved script about AI with detailed technical information and practical examples.",
            provider="mock",
            model="mock-model"
        )
        
        with patch('youtube_ai.content.script_generator.llm_manager') as mock_llm:
            mock_llm.generate_completion = AsyncMock(return_value=improved_response)
            
            improved_script = await script_generator.improve_script(
                script=original_script,
                feedback=feedback
            )
            
            assert isinstance(improved_script, str)
            assert len(improved_script) > len(original_script)
    
    def test_style_prompts_exist(self, script_generator):
        """Test that style prompts are properly defined."""
        assert "educational" in script_generator.style_prompts
        assert "entertaining" in script_generator.style_prompts
        assert "informative" in script_generator.style_prompts
        assert "tutorial" in script_generator.style_prompts
        assert "review" in script_generator.style_prompts
        
        # Check that prompts have content
        for style, prompt in script_generator.style_prompts.items():
            assert len(prompt.strip()) > 50  # Substantial prompt content


class TestVideoScript:
    """Test cases for VideoScript data class."""
    
    def test_video_script_creation(self):
        """Test VideoScript creation and properties."""
        sections = [
            ScriptSection(type="intro", content="Welcome to the video"),
            ScriptSection(type="main", content="Main content here"),
            ScriptSection(type="conclusion", content="Thanks for watching")
        ]
        
        script = VideoScript(
            title="Test Video",
            sections=sections,
            total_duration=180,
            topic="Test Topic",
            style="educational",
            audience="general",
            word_count=150,
            estimated_reading_time=60
        )
        
        assert script.title == "Test Video"
        assert len(script.sections) == 3
        assert script.total_duration == 180
        assert script.word_count == 150
    
    def test_to_text_conversion(self):
        """Test script to text conversion."""
        sections = [
            ScriptSection(type="intro", content="Welcome"),
            ScriptSection(type="main", content="Main content"),
            ScriptSection(type="conclusion", content="Goodbye")
        ]
        
        script = VideoScript(
            title="Test Video",
            sections=sections,
            total_duration=60,
            topic="Test",
            style="educational",
            audience="general",
            word_count=20,
            estimated_reading_time=30
        )
        
        text = script.to_text()
        assert "# Test Video" in text
        assert "Topic: Test" in text
        assert "Welcome" in text
        assert "Main content" in text
        assert "Goodbye" in text


class TestScriptSection:
    """Test cases for ScriptSection data class."""
    
    def test_script_section_creation(self):
        """Test ScriptSection creation."""
        section = ScriptSection(
            type="intro",
            content="This is the introduction",
            duration=30,
            notes="Keep it engaging"
        )
        
        assert section.type == "intro"
        assert section.content == "This is the introduction"
        assert section.duration == 30
        assert section.notes == "Keep it engaging"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])