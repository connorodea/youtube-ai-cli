import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.ai.ai_manager import llm_manager

logger = get_logger(__name__)


@dataclass
class ScriptSection:
    """Represents a section of a video script."""
    type: str  # "intro", "main", "transition", "conclusion"
    content: str
    duration: Optional[int] = None  # estimated duration in seconds
    notes: Optional[str] = None


@dataclass
class VideoScript:
    """Complete video script with metadata."""
    title: str
    sections: List[ScriptSection]
    total_duration: int
    topic: str
    style: str
    audience: str
    word_count: int
    estimated_reading_time: int
    
    def to_text(self) -> str:
        """Convert script to plain text format."""
        lines = [
            f"# {self.title}",
            f"Topic: {self.topic}",
            f"Style: {self.style}",
            f"Audience: {self.audience}",
            f"Duration: {self.total_duration} seconds",
            f"Word Count: {self.word_count}",
            "",
            "## Script",
            ""
        ]
        
        for section in self.sections:
            if section.type != "main":
                lines.append(f"### {section.type.title()}")
            lines.append(section.content)
            if section.notes:
                lines.append(f"*[Note: {section.notes}]*")
            lines.append("")
        
        return "\n".join(lines)


class ScriptGenerator:
    """Generates video scripts using AI."""
    
    def __init__(self):
        self.config = config_manager.load_config()
        self.style_prompts = self._load_style_prompts()
    
    def _load_style_prompts(self) -> Dict[str, str]:
        """Load style-specific prompts."""
        return {
            "educational": """
                Create an educational video script that:
                - Explains concepts clearly and step-by-step
                - Uses analogies and examples to illustrate points
                - Includes engaging questions to maintain viewer interest
                - Has a logical flow from basic to advanced concepts
                - Encourages learning and curiosity
            """,
            "entertaining": """
                Create an entertaining video script that:
                - Uses humor and storytelling to engage viewers
                - Includes surprising facts or interesting anecdotes
                - Has a conversational, friendly tone
                - Uses relatable examples and pop culture references
                - Keeps energy high throughout
            """,
            "informative": """
                Create an informative video script that:
                - Presents facts and data clearly
                - Uses credible sources and examples
                - Maintains objectivity and balance
                - Includes key statistics and insights
                - Has a professional, authoritative tone
            """,
            "tutorial": """
                Create a tutorial video script that:
                - Provides step-by-step instructions
                - Includes practical tips and best practices
                - Anticipates common questions and problems
                - Uses clear, actionable language
                - Includes checkpoints and summaries
            """,
            "review": """
                Create a review video script that:
                - Provides balanced analysis of pros and cons
                - Includes personal experience and opinions
                - Compares with alternatives when relevant
                - Uses specific examples and demonstrations
                - Ends with clear recommendations
            """
        }
    
    async def generate_script(
        self,
        topic: str,
        style: str = "educational",
        duration: int = 300,
        audience: str = "general",
        include_intro: bool = True,
        include_outro: bool = True,
        provider: Optional[str] = None
    ) -> str:
        """Generate a complete video script."""
        try:
            logger.info(f"Generating script for topic: {topic}")
            
            # Calculate approximate word count (150 words per minute)
            target_words = int((duration / 60) * 150)
            
            # Build the prompt
            system_prompt = self._build_system_prompt(style, audience, duration)
            user_prompt = self._build_user_prompt(
                topic, duration, target_words, include_intro, include_outro
            )
            
            # Generate the script
            response = await llm_manager.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider,
                temperature=0.7,
                max_tokens=target_words * 2  # Allow more tokens for formatting
            )
            
            script_text = response.content.strip()
            
            # Parse and structure the script
            video_script = self._parse_script(
                script_text, topic, style, audience, duration
            )
            
            logger.info(f"Script generated successfully: {video_script.word_count} words")
            return video_script.to_text()
            
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            raise
    
    def _build_system_prompt(self, style: str, audience: str, duration: int) -> str:
        """Build the system prompt for script generation."""
        style_instruction = self.style_prompts.get(style, self.style_prompts["educational"])
        
        return f"""
        You are an expert YouTube content creator and scriptwriter. Your task is to create engaging, well-structured video scripts that capture and maintain viewer attention.
        
        {style_instruction}
        
        General Guidelines:
        - Write in a conversational, engaging tone appropriate for {audience} audience
        - Structure the content logically with smooth transitions
        - Include hooks to maintain engagement throughout
        - Use active voice and varied sentence lengths
        - Write for a {duration}-second video duration
        - Include natural speaking cues and pauses where appropriate
        - Make content accessible and easy to follow
        
        The script should be ready for voice-over recording and video production.
        """
    
    def _build_user_prompt(
        self,
        topic: str,
        duration: int,
        target_words: int,
        include_intro: bool,
        include_outro: bool
    ) -> str:
        """Build the user prompt for script generation."""
        structure_request = []
        
        if include_intro:
            structure_request.append("- A compelling intro that hooks the viewer (10-15 seconds)")
        
        structure_request.extend([
            "- Main content divided into clear sections",
            "- Smooth transitions between sections",
            "- Engaging examples and explanations"
        ])
        
        if include_outro:
            structure_request.append("- A strong conclusion with call-to-action (10-15 seconds)")
        
        structure = "\n".join(structure_request)
        
        return f"""
        Create a video script about: "{topic}"
        
        Requirements:
        - Target duration: {duration} seconds ({duration // 60} minutes {duration % 60} seconds)
        - Approximate word count: {target_words} words
        - Speaking pace: ~150 words per minute
        
        Structure the script with:
        {structure}
        
        Please provide only the script content, written in a natural speaking style that would work well for voice-over. Do not include technical directions or camera instructions.
        """
    
    def _parse_script(
        self,
        script_text: str,
        topic: str,
        style: str,
        audience: str,
        duration: int
    ) -> VideoScript:
        """Parse the generated script into structured sections."""
        # Remove any markdown formatting
        script_text = re.sub(r'[#*`]', '', script_text)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in script_text.split('\n\n') if p.strip()]
        
        sections = []
        
        # Try to identify intro (first paragraph)
        if paragraphs:
            intro_text = paragraphs[0]
            sections.append(ScriptSection(
                type="intro",
                content=intro_text,
                duration=self._estimate_duration(intro_text)
            ))
        
        # Main content (middle paragraphs)
        if len(paragraphs) > 2:
            main_content = "\n\n".join(paragraphs[1:-1])
            sections.append(ScriptSection(
                type="main",
                content=main_content,
                duration=self._estimate_duration(main_content)
            ))
        elif len(paragraphs) == 2:
            # If only 2 paragraphs, second one is main content
            sections.append(ScriptSection(
                type="main",
                content=paragraphs[1],
                duration=self._estimate_duration(paragraphs[1])
            ))
        
        # Outro (last paragraph if more than one)
        if len(paragraphs) > 1:
            outro_text = paragraphs[-1]
            # Check if this looks like a conclusion
            if any(word in outro_text.lower() for word in ['conclusion', 'remember', 'thanks', 'subscribe', 'like', 'comment']):
                sections.append(ScriptSection(
                    type="conclusion",
                    content=outro_text,
                    duration=self._estimate_duration(outro_text)
                ))
            else:
                # Merge with main content if it doesn't look like an outro
                if sections and sections[-1].type == "main":
                    sections[-1].content += "\n\n" + outro_text
                    sections[-1].duration = self._estimate_duration(sections[-1].content)
        
        # Calculate metrics
        word_count = len(script_text.split())
        estimated_reading_time = self._estimate_duration(script_text)
        
        # Generate title if not provided
        title = self._generate_title(topic, style)
        
        return VideoScript(
            title=title,
            sections=sections,
            total_duration=duration,
            topic=topic,
            style=style,
            audience=audience,
            word_count=word_count,
            estimated_reading_time=estimated_reading_time
        )
    
    def _estimate_duration(self, text: str) -> int:
        """Estimate speaking duration for text (assuming 150 words per minute)."""
        word_count = len(text.split())
        return int((word_count / 150) * 60)
    
    def _generate_title(self, topic: str, style: str) -> str:
        """Generate a title based on topic and style."""
        # Simple title generation - could be enhanced with AI
        style_prefixes = {
            "educational": "How to",
            "entertaining": "The Amazing",
            "informative": "Everything About",
            "tutorial": "Complete Guide to",
            "review": "Honest Review of"
        }
        
        prefix = style_prefixes.get(style, "")
        if prefix:
            return f"{prefix} {topic}"
        return topic.title()
    
    async def improve_script(
        self,
        script: str,
        feedback: str,
        provider: Optional[str] = None
    ) -> str:
        """Improve an existing script based on feedback."""
        try:
            system_prompt = """
            You are an expert script editor. Your task is to improve the given video script based on the provided feedback while maintaining its core message and structure.
            
            Guidelines:
            - Address the specific feedback provided
            - Maintain the original script's tone and style
            - Keep the same approximate length unless requested otherwise
            - Ensure smooth flow and natural speaking rhythm
            - Preserve any good elements from the original
            """
            
            user_prompt = f"""
            Original Script:
            {script}
            
            Feedback to address:
            {feedback}
            
            Please provide an improved version of the script that addresses the feedback while maintaining quality and engagement.
            """
            
            response = await llm_manager.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider,
                temperature=0.6
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error improving script: {e}")
            raise