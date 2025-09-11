import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.ai.ai_manager import llm_manager

logger = get_logger(__name__)


@dataclass
class SEOAnalysis:
    """SEO analysis results for video content."""
    title_score: int  # 0-100
    description_score: int  # 0-100
    tags_score: int  # 0-100
    overall_score: int  # 0-100
    recommendations: List[str]
    keywords_found: List[str]
    keywords_missing: List[str]


@dataclass
class VideoMetadata:
    """Complete video metadata for SEO optimization."""
    title: str
    description: str
    tags: List[str]
    thumbnail_text: Optional[str] = None
    category: Optional[str] = None


class SEOOptimizer:
    """Optimizes video content for YouTube SEO."""
    
    def __init__(self):
        self.config = config_manager.load_config()
        self.youtube_best_practices = {
            "title_max_length": 60,
            "title_min_length": 10,
            "description_max_length": 5000,
            "description_min_length": 100,
            "tags_max_count": 15,
            "tags_min_count": 5,
        }
    
    async def generate_titles(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        count: int = 5,
        style: str = "engaging",
        provider: Optional[str] = None
    ) -> List[str]:
        """Generate optimized video titles."""
        try:
            logger.info(f"Generating {count} titles for content")
            
            system_prompt = self._build_title_system_prompt(style)
            user_prompt = self._build_title_user_prompt(content, keywords, count)
            
            response = await llm_manager.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider,
                temperature=0.8
            )
            
            # Parse titles from response
            titles = self._parse_titles_from_response(response.content)
            
            # Validate and filter titles
            valid_titles = []
            for title in titles:
                if self._validate_title(title):
                    valid_titles.append(title.strip())
            
            logger.info(f"Generated {len(valid_titles)} valid titles")
            return valid_titles[:count]
            
        except Exception as e:
            logger.error(f"Error generating titles: {e}")
            raise
    
    async def generate_description(
        self,
        content: str,
        title: str,
        keywords: Optional[List[str]] = None,
        include_timestamps: bool = False,
        include_links: bool = True,
        provider: Optional[str] = None
    ) -> str:
        """Generate optimized video description."""
        try:
            logger.info("Generating video description")
            
            system_prompt = self._build_description_system_prompt()
            user_prompt = self._build_description_user_prompt(
                content, title, keywords, include_timestamps, include_links
            )
            
            response = await llm_manager.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider,
                temperature=0.7
            )
            
            description = response.content.strip()
            
            # Validate description length
            if len(description) > self.youtube_best_practices["description_max_length"]:
                description = description[:self.youtube_best_practices["description_max_length"]]
            
            logger.info(f"Generated description: {len(description)} characters")
            return description
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            raise
    
    async def generate_tags(
        self,
        content: str,
        title: str,
        keywords: Optional[List[str]] = None,
        max_tags: int = 15,
        provider: Optional[str] = None
    ) -> List[str]:
        """Generate optimized tags for the video."""
        try:
            logger.info("Generating video tags")
            
            system_prompt = self._build_tags_system_prompt()
            user_prompt = self._build_tags_user_prompt(content, title, keywords, max_tags)
            
            response = await llm_manager.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider,
                temperature=0.6
            )
            
            # Parse tags from response
            tags = self._parse_tags_from_response(response.content)
            
            # Validate and clean tags
            valid_tags = []
            for tag in tags:
                cleaned_tag = self._clean_tag(tag)
                if cleaned_tag and len(cleaned_tag) <= 100:  # YouTube tag limit
                    valid_tags.append(cleaned_tag)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_tags = []
            for tag in valid_tags:
                if tag.lower() not in seen:
                    seen.add(tag.lower())
                    unique_tags.append(tag)
            
            logger.info(f"Generated {len(unique_tags)} valid tags")
            return unique_tags[:max_tags]
            
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            raise
    
    async def optimize_metadata(
        self,
        content: str,
        current_metadata: Optional[VideoMetadata] = None,
        keywords: Optional[List[str]] = None,
        provider: Optional[str] = None
    ) -> VideoMetadata:
        """Generate complete optimized metadata for a video."""
        try:
            logger.info("Optimizing complete video metadata")
            
            # Generate title if not provided
            if current_metadata and current_metadata.title:
                title = current_metadata.title
            else:
                titles = await self.generate_titles(content, keywords, count=1, provider=provider)
                title = titles[0] if titles else "Untitled Video"
            
            # Generate description
            description = await self.generate_description(
                content, title, keywords, provider=provider
            )
            
            # Generate tags
            tags = await self.generate_tags(
                content, title, keywords, provider=provider
            )
            
            return VideoMetadata(
                title=title,
                description=description,
                tags=tags
            )
            
        except Exception as e:
            logger.error(f"Error optimizing metadata: {e}")
            raise
    
    def analyze_seo(
        self,
        metadata: VideoMetadata,
        target_keywords: Optional[List[str]] = None
    ) -> SEOAnalysis:
        """Analyze SEO quality of video metadata."""
        title_score = self._analyze_title_seo(metadata.title, target_keywords)
        description_score = self._analyze_description_seo(metadata.description, target_keywords)
        tags_score = self._analyze_tags_seo(metadata.tags, target_keywords)
        
        overall_score = int((title_score + description_score + tags_score) / 3)
        
        recommendations = self._generate_seo_recommendations(
            metadata, target_keywords, title_score, description_score, tags_score
        )
        
        keywords_found, keywords_missing = self._analyze_keyword_coverage(
            metadata, target_keywords
        )
        
        return SEOAnalysis(
            title_score=title_score,
            description_score=description_score,
            tags_score=tags_score,
            overall_score=overall_score,
            recommendations=recommendations,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing
        )
    
    def _build_title_system_prompt(self, style: str) -> str:
        """Build system prompt for title generation."""
        style_instructions = {
            "engaging": "Create titles that are compelling, exciting, and click-worthy while remaining honest",
            "informative": "Create titles that clearly communicate the video's educational value",
            "curiosity": "Create titles that spark curiosity and make viewers want to learn more",
            "benefit": "Create titles that highlight specific benefits or outcomes for viewers",
            "listicle": "Create titles in list format (e.g., '5 Ways to...', 'Top 10...')"
        }
        
        instruction = style_instructions.get(style, style_instructions["engaging"])
        
        return f"""
        You are a YouTube SEO expert specializing in creating high-performing video titles.
        
        {instruction}
        
        Title Best Practices:
        - Keep titles between 10-60 characters for optimal display
        - Include primary keywords naturally
        - Use power words that drive engagement
        - Create emotional hooks that make people want to click
        - Avoid clickbait - be compelling but truthful
        - Consider trending phrases and current events when relevant
        - Make titles specific and clear about the video's value
        
        Format your response as a numbered list of titles only.
        """
    
    def _build_title_user_prompt(
        self, content: str, keywords: Optional[List[str]], count: int
    ) -> str:
        """Build user prompt for title generation."""
        content_summary = content[:500] + "..." if len(content) > 500 else content
        keywords_text = f"\nTarget keywords: {', '.join(keywords)}" if keywords else ""
        
        return f"""
        Create {count} compelling YouTube video titles for this content:
        
        {content_summary}
        {keywords_text}
        
        Generate {count} different title options that would perform well on YouTube.
        """
    
    def _build_description_system_prompt(self) -> str:
        """Build system prompt for description generation."""
        return """
        You are a YouTube SEO expert creating optimized video descriptions.
        
        Description Best Practices:
        - Start with a compelling hook in the first 125 characters (mobile preview)
        - Include primary keywords naturally throughout
        - Provide valuable context about the video content
        - Include relevant timestamps if requested
        - Add call-to-actions (subscribe, like, comment)
        - Include social media links and website if requested
        - Use line breaks for readability
        - Include relevant hashtags at the end
        - Aim for 100-5000 characters total
        
        Structure:
        1. Opening hook (first paragraph)
        2. Video overview and key points
        3. Additional context or related information
        4. Call-to-action
        5. Timestamps (if requested)
        6. Links and social media (if requested)
        7. Hashtags
        """
    
    def _build_description_user_prompt(
        self,
        content: str,
        title: str,
        keywords: Optional[List[str]],
        include_timestamps: bool,
        include_links: bool
    ) -> str:
        """Build user prompt for description generation."""
        content_summary = content[:1000] + "..." if len(content) > 1000 else content
        keywords_text = f"\nTarget keywords: {', '.join(keywords)}" if keywords else ""
        
        additional_requests = []
        if include_timestamps:
            additional_requests.append("- Include placeholder timestamps")
        if include_links:
            additional_requests.append("- Include placeholders for social media links")
        
        additional_text = "\n".join(additional_requests) if additional_requests else ""
        
        return f"""
        Create an optimized YouTube video description for:
        
        Title: {title}
        
        Content:
        {content_summary}
        {keywords_text}
        
        Additional requirements:
        {additional_text}
        
        Create a compelling, SEO-optimized description that encourages engagement.
        """
    
    def _build_tags_system_prompt(self) -> str:
        """Build system prompt for tags generation."""
        return """
        You are a YouTube SEO expert creating optimized video tags.
        
        Tag Best Practices:
        - Use 5-15 relevant tags
        - Include primary keyword variations
        - Mix broad and specific tags
        - Include related topics and synonyms
        - Use both single words and phrases
        - Consider trending topics in the niche
        - Avoid irrelevant or misleading tags
        - Each tag should be under 100 characters
        
        Format your response as a comma-separated list of tags only.
        """
    
    def _build_tags_user_prompt(
        self,
        content: str,
        title: str,
        keywords: Optional[List[str]],
        max_tags: int
    ) -> str:
        """Build user prompt for tags generation."""
        content_summary = content[:500] + "..." if len(content) > 500 else content
        keywords_text = f"\nTarget keywords: {', '.join(keywords)}" if keywords else ""
        
        return f"""
        Generate optimized YouTube tags for this video:
        
        Title: {title}
        
        Content:
        {content_summary}
        {keywords_text}
        
        Generate up to {max_tags} relevant tags as a comma-separated list.
        """
    
    def _parse_titles_from_response(self, response: str) -> List[str]:
        """Parse titles from AI response."""
        lines = response.strip().split('\n')
        titles = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove numbering (1., 2., etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                # Remove quotes if present
                line = re.sub(r'^["\']|["\']$', '', line)
                if line:
                    titles.append(line)
        
        return titles
    
    def _parse_tags_from_response(self, response: str) -> List[str]:
        """Parse tags from AI response."""
        # Split by commas and clean up
        tags = [tag.strip() for tag in response.split(',')]
        
        # Remove any numbering or formatting
        cleaned_tags = []
        for tag in tags:
            tag = re.sub(r'^\d+\.\s*', '', tag)  # Remove numbering
            tag = re.sub(r'^["\']|["\']$', '', tag)  # Remove quotes
            tag = tag.strip()
            if tag:
                cleaned_tags.append(tag)
        
        return cleaned_tags
    
    def _validate_title(self, title: str) -> bool:
        """Validate title meets YouTube requirements."""
        if not title:
            return False
        
        length = len(title)
        min_length = self.youtube_best_practices["title_min_length"]
        max_length = self.youtube_best_practices["title_max_length"]
        
        return min_length <= length <= max_length
    
    def _clean_tag(self, tag: str) -> str:
        """Clean and validate a tag."""
        tag = tag.strip()
        # Remove special characters that might cause issues
        tag = re.sub(r'[<>"]', '', tag)
        return tag
    
    def _analyze_title_seo(self, title: str, keywords: Optional[List[str]]) -> int:
        """Analyze title SEO quality (0-100)."""
        score = 0
        
        # Length check
        length = len(title)
        if 30 <= length <= 60:
            score += 30
        elif 10 <= length <= 30 or 60 <= length <= 100:
            score += 20
        else:
            score += 10
        
        # Keyword presence
        if keywords:
            title_lower = title.lower()
            keyword_found = any(keyword.lower() in title_lower for keyword in keywords)
            if keyword_found:
                score += 25
        else:
            score += 25  # No keywords to check
        
        # Engagement indicators
        power_words = ['how', 'why', 'what', 'best', 'top', 'ultimate', 'complete', 'guide', 'tips', 'secrets']
        if any(word in title.lower() for word in power_words):
            score += 20
        
        # Numbers and specificity
        if re.search(r'\d+', title):
            score += 15
        
        # Capitalization (title case is better)
        if title.istitle() or title[0].isupper():
            score += 10
        
        return min(score, 100)
    
    def _analyze_description_seo(self, description: str, keywords: Optional[List[str]]) -> int:
        """Analyze description SEO quality (0-100)."""
        score = 0
        
        # Length check
        length = len(description)
        if 150 <= length <= 1000:
            score += 25
        elif 100 <= length <= 150 or 1000 <= length <= 5000:
            score += 20
        else:
            score += 10
        
        # Keyword presence
        if keywords:
            description_lower = description.lower()
            keywords_found = sum(1 for keyword in keywords if keyword.lower() in description_lower)
            score += min(keywords_found * 10, 30)
        else:
            score += 30
        
        # Call-to-action presence
        cta_words = ['subscribe', 'like', 'comment', 'share', 'follow']
        if any(word in description.lower() for word in cta_words):
            score += 20
        
        # Link/hashtag presence
        if '#' in description or 'http' in description:
            score += 15
        
        # Structure (multiple paragraphs)
        if description.count('\n\n') >= 1:
            score += 10
        
        return min(score, 100)
    
    def _analyze_tags_seo(self, tags: List[str], keywords: Optional[List[str]]) -> int:
        """Analyze tags SEO quality (0-100)."""
        score = 0
        
        # Count check
        tag_count = len(tags)
        if 8 <= tag_count <= 15:
            score += 30
        elif 5 <= tag_count <= 8:
            score += 25
        else:
            score += 15
        
        # Keyword coverage
        if keywords:
            tags_lower = [tag.lower() for tag in tags]
            keywords_in_tags = sum(1 for keyword in keywords if any(keyword.lower() in tag for tag in tags_lower))
            score += min(keywords_in_tags * 15, 35)
        else:
            score += 35
        
        # Tag variety (mix of short and long tags)
        short_tags = sum(1 for tag in tags if len(tag.split()) == 1)
        long_tags = len(tags) - short_tags
        if short_tags > 0 and long_tags > 0:
            score += 20
        
        # No duplicate tags
        if len(tags) == len(set(tag.lower() for tag in tags)):
            score += 15
        
        return min(score, 100)
    
    def _generate_seo_recommendations(
        self,
        metadata: VideoMetadata,
        keywords: Optional[List[str]],
        title_score: int,
        description_score: int,
        tags_score: int
    ) -> List[str]:
        """Generate SEO improvement recommendations."""
        recommendations = []
        
        # Title recommendations
        if title_score < 70:
            if len(metadata.title) < 30:
                recommendations.append("Consider making your title longer to include more keywords")
            elif len(metadata.title) > 60:
                recommendations.append("Shorten your title to under 60 characters for better display")
            
            if keywords and not any(keyword.lower() in metadata.title.lower() for keyword in keywords):
                recommendations.append("Include your primary keyword in the title")
        
        # Description recommendations
        if description_score < 70:
            if len(metadata.description) < 150:
                recommendations.append("Expand your description to provide more context and keywords")
            
            if not any(word in metadata.description.lower() for word in ['subscribe', 'like', 'comment']):
                recommendations.append("Add call-to-actions to encourage engagement")
        
        # Tags recommendations
        if tags_score < 70:
            if len(metadata.tags) < 8:
                recommendations.append("Add more tags to improve discoverability")
            elif len(metadata.tags) > 15:
                recommendations.append("Reduce the number of tags to focus on most relevant ones")
            
            if keywords and not any(keyword.lower() in ' '.join(metadata.tags).lower() for keyword in keywords):
                recommendations.append("Include your target keywords as tags")
        
        return recommendations
    
    def _analyze_keyword_coverage(
        self, metadata: VideoMetadata, keywords: Optional[List[str]]
    ) -> Tuple[List[str], List[str]]:
        """Analyze which keywords are covered and which are missing."""
        if not keywords:
            return [], []
        
        # Combine all text for analysis
        all_text = f"{metadata.title} {metadata.description} {' '.join(metadata.tags)}".lower()
        
        found = []
        missing = []
        
        for keyword in keywords:
            if keyword.lower() in all_text:
                found.append(keyword)
            else:
                missing.append(keyword)
        
        return found, missing