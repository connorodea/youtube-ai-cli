#!/usr/bin/env python3
"""
Basic usage example for YouTube AI CLI.

This script demonstrates how to use the library programmatically
to generate content for YouTube videos.
"""

import asyncio
import os
from pathlib import Path

# Add the src directory to the path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from youtube_ai import ScriptGenerator, SEOOptimizer, config_manager


async def main():
    """Demonstrate basic usage of YouTube AI CLI."""
    print("🎬 YouTube AI CLI - Basic Usage Example")
    print("=" * 50)
    
    # Check configuration
    print("\n1. Checking configuration...")
    config = config_manager.load_config()
    is_valid, issues = config_manager.validate_config()
    
    if not is_valid:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nPlease run 'youtube-ai config init' to set up your API keys.")
        return
    else:
        print("✅ Configuration is valid!")
    
    # Example topic for demonstration
    topic = "The Future of Artificial Intelligence in 2025"
    
    try:
        # Generate a script
        print(f"\n2. Generating script for: '{topic}'")
        script_generator = ScriptGenerator()
        
        script = await script_generator.generate_script(
            topic=topic,
            style="educational",
            duration=300,  # 5 minutes
            audience="tech enthusiasts"
        )
        
        print("✅ Script generated successfully!")
        print(f"   Length: {len(script.split())} words")
        
        # Save script to file
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)
        script_file = output_dir / "example_script.txt"
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"   Saved to: {script_file}")
        
        # Generate SEO-optimized titles
        print(f"\n3. Generating optimized titles...")
        seo_optimizer = SEOOptimizer()
        
        titles = await seo_optimizer.generate_titles(
            content=script,
            keywords=["AI", "artificial intelligence", "technology", "2025"],
            count=5,
            style="engaging"
        )
        
        print("✅ Generated title options:")
        for i, title in enumerate(titles, 1):
            print(f"   {i}. {title}")
        
        # Generate description
        print(f"\n4. Generating video description...")
        description = await seo_optimizer.generate_description(
            content=script,
            title=titles[0] if titles else topic,
            keywords=["AI", "artificial intelligence", "technology", "2025"],
            include_timestamps=True,
            include_links=True
        )
        
        print("✅ Description generated successfully!")
        print(f"   Length: {len(description)} characters")
        
        # Save description
        desc_file = output_dir / "example_description.txt"
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"   Saved to: {desc_file}")
        
        # Generate tags
        print(f"\n5. Generating video tags...")
        tags = await seo_optimizer.generate_tags(
            content=script,
            title=titles[0] if titles else topic,
            keywords=["AI", "artificial intelligence", "technology", "2025"],
            max_tags=12
        )
        
        print("✅ Generated tags:")
        print(f"   {', '.join(tags)}")
        
        # Create complete metadata file
        print(f"\n6. Creating complete metadata...")
        metadata = {
            "title": titles[0] if titles else topic,
            "description": description,
            "tags": tags,
            "topic": topic,
            "style": "educational",
            "duration": 300,
            "audience": "tech enthusiasts"
        }
        
        import json
        metadata_file = output_dir / "example_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   Saved complete metadata to: {metadata_file}")
        
        # Summary
        print(f"\n🎉 Example completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Files created:")
        print(f"   • {script_file.name} - Video script")
        print(f"   • {desc_file.name} - Video description")
        print(f"   • {metadata_file.name} - Complete metadata")
        
        print(f"\n💡 Next steps:")
        print(f"   • Review and edit the generated content")
        print(f"   • Create video using the script")
        print(f"   • Upload to YouTube with the optimized metadata")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    optional_vars = ["YOUTUBE_API_KEY", "ELEVENLABS_API_KEY"]
    
    print("🔍 Environment Check:")
    
    has_ai_key = False
    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var} is set")
            has_ai_key = True
        else:
            print(f"   ⚠️  {var} not set")
    
    if not has_ai_key:
        print("\n❌ No AI API keys found!")
        print("   Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("\n   Example:")
        print("   export OPENAI_API_KEY='your-openai-key-here'")
        print("   # OR")
        print("   export ANTHROPIC_API_KEY='your-anthropic-key-here'")
        return False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"   ✅ {var} is set")
        else:
            print(f"   ℹ️  {var} not set (optional)")
    
    return True


if __name__ == "__main__":
    print("YouTube AI CLI - Basic Usage Example")
    print("=" * 40)
    
    # Check environment first
    if check_environment():
        print("\n✅ Environment check passed!")
        asyncio.run(main())
    else:
        print("\n❌ Please set up your API keys first.")
        print("   Run this after setting your environment variables:")
        print("   python examples/basic_usage.py")