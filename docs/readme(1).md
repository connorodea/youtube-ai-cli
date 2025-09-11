# YouTube AI CLI

🎬 **AI-powered YouTube automation CLI library** - Automate your entire YouTube content creation pipeline from ideation to publication using cutting-edge AI.

## ✨ Features

- 🤖 **AI Script Generation** - Create engaging video scripts using GPT-4, Claude, or local models
- 🎯 **SEO Optimization** - Generate optimized titles, descriptions, and tags for maximum reach
- 🎥 **Video Creation** - Automated video generation with AI-powered visuals and audio
- 🎵 **Voice Synthesis** - High-quality text-to-speech with multiple voice options
- 📊 **Analytics Integration** - Performance tracking and optimization suggestions
- 🔄 **Workflow Automation** - Batch processing and scheduled content creation
- 🛠️ **Multi-Provider Support** - Works with OpenAI, Anthropic, ElevenLabs, and more

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (coming soon)
pip install youtube-ai-cli

# Or install from source
git clone https://github.com/yourusername/youtube-ai-cli.git
cd youtube-ai-cli
pip install -e .
```

### Initial Setup

1. **Configure API Keys**
   ```bash
   youtube-ai config init
   ```

2. **Verify Configuration**
   ```bash
   youtube-ai config show
   youtube-ai config validate
   ```

### Basic Usage

```bash
# Generate a video script
youtube-ai generate script --topic "AI in 2025" --style educational --duration 300

# Create optimized titles
youtube-ai generate title --script my_script.txt --keywords "AI,technology,future"

# Generate complete video metadata
youtube-ai optimize seo --video-data script.txt --keywords "AI,automation"
```

## 📖 Detailed Usage

### Content Generation

**Generate Video Scripts**
```bash
# Basic script generation
youtube-ai generate script \
  --topic "How to learn Python programming" \
  --style tutorial \
  --duration 600 \
  --audience beginners

# Advanced options
youtube-ai generate script \
  --topic "Machine Learning Basics" \
  --style educational \
  --duration 900 \
  --audience "developers" \
  --output custom_script.txt
```

**SEO Optimization**
```bash
# Generate multiple title options
youtube-ai generate title \
  --topic "Python Tutorial" \
  --keywords "python,programming,tutorial" \
  --count 10

# Create optimized description
youtube-ai generate description \
  --script my_script.txt \
  --title "Complete Python Guide" \
  --keywords "python,coding"

# Generate tags
youtube-ai generate tags \
  --script my_script.txt \
  --title "Python Programming Tutorial"
```

### Configuration Management

```bash
# Set specific configuration values
youtube-ai config set ai.openai_api_key "your-api-key"
youtube-ai config set youtube.channel_id "your-channel-id"
youtube-ai config set video.resolution "1080p"

# View current configuration
youtube-ai config show

# Validate setup
youtube-ai config validate
```

### Advanced Workflows

```bash
# Create a complete workflow
youtube-ai workflow create \
  --name "tech-review" \
  --template tech_video.yml

# Run automated workflow
youtube-ai workflow run \
  --name "tech-review" \
  --topic "iPhone 15 Review"

# Batch processing
youtube-ai batch \
  --input topics.csv \
  --template educational.yml \
  --output-dir ./batch_output
```

## ⚙️ Configuration

### API Keys Required

- **OpenAI API Key** - For GPT-4 script generation
- **Anthropic API Key** - For Claude-based content creation (alternative)
- **YouTube Data API Key** - For video upload and analytics
- **ElevenLabs API Key** - For high-quality voice synthesis (optional)

### Configuration File

Configuration is stored in `~/.youtube-ai/config.yml`:

```yaml
youtube:
  api_key: "your-youtube-api-key"
  channel_id: "your-channel-id"
  default_privacy: "private"

ai:
  openai_api_key: "your-openai-key"
  anthropic_api_key: "your-anthropic-key"
  elevenlabs_api_key: "your-elevenlabs-key"
  default_llm: "openai"
  default_tts: "openai"

video:
  resolution: "1080p"
  fps: 30
  format: "mp4"
  quality: "high"

content:
  language: "en"
  target_audience: "general"
  default_style: "educational"
```

### Environment Variables

You can also use environment variables:

```bash
export YOUTUBE_API_KEY="your-youtube-api-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

## 🎯 Content Styles

The CLI supports multiple content styles:

- **educational** - Clear, informative content with step-by-step explanations
- **entertaining** - Engaging, humorous content with storytelling elements
- **informative** - Fact-based content with data and research
- **tutorial** - Hands-on, practical guides with actionable steps
- **review** - Balanced analysis with pros, cons, and recommendations

## 🏗️ Project Structure

```
youtube-ai-cli/
├── src/youtube_ai/
│   ├── core/                 # Core utilities
│   │   ├── config.py        # Configuration management
│   │   ├── logger.py        # Logging system
│   │   └── exceptions.py    # Custom exceptions
│   ├── modules/
│   │   ├── ai/              # AI integration
│   │   │   ├── llm_client.py    # LLM providers
│   │   │   ├── tts_client.py    # Text-to-speech
│   │   │   └── image_generator.py
│   │   ├── content/         # Content generation
│   │   │   ├── script_generator.py
│   │   │   ├── seo_optimizer.py
│   │   │   └── topic_research.py
│   │   ├── media/           # Media processing
│   │   │   ├── video_generator.py
│   │   │   ├── audio_processor.py
│   │   │   └── thumbnail_creator.py
│   │   └── youtube/         # YouTube integration
│   │       ├── uploader.py
│   │       ├── analytics.py
│   │       └── scheduler.py
│   └── cli/                 # CLI interface
│       ├── main.py          # Main CLI entry point
│       └── commands/        # Command modules
├── config/                  # Configuration templates
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## 🧪 Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-ai-cli.git
cd youtube-ai-cli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=youtube_ai

# Run specific test file
pytest tests/test_script_generator.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## 📚 API Reference

### ScriptGenerator

```python
from youtube_ai import ScriptGenerator

generator = ScriptGenerator()
script = await generator.generate_script(
    topic="AI in Healthcare",
    style="educational",
    duration=300,
    audience="medical professionals"
)
```

### SEOOptimizer

```python
from youtube_ai import SEOOptimizer

optimizer = SEOOptimizer()
titles = await optimizer.generate_titles(
    content="Script content...",
    keywords=["AI", "healthcare", "technology"],
    count=5
)
```

### Configuration

```python
from youtube_ai import config_manager

# Get configuration
config = config_manager.load_config()

# Set configuration value
config_manager.set_value("ai.default_llm", "anthropic")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 [Documentation](https://youtube-ai-cli.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/yourusername/youtube-ai-cli/issues)
- 💬 [Discussions](https://github.com/yourusername/youtube-ai-cli/discussions)
- 📧 [Email Support](mailto:support@youtube-ai-cli.com)

## 🗺️ Roadmap

- [ ] **v0.2.0** - Video generation and editing
- [ ] **v0.3.0** - Advanced voice synthesis and audio processing
- [ ] **v0.4.0** - Thumbnail generation and A/B testing
- [ ] **v0.5.0** - Analytics and performance optimization
- [ ] **v1.0.0** - Full automation workflows and scheduling

## ⭐ Show Your Support

If this project helps you automate your YouTube content creation, please give it a star! ⭐

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/youtube-ai-cli)
![GitHub forks](https://img.shields.io/github/forks/yourusername/youtube-ai-cli)
![GitHub issues](https://img.shields.io/github/issues/yourusername/youtube-ai-cli)
![PyPI version](https://img.shields.io/pypi/v/youtube-ai-cli)
![Python version](https://img.shields.io/pypi/pyversions/youtube-ai-cli)

---

**Made with ❤️ for the YouTube creator community**