# Contributing to YouTube AI CLI

🎉 First off, thank you for considering contributing to YouTube AI CLI! It's people like you that make this tool amazing for the entire creator community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@youtube-ai-cli.com](mailto:conduct@youtube-ai-cli.com).

## Getting Started

### Types of Contributions

We welcome many different types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve or add to our documentation
- 🔧 **Code Contributions**: Fix bugs or implement new features
- 🧪 **Testing**: Add tests or improve test coverage
- 🌍 **Internationalization**: Help translate the tool
- 🎨 **Design**: Improve user experience and interfaces
- 📊 **Performance**: Optimize code and improve efficiency

### Before You Start

1. **Check existing issues**: Look through existing [issues](https://github.com/yourusername/youtube-ai-cli/issues) and [pull requests](https://github.com/yourusername/youtube-ai-cli/pulls)
2. **Create an issue**: For new features or significant changes, create an issue first to discuss
3. **Small fixes**: For small bug fixes or documentation improvements, feel free to submit a PR directly

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (for containerized development)
- FFmpeg (for video processing)

### Setup Instructions

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/youtube-ai-cli.git
   cd youtube-ai-cli
   
   # Add the upstream repository
   git remote add upstream https://github.com/yourusername/youtube-ai-cli.git
   ```

2. **Create Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Set Up Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit with your API keys (for testing)
   # Note: Use test/development keys, never production keys
   ```

4. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   make test
   
   # Check code quality
   make quality
   
   # Verify CLI works
   youtube-ai --help
   ```

### Development with Docker

```bash
# Build development image
docker build -f Dockerfile.dev -t youtube-ai-cli:dev .

# Run development container
docker run -it --rm \
  -v $(pwd):/workspace \
  -v youtube_ai_dev_output:/app/output \
  youtube-ai-cli:dev bash

# Inside container
pip install -e ".[dev]"
pytest
```

## How to Contribute

### Reporting Bugs

When filing a bug report, please include:

1. **Clear title**: Summarize the issue in the title
2. **Environment details**: 
   - OS and version
   - Python version
   - YouTube AI CLI version
   - Relevant dependencies
3. **Steps to reproduce**: Detailed steps to reproduce the issue
4. **Expected behavior**: What should have happened
5. **Actual behavior**: What actually happened
6. **Error messages**: Full error messages and stack traces
7. **Configuration**: Relevant configuration (with secrets redacted)

**Bug Report Template:**
```markdown
## Bug Description
A clear description of the bug.

## Environment
- OS: [e.g., Ubuntu 20.04, macOS 12.0, Windows 11]
- Python: [e.g., 3.11.0]
- YouTube AI CLI: [e.g., 0.1.0]

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Error Messages
```
Paste error messages here
```

## Additional Context
Add any other context about the problem here.
```

### Suggesting Features

For feature requests, please include:

1. **Problem statement**: What problem does this solve?
2. **Proposed solution**: Detailed description of the feature
3. **Alternatives considered**: Other solutions you've considered
4. **Use cases**: Real-world scenarios where this would be useful
5. **Implementation ideas**: Technical approach (if you have ideas)

### Contributing Code

#### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/feature-name`: Individual feature branches
- `bugfix/bug-description`: Bug fix branches
- `hotfix/critical-fix`: Critical production fixes

#### Workflow

1. **Create a branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   make test
   
   # Run specific tests
   pytest tests/test_your_feature.py
   
   # Check code quality
   make quality
   ```

4. **Commit your changes**
   ```bash
   # Stage changes
   git add .
   
   # Commit with descriptive message
   git commit -m "feat: add video thumbnail generation with AI optimization
   
   - Implement ThumbnailGenerator class with multiple styles
   - Add AI-powered optimization using GPT-4
   - Support for batch thumbnail generation
   - Include A/B testing capabilities
   
   Closes #123"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 120 characters (not 79)
- **String quotes**: Use double quotes for strings
- **Imports**: Use absolute imports, group by standard/third-party/local

### Code Formatting

We use automated tools for consistency:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Type Hints

- Use type hints for all public functions and methods
- Use `typing` module for complex types
- Example:
  ```python
  from typing import Dict, List, Optional, Union
  
  async def generate_script(
      self,
      topic: str,
      style: str = "educational",
      duration: int = 300,
      audience: str = "general"
  ) -> str:
      """Generate a video script using AI."""
      pass
  ```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this exception is raised.
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        "success"
    """
    pass
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Example:
  ```python
  try:
      result = await api_call()
  except APIError as e:
      logger.error(f"API call failed: {e}")
      raise ScriptGenerationError(f"Failed to generate script: {e}") from e
  ```

### Async/Await

- Use async/await for I/O operations
- Don't use async for CPU-bound operations
- Handle exceptions in async code properly

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests with external services
├── fixtures/       # Test data and fixtures
├── conftest.py     # Pytest configuration
└── utils/          # Test utilities
```

### Writing Tests

1. **Test naming**: Use descriptive names
   ```python
   def test_script_generator_creates_educational_content():
       pass
   
   def test_video_generator_handles_missing_audio_file():
       pass
   ```

2. **Test structure**: Follow Arrange-Act-Assert pattern
   ```python
   def test_seo_optimizer_generates_multiple_titles():
       # Arrange
       optimizer = SEOOptimizer()
       content = "Sample video content about AI"
       keywords = ["AI", "technology"]
       
       # Act
       titles = await optimizer.generate_titles(
           content=content,
           keywords=keywords,
           count=5
       )
       
       # Assert
       assert len(titles) == 5
       assert all(isinstance(title, str) for title in titles)
       assert any("AI" in title for title in titles)
   ```

3. **Mocking external services**:
   ```python
   @pytest.fixture
   def mock_openai_client():
       with patch('youtube_ai.modules.ai.llm_client.OpenAIClient') as mock:
           mock.return_value.generate_completion.return_value = AIResponse(
               content="Test response",
               provider="openai",
               model="gpt-4"
           )
           yield mock
   ```

### Test Categories

Use pytest markers for different test types:

```python
@pytest.mark.unit
def test_unit_functionality():
    pass

@pytest.mark.integration
def test_api_integration():
    pass

@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.requires_api
def test_with_real_api():
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit
pytest -m "not slow"
pytest -m integration

# Run with coverage
pytest --cov=youtube_ai --cov-report=html

# Run specific test file
pytest tests/test_script_generator.py

# Run with verbose output
pytest -v

# Run in parallel (faster)
pytest -n auto
```

## Documentation

### Types of Documentation

1. **Code comments**: Explain complex logic
2. **Docstrings**: Document functions, classes, and modules
3. **README files**: Explain project structure and usage
4. **API documentation**: Comprehensive API reference
5. **Tutorials**: Step-by-step guides for common tasks
6. **Examples**: Real-world usage examples

### Documentation Standards

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include examples where helpful
- Test code examples to ensure they work

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve documentation locally
make serve
```

## Submitting Changes

### Pull Request Guidelines

1. **PR Title**: Use conventional commits format
   - `feat: add new feature`
   - `fix: resolve bug in script generation`
   - `docs: update API documentation`
   - `test: add tests for video generator`
   - `refactor: improve error handling`

2. **PR Description**: Use the template
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Code refactoring
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

3. **Small, focused changes**: Keep PRs small and focused on a single feature or fix

4. **Link issues**: Reference related issues in the PR description

### Review Process

1. **Automated checks**: All tests and quality checks must pass
2. **Code review**: At least one maintainer must review and approve
3. **Documentation review**: Ensure documentation is updated
4. **Testing**: Verify that tests cover new functionality

### After Your PR is Merged

1. **Delete your branch**: Clean up feature branches after merging
2. **Update your fork**: Keep your fork up-to-date
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Tag release
4. Build and test packages
5. Deploy to PyPI
6. Update documentation
7. Announce release

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat and community support
- **Email**: [maintainers@youtube-ai-cli.com](mailto:maintainers@youtube-ai-cli.com)

### Getting Help

- **Documentation**: Check our comprehensive docs first
- **GitHub Discussions**: Ask questions in discussions
- **Discord**: Get quick help from the community
- **Stack Overflow**: Tag questions with `youtube-ai-cli`

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor spotlight
- Special mentions in documentation

### Becoming a Maintainer

Regular contributors may be invited to become maintainers. Maintainers have:
- Commit access to the repository
- Ability to review and merge PRs
- Responsibility for project direction
- Access to maintainer resources

## Thank You! 🙏

Every contribution, no matter how small, makes a difference. Whether you're reporting a bug, suggesting a feature, or contributing code, you're helping make YouTube AI CLI better for everyone.

## Additional Resources

- [Development Setup Guide](docs/development.md)
- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Security Policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Changelog](CHANGELOG.md)

---

*Happy coding! 🚀*