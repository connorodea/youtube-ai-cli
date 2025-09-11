from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="youtube-ai-cli",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered YouTube automation CLI library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/youtube-ai-cli",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "local": ["torch>=2.0.0", "transformers>=4.30.0", "diffusers>=0.20.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0", "mypy>=1.5.0"],
    },
    entry_points={
        "console_scripts": [
            "youtube-ai=youtube_ai.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "youtube_ai": ["config/*.yml", "config/templates/*.yml", "config/presets/*.yml"],
    },
)