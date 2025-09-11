"""Custom exceptions for YouTube AI CLI."""


class YouTubeAIError(Exception):
    """Base exception for YouTube AI CLI."""
    pass


class ConfigurationError(YouTubeAIError):
    """Raised when there's a configuration issue."""
    pass


class APIError(YouTubeAIError):
    """Raised when there's an API-related error."""
    
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class AIProviderError(APIError):
    """Raised when there's an error with AI providers."""
    pass


class TTSError(YouTubeAIError):
    """Raised when there's a text-to-speech error."""
    pass


class VideoGenerationError(YouTubeAIError):
    """Raised when there's a video generation error."""
    pass


class YouTubeUploadError(YouTubeAIError):
    """Raised when there's a YouTube upload error."""
    
    def __init__(self, message: str, video_id: str = None, error_code: str = None):
        super().__init__(message)
        self.video_id = video_id
        self.error_code = error_code


class ValidationError(YouTubeAIError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: str = None, value: str = None):
        super().__init__(message)
        self.field = field
        self.value = value


class FileNotFoundError(YouTubeAIError):
    """Raised when a required file is not found."""
    pass


class AuthenticationError(YouTubeAIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, service: str = None):
        super().__init__(message)
        self.service = service


class QuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    
    def __init__(self, message: str, provider: str = None, reset_time: str = None):
        super().__init__(message, provider)
        self.reset_time = reset_time


class ContentPolicyError(YouTubeAIError):
    """Raised when content violates platform policies."""
    
    def __init__(self, message: str, policy: str = None):
        super().__init__(message)
        self.policy = policy


class RateLimitError(APIError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, provider: str = None, retry_after: int = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class DependencyError(YouTubeAIError):
    """Raised when a required dependency is missing."""
    
    def __init__(self, message: str, package: str = None, install_command: str = None):
        super().__init__(message)
        self.package = package
        self.install_command = install_command


class WorkflowError(YouTubeAIError):
    """Raised when there's a workflow execution error."""
    
    def __init__(self, message: str, step: str = None):
        super().__init__(message)
        self.step = step


class TemplateError(YouTubeAIError):
    """Raised when there's a template processing error."""
    
    def __init__(self, message: str, template: str = None):
        super().__init__(message)
        self.template = template