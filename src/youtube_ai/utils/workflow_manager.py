import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.core.exceptions import WorkflowError, ValidationError
from youtube_ai.content.script_generator import ScriptGenerator
from youtube_ai.content.seo_optimizer import SEOOptimizer
from youtube_ai.ai.tts_client import tts_manager
from youtube_ai.media.video_generator import video_generator, VideoStyle
from youtube_ai.utils.youtube_uploader import youtube_uploader, VideoMetadata, PrivacyStatus

logger = get_logger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepType(Enum):
    GENERATE_SCRIPT = "generate_script"
    GENERATE_AUDIO = "generate_audio"
    GENERATE_VIDEO = "generate_video"
    GENERATE_THUMBNAIL = "generate_thumbnail"
    OPTIMIZE_SEO = "optimize_seo"
    UPLOAD_VIDEO = "upload_video"
    SCHEDULE_VIDEO = "schedule_video"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    id: str
    type: WorkflowStepType
    name: str
    config: Dict[str, Any]
    depends_on: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds
    enabled: bool = True

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class WorkflowExecution:
    """Tracks execution of a workflow run."""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = None
    failed_steps: List[str] = None
    outputs: Dict[str, Any] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []
        if self.outputs is None:
            self.outputs = {}


@dataclass
class WorkflowTemplate:
    """Template for creating workflows."""
    id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    default_config: Dict[str, Any]
    tags: List[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()


class WorkflowManager:
    """Manages workflow creation, execution, and monitoring."""

    def __init__(self):
        self.config = config_manager.load_config()
        self.workflow_dir = Path.home() / ".youtube-ai" / "workflows"
        self.execution_dir = Path.home() / ".youtube-ai" / "executions"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize step executors
        self.step_executors = {
            WorkflowStepType.GENERATE_SCRIPT: self._execute_generate_script,
            WorkflowStepType.GENERATE_AUDIO: self._execute_generate_audio,
            WorkflowStepType.GENERATE_VIDEO: self._execute_generate_video,
            WorkflowStepType.OPTIMIZE_SEO: self._execute_optimize_seo,
            WorkflowStepType.UPLOAD_VIDEO: self._execute_upload_video,
        }

    async def create_workflow_template(
        self,
        name: str,
        description: str,
        steps: List[WorkflowStep],
        default_config: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> WorkflowTemplate:
        """Create a new workflow template."""
        template_id = name.lower().replace(' ', '_').replace('-', '_')
        
        template = WorkflowTemplate(
            id=template_id,
            name=name,
            description=description,
            version="1.0.0",
            steps=steps,
            default_config=default_config or {},
            tags=tags or []
        )
        
        # Validate workflow
        await self._validate_workflow(template)
        
        # Save template
        template_file = self.workflow_dir / f"{template_id}.yml"
        with open(template_file, 'w') as f:
            yaml.dump(asdict(template), f, default_flow_style=False)
        
        logger.info(f"Created workflow template: {name}")
        return template

    async def load_workflow_template(self, template_id: str) -> WorkflowTemplate:
        """Load a workflow template."""
        template_file = self.workflow_dir / f"{template_id}.yml"
        
        if not template_file.exists():
            raise WorkflowError(f"Workflow template not found: {template_id}")
        
        with open(template_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert steps back to WorkflowStep objects
        steps = []
        for step_data in data['steps']:
            steps.append(WorkflowStep(
                id=step_data['id'],
                type=WorkflowStepType(step_data['type']),
                name=step_data['name'],
                config=step_data['config'],
                depends_on=step_data.get('depends_on', []),
                retry_count=step_data.get('retry_count', 0),
                max_retries=step_data.get('max_retries', 3),
                timeout=step_data.get('timeout', 300),
                enabled=step_data.get('enabled', True)
            ))
        
        template = WorkflowTemplate(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            version=data['version'],
            steps=steps,
            default_config=data['default_config'],
            tags=data.get('tags', []),
            created_at=datetime.fromisoformat(data['created_at'])
        )
        
        return template

    async def execute_workflow(
        self,
        template_id: str,
        inputs: Dict[str, Any],
        execution_config: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a workflow from a template."""
        template = await self.load_workflow_template(template_id)
        
        execution_id = f"{template_id}_{int(time.time())}"
        execution = WorkflowExecution(
            workflow_id=template_id,
            execution_id=execution_id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now()
        )
        
        try:
            execution.status = WorkflowStatus.RUNNING
            await self._save_execution_state(execution)
            
            logger.info(f"Starting workflow execution: {execution_id}")
            
            # Merge configs
            config = {**template.default_config, **(execution_config or {}), **inputs}
            
            # Execute steps in dependency order
            sorted_steps = self._sort_steps_by_dependencies(template.steps)
            
            for step in sorted_steps:
                if not step.enabled:
                    logger.info(f"Skipping disabled step: {step.name}")
                    continue
                
                execution.current_step = step.id
                await self._save_execution_state(execution)
                
                logger.info(f"Executing step: {step.name}")
                
                try:
                    # Execute step with retry logic
                    step_output = await self._execute_step_with_retry(step, config, execution.outputs)
                    
                    # Store step output
                    execution.outputs[step.id] = step_output
                    execution.completed_steps.append(step.id)
                    
                    logger.info(f"Completed step: {step.name}")
                    
                except Exception as e:
                    logger.error(f"Step failed: {step.name} - {e}")
                    execution.failed_steps.append(step.id)
                    execution.error_message = str(e)
                    execution.status = WorkflowStatus.FAILED
                    await self._save_execution_state(execution)
                    raise WorkflowError(f"Step '{step.name}' failed: {e}", step=step.id)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.current_step = None
            
            logger.info(f"Workflow execution completed: {execution_id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.now()
            execution.error_message = str(e)
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
        
        finally:
            await self._save_execution_state(execution)
        
        return execution

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        config: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Any:
        """Execute a step with retry logic."""
        last_error = None
        
        for attempt in range(step.max_retries + 1):
            try:
                if step.type in self.step_executors:
                    return await asyncio.wait_for(
                        self.step_executors[step.type](step, config, previous_outputs),
                        timeout=step.timeout
                    )
                else:
                    raise WorkflowError(f"Unknown step type: {step.type}")
                    
            except Exception as e:
                last_error = e
                step.retry_count = attempt + 1
                
                if attempt < step.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Step {step.name} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Step {step.name} failed after {step.max_retries + 1} attempts")
                    raise last_error

    async def _execute_generate_script(
        self,
        step: WorkflowStep,
        config: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute script generation step."""
        script_config = {**config, **step.config}
        
        generator = ScriptGenerator()
        script = await generator.generate_script(
            topic=script_config['topic'],
            style=script_config.get('style', 'educational'),
            duration=script_config.get('duration', 300),
            audience=script_config.get('audience', 'general'),
            provider=script_config.get('ai_provider')
        )
        
        # Save script to file
        output_dir = Path(config.get('output_dir', self.config.output_dir))
        script_file = output_dir / f"script_{int(time.time())}.txt"
        script_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script)
        
        return {
            "script_content": script,
            "script_file": str(script_file),
            "word_count": len(script.split()),
            "estimated_duration": len(script.split()) / 150 * 60  # 150 WPM
        }

    async def _execute_generate_audio(
        self,
        step: WorkflowStep,
        config: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute audio generation step."""
        audio_config = {**config, **step.config}
        
        # Get script from previous step
        script_content = None
        for output in previous_outputs.values():
            if isinstance(output, dict) and 'script_content' in output:
                script_content = output['script_content']
                break
        
        if not script_content:
            raise WorkflowError("No script content found for audio generation")
        
        # Generate audio
        output_dir = Path(config.get('output_dir', self.config.output_dir))
        audio_file = output_dir / f"audio_{int(time.time())}.mp3"
        
        response = await tts_manager.synthesize_speech(
            text=script_content,
            voice=audio_config.get('voice'),
            provider=audio_config.get('tts_provider'),
            speed=audio_config.get('speed', 1.0),
            output_file=audio_file
        )
        
        return {
            "audio_file": str(audio_file),
            "provider": response.provider,
            "voice": response.voice,
            "format": response.format,
            "file_size": audio_file.stat().st_size
        }

    async def _execute_generate_video(
        self,
        step: WorkflowStep,
        config: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute video generation step."""
        if not video_generator:
            raise WorkflowError("Video generator not available")
        
        video_config = {**config, **step.config}
        
        # Get script from previous step
        script_content = None
        for output in previous_outputs.values():
            if isinstance(output, dict) and 'script_content' in output:
                script_content = output['script_content']
                break
        
        if not script_content:
            raise WorkflowError("No script content found for video generation")
        
        # Generate video
        output_dir = Path(config.get('output_dir', self.config.output_dir))
        video_file = output_dir / f"video_{int(time.time())}.mp4"
        
        result = await video_generator.create_video_from_script(
            script=script_content,
            output_file=video_file,
            style=VideoStyle(video_config.get('style', 'slideshow')),
            voice=video_config.get('voice'),
            provider=video_config.get('tts_provider'),
            background_color=video_config.get('background_color', '#1a1a1a'),
            text_color=video_config.get('text_color', '#ffffff')
        )
        
        return {
            "video_file": str(result.video_file),
            "duration": result.duration,
            "file_size": result.file_size,
            "metadata": result.metadata
        }

    async def _execute_optimize_seo(
        self,
        step: WorkflowStep,
        config: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute SEO optimization step."""
        seo_config = {**config, **step.config}
        
        # Get script from previous step
        script_content = None
        for output in previous_outputs.values():
            if isinstance(output, dict) and 'script_content' in output:
                script_content = output['script_content']
                break
        
        if not script_content:
            raise WorkflowError("No script content found for SEO optimization")
        
        optimizer = SEOOptimizer()
        
        # Generate optimized metadata
        metadata = await optimizer.optimize_metadata(
            content=script_content,
            keywords=seo_config.get('keywords'),
            provider=seo_config.get('ai_provider')
        )
        
        return {
            "title": metadata.title,
            "description": metadata.description,
            "tags": metadata.tags,
            "seo_optimized": True
        }

    async def _execute_upload_video(
        self,
        step: WorkflowStep,
        config: Dict[str, Any],
        previous_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute video upload step."""
        if not youtube_uploader:
            raise WorkflowError("YouTube uploader not available")
        
        upload_config = {**config, **step.config}
        
        # Get video file and metadata from previous steps
        video_file = None
        title = upload_config.get('title', 'Generated Video')
        description = upload_config.get('description', '')
        tags = upload_config.get('tags', [])
        
        for output in previous_outputs.values():
            if isinstance(output, dict):
                if 'video_file' in output:
                    video_file = output['video_file']
                if 'title' in output:
                    title = output['title']
                if 'description' in output:
                    description = output['description']
                if 'tags' in output:
                    tags = output['tags']
        
        if not video_file:
            raise WorkflowError("No video file found for upload")
        
        # Create metadata
        metadata = VideoMetadata(
            title=title,
            description=description,
            tags=tags,
            privacy_status=upload_config.get('privacy', PrivacyStatus.PRIVATE.value)
        )
        
        # Upload video
        result = await youtube_uploader.upload_video(
            video_file=Path(video_file),
            metadata=metadata
        )
        
        return {
            "video_id": result.video_id,
            "video_url": result.video_url,
            "upload_status": result.status,
            "privacy_status": result.privacy_status
        }

    def _sort_steps_by_dependencies(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Sort steps by their dependencies using topological sort."""
        step_map = {step.id: step for step in steps}
        sorted_steps = []
        visited = set()
        temp_visited = set()
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise WorkflowError(f"Circular dependency detected involving step: {step_id}")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep_id in step.depends_on:
                    if dep_id in step_map:
                        visit(dep_id)
                    else:
                        raise WorkflowError(f"Dependency not found: {dep_id}")
                
                visited.add(step_id)
                sorted_steps.append(step)
            temp_visited.remove(step_id)
        
        for step in steps:
            if step.id not in visited:
                visit(step.id)
        
        return sorted_steps

    async def _validate_workflow(self, template: WorkflowTemplate):
        """Validate workflow template."""
        if not template.steps:
            raise ValidationError("Workflow must have at least one step")
        
        step_ids = {step.id for step in template.steps}
        
        # Check for duplicate step IDs
        if len(step_ids) != len(template.steps):
            raise ValidationError("Duplicate step IDs found")
        
        # Check dependencies
        for step in template.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValidationError(f"Step {step.id} depends on non-existent step: {dep_id}")
        
        # Check for cycles
        try:
            self._sort_steps_by_dependencies(template.steps)
        except WorkflowError as e:
            raise ValidationError(f"Workflow validation failed: {e}")

    async def _save_execution_state(self, execution: WorkflowExecution):
        """Save execution state to disk."""
        execution_file = self.execution_dir / f"{execution.execution_id}.json"
        
        # Convert datetime objects to strings for JSON serialization
        execution_dict = asdict(execution)
        execution_dict['started_at'] = execution.started_at.isoformat()
        if execution.completed_at:
            execution_dict['completed_at'] = execution.completed_at.isoformat()
        execution_dict['status'] = execution.status.value
        
        with open(execution_file, 'w') as f:
            json.dump(execution_dict, f, indent=2)

    async def get_execution_status(self, execution_id: str) -> WorkflowExecution:
        """Get the status of a workflow execution."""
        execution_file = self.execution_dir / f"{execution_id}.json"
        
        if not execution_file.exists():
            raise WorkflowError(f"Execution not found: {execution_id}")
        
        with open(execution_file, 'r') as f:
            data = json.load(f)
        
        # Convert back to WorkflowExecution
        execution = WorkflowExecution(
            workflow_id=data['workflow_id'],
            execution_id=data['execution_id'],
            status=WorkflowStatus(data['status']),
            started_at=datetime.fromisoformat(data['started_at']),
            completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None,
            current_step=data['current_step'],
            completed_steps=data['completed_steps'],
            failed_steps=data['failed_steps'],
            outputs=data['outputs'],
            error_message=data['error_message']
        )
        
        return execution

    def list_workflow_templates(self) -> List[Dict[str, Any]]:
        """List all available workflow templates."""
        templates = []
        
        for template_file in self.workflow_dir.glob("*.yml"):
            try:
                with open(template_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                templates.append({
                    "id": data['id'],
                    "name": data['name'],
                    "description": data['description'],
                    "version": data['version'],
                    "steps": len(data['steps']),
                    "tags": data.get('tags', []),
                    "created_at": data['created_at']
                })
            except Exception as e:
                logger.warning(f"Error loading template {template_file}: {e}")
        
        return templates

    def list_executions(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow executions."""
        executions = []
        
        for execution_file in self.execution_dir.glob("*.json"):
            try:
                with open(execution_file, 'r') as f:
                    data = json.load(f)
                
                if workflow_id and data['workflow_id'] != workflow_id:
                    continue
                
                executions.append({
                    "execution_id": data['execution_id'],
                    "workflow_id": data['workflow_id'],
                    "status": data['status'],
                    "started_at": data['started_at'],
                    "completed_at": data['completed_at'],
                    "completed_steps": len(data['completed_steps']),
                    "failed_steps": len(data['failed_steps'])
                })
            except Exception as e:
                logger.warning(f"Error loading execution {execution_file}: {e}")
        
        return sorted(executions, key=lambda x: x['started_at'], reverse=True)


# Global workflow manager instance
workflow_manager = WorkflowManager()