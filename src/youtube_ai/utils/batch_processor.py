"""Batch processing system for YouTube AI CLI operations."""

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from youtube_ai.core.config import config_manager
from youtube_ai.core.logger import get_logger
from youtube_ai.core.exceptions import WorkflowError, ValidationError
from youtube_ai.utils.workflow_manager import workflow_manager, WorkflowExecution
from youtube_ai.utils.analytics_tracker import analytics_tracker, EventType, MetricType
from youtube_ai.utils.file_manager import file_manager

logger = get_logger(__name__)


class BatchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchTask:
    """Individual task in a batch operation."""
    id: str
    inputs: Dict[str, Any]
    workflow_template: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    outputs: Dict[str, Any] = None
    execution_id: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


@dataclass
class BatchJob:
    """Batch job configuration and status."""
    id: str
    name: str
    workflow_template: str
    tasks: List[BatchTask]
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None
    max_concurrent: int = 3
    retry_failed: bool = True
    max_retries: int = 2
    continue_on_error: bool = True
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BatchProgress:
    """Batch processing progress information."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    running_tasks: int
    pending_tasks: int
    percentage: float
    estimated_remaining_time: Optional[float] = None
    current_task_ids: List[str] = None

    def __post_init__(self):
        if self.current_task_ids is None:
            self.current_task_ids = []


class BatchProcessor:
    """Processes multiple tasks in parallel using workflows."""

    def __init__(self):
        self.config = config_manager.load_config()
        self.batch_dir = Path.home() / ".youtube-ai" / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_jobs: Dict[str, BatchJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)

    async def create_batch_from_csv(
        self,
        csv_file: Path,
        workflow_template: str,
        name: Optional[str] = None,
        max_concurrent: int = 3
    ) -> BatchJob:
        """Create a batch job from CSV file."""
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        tasks = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader, 1):
                    # Clean and validate row data
                    inputs = {k.strip(): v.strip() for k, v in row.items() if v.strip()}
                    
                    if not inputs.get('topic'):
                        logger.warning(f"Row {i}: Missing required 'topic' field, skipping")
                        continue
                    
                    task = BatchTask(
                        id=f"task_{i:04d}",
                        inputs=inputs,
                        workflow_template=workflow_template
                    )
                    tasks.append(task)
            
            if not tasks:
                raise ValidationError("No valid tasks found in CSV file")
            
            # Generate job ID and name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"batch_{timestamp}"
            job_name = name or f"Batch job from {csv_file.name}"
            
            batch_job = BatchJob(
                id=job_id,
                name=job_name,
                workflow_template=workflow_template,
                tasks=tasks,
                max_concurrent=max_concurrent,
                output_dir=self.batch_dir / job_id
            )
            
            # Save job configuration
            await self._save_batch_job(batch_job)
            
            logger.info(f"Created batch job '{job_name}' with {len(tasks)} tasks")
            return batch_job
            
        except Exception as e:
            logger.error(f"Error creating batch from CSV: {e}")
            raise

    async def create_batch_from_topics(
        self,
        topics: List[str],
        workflow_template: str,
        name: Optional[str] = None,
        base_config: Dict[str, Any] = None,
        max_concurrent: int = 3
    ) -> BatchJob:
        """Create a batch job from a list of topics."""
        if not topics:
            raise ValidationError("No topics provided")
        
        tasks = []
        base_inputs = base_config or {}
        
        for i, topic in enumerate(topics, 1):
            inputs = {**base_inputs, "topic": topic.strip()}
            
            task = BatchTask(
                id=f"task_{i:04d}",
                inputs=inputs,
                workflow_template=workflow_template
            )
            tasks.append(task)
        
        # Generate job ID and name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"batch_{timestamp}"
        job_name = name or f"Batch job - {len(topics)} topics"
        
        batch_job = BatchJob(
            id=job_id,
            name=job_name,
            workflow_template=workflow_template,
            tasks=tasks,
            max_concurrent=max_concurrent,
            output_dir=self.batch_dir / job_id
        )
        
        # Save job configuration
        await self._save_batch_job(batch_job)
        
        logger.info(f"Created batch job '{job_name}' with {len(tasks)} tasks")
        return batch_job

    async def execute_batch(
        self,
        batch_job: BatchJob,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchJob:
        """Execute a batch job."""
        batch_job.status = BatchStatus.RUNNING
        batch_job.started_at = datetime.now()
        
        # Create output directory
        if batch_job.output_dir:
            batch_job.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track batch execution
        analytics_tracker.track_event(
            EventType.WORKFLOW_COMPLETED,
            metadata={
                "batch_id": batch_job.id,
                "total_tasks": len(batch_job.tasks),
                "workflow": batch_job.workflow_template
            }
        )
        
        try:
            self.active_jobs[batch_job.id] = batch_job
            
            # Execute tasks with concurrency control
            semaphore = asyncio.Semaphore(batch_job.max_concurrent)
            
            async def execute_task_with_semaphore(task: BatchTask) -> BatchTask:
                async with semaphore:
                    return await self._execute_single_task(batch_job, task)
            
            # Create task coroutines
            task_coroutines = [
                execute_task_with_semaphore(task) for task in batch_job.tasks
            ]
            
            # Execute tasks and track progress
            completed_tasks = []
            
            for future in asyncio.as_completed(task_coroutines):
                try:
                    completed_task = await future
                    completed_tasks.append(completed_task)
                    
                    # Update progress
                    if progress_callback:
                        progress = self._calculate_progress(batch_job)
                        progress_callback(progress)
                    
                    # Save progress
                    await self._save_batch_job(batch_job)
                    
                except Exception as e:
                    logger.error(f"Task execution error: {e}")
                    if not batch_job.continue_on_error:
                        break
            
            # Calculate final statistics
            successful_tasks = sum(1 for task in batch_job.tasks if task.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for task in batch_job.tasks if task.status == TaskStatus.FAILED)
            
            batch_job.status = BatchStatus.COMPLETED if failed_tasks == 0 else BatchStatus.FAILED
            batch_job.completed_at = datetime.now()
            batch_job.total_duration = (batch_job.completed_at - batch_job.started_at).total_seconds()
            
            # Generate summary report
            await self._generate_batch_report(batch_job)
            
            logger.info(f"Batch job completed: {successful_tasks} successful, {failed_tasks} failed")
            
        except Exception as e:
            batch_job.status = BatchStatus.FAILED
            batch_job.completed_at = datetime.now()
            logger.error(f"Batch job failed: {e}")
            raise
        
        finally:
            # Clean up
            if batch_job.id in self.active_jobs:
                del self.active_jobs[batch_job.id]
            
            await self._save_batch_job(batch_job)
        
        return batch_job

    async def _execute_single_task(self, batch_job: BatchJob, task: BatchTask) -> BatchTask:
        """Execute a single task within a batch."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            logger.info(f"Executing task {task.id}: {task.inputs.get('topic', 'Unknown topic')}")
            
            # Add batch-specific configuration
            execution_config = {
                "output_dir": str(batch_job.output_dir / task.id) if batch_job.output_dir else None,
                "batch_mode": True,
                "task_id": task.id
            }
            
            # Execute workflow
            execution = await workflow_manager.execute_workflow(
                template_id=batch_job.workflow_template,
                inputs=task.inputs,
                execution_config=execution_config
            )
            
            task.execution_id = execution.execution_id
            
            if execution.status.value == "completed":
                task.status = TaskStatus.COMPLETED
                task.outputs = execution.outputs
                logger.info(f"Task {task.id} completed successfully")
            else:
                task.status = TaskStatus.FAILED
                task.error_message = execution.error_message
                logger.error(f"Task {task.id} failed: {execution.error_message}")
                
                # Retry if configured
                if batch_job.retry_failed and task.retry_count < batch_job.max_retries:
                    task.retry_count += 1
                    logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1})")
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    return await self._execute_single_task(batch_job, task)
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            logger.error(f"Task {task.id} execution error: {e}")
        
        finally:
            task.end_time = datetime.now()
            if task.start_time:
                task.duration = (task.end_time - task.start_time).total_seconds()
        
        return task

    def _calculate_progress(self, batch_job: BatchJob) -> BatchProgress:
        """Calculate current progress of batch job."""
        total = len(batch_job.tasks)
        completed = sum(1 for task in batch_job.tasks if task.status == TaskStatus.COMPLETED)
        failed = sum(1 for task in batch_job.tasks if task.status == TaskStatus.FAILED)
        running = sum(1 for task in batch_job.tasks if task.status == TaskStatus.RUNNING)
        pending = sum(1 for task in batch_job.tasks if task.status == TaskStatus.PENDING)
        
        percentage = ((completed + failed) / total) * 100 if total > 0 else 0
        
        # Estimate remaining time
        estimated_remaining = None
        if completed > 0 and batch_job.started_at:
            elapsed = (datetime.now() - batch_job.started_at).total_seconds()
            avg_time_per_task = elapsed / (completed + failed)
            estimated_remaining = avg_time_per_task * (pending + running)
        
        # Get currently running task IDs
        current_task_ids = [task.id for task in batch_job.tasks if task.status == TaskStatus.RUNNING]
        
        return BatchProgress(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            running_tasks=running,
            pending_tasks=pending,
            percentage=percentage,
            estimated_remaining_time=estimated_remaining,
            current_task_ids=current_task_ids
        )

    async def _save_batch_job(self, batch_job: BatchJob):
        """Save batch job state to disk."""
        job_file = self.batch_dir / f"{batch_job.id}.json"
        
        # Convert to serializable format
        job_data = asdict(batch_job)
        
        # Convert datetime objects to strings
        for key in ['created_at', 'started_at', 'completed_at']:
            if job_data[key]:
                job_data[key] = job_data[key].isoformat()
        
        # Convert task datetime objects
        for task_data in job_data['tasks']:
            for key in ['start_time', 'end_time']:
                if task_data[key]:
                    task_data[key] = task_data[key].isoformat()
            task_data['status'] = task_data['status'].value if hasattr(task_data['status'], 'value') else task_data['status']
        
        job_data['status'] = job_data['status'].value if hasattr(job_data['status'], 'value') else job_data['status']
        
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)

    async def load_batch_job(self, job_id: str) -> Optional[BatchJob]:
        """Load batch job from disk."""
        job_file = self.batch_dir / f"{job_id}.json"
        
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            # Convert datetime strings back to objects
            for key in ['created_at', 'started_at', 'completed_at']:
                if job_data[key]:
                    job_data[key] = datetime.fromisoformat(job_data[key])
            
            # Convert task datetime strings and enums
            tasks = []
            for task_data in job_data['tasks']:
                for key in ['start_time', 'end_time']:
                    if task_data[key]:
                        task_data[key] = datetime.fromisoformat(task_data[key])
                
                task_data['status'] = TaskStatus(task_data['status'])
                tasks.append(BatchTask(**task_data))
            
            job_data['tasks'] = tasks
            job_data['status'] = BatchStatus(job_data['status'])
            
            if job_data.get('output_dir'):
                job_data['output_dir'] = Path(job_data['output_dir'])
            
            return BatchJob(**job_data)
            
        except Exception as e:
            logger.error(f"Error loading batch job {job_id}: {e}")
            return None

    async def _generate_batch_report(self, batch_job: BatchJob):
        """Generate a comprehensive batch execution report."""
        if not batch_job.output_dir:
            return
        
        # Calculate statistics
        total_tasks = len(batch_job.tasks)
        completed_tasks = sum(1 for task in batch_job.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in batch_job.tasks if task.status == TaskStatus.FAILED)
        success_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate timing statistics
        task_durations = [task.duration for task in batch_job.tasks if task.duration]
        avg_duration = sum(task_durations) / len(task_durations) if task_durations else 0
        
        report = {
            "batch_id": batch_job.id,
            "batch_name": batch_job.name,
            "workflow_template": batch_job.workflow_template,
            "execution_summary": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": success_rate,
                "total_duration": batch_job.total_duration,
                "average_task_duration": avg_duration
            },
            "timing": {
                "created_at": batch_job.created_at.isoformat(),
                "started_at": batch_job.started_at.isoformat() if batch_job.started_at else None,
                "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None
            },
            "configuration": {
                "max_concurrent": batch_job.max_concurrent,
                "retry_failed": batch_job.retry_failed,
                "max_retries": batch_job.max_retries,
                "continue_on_error": batch_job.continue_on_error
            },
            "tasks": []
        }
        
        # Add task details
        for task in batch_job.tasks:
            task_info = {
                "id": task.id,
                "status": task.status.value,
                "inputs": task.inputs,
                "duration": task.duration,
                "retry_count": task.retry_count
            }
            
            if task.error_message:
                task_info["error_message"] = task.error_message
            
            if task.outputs:
                # Include file paths but not full content
                task_info["outputs"] = {
                    k: v for k, v in task.outputs.items() 
                    if isinstance(v, (str, int, float, bool)) or k.endswith('_file')
                }
            
            report["tasks"].append(task_info)
        
        # Save report
        report_file = batch_job.output_dir / "batch_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create CSV summary
        csv_file = batch_job.output_dir / "batch_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['task_id', 'status', 'topic', 'duration', 'error_message']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for task in batch_job.tasks:
                writer.writerow({
                    'task_id': task.id,
                    'status': task.status.value,
                    'topic': task.inputs.get('topic', ''),
                    'duration': task.duration,
                    'error_message': task.error_message or ''
                })
        
        logger.info(f"Generated batch report: {report_file}")

    def list_batch_jobs(self) -> List[Dict[str, Any]]:
        """List all batch jobs."""
        jobs = []
        
        for job_file in self.batch_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                
                # Extract summary information
                total_tasks = len(job_data['tasks'])
                completed_tasks = sum(1 for task in job_data['tasks'] if task['status'] == 'completed')
                failed_tasks = sum(1 for task in job_data['tasks'] if task['status'] == 'failed')
                
                job_summary = {
                    "id": job_data['id'],
                    "name": job_data['name'],
                    "status": job_data['status'],
                    "workflow_template": job_data['workflow_template'],
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "created_at": job_data['created_at'],
                    "started_at": job_data.get('started_at'),
                    "completed_at": job_data.get('completed_at'),
                    "total_duration": job_data.get('total_duration')
                }
                
                jobs.append(job_summary)
                
            except Exception as e:
                logger.warning(f"Error loading job summary from {job_file}: {e}")
        
        return sorted(jobs, key=lambda x: x['created_at'], reverse=True)

    async def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        if job_id in self.active_jobs:
            batch_job = self.active_jobs[job_id]
            batch_job.status = BatchStatus.CANCELLED
            
            # Cancel pending tasks
            for task in batch_job.tasks:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SKIPPED
            
            await self._save_batch_job(batch_job)
            logger.info(f"Cancelled batch job: {job_id}")
            return True
        
        return False


# Global batch processor instance
batch_processor = BatchProcessor()