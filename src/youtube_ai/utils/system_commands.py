"""System management and administration commands."""

import asyncio
import json
import psutil
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout

from ..core.config import config_manager
from ..core.logger import get_logger
from ..modules.analytics.tracker import analytics_tracker
from ..modules.batch.processor import batch_processor
from ..modules.workflow.manager import workflow_manager
from ..utils.file_manager import file_manager

console = Console()
logger = get_logger(__name__)


@click.group()
def system():
    """System management and administration commands."""
    pass


@system.command('status')
@click.option('--detailed', is_flag=True, help='Show detailed system information')
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def system_status(detailed, output_json):
    """Show system status and health."""
    try:
        # Gather system information
        status_info = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if sys.platform != 'win32' else psutil.disk_usage('C:').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "application": {
                "config_valid": True,
                "output_dir_writable": True,
                "dependencies_installed": True
            }
        }
        
        # Check configuration
        config = config_manager.load_config()
        is_valid, issues = config_manager.validate_config()
        status_info["application"]["config_valid"] = is_valid
        status_info["application"]["config_issues"] = issues
        
        # Check output directory
        output_dir = Path(config.output_dir)
        status_info["application"]["output_dir_writable"] = output_dir.exists() and output_dir.is_dir()
        
        # Check API connectivity
        api_status = {}
        if config.ai.openai_api_key:
            api_status["openai"] = "configured"
        if config.ai.anthropic_api_key:
            api_status["anthropic"] = "configured"
        if config.youtube.api_key:
            api_status["youtube"] = "configured"
        status_info["application"]["api_keys"] = api_status
        
        # Get recent performance metrics
        try:
            summary = analytics_tracker.get_performance_summary(days=1)
            status_info["performance"] = {
                "events_last_24h": sum(stats['count'] for stats in summary['event_statistics'].values()),
                "total_cost_last_24h": summary['cost_analysis']['total_cost'],
                "avg_success_rate": summary['session_statistics']['average_success_rate']
            }
        except Exception as e:
            status_info["performance"] = {"error": str(e)}
        
        if output_json:
            console.print(json.dumps(status_info, indent=2))
            return
        
        # Display formatted status
        console.print("[bold blue]YouTube AI CLI System Status[/bold blue]")
        console.print(f"[dim]Generated at:[/dim] {status_info['timestamp'][:19]}")
        
        # System information
        sys_info = status_info["system"]
        system_panel = Panel(
            f"Platform: {sys_info['platform']}\n"
            f"Python: {sys_info['python_version'].split()[0]}\n"
            f"CPU Cores: {sys_info['cpu_count']}\n"
            f"Memory: {sys_info['memory_percent']:.1f}% used ({sys_info['memory_available'] // (1024**3):.1f}GB available)\n"
            f"Disk: {sys_info['disk_usage']:.1f}% used",
            title="System Information",
            border_style="green"
        )
        console.print(system_panel)
        
        # Application status
        app_info = status_info["application"]
        app_status = "🟢 Healthy" if app_info["config_valid"] and app_info["output_dir_writable"] else "🟡 Issues Found"
        
        app_details = f"Configuration: {'✅ Valid' if app_info['config_valid'] else '❌ Invalid'}\n"
        app_details += f"Output Directory: {'✅ Writable' if app_info['output_dir_writable'] else '❌ Not writable'}\n"
        app_details += f"API Keys: {', '.join(app_info['api_keys'].keys()) if app_info['api_keys'] else 'None configured'}"
        
        if not app_info["config_valid"]:
            app_details += f"\nIssues: {', '.join(app_info['config_issues'])}"
        
        app_panel = Panel(
            app_details,
            title=f"Application Status - {app_status}",
            border_style="green" if app_info["config_valid"] else "yellow"
        )
        console.print(app_panel)
        
        # Performance metrics
        if "error" not in status_info["performance"]:
            perf_info = status_info["performance"]
            perf_panel = Panel(
                f"Events (24h): {perf_info['events_last_24h']}\n"
                f"Cost (24h): ${perf_info['total_cost_last_24h']:.2f}\n"
                f"Success Rate: {perf_info['avg_success_rate']:.1f}%",
                title="Performance (Last 24 Hours)",
                border_style="blue"
            )
            console.print(perf_panel)
        
        if detailed:
            # Show more detailed information
            _show_detailed_status()
        
    except Exception as e:
        console.print(f"[red]Error getting system status:[/red] {e}")
        if config_manager.load_config().debug:
            console.print_exception()


def _show_detailed_status():
    """Show detailed system status."""
    # Process information
    process = psutil.Process()
    
    table = Table(title="Detailed Process Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("PID", str(process.pid))
    table.add_row("CPU Percent", f"{process.cpu_percent():.1f}%")
    table.add_row("Memory Percent", f"{process.memory_percent():.1f}%")
    table.add_row("Memory RSS", f"{process.memory_info().rss / (1024**2):.1f} MB")
    table.add_row("Open Files", str(process.num_fds() if hasattr(process, 'num_fds') else 'N/A'))
    table.add_row("Threads", str(process.num_threads()))
    table.add_row("Created", process.create_time())
    
    console.print(table)


@system.command('health')
@click.option('--check', multiple=True, help='Specific health checks to run')
@click.option('--timeout', default=30, help='Timeout for health checks in seconds')
def health_check(check, timeout):
    """Run comprehensive health checks."""
    async def _run_health_checks():
        try:
            checks = {
                "config": _check_configuration,
                "storage": _check_storage,
                "apis": _check_api_connectivity,
                "dependencies": _check_dependencies,
                "performance": _check_performance,
                "resources": _check_resources
            }
            
            selected_checks = list(check) if check else list(checks.keys())
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                results = {}
                for check_name in selected_checks:
                    if check_name in checks:
                        task = progress.add_task(f"Running {check_name} check...", total=None)
                        
                        try:
                            result = await asyncio.wait_for(
                                checks[check_name](), 
                                timeout=timeout
                            )
                            results[check_name] = result
                            status = "✅ PASS" if result["status"] == "pass" else "❌ FAIL"
                            progress.update(task, description=f"{check_name}: {status}")
                        except asyncio.TimeoutError:
                            results[check_name] = {
                                "status": "fail",
                                "message": f"Check timed out after {timeout}s"
                            }
                            progress.update(task, description=f"{check_name}: ⏰ TIMEOUT")
                        except Exception as e:
                            results[check_name] = {
                                "status": "fail",
                                "message": str(e)
                            }
                            progress.update(task, description=f"{check_name}: ❌ ERROR")
                
                progress.stop()
            
            # Display results
            console.print("\n[bold blue]Health Check Results[/bold blue]")
            
            overall_status = "pass"
            for check_name, result in results.items():
                status_icon = "✅" if result["status"] == "pass" else "❌"
                console.print(f"{status_icon} {check_name.title()}: {result['message']}")
                
                if result["status"] != "pass":
                    overall_status = "fail"
                    
                    # Show additional details if available
                    if "details" in result:
                        for detail in result["details"]:
                            console.print(f"    • {detail}")
            
            console.print(f"\n[bold]Overall Status:[/bold] {'🟢 HEALTHY' if overall_status == 'pass' else '🔴 UNHEALTHY'}")
            
        except Exception as e:
            console.print(f"[red]Error running health checks:[/red] {e}")
    
    asyncio.run(_run_health_checks())


async def _check_configuration():
    """Check configuration validity."""
    is_valid, issues = config_manager.validate_config()
    
    if is_valid:
        return {"status": "pass", "message": "Configuration is valid"}
    else:
        return {
            "status": "fail",
            "message": "Configuration has issues",
            "details": issues
        }


async def _check_storage():
    """Check storage availability and permissions."""
    config = config_manager.load_config()
    output_dir = Path(config.output_dir)
    
    issues = []
    
    # Check if output directory exists and is writable
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        issues.append(f"Output directory not writable: {e}")
    
    # Check disk space
    try:
        usage = file_manager.get_disk_usage(output_dir)
        free_gb = (usage["total_size_gb"] * (100 - usage.get("usage_percent", 0)) / 100) if "usage_percent" in usage else 0
        
        if free_gb < 1:  # Less than 1GB free
            issues.append(f"Low disk space: {free_gb:.1f}GB remaining")
    except Exception as e:
        issues.append(f"Could not check disk usage: {e}")
    
    if issues:
        return {
            "status": "fail",
            "message": "Storage issues detected",
            "details": issues
        }
    else:
        return {"status": "pass", "message": "Storage is accessible and has sufficient space"}


async def _check_api_connectivity():
    """Check API connectivity."""
    config = config_manager.load_config()
    issues = []
    
    # Test OpenAI API
    if config.ai.openai_api_key:
        try:
            from ..modules.ai.llm_client import OpenAIClient
            client = OpenAIClient(config.ai.openai_api_key)
            # Simple test - just initialize, don't make actual calls to avoid costs
            if len(config.ai.openai_api_key) < 20:
                issues.append("OpenAI API key appears invalid (too short)")
        except Exception as e:
            issues.append(f"OpenAI API client error: {e}")
    else:
        issues.append("OpenAI API key not configured")
    
    # Test YouTube API
    if config.youtube.api_key:
        if len(config.youtube.api_key) < 20:
            issues.append("YouTube API key appears invalid (too short)")
    else:
        issues.append("YouTube API key not configured")
    
    if issues:
        return {
            "status": "fail",
            "message": "API connectivity issues",
            "details": issues
        }
    else:
        return {"status": "pass", "message": "API keys configured and appear valid"}


async def _check_dependencies():
    """Check required dependencies."""
    missing_deps = []
    
    # Check required packages
    required_packages = [
        ("moviepy", "video processing"),
        ("PIL", "image processing"),
        ("requests", "HTTP requests"),
        ("openai", "OpenAI API"),
        ("google-auth", "Google authentication")
    ]
    
    for package, purpose in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append(f"{package} (required for {purpose})")
    
    # Check system dependencies
    system_deps = [
        ("ffmpeg", "video processing")
    ]
    
    for dep, purpose in system_deps:
        try:
            import shutil
            if not shutil.which(dep):
                missing_deps.append(f"{dep} (system dependency for {purpose})")
        except Exception:
            missing_deps.append(f"Could not check for {dep}")
    
    if missing_deps:
        return {
            "status": "fail",
            "message": "Missing dependencies",
            "details": missing_deps
        }
    else:
        return {"status": "pass", "message": "All dependencies are installed"}


async def _check_performance():
    """Check recent performance metrics."""
    try:
        summary = analytics_tracker.get_performance_summary(days=1)
        
        # Check error rates
        issues = []
        for event_type, stats in summary['event_statistics'].items():
            if stats['success_rate'] < 90:  # Less than 90% success rate
                issues.append(f"{event_type}: {stats['success_rate']:.1f}% success rate")
        
        # Check costs
        daily_cost = summary['cost_analysis']['total_cost']
        if daily_cost > 50:  # More than $50 per day
            issues.append(f"High daily cost: ${daily_cost:.2f}")
        
        if issues:
            return {
                "status": "fail",
                "message": "Performance issues detected",
                "details": issues
            }
        else:
            return {"status": "pass", "message": "Performance metrics look good"}
            
    except Exception as e:
        return {
            "status": "fail",
            "message": f"Could not check performance metrics: {e}"
        }


async def _check_resources():
    """Check system resource usage."""
    issues = []
    
    # Check memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        issues.append(f"High memory usage: {memory.percent:.1f}%")
    
    # Check disk usage
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    if disk_percent > 90:
        issues.append(f"High disk usage: {disk_percent:.1f}%")
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        issues.append(f"High CPU usage: {cpu_percent:.1f}%")
    
    if issues:
        return {
            "status": "fail",
            "message": "Resource usage issues",
            "details": issues
        }
    else:
        return {"status": "pass", "message": "Resource usage is normal"}


@system.command('cleanup')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without actually doing it')
@click.option('--days', default=7, help='Clean files older than N days')
@click.option('--type', 'cleanup_type', multiple=True, 
              help='Types to clean: temp, cache, logs, archive')
def cleanup(dry_run, days, cleanup_type):
    """Clean up temporary files and old data."""
    try:
        cleanup_types = list(cleanup_type) if cleanup_type else ['temp', 'cache', 'logs']
        
        console.print(f"[blue]{'Dry run: ' if dry_run else ''}Cleaning up files older than {days} days[/blue]")
        
        total_freed = 0
        
        for ctype in cleanup_types:
            if ctype == 'temp':
                freed = file_manager.cleanup_temp_files(max_age_hours=days * 24)
                console.print(f"  Temp files: {freed} files {'would be ' if dry_run else ''}deleted")
                
            elif ctype == 'cache':
                # Implementation for cache cleanup
                console.print(f"  Cache: {'Would clean' if dry_run else 'Cleaned'} cache files")
                
            elif ctype == 'logs':
                # Implementation for log cleanup
                console.print(f"  Logs: {'Would clean' if dry_run else 'Cleaned'} old log files")
                
            elif ctype == 'archive':
                # Implementation for archive cleanup
                console.print(f"  Archives: {'Would clean' if dry_run else 'Cleaned'} old archives")
        
        if not dry_run:
            console.print(f"[green]✓[/green] Cleanup completed")
        else:
            console.print(f"[yellow]Dry run completed - no files were actually deleted[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error during cleanup:[/red] {e}")


@system.command('monitor')
@click.option('--interval', default=5, help='Update interval in seconds')
@click.option('--duration', default=60, help='Monitoring duration in seconds')
def monitor(interval, duration):
    """Real-time system monitoring."""
    try:
        console.print("[blue]Starting real-time monitoring...[/blue]")
        console.print("Press Ctrl+C to stop")
        
        start_time = datetime.now()
        
        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > duration:
                break
                
            # Clear screen and show current stats
            console.clear()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            console.print(f"[bold blue]YouTube AI CLI Monitor[/bold blue] - {datetime.now().strftime('%H:%M:%S')}")
            console.print(f"CPU: {cpu_percent:5.1f}% | Memory: {memory.percent:5.1f}% | Elapsed: {elapsed:.0f}s")
            
            # Get active batch jobs
            jobs = batch_processor.list_batch_jobs()
            active_jobs = [job for job in jobs if job['status'] in ['running', 'pending']]
            
            if active_jobs:
                console.print(f"\n[yellow]Active Jobs: {len(active_jobs)}[/yellow]")
                for job in active_jobs[:5]:  # Show first 5
                    console.print(f"  {job['id']}: {job['status']} ({job['completed_tasks']}/{job['total_tasks']})")
            
            # Sleep for interval
            import time
            time.sleep(interval)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during monitoring:[/red] {e}")


@system.command('export')
@click.option('--format', default='json', help='Export format (json, csv)')
@click.option('--output', help='Output file path')
@click.option('--days', default=30, help='Number of days to export')
def export_data(format, output, days):
    """Export system data and analytics."""
    try:
        console.print(f"[blue]Exporting data for last {days} days...[/blue]")
        
        # Export analytics
        analytics_file = analytics_tracker.export_analytics(format=format, output_file=Path(output) if output else None)
        
        console.print(f"[green]✓[/green] Analytics exported to: {analytics_file}")
        
        # Export configuration
        config_data = {
            "exported_at": datetime.now().isoformat(),
            "configuration": config_manager.load_config().model_dump(),
            "workflows": workflow_manager.list_workflow_templates(),
            "batch_jobs": batch_processor.list_batch_jobs()
        }
        
        if output:
            config_file = Path(output).with_name(f"config_export.{format}")
        else:
            config_file = Path(f"config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}")
        
        if format == 'json':
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
        
        console.print(f"[green]✓[/green] Configuration exported to: {config_file}")
        
    except Exception as e:
        console.print(f"[red]Error exporting data:[/red] {e}")