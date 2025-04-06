"""
Distributed processing framework using Dask.

This module provides a framework for distributed processing of large datasets
using Dask, with support for clusters, task tracking, and checkpoint management.
"""

import os
import time
import logging
import traceback
import numpy as np
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional, Iterator, Set
from dataclasses import dataclass
import contextlib
import atexit

# Dask-specific imports
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster, progress, wait, as_completed
from dask.diagnostics import ProgressBar
import distributed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DistributedTask:
    """Task container for distributed processing."""
    id: int
    params: Dict[str, Any]
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "params": self.params,
            "retries": self.retries,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            params=data["params"],
            retries=data["retries"],
            max_retries=data["max_retries"]
        )


class DistributedCheckpointManager:
    """Manages checkpoints for restartable processing in a distributed environment."""
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.completed_tasks: Set[int] = set()
        self.failed_tasks: Dict[int, str] = {}
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Register cleanup handler
        atexit.register(self._save_checkpoint)
    
    def _load_checkpoint(self) -> None:
        """Load checkpoint from file if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.completed_tasks = set(data.get("completed_tasks", []))
                    self.failed_tasks = {int(k): v for k, v in data.get("failed_tasks", {}).items()}
                logger.info(f"Loaded checkpoint with {len(self.completed_tasks)} completed tasks and {len(self.failed_tasks)} failed tasks")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint to file."""
        try:
            os.makedirs(self.checkpoint_file.parent, exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    "completed_tasks": list(self.completed_tasks),
                    "failed_tasks": self.failed_tasks
                }, f)
            logger.info(f"Saved checkpoint with {len(self.completed_tasks)} completed tasks")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def mark_completed(self, task_id: int) -> None:
        """Mark task as completed."""
        self.completed_tasks.add(task_id)
        if task_id in self.failed_tasks:
            del self.failed_tasks[task_id]
        
        # Periodically save checkpoint
        if len(self.completed_tasks) % 10 == 0:
            self._save_checkpoint()
    
    def mark_failed(self, task_id: int, error: str) -> None:
        """Mark task as failed."""
        self.failed_tasks[task_id] = error
        
        # Save checkpoint on each failure
        self._save_checkpoint()
    
    def is_completed(self, task_id: int) -> bool:
        """Check if task is completed."""
        return task_id in self.completed_tasks
    
    def get_failed_tasks(self) -> Dict[int, str]:
        """Get all failed tasks."""
        return self.failed_tasks.copy()


class DaskTaskManager:
    """Manages distributed task generation and tracking."""
    def __init__(
        self, 
        task_generator: Callable[[], Iterator[Dict[str, Any]]],
        checkpoint_file: str = "dask_checkpoint.json"
    ):
        self.task_generator = task_generator
        self.checkpoint = DistributedCheckpointManager(checkpoint_file)
        self.total_tasks = 0
        self.completed_tasks = 0
    
    def generate_tasks(self) -> List[DistributedTask]:
        """Generate tasks from generator function."""
        tasks = []
        for i, params in enumerate(self.task_generator()):
            task = DistributedTask(id=i, params=params)
            if not self.checkpoint.is_completed(task.id):
                tasks.append(task)
            
        self.total_tasks = len(tasks)
        logger.info(f"Generated {self.total_tasks} tasks")
        return tasks
    
    def log_progress(self, completed: int, total: int) -> None:
        """Log progress information."""
        percent = (completed / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {completed}/{total} ({percent:.2f}%)")
        self.completed_tasks = completed


class DaskDistributedFramework:
    """
    Distributed computing framework using Dask for large-scale processing.
    
    This class handles:
    1. Setup of Dask cluster (local or remote)
    2. Task distribution and execution
    3. Error handling with retries
    4. Progress tracking and visualization
    5. Resource monitoring and adaptive scaling
    """
    def __init__(
        self,
        process_func: Callable[[DistributedTask, Dict[str, Any]], Any],
        task_generator: Callable[[], Iterator[Dict[str, Any]]],
        shared_data: Dict[str, Any] = None,
        scheduler_address: str = None,
        n_workers: int = None,
        threads_per_worker: int = 1,
        memory_limit: str = "4GB",
        checkpoint_file: str = "dask_checkpoint.json",
        max_retries: int = 3,
        adaptive_scaling: bool = True,
        worker_timeout: str = "60 minutes",
        resources_per_worker: Dict[str, float] = None,
        dashboard_address: str = ":8787"
    ):
        """
        Initialize the distributed framework.
        
        Args:
            process_func: Function to process a single task
            task_generator: Function that generates task parameters
            shared_data: Data shared across all workers
            scheduler_address: Address of existing Dask scheduler (if None, will create local cluster)
            n_workers: Number of worker processes (for local cluster)
            threads_per_worker: Threads per worker process
            memory_limit: Memory limit per worker
            checkpoint_file: File to store checkpoint data
            max_retries: Maximum number of retries for failed tasks
            adaptive_scaling: Whether to use adaptive scaling
            worker_timeout: Worker timeout duration
            resources_per_worker: Resources available on each worker
            dashboard_address: Address for the Dask dashboard
        """
        self.process_func = process_func
        self.shared_data = shared_data or {}
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers or os.cpu_count()
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.max_retries = max_retries
        self.adaptive_scaling = adaptive_scaling
        self.worker_timeout = worker_timeout
        self.resources_per_worker = resources_per_worker
        self.dashboard_address = dashboard_address
        
        self.client = None
        self.cluster = None
        self.task_manager = DaskTaskManager(task_generator, checkpoint_file)
        
        # Store futures for tracking
        self.futures = []
    
    def _setup_cluster(self) -> distributed.Client:
        """Set up Dask cluster and client."""
        if self.scheduler_address:
            # Connect to existing cluster
            logger.info(f"Connecting to existing Dask cluster at {self.scheduler_address}")
            client = Client(self.scheduler_address)
        else:
            # Create local cluster
            logger.info(f"Creating local Dask cluster with {self.n_workers} workers")
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                dashboard_address=self.dashboard_address,
                resources=self.resources_per_worker
            )
            client = Client(self.cluster)
            
            # Set up adaptive scaling if requested
            if self.adaptive_scaling and self.n_workers > 1:
                logger.info("Setting up adaptive scaling")
                self.cluster.adapt(minimum=1, maximum=self.n_workers)
        
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        return client
    
    def _process_task_wrapper(self, task: DistributedTask, shared_data: Dict[str, Any]) -> Tuple[int, str]:
        """
        Wrapper for task processing to handle errors and retries.
        
        Returns:
            Tuple of (task_id, status)
        """
        task_id = task.id
        try:
            start_time = time.time()
            result = self.process_func(task, shared_data)
            end_time = time.time()
            processing_time = end_time - start_time
            
            return task_id, f"completed:{processing_time:.2f}s"
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error processing task {task_id}: {error_msg}")
            logger.error(traceback.format_exc())
            
            return task_id, f"failed:{error_msg}"
    
    def _handle_task_result(self, task_id: int, status: str) -> bool:
        """
        Handle task result, updating checkpoints and determining if retry is needed.
        
        Returns:
            bool: True if task completed successfully, False otherwise
        """
        if status.startswith("completed:"):
            self.task_manager.checkpoint.mark_completed(task_id)
            processing_time = status.split(":")[1]
            logger.debug(f"Task {task_id} completed in {processing_time}")
            return True
        elif status.startswith("failed:"):
            error_msg = status[7:]  # Remove "failed:" prefix
            self.task_manager.checkpoint.mark_failed(task_id, error_msg)
            return False
        else:
            logger.warning(f"Unknown task status: {status}")
            return False
    
    def process(self) -> Dict[str, Any]:
        """
        Process all tasks using Dask distributed computing.
        
        Returns:
            Dict with statistics about the processing
        """
        logger.info("Starting distributed processing with Dask")
        
        # Set up cluster and client
        self.client = self._setup_cluster()
        
        # Distribute shared data to all workers
        self.client.publish_dataset(shared_data=self.shared_data)
        
        # Generate tasks
        tasks = self.task_manager.generate_tasks()
        total_tasks = len(tasks)
        
        if total_tasks == 0:
            logger.info("No tasks to process, exiting")
            return {
                "status": "completed",
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0
            }
        
        # Submit tasks to Dask
        logger.info(f"Submitting {total_tasks} tasks to Dask")
        futures = []
        
        for task in tasks:
            # Skip already completed tasks
            if self.task_manager.checkpoint.is_completed(task.id):
                continue
                
            # Submit task to Dask
            future = self.client.submit(
                self._process_task_wrapper,
                task,
                self.client.get_dataset("shared_data"),
                retries=self.max_retries,
                key=f"task-{task.id}"
            )
            futures.append(future)
        
        self.futures = futures
        logger.info(f"Submitted {len(futures)} tasks to Dask")
        
        try:
            # Set up progress monitoring
            progress_bar = ProgressBar()
            progress_bar.register()
            
            # Process results as they complete
            completed_count = 0
            failed_count = 0
            for future, result in as_completed(futures, with_results=True, raise_errors=False):
                if future.status == 'error':
                    # Handle task error
                    task_id = int(future.key.split('-')[1])
                    error = future.exception()
                    logger.error(f"Task {task_id} failed with error: {error}")
                    self.task_manager.checkpoint.mark_failed(task_id, str(error))
                    failed_count += 1
                else:
                    # Handle task success
                    task_id, status = result
                    success = self._handle_task_result(task_id, status)
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                
                # Update progress
                if (completed_count + failed_count) % 10 == 0 or (completed_count + failed_count) == total_tasks:
                    percent = ((completed_count + failed_count) / total_tasks) * 100
                    logger.info(f"Progress: {completed_count + failed_count}/{total_tasks} ({percent:.2f}%)")
                    logger.info(f"Completed: {completed_count}, Failed: {failed_count}")
            
            # All tasks have been processed
            progress_bar.unregister()
            
            # Get final counts
            completed_tasks = len(self.task_manager.checkpoint.completed_tasks)
            failed_tasks = self.task_manager.checkpoint.get_failed_tasks()
            
            result = {
                "status": "completed",
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": len(failed_tasks),
                "failed_task_ids": list(failed_tasks.keys())
            }
            
            # Log summary
            logger.info(f"Processing completed: {result['completed_tasks']}/{result['total_tasks']} tasks processed")
            if result["failed_tasks"] > 0:
                logger.warning(f"{result['failed_tasks']} tasks failed")
            
            # Close client and cluster
            if self.cluster:
                self.cluster.close()
            self.client.close()
            
            return result
            
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down")
            
            # Cancel all running tasks
            for future in futures:
                if not future.done():
                    future.cancel()
            
            # Close client and cluster
            if self.cluster:
                self.cluster.close()
            self.client.close()
            
            # Return current progress
            completed_tasks = len(self.task_manager.checkpoint.completed_tasks)
            failed_tasks = self.task_manager.checkpoint.get_failed_tasks()
            
            return {
                "status": "interrupted",
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": len(failed_tasks)
            }
    
    def retry_failed_tasks(self) -> Dict[str, Any]:
        """
        Retry tasks that previously failed.
        
        Returns:
            Dict with statistics about the retry process
        """
        # Set up cluster and client if not already done
        if self.client is None:
            self.client = self._setup_cluster()
            
        # Distribute shared data to all workers
        self.client.publish_dataset(shared_data=self.shared_data)
        
        # Get failed tasks
        failed_tasks = self.task_manager.checkpoint.get_failed_tasks()
        if not failed_tasks:
            logger.info("No failed tasks to retry")
            return {"status": "completed", "retried_tasks": 0}
        
        # Generate task objects for failed tasks
        retry_tasks = []
        tasks = list(self.task_manager.generate_tasks())
        task_dict = {task.id: task for task in tasks}
        
        for task_id in failed_tasks:
            if task_id in task_dict:
                task = task_dict[task_id]
                task.retries = 0  # Reset retry counter
                retry_tasks.append(task)
        
        logger.info(f"Retrying {len(retry_tasks)} failed tasks")
        
        # Submit retry tasks to Dask
        futures = []
        for task in retry_tasks:
            future = self.client.submit(
                self._process_task_wrapper,
                task,
                self.client.get_dataset("shared_data"),
                retries=self.max_retries * 2,  # Double retry count for failed tasks
                key=f"retry-{task.id}"
            )
            futures.append(future)
        
        try:
            # Set up progress monitoring
            progress_bar = ProgressBar()
            progress_bar.register()
            
            # Process results as they complete
            successful_retries = 0
            failed_retries = 0
            
            for future, result in as_completed(futures, with_results=True, raise_errors=False):
                if future.status == 'error':
                    # Handle task error
                    task_id = int(future.key.split('-')[1])
                    error = future.exception()
                    logger.error(f"Retry task {task_id} failed with error: {error}")
                    self.task_manager.checkpoint.mark_failed(task_id, str(error))
                    failed_retries += 1
                else:
                    # Handle task success
                    task_id, status = result
                    success = self._handle_task_result(task_id, status)
                    if success:
                        successful_retries += 1
                    else:
                        failed_retries += 1
                
                # Update progress
                if (successful_retries + failed_retries) % 5 == 0 or (successful_retries + failed_retries) == len(retry_tasks):
                    percent = ((successful_retries + failed_retries) / len(retry_tasks)) * 100
                    logger.info(f"Retry progress: {successful_retries + failed_retries}/{len(retry_tasks)} ({percent:.2f}%)")
                    logger.info(f"Successful: {successful_retries}, Failed: {failed_retries}")
            
            # All retries have been processed
            progress_bar.unregister()
            
            # Get final counts
            still_failed = self.task_manager.checkpoint.get_failed_tasks()
            
            result = {
                "status": "completed",
                "retried_tasks": len(retry_tasks),
                "successful_retries": successful_retries,
                "failed_retries": failed_retries,
                "still_failed_task_ids": list(still_failed.keys())
            }
            
            logger.info(f"Retry completed: {result['successful_retries']}/{result['retried_tasks']} tasks successfully retried")
            if result["failed_retries"] > 0:
                logger.warning(f"{result['failed_retries']} tasks still failing")
            
            # Close client and cluster
            if self.cluster:
                self.cluster.close()
            self.client.close()
            
            return result
            
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down")
            
            # Cancel all running tasks
            for future in futures:
                if not future.done():
                    future.cancel()
            
            # Close client and cluster
            if self.cluster:
                self.cluster.close()
            self.client.close()
            
            return {
                "status": "interrupted",
                "retried_tasks": len(retry_tasks),
                "completed_retries": successful_retries + failed_retries
            }


# Helper functions for distributed processing

def create_dask_array_from_zarr(zarr_path: str, mode: str = "r") -> da.Array:
    """
    Create a Dask array from a Zarr array for distributed processing.
    
    Args:
        zarr_path: Path to the zarr array
        mode: Mode to open the zarr array
        
    Returns:
        Dask array
    """
    import zarr
    z_array = zarr.open(zarr_path, mode=mode)
    d_array = da.from_zarr(zarr_path)
    return d_array


def write_dask_array_to_zarr(dask_array: da.Array, zarr_path: str, compute: bool = True) -> None:
    """
    Write a Dask array to a Zarr array.
    
    Args:
        dask_array: Dask array to write
        zarr_path: Path to write to
        compute: Whether to compute immediately
    """
    dask_array.to_zarr(zarr_path)
    if compute:
        with ProgressBar():
            dask_array.compute()


def setup_dask_slurm_cluster(
    job_name: str = "dask-worker",
    queue: str = "normal",
    project: str = None,
    cores: int = 1,
    memory: str = "4GB",
    walltime: str = "01:00:00",
    n_workers: int = 10,
    adaptive: bool = True,
    minimum_jobs: int = 1,
    maximum_jobs: int = 20,
    dashboard_address: str = ":8787"
) -> Tuple[Any, Client]:
    """
    Set up a Dask cluster on a SLURM cluster.
    
    Args:
        job_name: Name of the SLURM job
        queue: Queue to submit jobs to
        project: Project to charge resources to
        cores: Number of cores per worker
        memory: Memory per worker
        walltime: Wall time limit
        n_workers: Number of workers
        adaptive: Whether to use adaptive scaling
        minimum_jobs: Minimum number of jobs for adaptive scaling
        maximum_jobs: Maximum number of jobs for adaptive scaling
        dashboard_address: Address for the Dask dashboard
        
    Returns:
        Tuple of (cluster, client)
    """
    try:
        from dask_jobqueue import SLURMCluster
    except ImportError:
        raise ImportError("dask_jobqueue is required for this function. Install it with: pip install dask_jobqueue")
    
    # Create SLURM cluster
    cluster = SLURMCluster(
        queue=queue,
        project=project,
        cores=cores,
        memory=memory,
        walltime=walltime,
        name=job_name,
        job_extra=[],
        dashboard_address=dashboard_address
    )
    
    # Set up scaling
    if adaptive:
        cluster.adapt(minimum=minimum_jobs, maximum=maximum_jobs)
    else:
        cluster.scale(n_workers)
    
    # Create client
    client = Client(cluster)
    
    logger.info(f"Dask SLURM cluster set up with {n_workers} workers")
    logger.info(f"Dashboard available at: {client.dashboard_link}")
    
    return cluster, client


# Make classes and functions available at the module level
__all__ = [
    'DistributedTask',
    'DistributedCheckpointManager',
    'DaskTaskManager',
    'DaskDistributedFramework',
    'create_dask_array_from_zarr',
    'write_dask_array_to_zarr',
    'setup_dask_slurm_cluster'
]