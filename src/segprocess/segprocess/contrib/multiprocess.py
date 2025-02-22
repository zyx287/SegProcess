'''
author: zyx
date: 2025-02-21
last_modified: 2025-02-21
description: 
'''

import os
import time
import logging
import traceback
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Union, Optional, Iterator
from dataclasses import dataclass
from multiprocessing import Process, JoinableQueue, Value, Lock, cpu_count
import signal
import json
import contextlib
import atexit


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multiprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task container for worker processes."""
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
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            params=data["params"],
            retries=data["retries"],
            max_retries=data["max_retries"]
        )


class AtomicCounter:
    """Thread-safe counter for tracking progress."""
    def __init__(self, initial_value: int = 0):
        self.value = Value('i', initial_value)
        self.lock = Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self.lock:
            self.value.value += 1
            return self.value.value
    
    def get(self) -> int:
        """Get current value."""
        return self.value.value


class CheckpointManager:
    """Manages checkpoints for restartable processing."""
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.lock = Lock()
        self.completed_tasks = set()
        self.failed_tasks = {}
        
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
        with self.lock:
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
        with self.lock:
            self.completed_tasks.add(task_id)
            if task_id in self.failed_tasks:
                del self.failed_tasks[task_id]
            
            # Periodically save checkpoint
            if len(self.completed_tasks) % 10 == 0:
                self._save_checkpoint()
    
    def mark_failed(self, task_id: int, error: str) -> None:
        """Mark task as failed."""
        with self.lock:
            self.failed_tasks[task_id] = error
            
            # Save checkpoint on each failure
            self._save_checkpoint()
    
    def is_completed(self, task_id: int) -> bool:
        """Check if task is completed."""
        return task_id in self.completed_tasks
    
    def get_failed_tasks(self) -> Dict[int, str]:
        """Get all failed tasks."""
        with self.lock:
            return self.failed_tasks.copy()


class TaskManager:
    """Manages task generation and distribution."""
    def __init__(
        self, 
        task_generator: Callable[[], Iterator[Dict[str, Any]]],
        checkpoint_file: str = "process_checkpoint.json"
    ):
        self.task_generator = task_generator
        self.checkpoint = CheckpointManager(checkpoint_file)
        self.counter = AtomicCounter()
        self.total_tasks = 0
    
    def generate_tasks(self) -> List[Task]:
        """Generate tasks from generator function."""
        tasks = []
        for i, params in enumerate(self.task_generator()):
            task = Task(id=i, params=params)
            if not self.checkpoint.is_completed(task.id):
                tasks.append(task)
            
        self.total_tasks = len(tasks)
        logger.info(f"Generated {self.total_tasks} tasks")
        return tasks
    
    def log_progress(self, completed: int, total: int) -> None:
        """Log progress information."""
        percent = (completed / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {completed}/{total} ({percent:.2f}%)")


class MultiprocessingFramework:
    """
    General framework for multiprocessing chunk-based operations.
    
    This class handles:
    1. Task generation and distribution
    2. Worker process management
    3. Error handling and retries
    4. Progress tracking and reporting
    5. Graceful shutdown
    """
    def __init__(
        self,
        process_func: Callable[[Task, Dict[str, Any]], Any],
        task_generator: Callable[[], Iterator[Dict[str, Any]]],
        shared_data: Dict[str, Any] = None,
        num_workers: int = None,
        checkpoint_file: str = "process_checkpoint.json",
        max_retries: int = 3,
        retry_delay: int = 5,
        progress_interval: int = 10
    ):
        """
        Initialize the framework.
        
        Args:
            process_func: Function to process a single task
            task_generator: Function that generates task parameters
            shared_data: Data shared across all workers
            num_workers: Number of worker processes
            checkpoint_file: File to store checkpoint data
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Delay between retries in seconds
            progress_interval: Interval for progress reporting in seconds
        """
        self.process_func = process_func
        self.shared_data = shared_data or {}
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.progress_interval = progress_interval
        
        self.task_queue = JoinableQueue()
        self.workers = []
        self.completed_count = AtomicCounter()
        self.task_manager = TaskManager(task_generator, checkpoint_file)
        
        # Handle interrupts gracefully
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info("Interrupt received, shutting down gracefully...")
            self._terminate_workers()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _worker(self) -> None:
        """Worker process function."""
        # Each worker process needs its own random seed
        np.random.seed()
        
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.task_done()
                break
                
            try:
                if self.task_manager.checkpoint.is_completed(task.id):
                    logger.debug(f"Task {task.id} already completed, skipping")
                    self.task_queue.task_done()
                    continue
                    
                # Process the task
                self.process_func(task, self.shared_data)
                
                # Mark as completed
                self.task_manager.checkpoint.mark_completed(task.id)
                completed = self.completed_count.increment()
                
                self.task_queue.task_done()
                
            except Exception as e:
                task.retries += 1
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Error processing task {task.id}: {error_msg}")
                logger.error(traceback.format_exc())
                
                if task.retries < task.max_retries:
                    logger.info(f"Retrying task {task.id} ({task.retries}/{task.max_retries})")
                    time.sleep(self.retry_delay)
                    self.task_queue.put(task)
                else:
                    logger.error(f"Task {task.id} failed after {task.retries} retries")
                    self.task_manager.checkpoint.mark_failed(task.id, error_msg)
                    completed = self.completed_count.increment()
                
                self.task_queue.task_done()
    
    def _start_workers(self) -> None:
        """Start worker processes."""
        for _ in range(self.num_workers):
            worker = Process(target=self._worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} worker processes")
    
    def _terminate_workers(self) -> None:
        """Terminate all worker processes."""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
        
        logger.info("All workers terminated")
    
    def _monitor_progress(self, total_tasks: int) -> None:
        """Monitor and report progress."""
        start_time = time.time()
        last_report_time = start_time
        last_completed = 0
        
        try:
            while True:
                time.sleep(1)
                current_completed = self.completed_count.get()
                current_time = time.time()
                
                # Report progress at specified intervals
                if current_time - last_report_time >= self.progress_interval:
                    elapsed = current_time - start_time
                    rate = current_completed / elapsed if elapsed > 0 else 0
                    
                    # Calculate time remaining
                    remaining = 0
                    if rate > 0:
                        remaining = (total_tasks - current_completed) / rate
                    
                    # Log progress
                    percent = (current_completed / total_tasks) * 100 if total_tasks > 0 else 0
                    logger.info(f"Progress: {current_completed}/{total_tasks} ({percent:.2f}%)")
                    logger.info(f"Rate: {rate:.2f} tasks/sec, Estimated time remaining: {remaining/60:.2f} minutes")
                    
                    last_report_time = current_time
                    last_completed = current_completed
                
                # Check if all tasks are done
                if current_completed >= total_tasks:
                    break
                
                # Check if all workers are done
                active_workers = sum(1 for w in self.workers if w.is_alive())
                if active_workers == 0:
                    logger.warning("All workers have exited but not all tasks are complete")
                    break
        
        except KeyboardInterrupt:
            logger.info("Progress monitoring interrupted")
    
    def process(self) -> Dict[str, Any]:
        """
        Process all tasks using the multiprocessing framework.
        
        Returns:
            Dict with statistics about the processing
        """
        logger.info("Starting processing with multiprocessing framework")
        logger.info(f"Using {self.num_workers} worker processes")
        
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
        
        # Start workers
        self._start_workers()
        
        # Queue tasks
        for task in tasks:
            self.task_queue.put(task)
        
        # Start progress monitoring in the main thread
        progress_process = Process(target=self._monitor_progress, args=(total_tasks,))
        progress_process.daemon = True
        progress_process.start()
        
        try:
            # Wait for all tasks to complete
            self.task_queue.join()
            
            # Send termination signal to all workers
            for _ in range(self.num_workers):
                self.task_queue.put(None)
            
            # Wait for all workers to exit
            for worker in self.workers:
                worker.join()
            
            # Terminate progress monitor
            if progress_process.is_alive():
                progress_process.terminate()
                progress_process.join()
            
            # Get failed tasks
            failed_tasks = self.task_manager.checkpoint.get_failed_tasks()
            
            result = {
                "status": "completed",
                "total_tasks": total_tasks,
                "completed_tasks": self.completed_count.get(),
                "failed_tasks": len(failed_tasks),
                "failed_task_ids": list(failed_tasks.keys())
            }
            
            # Log summary
            logger.info(f"Processing completed: {result['completed_tasks']}/{result['total_tasks']} tasks processed")
            if result["failed_tasks"] > 0:
                logger.warning(f"{result['failed_tasks']} tasks failed")
            
            return result
            
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down")
            self._terminate_workers()
            if progress_process.is_alive():
                progress_process.terminate()
            
            return {
                "status": "interrupted",
                "total_tasks": total_tasks,
                "completed_tasks": self.completed_count.get(),
                "failed_tasks": len(self.task_manager.checkpoint.get_failed_tasks())
            }
    
    def retry_failed_tasks(self) -> Dict[str, Any]:
        """
        Retry tasks that previously failed.
        
        Returns:
            Dict with statistics about the retry process
        """
        failed_tasks = self.task_manager.checkpoint.get_failed_tasks()
        if not failed_tasks:
            logger.info("No failed tasks to retry")
            return {"status": "completed", "retried_tasks": 0}
        
        logger.info(f"Retrying {len(failed_tasks)} failed tasks")
        
        # Start workers
        self._start_workers()
        
        # Queue failed tasks
        for task_id, _ in failed_tasks.items():
            # Regenerate the task
            for params in self.task_manager.task_generator():
                if params.get("id") == task_id:
                    task = Task(id=task_id, params=params, retries=0, max_retries=self.max_retries)
                    self.task_queue.put(task)
                    break
        
        total_retry_tasks = len(failed_tasks)
        
        # Start progress monitoring
        progress_process = Process(target=self._monitor_progress, args=(total_retry_tasks,))
        progress_process.daemon = True
        progress_process.start()
        
        try:
            # Wait for all tasks to complete
            self.task_queue.join()
            
            # Send termination signal to all workers
            for _ in range(self.num_workers):
                self.task_queue.put(None)
            
            # Wait for all workers to exit
            for worker in self.workers:
                worker.join()
            
            # Terminate progress monitor
            if progress_process.is_alive():
                progress_process.terminate()
            
            # Get still failed tasks
            still_failed = self.task_manager.checkpoint.get_failed_tasks()
            
            result = {
                "status": "completed",
                "retried_tasks": total_retry_tasks,
                "successful_retries": total_retry_tasks - len(still_failed),
                "failed_retries": len(still_failed),
                "failed_task_ids": list(still_failed.keys())
            }
            
            logger.info(f"Retry completed: {result['successful_retries']}/{result['retried_tasks']} tasks successfully retried")
            if result["failed_retries"] > 0:
                logger.warning(f"{result['failed_retries']} tasks still failing")
            
            return result
            
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down")
            self._terminate_workers()
            if progress_process.is_alive():
                progress_process.terminate()
            
            return {
                "status": "interrupted",
                "retried_tasks": total_retry_tasks,
                "completed_retries": self.completed_count.get()
            }


# Helper functions for common chunk processing patterns

def compute_chunks_from_shape(total_shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
    """
    Compute chunk parameters from total shape and chunk shape.
    
    Args:
        total_shape: Total shape of the data (x, y, z)
        chunk_shape: Shape of each chunk (x, y, z)
        
    Returns:
        List of dictionaries with slices for each chunk
    """
    chunk_params = []
    for chunk_indices in np.ndindex(*[int(np.ceil(total_shape[dim] / chunk_shape[dim])) for dim in range(len(total_shape))]):
        slices = []
        for dim, idx in enumerate(chunk_indices):
            start = idx * chunk_shape[dim]
            stop = min((idx + 1) * chunk_shape[dim], total_shape[dim])
            slices.append(slice(start, stop))
        
        chunk_params.append({
            "slices": slices,
            "offset": [s.start for s in slices],
            "size": [s.stop - s.start for s in slices],
            "index": chunk_indices
        })
    
    return chunk_params


@contextlib.contextmanager
def measure_time(task_name: str) -> None:
    """Context manager to measure execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{task_name} completed in {end_time - start_time:.2f} seconds")


def zarr_chunk_exists(zarr_obj, slices: List[slice]) -> bool:
    """Check if a chunk already exists in a Zarr array."""
    try:
        chunk = zarr_obj[tuple(slices)]
        return np.any(chunk != 0)
    except Exception:
        return False