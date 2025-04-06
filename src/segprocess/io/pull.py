"""
Functions for downloading and processing segmentation data from Ariadne.
"""

import zarr
import numpy as np
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Tuple, List, Dict, Any, Optional, Union
from multiprocessing import Process, JoinableQueue, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_slices(total_volume_size: Tuple[int, ...], 
                 chunk_size: Tuple[int, ...]) -> List[List[slice]]:
    """
    Precompute slices for all chunks in the volume.

    Args:
        total_volume_size: The total size of the volume (x, y, z)
        chunk_size: The size of each chunk (x, y, z)

    Returns:
        List of slices for each chunk
    """
    slices_list = []
    for chunk_indices in np.ndindex(*[int(np.ceil(total_volume_size[dim] / chunk_size[dim])) for dim in range(len(total_volume_size))]):
        volume_offset = [idx * chunk_size[dim] for dim, idx in enumerate(chunk_indices)]
        volume_size = [
            min(chunk_size[dim], total_volume_size[dim] - volume_offset[dim])  # Handle boundary cases
            for dim in range(len(chunk_size))
        ]
        slices = [
            slice(volume_offset[dim], volume_offset[dim] + volume_size[dim])
            for dim in range(len(volume_offset))
        ]
        slices_list.append(slices)
    return slices_list


def create_retry_session(retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Args:
        retries: Number of retry attempts
        backoff_factor: Backoff factor for retries
        
    Returns:
        Session with retry logic
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def chunk_already_processed(output_zarr: zarr.Array, downsampled_slices: List[slice]) -> bool:
    """
    Check if a chunk has already been processed in the Zarr dataset.
    
    Args:
        output_zarr: Output Zarr array
        downsampled_slices: Slices defining the chunk
        
    Returns:
        True if the chunk has been processed, False otherwise
    """
    try:
        data = output_zarr[tuple(downsampled_slices)]
        return np.any(data)
    except Exception:
        return False


def load_and_write_chunk(slices: List[slice], 
                       toml_path: str, 
                       output_zarr_path: str, 
                       mag_size: int, 
                       downsampled_size: Tuple[int, ...]) -> None:
    """
    Load segmentation data for a specific slice and write to Zarr.
    
    Args:
        slices: Precomputed slices defining the chunk
        toml_path: Path to the TOML file for configuration
        output_zarr_path: Path to the output Zarr dataset
        mag_size: Magnification level for segmentation
        downsampled_size: Downsampled size of the dataset
    """
    try:
        from segprocess.utils.knossos import load_knossos_dataset
    except ImportError:
        raise ImportError("knossos_utils is required for this function")
    
    # Compute the offset and size from the slices
    volume_offset = [s.start for s in slices]
    volume_size = [s.stop - s.start for s in slices]
    logger.info(f"Processing chunk: offset={volume_offset}, size={volume_size}")

    # Create downsampled slices
    downsampled_slices = [
        slice(s.start // mag_size, s.stop // mag_size)
        for s in slices
    ]
    
    # Check if already processed
    output_zarr = zarr.open(output_zarr_path, mode="a")
    if chunk_already_processed(output_zarr, downsampled_slices):
        logger.info(f"Skipping already processed chunk: {downsampled_slices}")
        return

    # Load knossos dataset
    try:
        # Create retry session
        session = create_retry_session()
        
        # Offset sequence: XYZ, output order: ZYX
        chunk_data = load_knossos_dataset(
            toml_path=toml_path,
            volume_offset=tuple(volume_offset), 
            volume_size=tuple(volume_size), 
            mag_size=mag_size
        )
    except Exception as e:
        logger.error(f"Failed to load segment: {e}")
        raise

    # Convert chunk_data from ZYX to XYZ order 
    chunk_data = np.transpose(chunk_data, (2, 1, 0))

    # Write to output zarr
    output_zarr[tuple(downsampled_slices)] = chunk_data
    logger.info(f"Processed chunk with offset {volume_offset} and size {volume_size}, written to {downsampled_slices}")


def worker(task_queue: JoinableQueue, 
         toml_path: str, 
         output_zarr_path: str, 
         mag_size: int, 
         downsampled_size: Tuple[int, ...]) -> None:
    """
    Worker function to process tasks from the queue.
    
    Args:
        task_queue: Queue containing tasks to process
        toml_path: Path to the TOML file for configuration
        output_zarr_path: Path to the output Zarr dataset
        mag_size: Magnification level for segmentation
        downsampled_size: Downsampled size of the dataset
    """
    while True:
        slices = task_queue.get()
        if slices is None:
            break
        try:
            load_and_write_chunk(slices, toml_path, output_zarr_path, mag_size, downsampled_size)
        except Exception as e:
            logger.error(f"Error processing chunk {slices}: {e}")
        task_queue.task_done()


def process_volume_with_precomputed_slices(toml_path: str, 
                                        output_zarr_path: str, 
                                        total_volume_size: Tuple[int, ...], 
                                        chunk_size: Tuple[int, ...], 
                                        mag_size: int, 
                                        num_workers: Optional[int] = None) -> None:
    """
    Process a large volume with precomputed slices and multiprocessing.
    
    Args:
        toml_path: Path to the TOML file for configuration
        output_zarr_path: Path to the output Zarr dataset
        total_volume_size: The total size of the volume (x, y, z)
        chunk_size: The size of each chunk (x, y, z)
        mag_size: Magnification level for segmentation
        num_workers: Number of worker processes (default: number of CPUs)
    """
    # Calculate downsampled size
    downsampled_size = tuple(d // mag_size for d in total_volume_size)
    downsampled_chunk_size = tuple(d // mag_size for d in chunk_size)
    logger.info(f"Downsampled size: {downsampled_size}, Chunk size: {downsampled_chunk_size}")
    
    # Create output zarr
    zarr.open(output_zarr_path, mode="w", shape=downsampled_size, chunks=downsampled_chunk_size, dtype="uint32")

    # Precompute slices
    slices_list = compute_slices(total_volume_size, chunk_size)
    logger.info(f"Precomputed {len(slices_list)} chunk slices")

    # Create task queue
    task_queue = JoinableQueue()
    for slices in slices_list:
        task_queue.put(slices)

    # Start worker processes
    num_workers = num_workers or cpu_count()
    workers = []
    for _ in range(num_workers):
        worker_process = Process(
            target=worker, 
            args=(task_queue, toml_path, output_zarr_path, mag_size, downsampled_size)
        )
        worker_process.start()
        workers.append(worker_process)

    # Wait for all tasks to complete
    task_queue.join()

    # Send termination signal to workers
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Wait for workers to terminate
    for worker_process in workers:
        worker_process.join()

    logger.info("Processing completed")


def retry_failed_chunks(error_log_path: str, 
                      toml_path: str, 
                      output_zarr_path: str, 
                      mag_size: int) -> None:
    """
    Retry failed chunks based on error log.
    
    Args:
        error_log_path: Path to the error log
        toml_path: Path to the TOML file for configuration
        output_zarr_path: Path to the output Zarr dataset
        mag_size: Magnification level for segmentation
    """
    import re
    
    # Open output zarr to get shape
    output_zarr = zarr.open(output_zarr_path, mode="r")
    downsampled_size = output_zarr.shape
    
    # Extract failed chunk information from error log
    failed_chunks = []
    with open(error_log_path, 'r') as log_file:
        for line in log_file:
            if "Failed to load segment" in line or "Error processing chunk" in line:
                match = re.search(r"\[slice\((\d+), (\d+), None\), slice\((\d+), (\d+), None\), slice\((\d+), (\d+), None\)\]", line)
                if match:
                    slices = [
                        slice(int(match.group(1)), int(match.group(2))),
                        slice(int(match.group(3)), int(match.group(4))),
                        slice(int(match.group(5)), int(match.group(6)))
                    ]
                    failed_chunks.append(slices)
    
    # Retry each failed chunk
    logger.info(f"Retrying {len(failed_chunks)} failed chunks")
    for slices in failed_chunks:
        try:
            load_and_write_chunk(slices, toml_path, output_zarr_path, mag_size, downsampled_size)
        except Exception as e:
            logger.error(f"Failed to retry chunk {slices}: {e}")


# Make functions available at the module level
__all__ = [
    'process_volume_with_precomputed_slices',
    'retry_failed_chunks'
]