"""
Base utility functions for segmentation data processing.
"""

import numpy as np
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def id_to_label(data_chunk: np.ndarray, lookup: Dict[int, int]) -> np.ndarray:
    """
    Process dense labeled segmentation data using the look-up table
    
    Args:
        data_chunk: The dense labeled segmentation data
        lookup: The look-up table
        
    Returns:
        Processed segmentation data
    """
    logger.debug('Processing chunk...')
    vectorized_id_to_label = np.vectorize(lambda x: lookup.get(x, 0), otypes=[np.uint32])
    result = vectorized_id_to_label(data_chunk)
    logger.debug("Finished processing chunk.")
    return result


def filter_labels(data_array: np.ndarray, 
                 labels: Union[np.ndarray, List[int], Set[int]]) -> np.ndarray:
    """
    Filter out labels from a dense labeled segmentation data
    
    Args:
        data_array: The dense labeled segmentation data
        labels: The labels to keep (array, list, or set)
        
    Returns:
        The filtered dense labeled segmentation data
    """
    # Handle None values
    data_array = np.where(np.equal(data_array, None), 0, data_array)
    
    # Convert labels to a set for faster lookup
    valid_labels_set = set(labels)
    
    # Define filter function
    def _filter_labels(x):
        return x if x in valid_labels_set else 0
        
    # Apply filter
    vectorized_filter_labels = np.vectorize(_filter_labels)
    return vectorized_filter_labels(data_array)


def process_volume_dask(data: np.ndarray, 
                      lookup: Dict[int, Any], 
                      chunk_size: Tuple[int, ...] = (100, 100, 100), 
                      mode: str = "threads", 
                      n_cpus: int = 4) -> Tuple[np.ndarray, int, int]:
    """
    Process a large 3D volume using a lookup table with Dask.
    
    Args:
        data: The large 3D volume to process
        lookup: A lookup table where keys map to labels
        chunk_size: Size of 3D chunks to divide the volume into
        mode: Execution mode: "threads", "processes", or "distributed"
        n_cpus: Number of CPUs or workers to use
        
    Returns:
        Tuple containing:
            - The processed 3D volume with labels applied
            - Number of processed chunks
            - Total number of chunks
    """
    processed_chunks = []  # Track processed chunks
    
    def apply_lookup(data_chunk, block_id=None):
        """
        Apply lookup to a single chunk
        """
        try:
            if not isinstance(block_id, tuple):
                logger.warning(f"Unexpected block_id: {block_id}. Skipping.")
                return None

            logger.debug(f"Processing chunk {block_id} with shape {data_chunk.shape}")
            vectorized_lookup = np.vectorize(lambda x: lookup.get(x, 0), otypes=[np.uint32])
            result = vectorized_lookup(data_chunk)
            processed_chunks.append(block_id)

            return result
        except Exception as e:
            logger.error(f"Error processing chunk {block_id}: {e}")
            raise
            
    # Create dask array
    dask_array = da.from_array(data, chunks=chunk_size)

    # Apply lookup
    processed_dask_array = dask_array.map_blocks(apply_lookup, dtype=np.uint32, block_id=True)

    # Set up computation
    client = None
    if mode == "distributed":
        cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1)
        client = Client(cluster)
        logger.info(f"Running on distributed cluster with {n_cpus} workers.")
    elif mode == "processes":
        scheduler = "processes"
        logger.info(f"Running in multiprocessing mode on {n_cpus} CPUs.")
    else:  # Default to threads
        scheduler = "threads"
        logger.info(f"Running in multithreaded mode on {n_cpus} threads.")

    # Compute result
    with ProgressBar():
        result = processed_dask_array.compute(scheduler=scheduler if mode != "distributed" else None)

    # Clean up
    if client:
        client.close()

    processed_chunks_number = len(processed_chunks)
    total_chunks_number = dask_array.npartitions
    logger.info(f"Processing completed: {processed_chunks_number}/{total_chunks_number} chunks")

    return result, processed_chunks_number, total_chunks_number


def npy_to_zarr(input_npy_path: str, 
              output_zarr_path: str, 
              chunk_size: Optional[Tuple[int, ...]] = None) -> str:
    """
    Convert a numpy array file (.npy) to a zarr array
    
    Args:
        input_npy_path: Path to the input numpy array file
        output_zarr_path: Path to the output zarr array
        chunk_size: Chunk size for the zarr array
        
    Returns:
        Path to the output zarr array
    """
    # Load numpy array (use memory mapping to avoid loading the entire array)
    data_array = np.load(input_npy_path, mmap_mode="r")
    
    # Default chunk size if not provided
    if chunk_size is None:
        chunk_size = (64, 64, 64) if data_array.ndim == 3 else tuple(min(512, s) for s in data_array.shape)
    
    # Create zarr array
    zarr_array = zarr.open(
        output_zarr_path,
        mode="w",
        shape=data_array.shape,
        chunks=chunk_size,
        dtype=data_array.dtype
    )
    
    # Copy data
    zarr_array[:] = data_array
    
    logger.info(f"Converted numpy array to zarr: {output_zarr_path}")
    logger.info(f"Shape: {data_array.shape}, Chunks: {chunk_size}, Dtype: {data_array.dtype}")
    
    return output_zarr_path


# Make functions available at the module level
__all__ = [
    'id_to_label',
    'filter_labels',
    'process_volume_dask',
    'npy_to_zarr'
]