"""
Functions for merging nuclei segmentation into cell segmentation.
"""

import zarr
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union, List

from segprocess.processing.multiprocess import process_large_zarr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def downsample_and_merge_chunk(cell_chunk: np.ndarray, 
                             nuclei_chunk: np.ndarray) -> np.ndarray:
    """
    Downsample and merge cell and nuclei segmentation chunks.
    
    Args:
        cell_chunk: Cell segmentation chunk (2x resolution)
        nuclei_chunk: Nuclei segmentation chunk
        
    Returns:
        Merged segmentation chunk
    """
    # Downsample cell chunk by taking every other voxel
    downsampled_cell = cell_chunk[::2, ::2, ::2]
    
    # Create output chunk
    output_chunk = np.zeros_like(nuclei_chunk, dtype=np.uint32)
    
    # Merge logic: if both cell and nuclei are labeled, use cell label
    mask_both_labeled = (downsampled_cell > 0) & (nuclei_chunk > 0)
    
    output_chunk[mask_both_labeled] = downsampled_cell[mask_both_labeled]
    
    return output_chunk


@process_large_zarr
def merge_cell_nuclei(data_chunk: np.ndarray,
                    cell_array: np.ndarray,
                    downsampling_factor: int = 2) -> np.ndarray:
    """
    Process function for merging cell and nuclei segmentation.
    
    Args:
        data_chunk: Nuclei segmentation chunk
        cell_array: Cell segmentation array (higher resolution)
        downsampling_factor: Factor by which cell segmentation is downsampled
        
    Returns:
        Merged segmentation chunk
    """
    # Get cell segmentation chunk at a higher resolution
    # Convert the slices from nuclei resolution to cell resolution
    cell_slices = []
    for s in data_chunk.slices:
        start = s.start * downsampling_factor
        stop = s.stop * downsampling_factor
        cell_slices.append(slice(start, stop))
    
    cell_chunk = cell_array[tuple(cell_slices)]
    
    # Downsample and merge
    downsampled_cell = cell_chunk[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]
    
    # Create output chunk
    output_chunk = np.zeros_like(data_chunk, dtype=np.uint32)
    
    # Merge logic: if both cell and nuclei are labeled, use cell label
    mask_both_labeled = (downsampled_cell > 0) & (data_chunk > 0)
    output_chunk[mask_both_labeled] = downsampled_cell[mask_both_labeled]
    
    return output_chunk


def downsample_and_merge_zarr(cell_zarr_path: str, 
                            nuclei_zarr_path: str,
                            output_zarr_path: str,
                            downsampling_factor: int = 2,
                            num_workers: Optional[int] = None) -> str:
    """
    Downsample and merge cell and nuclei segmentation.
    
    Args:
        cell_zarr_path: Path to cell segmentation zarr
        nuclei_zarr_path: Path to nuclei segmentation zarr
        output_zarr_path: Path to output zarr
        downsampling_factor: Factor by which cell segmentation is downsampled
        num_workers: Number of worker processes
        
    Returns:
        Path to output zarr
    """
    # Open zarr arrays
    cell_zarr = zarr.open(cell_zarr_path, mode="r")
    nuclei_zarr = zarr.open(nuclei_zarr_path, mode="r")
    
    # Check shapes
    expected_cell_shape = tuple(d * downsampling_factor for d in nuclei_zarr.shape)
    if cell_zarr.shape != expected_cell_shape:
        logger.warning(f"Cell shape {cell_zarr.shape} doesn't match expected shape {expected_cell_shape}")
        logger.warning(f"Will use downsampling factors derived from the shape ratio")
        
        # Calculate actual downsampling factors
        downsampling_factors = [c / n for c, n in zip(cell_zarr.shape, nuclei_zarr.shape)]
        logger.info(f"Actual downsampling factors: {downsampling_factors}")
        
        if any(abs(f - downsampling_factor) > 0.1 for f in downsampling_factors):
            logger.warning(f"Downsampling factors differ significantly from requested factor {downsampling_factor}")
    
    # Create output zarr
    output_zarr = zarr.open(
        output_zarr_path,
        mode="w",
        shape=nuclei_zarr.shape,
        chunks=nuclei_zarr.chunks,
        dtype=np.uint32
    )
    
    from segprocess.processing.parallel import process_zarr_dynamically
    
    # Define processing function for each chunk
    def process_chunk(task, input_zarr_path, output_zarr_path, shared_data):
        index, slices = task
        
        try:
            # Open zarr arrays
            nuclei_zarr = zarr.open(input_zarr_path, mode="r")
            cell_zarr = zarr.open(shared_data["cell_zarr_path"], mode="r")
            output_zarr = zarr.open(output_zarr_path, mode="a")
            
            # Load nuclei chunk
            nuclei_chunk = nuclei_zarr[tuple(slices)]
            
            # Convert slices to cell resolution
            cell_slices = []
            for s in slices:
                start = s.start * downsampling_factor
                stop = s.stop * downsampling_factor
                cell_slices.append(slice(start, stop))
            
            # Load cell chunk
            cell_chunk = cell_zarr[tuple(cell_slices)]
            
            # Downsample and merge
            downsampled_cell = cell_chunk[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]
            
            # Create output chunk
            output_chunk = np.zeros_like(nuclei_chunk, dtype=np.uint32)
            
            # Merge logic: if both cell and nuclei are labeled, use cell label
            mask_both_labeled = (downsampled_cell > 0) & (nuclei_chunk > 0)
            output_chunk[mask_both_labeled] = downsampled_cell[mask_both_labeled]
            
            # Write output chunk
            output_zarr[tuple(slices)] = output_chunk
            
            logger.info(f"Processed chunk at {slices}")
            
        except Exception as e:
            logger.error(f"Error processing chunk {index}: {e}")
            raise
    
    # Process all chunks
    shared_data = {
        "cell_zarr_path": cell_zarr_path,
        "downsampling_factor": downsampling_factor
    }
    
    # Process chunks
    process_zarr_dynamically(
        input_zarr_path=nuclei_zarr_path,
        output_zarr_path=output_zarr_path,
        filtered_zarr_path=None,
        lookup=None,
        valid_labels_array=None,
        chunk_processor=process_chunk,
        shared_data=shared_data,
        num_workers=num_workers
    )
    
    logger.info(f"Downsampling and merging completed: {output_zarr_path}")
    return output_zarr_path


# Make functions available at the module level
__all__ = [
    'downsample_and_merge_chunk',
    'downsample_and_merge_zarr'
]