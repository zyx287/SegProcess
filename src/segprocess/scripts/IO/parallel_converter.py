'''
author: zyx
date: 2025-03-09
last modified: 2025-03-09
description: 
    Convert uint16 Zarr dataset to uint8 with normalization using multiprocessing
    Each worker processes a fixed 1024x1024x1024 chunk
'''
import zarr
import numpy as np
from multiprocessing import Process, JoinableQueue, cpu_count
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zarr_conversion.log'),
        logging.StreamHandler()
    ]
)

def normalize_and_convert_to_uint8(img_uint16, min_val=34939, max_val=36096):
    """
    Normalize a uint16 image within a specific range and convert to uint8.
    
    Args:
        img_uint16: Input uint16 image
        min_val: Minimum value for normalization (default: 34939)
        max_val: Maximum value for normalization (default: 36096)
        
    Returns:
        uint8 image normalized within the specified range
    """
    # Clip the image values to the specified range
    img_clipped = np.clip(img_uint16, min_val, max_val)
    
    # Normalize to 0-1 range based on the specified min/max values
    img_normalized = (img_clipped - min_val) / (max_val - min_val)
    
    # Convert to uint8 (0-255)
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    return img_uint8

def compute_processing_slices(total_volume_size, processing_chunk_size):
    '''
    Precompute slices for processing chunks in the volume.

    Args:
        total_volume_size: tuple
            The total size of the volume (z, y, x).
        processing_chunk_size: tuple
            The size of each processing chunk (z, y, x).

    Returns:
        list of tuple
            A list of slices for each processing chunk.
    '''
    slices_list = []
    
    # Calculate number of chunks in each dimension
    chunks_per_dim = [int(np.ceil(total_volume_size[dim] / processing_chunk_size[dim])) 
                     for dim in range(len(total_volume_size))]
    
    for chunk_indices in np.ndindex(*chunks_per_dim):
        # Calculate starting offset for this chunk
        volume_offset = [idx * processing_chunk_size[dim] for dim, idx in enumerate(chunk_indices)]
        
        # Calculate actual chunk size (handling boundary cases)
        volume_size = [
            min(processing_chunk_size[dim], total_volume_size[dim] - volume_offset[dim])
            for dim in range(len(processing_chunk_size))
        ]
        
        # Create slices
        slices = [
            slice(volume_offset[dim], volume_offset[dim] + volume_size[dim])
            for dim in range(len(volume_offset))
        ]
        
        slices_list.append((chunk_indices, slices))
    
    return slices_list

def is_chunk_processed(output_zarr, slices):
    """
    Check if a chunk has already been processed in the output Zarr dataset.
    
    Args:
        output_zarr: zarr array
            The output Zarr dataset
        slices: list of slice
            The slices defining the chunk
            
    Returns:
        bool: True if processed, False otherwise
    """
    try:
        # Create a small sample from the chunk to check if it contains data
        sample_slices = []
        for s in slices:
            start = s.start
            end = min(s.start + 10, s.stop)  # Sample first 10 elements or fewer
            sample_slices.append(slice(start, end))
        
        sample = output_zarr[tuple(sample_slices)]
        return np.any(sample)
    except Exception as e:
        logging.debug(f"Error checking if chunk is processed: {e}")
        return False

def convert_processing_chunk(chunk_info, input_zarr_path, output_zarr_path, min_val, max_val):
    '''
    Convert a processing chunk from uint16 to uint8 with normalization.

    Args:
        chunk_info: tuple
            (chunk_indices, slices) where chunk_indices is the n-dimensional index
            and slices are the slice objects for accessing the chunk.
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to the output Zarr dataset.
        min_val: int
            Minimum value for normalization.
        max_val: int
            Maximum value for normalization.
    '''
    chunk_indices, slices = chunk_info
    
    try:
        # Open zarr arrays
        input_zarr = zarr.open(input_zarr_path, mode='r')
        output_zarr = zarr.open(output_zarr_path, mode='a')
        
        # Check if already processed
        if is_chunk_processed(output_zarr, slices):
            logging.info(f"Skipping already processed chunk: {chunk_indices}")
            return
        
        # Load chunk
        start_time = time.time()
        chunk_data = input_zarr[tuple(slices)]
        load_time = time.time() - start_time
        
        # Convert chunk
        start_time = time.time()
        converted_chunk = normalize_and_convert_to_uint8(chunk_data, min_val, max_val)
        convert_time = time.time() - start_time
        
        # Write converted chunk
        start_time = time.time()
        output_zarr[tuple(slices)] = converted_chunk
        write_time = time.time() - start_time
        
        total_time = load_time + convert_time + write_time
        chunk_shape = tuple(s.stop - s.start for s in slices)
        chunk_size_gb = np.prod(chunk_shape) * 2 / (1024**3)  # uint16 input size in GB
        
        logging.info(f"Processed chunk {chunk_indices}: shape={chunk_shape}, "
                    f"size={chunk_size_gb:.2f}GB, time={total_time:.2f}s "
                    f"(load={load_time:.2f}s, convert={convert_time:.2f}s, write={write_time:.2f}s)")
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_indices}: {e}")
        raise

def worker(task_queue, input_zarr_path, output_zarr_path, min_val, max_val):
    """
    Worker function to process tasks from the queue.

    Args:
        task_queue: JoinableQueue
            Queue containing tasks to process.
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to the output Zarr dataset.
        min_val: int
            Minimum value for normalization.
        max_val: int
            Maximum value for normalization.
    """
    while True:
        chunk_info = task_queue.get()
        if chunk_info is None:
            break
        try:
            convert_processing_chunk(chunk_info, input_zarr_path, output_zarr_path, min_val, max_val)
        except Exception as e:
            logging.error(f"Error in worker processing chunk {chunk_info[0]}: {e}")
        finally:
            task_queue.task_done()

def convert_zarr_uint16_to_uint8(
        input_zarr_path, 
        output_zarr_path, 
        min_val=34939, 
        max_val=36096, 
        storage_chunk_size=None, 
        processing_chunk_size=(1024, 1024, 1024),
        num_workers=None):
    """
    Convert a uint16 Zarr array to uint8 using multiprocessing.
    
    Args:
        input_zarr_path: str
            Path to the input Zarr dataset (uint16).
        output_zarr_path: str
            Path to the output Zarr dataset (uint8).
        min_val: int
            Minimum value for normalization (default: 34939).
        max_val: int
            Maximum value for normalization (default: 36096).
        storage_chunk_size: tuple or None
            Physical chunk size for Zarr storage. If None, use the input Zarr's chunk size.
        processing_chunk_size: tuple
            Size of data each worker processes (default: 1024x1024x1024).
        num_workers: int or None
            Number of worker processes to use. If None, use CPU count.
    """
    start_time = time.time()
    
    # Open input zarr to get shape and chunks
    input_zarr = zarr.open(input_zarr_path, mode='r')
    total_shape = input_zarr.shape
    
    # Use input chunk size for storage if not specified
    if storage_chunk_size is None:
        storage_chunk_size = input_zarr.chunks
    
    logging.info(f"Input Zarr shape: {total_shape}, storage chunk size: {storage_chunk_size}")
    logging.info(f"Processing chunk size: {processing_chunk_size}")
    logging.info(f"Normalization range: [{min_val}, {max_val}]")
    
    # Create output zarr if it doesn't exist
    if not os.path.exists(output_zarr_path):
        output_zarr = zarr.open(
            output_zarr_path,
            mode='w',
            shape=total_shape,
            chunks=storage_chunk_size,
            dtype=np.uint8
        )
        logging.info(f"Created output Zarr at {output_zarr_path}")
    else:
        output_zarr = zarr.open(output_zarr_path, mode='a')
        logging.info(f"Opened existing output Zarr at {output_zarr_path}")
    
    # Precompute processing slices
    processing_chunks = compute_processing_slices(total_shape, processing_chunk_size)
    total_chunks = len(processing_chunks)
    logging.info(f"Total processing chunks: {total_chunks}")
    
    # Set up multiprocessing
    num_workers = num_workers or cpu_count()
    logging.info(f"Using {num_workers} workers")
    
    # Create task queue
    task_queue = JoinableQueue()
    
    # Start worker processes
    workers = []
    for _ in range(num_workers):
        worker_process = Process(
            target=worker, 
            args=(task_queue, input_zarr_path, output_zarr_path, min_val, max_val)
        )
        worker_process.daemon = True
        worker_process.start()
        workers.append(worker_process)
    
    # Add tasks to queue
    for chunk_info in processing_chunks:
        task_queue.put(chunk_info)
    
    # Wait for all tasks to complete
    logging.info("Waiting for all chunks to be processed...")
    task_queue.join()
    
    # Signal workers to terminate
    logging.info("All chunks processed, shutting down workers...")
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Wait for workers to finish
    for worker_process in workers:
        worker_process.join()
    
    elapsed_time = time.time() - start_time
    processing_hours = elapsed_time / 3600
    
    logging.info(f"Conversion completed in {elapsed_time:.2f} seconds ({processing_hours:.2f} hours)")
    logging.info(f"Processed {total_chunks} chunks")
    
    return {
        "total_chunks": total_chunks,
        "elapsed_time": elapsed_time
    }

def main():
    # Cluster usage
    # input_zarr_path = "/nrs/cellmap/data/jrc_mus-cerebellum-2/jrc_mus-cerebellum-2.zarr/recon-1/em/fibsem-uint16/s1/"
    # output_zarr_path = "/groups/li/home/zhangy8/FIBProg/cerebellum2/RAW/uint8/S1/cerenellum2_s1_uint8_v3.zarr"
    # Local usage
    input_zarr_path = "/media/zhangy8/f76218c3-5b11-4bb6-b6f5-6f2117690496/S1RawCerebellum2/s1"
    output_zarr_path = "/media/zhangy8/f76218c3-5b11-4bb6-b6f5-6f2117690496/S1RawCerebellum2_int8/cerebellum2_s1_uint8_v2.zarr"
    
    # Normalization parameters
    min_val = 34600
    max_val = 36096
    
    # Fixed processing chunk size (1024×1024×1024)
    processing_chunk_size = (512, 512, 512)
    
    result = convert_zarr_uint16_to_uint8(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        min_val=min_val,
        max_val=max_val,
        storage_chunk_size=(512, 512, 512),
        processing_chunk_size=processing_chunk_size,
        num_workers=60  # Adjust based on your system's capabilities
    )
    
    print(f"Conversion completed: {result['total_chunks']} chunks processed in {result['elapsed_time']:.2f} seconds")

if __name__ == "__main__":
    main()
