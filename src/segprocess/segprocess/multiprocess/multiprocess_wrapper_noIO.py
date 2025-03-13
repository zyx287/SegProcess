'''
author: zyx
date: 2025-03-09
last modified: 2025-03-09
description: 
    Wrap a multiprocess function for general solution of large zarr dataset
        @process_large_zarr: Decorator to transform a simple array processing function (no io) into a function
'''
import zarr
import numpy as np
from multiprocessing import Process, JoinableQueue, cpu_count
import os
import time
import logging
import functools

from segprocess.segprocess.multiprocess.utils import compute_processing_slices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zarr_processing.log'),
        logging.StreamHandler()
    ]
)

#### No IO wrapper
def process_large_zarr(func):
    """
    Decorator that transforms a simple array processing function into a function
    that can process large Zarr datasets with multiprocessing.
    
    The wrapped function should take a numpy array as its first argument and return
    a processed numpy array.
    
    Usage:
        @process_large_zarr
        def my_function(data, param1=value1, param2=value2):
            # Process the data
            return processed_data
            
        result = my_function(input_zarr_path, output_zarr_path, output_dtype,
                            processing_chunk_size=(1024,1024,1024), 
                            num_workers=16, param1=value1, param2=value2)
    """
    @functools.wraps(func)
    def wrapper(input_zarr_path, output_zarr_path, output_dtype=None, 
                storage_chunk_size=None, processing_chunk_size=(1024, 1024, 1024),
                num_workers=None, **kwargs):
        
        start_time = time.time()
        
        # Get output_dtype from function if not provided
        if output_dtype is None:
            # Try to determine from a small test run
            input_zarr = zarr.open(input_zarr_path, mode='r')
            small_chunk = input_zarr[0:1, 0:1, 0:1]
            test_result = func(small_chunk, **kwargs)
            output_dtype = test_result.dtype
            logging.info(f"Auto-detected output dtype (set as input dtype) : {output_dtype}")
        
        # Open input zarr to get shape and chunks
        input_zarr = zarr.open(input_zarr_path, mode='r')
        total_shape = input_zarr.shape
        
        # Use input chunk size for storage if not specified
        if storage_chunk_size is None:
            storage_chunk_size = input_zarr.chunks
            logging.info(f"Using input Zarr chunk size for storage: {storage_chunk_size}")
        
        logging.info(f"Input Zarr shape: {total_shape}, storage chunk size: {storage_chunk_size}")
        logging.info(f"Processing chunk size: {processing_chunk_size}")
        logging.info(f"Function: {func.__name__}, output dtype: {output_dtype}")
        
        # Create output zarr if it doesn't exist
        if not os.path.exists(output_zarr_path):
            output_zarr = zarr.open(
                output_zarr_path,
                mode='w',
                shape=total_shape,
                chunks=storage_chunk_size,
                dtype=output_dtype
            )
            logging.info(f"Created output Zarr at {output_zarr_path}")
        else:
            output_zarr = zarr.open(output_zarr_path, mode='a')
            logging.info(f"Opened existing output Zarr at {output_zarr_path}")

        processing_chunks = compute_processing_slices(total_shape, processing_chunk_size)
        total_chunks = len(processing_chunks)
        logging.info(f"Total processing chunks: {total_chunks}")

        num_workers = num_workers or cpu_count()
        logging.info(f"Using {num_workers} workers")

        task_queue = JoinableQueue()
        
        # Start worker processes
        workers = []
        for _ in range(num_workers):
            worker_process = Process(
                target=worker_generic, 
                args=(task_queue, func, input_zarr_path, output_zarr_path, 
                     output_dtype, kwargs)
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
        
        logging.info(f"Processing completed in {elapsed_time:.2f} seconds ({processing_hours:.2f} hours)")
        logging.info(f"Processed {total_chunks} chunks using {func.__name__}")
        
        return {
            "total_chunks": total_chunks,
            "elapsed_time": elapsed_time,
            "function": func.__name__
        }
    
    return wrapper

def is_chunk_processed(output_zarr, slices):
    """
    Check if a chunk has already been processed in the output Zarr dataset.
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

def worker_generic(task_queue, process_function, input_zarr_path, output_zarr_path, 
                  output_dtype, func_args=None):
    """
    Worker function to process tasks from the queue.
    """
    while True:
        chunk_info = task_queue.get()
        if chunk_info is None:
            break
        try:
            process_chunk_generic(chunk_info, process_function, input_zarr_path, 
                                output_zarr_path, output_dtype, func_args)
        except Exception as e:
            logging.error(f"Error in worker processing chunk {chunk_info[0]}: {e}")
        finally:
            task_queue.task_done()

def process_chunk_generic(chunk_info, process_function, input_zarr_path, output_zarr_path, 
                         output_dtype, func_args=None):
    '''
    Process a chunk using the provided function.
    '''
    chunk_indices, slices = chunk_info
    func_args = func_args or {}
    
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
        
        # Process chunk with the provided function
        start_time = time.time()
        processed_chunk = process_function(chunk_data, **func_args)
        process_time = time.time() - start_time
        
        # Write processed chunk
        start_time = time.time()
        output_zarr[tuple(slices)] = processed_chunk
        write_time = time.time() - start_time
        
        total_time = load_time + process_time + write_time
        chunk_shape = tuple(s.stop - s.start for s in slices)
        input_size = np.prod(chunk_shape) * np.dtype(chunk_data.dtype).itemsize / (1024**3)  # size in GB
        
        logging.info(f"Processed chunk {chunk_indices}: shape={chunk_shape}, "
                    f"size={input_size:.2f}GB, time={total_time:.2f}s "
                    f"(load={load_time:.2f}s, process={process_time:.2f}s, write={write_time:.2f}s)")
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_indices}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    @process_large_zarr
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
        img_clipped = np.clip(img_uint16, min_val, max_val)
        img_normalized = (img_clipped - min_val) / (max_val - min_val)
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        
        return img_uint8

    # Call the function with appropriate arguments
    result = normalize_and_convert_to_uint8(
        input_zarr_path='input.zarr',
        output_zarr_path='output.zarr',
        output_dtype=np.uint8,
        processing_chunk_size=(1024, 1024, 1024),
        storage_chunk_size=(512, 512, 512),
        num_workers=45,
        min_val=34939,
        max_val=36096
    )
    print(result)