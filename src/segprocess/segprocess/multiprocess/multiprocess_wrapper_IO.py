'''
author: zyx
date: 2025-03-12
last_modified: 2025-03-12
description: 
    Decorator wrapper for multiprocessing functions that handle their own I/O operations
'''
import os
import time
import logging
import numpy as np
from multiprocessing import Process, JoinableQueue, cpu_count
import zarr
import functools
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

# Import the utility function for computing processing slices
from segprocess.segprocess.multiprocess.utils import compute_processing_slices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiprocess_io.log'),
        logging.StreamHandler()
    ]
)

def process_zarr_with_io(func):
    """
    Decorator that transforms a function that processes a single chunk with I/O operations
    into a function that processes a whole dataset using multiprocessing.
    
    The wrapped function should handle its own I/O operations and have the signature:
    func(chunk_info, input_zarr_path, output_paths, **kwargs) where chunk_info is (chunk_indices, slices).
    
    Usage:
        @process_with_io
        def process_chunk(chunk_info, input_zarr_path, output_paths, param1=value1):
            # Read, process and write data here
            
        result = process_chunk(
            input_zarr_path="input.zarr",
            output_paths=["output1.zarr", "output2.zarr"],
            output_dtypes=["uint32", "uint32"],
            processing_chunk_size=(512, 512, 512),
            num_workers=8,
            param1=value1
        )
    """
    @functools.wraps(func)
    def wrapper(input_zarr_path, output_paths, output_dtypes, 
                processing_chunk_size=None, storage_chunk_size=None, 
                num_workers=None, check_processed=True, **kwargs):
        """
        Processes a Zarr dataset using multiprocessing with I/O-handling functions.
        
        Parameters:
            input_zarr_path: str
                Path to the input Zarr dataset
            output_paths: List[str] or str
                Path(s) to output Zarr dataset(s)
            output_dtypes: List[str/np.dtype] or str/np.dtype
                Data type(s) for output Zarr dataset(s)
            processing_chunk_size: Tuple[int, ...], optional
                Size of chunks for processing (default: None, uses Zarr chunk size)
            storage_chunk_size: Tuple[int, ...], optional
                Size of chunks for output storage (default: None, uses Zarr chunk size)
            num_workers: int, optional
                Number of worker processes (default: None, uses CPU count)
            check_processed: bool, optional
                Whether to check if chunks are already processed (default: True)
            **kwargs:
                Additional arguments to pass to the processing function
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        start_time = time.time()
        
        # Standardize output paths and dtypes as lists
        if not isinstance(output_paths, list):
            output_paths = [output_paths]
            
        if not isinstance(output_dtypes, list):
            output_dtypes = [output_dtypes]
            
        if len(output_paths) != len(output_dtypes):
            raise ValueError("Number of output paths must match number of output dtypes")
            
        # Open input Zarr to get metadata
        input_zarr = zarr.open(input_zarr_path, mode='r')
        total_shape = input_zarr.shape
        input_chunks = input_zarr.chunks
        
        # Set chunk sizes with appropriate defaults
        if storage_chunk_size is None:
            storage_chunk_size = input_chunks
            logging.info(f"Using input Zarr chunk size for storage: {storage_chunk_size}")
            
        if processing_chunk_size is None:
            processing_chunk_size = storage_chunk_size
            logging.info(f"Using storage chunk size for processing: {processing_chunk_size}")
        
        # Create output Zarrs if they don't exist
        for i, output_path in enumerate(output_paths):
            if not os.path.exists(output_path):
                zarr.open(
                    output_path, mode='w',
                    shape=total_shape, chunks=storage_chunk_size,
                    dtype=output_dtypes[i]
                )
                logging.info(f"Created output Zarr at {output_path}")
            else:
                logging.info(f"Output Zarr at {output_path} already exists")
        
        # Generate processing chunks using the utility function
        processing_chunks = compute_processing_slices(total_shape, processing_chunk_size)
        total_chunks = len(processing_chunks)
        
        logging.info(f"Input Zarr shape: {total_shape}")
        logging.info(f"Processing chunk size: {processing_chunk_size}")
        logging.info(f"Storage chunk size: {storage_chunk_size}")
        logging.info(f"Total processing chunks: {total_chunks}")
        
        # Set up multiprocessing
        num_workers = min(num_workers or cpu_count(), total_chunks)
        logging.info(f"Using {num_workers} worker processes")
        
        task_queue = JoinableQueue()
        
        # Start worker processes
        workers = []
        for worker_id in range(num_workers):
            worker_process = Process(
                target=io_worker,
                args=(
                    task_queue, func, worker_id, 
                    input_zarr_path, output_paths, check_processed, kwargs
                )
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
        
        # Calculate and log timing statistics
        elapsed_time = time.time() - start_time
        hours = elapsed_time / 3600
        
        logging.info(f"Processing completed in {elapsed_time:.2f} seconds ({hours:.2f} hours)")
        logging.info(f"Processed {total_chunks} chunks using {func.__name__}")
        
        return {
            "total_chunks": total_chunks,
            "elapsed_time": elapsed_time,
            "function": func.__name__,
            "input_path": input_zarr_path,
            "output_paths": output_paths
        }
    
    return wrapper

def io_worker(task_queue, processing_func, worker_id, 
             input_zarr_path, output_paths, check_processed, kwargs):
    """
    Worker function that processes tasks from a queue.
    Each task is processed using the provided function that handles its own I/O.
    
    Parameters:
        task_queue: JoinableQueue
            Queue containing tasks to process
        processing_func: Callable
            Function to process each task
        worker_id: int
            ID of the worker process
        input_zarr_path: str
            Path to the input Zarr array
        output_paths: List[str]
            Paths to the output Zarr arrays
        check_processed: bool
            Whether to check if chunks are already processed
        kwargs: Dict[str, Any]
            Additional arguments to pass to the processing function
    """
    tasks_completed = 0
    
    try:
        # Check if we need to verify processed status
        if check_processed:
            # Open output Zarrs to check if chunks are processed
            output_zarrs = [zarr.open(path, mode='r') for path in output_paths]
        
        while True:
            # Get task from queue
            chunk_info = task_queue.get()
            
            # Check for termination signal
            if chunk_info is None:
                logging.debug(f"Worker {worker_id} received termination signal")
                break
            
            chunk_indices, slices = chunk_info
            
            # Check if already processed
            if check_processed:
                skip = True
                for output_zarr in output_zarrs:
                    # Create a small sample slice to check
                    sample_slices = []
                    for s in slices:
                        start = s.start
                        end = min(s.start + 10, s.stop)
                        sample_slices.append(slice(start, end))
                    
                    # Check if the chunk contains data
                    try:
                        if not np.any(output_zarr[tuple(sample_slices)]):
                            skip = False
                            break
                    except Exception as e:
                        # If there's an error checking the chunk, process it to be safe
                        logging.debug(f"Error checking if chunk {chunk_indices} is processed: {e}")
                        skip = False
                        break
                
                if skip:
                    logging.info(f"Worker {worker_id}: Skipping already processed chunk {chunk_indices}")
                    task_queue.task_done()
                    continue
            
            # Process task
            task_start_time = time.time()
            try:
                processing_func(chunk_info, input_zarr_path=input_zarr_path, 
                              output_paths=output_paths, **kwargs)
                task_time = time.time() - task_start_time
                tasks_completed += 1
                
                # Log progress periodically or for slow tasks
                if tasks_completed % 10 == 0 or task_time > 30:
                    logging.info(f"Worker {worker_id}: Completed chunk {chunk_indices} in {task_time:.2f}s " +
                                f"(total: {tasks_completed} chunks)")
                
            except Exception as e:
                logging.error(f"Worker {worker_id}: Error processing chunk {chunk_indices}: {str(e)}")
                
            finally:
                # Mark task as done
                task_queue.task_done()
                
    except Exception as e:
        logging.error(f"Worker {worker_id}: Unexpected error: {str(e)}")
        
    finally:
        logging.info(f"Worker {worker_id}: Shutting down after completing {tasks_completed} chunks")



if __name__ == "__main__":
    import pickle
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cell_segmentation_io.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load lookup table and valid labels
    with open('/home/zhangy8@hhmi.org/data1/Experiment/20250304_reconstruction_test/output/id_to_label_20250305.pkl', 'rb') as f:
        lookup_table = pickle.load(f)

    with open('/home/zhangy8@hhmi.org/data1/Experiment/20250304_reconstruction_test/output/proofread_labels_20250305.pkl', 'rb') as f:
        valid_labels_array = pickle.load(f)
    
    valid_labels_set = set(valid_labels_array.flatten())
    
    # Zarr paths
    input_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/source/cell_seg_mag2_20250305.zarr"
    output_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S1/cellshape_s1_converted_20250305_xyz_new.zarr"
    filtered_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S1/cellshape_s1_filtered_20250305_xyz_new.zarr"
    
    # Example application for cell segmentation
    @process_zarr_with_io
    def process_segment_chunk(chunk_info, input_zarr_path, output_paths, 
                            lookup_table, valid_labels_set):
        """
        Process a chunk of a cell segmentation Zarr array.
        
        Parameters:
            chunk_info: Tuple[Tuple, List[slice]]
                Task containing the chunk indices and slices
            input_zarr_path: str
                Path to the input Zarr array
            output_paths: List[str]
                Paths to the output Zarr arrays (processed and filtered)
            lookup_table: Dict
                Lookup table for converting input values to output values
            valid_labels_set: set
                Set of valid labels for filtering
        """
        chunk_indices, slices = chunk_info
        
        try:
            # Open Zarr arrays
            input_zarr = zarr.open(input_zarr_path, mode='r')
            output_zarr = zarr.open(output_paths[0], mode='a')
            filtered_zarr = zarr.open(output_paths[1], mode='a')
            
            # Load data
            start_time = time.time()
            data_chunk = input_zarr[tuple(slices)]
            load_time = time.time() - start_time
            
            # Apply lookup table
            start_time = time.time()
            vectorized_lookup = np.vectorize(lambda x: lookup_table.get(x, 0), otypes=[np.uint32])
            processed_chunk = vectorized_lookup(data_chunk)
            
            # Replace 'None' values with 0
            processed_chunk = np.where(processed_chunk == 'None', 0, processed_chunk)
            
            # Filter valid labels
            vectorized_filter = np.vectorize(lambda value: value if value in valid_labels_set else 0)
            filtered_chunk = vectorized_filter(processed_chunk)
            process_time = time.time() - start_time
            
            # Write results
            start_time = time.time()
            output_zarr[tuple(slices)] = processed_chunk
            filtered_zarr[tuple(slices)] = filtered_chunk
            write_time = time.time() - start_time
            
            total_time = load_time + process_time + write_time
            chunk_shape = tuple(s.stop - s.start for s in slices)
            input_size = np.prod(chunk_shape) * np.dtype(data_chunk.dtype).itemsize / (1024**3)  # size in GB
            
            # Log detailed timing for larger chunks or slow operations
            if total_time > 5 or input_size > 0.5:
                logging.info(f"Chunk {chunk_indices} details: shape={chunk_shape}, "
                            f"size={input_size:.3f}GB, "
                            f"load={load_time:.2f}s, process={process_time:.2f}s, write={write_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_indices}: {str(e)}")
        raise
    # Process using the decorated function
    result = process_segment_chunk(
        input_zarr_path=input_zarr_path,
        output_paths=[output_zarr_path, filtered_zarr_path],
        output_dtypes=["uint32", "uint32"],
        processing_chunk_size=(512, 512, 512),  # Adjust based on memory constraints
        num_workers=64,  # Adjust based on available CPUs
        lookup_table=lookup_table,
        valid_labels_set=valid_labels_set
    )
    
    logging.info(f"Zarr processing complete. Stats: {result}")