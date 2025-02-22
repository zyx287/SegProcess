'''
author: zyx
date: 2025-02-21
last_modified: 2025-02-21
description: 
    Pull segmentation from the ariadne dataset using the MultiprocessingFramework.
'''
import zarr
import numpy as np
from typing import Dict, Any, Iterator
from segprocess.segprocess.contrib.multiprocess import (MultiprocessingFramework, 
    Task, 
    compute_chunks_from_shape,
    measure_time,
    zarr_chunk_exists
)

###############################################
# Example 1: Adapting parallel_nuclei.py
###############################################

def nuclei_task_generator() -> Iterator[Dict[str, Any]]:
    """Generate tasks for nuclei processing."""
    # Load zarr datasets to get shapes
    cell_zarr = zarr.open("/path/to/cell_zarr.zarr", mode="r")
    nuclei_zarr = zarr.open("/path/to/nuclei_zarr.zarr", mode="r")
    
    # Generate tasks based on chunks
    chunk_shape = nuclei_zarr.chunks
    total_shape = nuclei_zarr.shape
    
    return iter(compute_chunks_from_shape(total_shape, chunk_shape))

def process_nuclei_chunk(task: Task, shared_data: Dict[str, Any]) -> None:
    """Process a single nuclei chunk."""
    # Extract parameters from task
    slices = task.params["slices"]
    
    # Get paths from shared data
    cell_zarr_path = shared_data["cell_zarr_path"]
    nuclei_zarr_path = shared_data["nuclei_zarr_path"]
    output_zarr_path = shared_data["output_zarr_path"]
    
    # Open datasets
    cell_zarr = zarr.open(cell_zarr_path, mode="r")
    nuclei_zarr = zarr.open(nuclei_zarr_path, mode="r")
    output_zarr = zarr.open(output_zarr_path, mode="a")
    
    # Check if chunk already processed
    if zarr_chunk_exists(output_zarr, slices):
        return
    
    # Process chunk (equivalent to downsample_and_merge_chunk from original)
    with measure_time(f"Process chunk {task.id}"):
        # Double the slices for cell zarr (for downsampling)
        cell_slices = [slice(s.start * 2, s.stop * 2) for s in slices]
        
        # Read chunks
        cell_chunk = cell_zarr[tuple(cell_slices)]
        downsampled_cell = cell_chunk[::2, ::2, ::2]
        nuclei_chunk = nuclei_zarr[tuple(slices)]
        
        # Create output chunk
        output_chunk = np.zeros_like(nuclei_chunk, dtype=np.uint32)
        
        # Merge logic: if both cell and nuclei are labeled, use cell label
        mask_both_labeled = (downsampled_cell > 0) & (nuclei_chunk > 0)
        output_chunk[mask_both_labeled] = downsampled_cell[mask_both_labeled]
        
        # Write result
        output_zarr[tuple(slices)] = output_chunk

def run_nuclei_processing():
    """Run the nuclei processing pipeline using the framework."""
    # Define shared data
    shared_data = {
        "cell_zarr_path": "/path/to/cell_zarr.zarr",
        "nuclei_zarr_path": "/path/to/nuclei_zarr.zarr",
        "output_zarr_path": "/path/to/output_zarr.zarr"
    }
    
    # Initialize output zarr if it doesn't exist
    cell_zarr = zarr.open(shared_data["cell_zarr_path"], mode="r")
    nuclei_zarr = zarr.open(shared_data["nuclei_zarr_path"], mode="r")
    
    # Create output zarr if it doesn't exist
    zarr.open(
        shared_data["output_zarr_path"], 
        mode="w", 
        shape=nuclei_zarr.shape, 
        chunks=nuclei_zarr.chunks, 
        dtype=np.uint32
    )
    
    # Create and run the framework
    framework = MultiprocessingFramework(
        process_func=process_nuclei_chunk,
        task_generator=nuclei_task_generator,
        shared_data=shared_data,
        num_workers=64,
        checkpoint_file="nuclei_processing_checkpoint.json"
    )
    
    # Run processing
    result = framework.process()
    print(f"Processing completed: {result}")
    
    # Retry any failed tasks
    if result["failed_tasks"] > 0:
        retry_result = framework.retry_failed_tasks()
        print(f"Retry completed: {retry_result}")


###############################################
# Example 2: Adapting parallel_pullSegmentation.py
###############################################

def segmentation_task_generator() -> Iterator[Dict[str, Any]]:
    """Generate tasks for pulling segmentation data."""
    # Define total volume and chunk sizes
    total_volume_size = (27491, 15255, 12548)
    chunk_size = (2048, 2048, 2048)
    
    return iter(compute_chunks_from_shape(total_volume_size, chunk_size))

def process_segmentation_chunk(task: Task, shared_data: Dict[str, Any]) -> None:
    """Process a single segmentation chunk."""
    from knossos_utils import KnossosDataset
    import requests
    
    # Extract parameters
    slices = task.params["slices"]
    volume_offset = task.params["offset"]
    volume_size = task.params["size"]
    
    # Get paths from shared data
    toml_path = shared_data["toml_path"]
    output_zarr_path = shared_data["output_zarr_path"]
    mag_size = shared_data["mag_size"]
    
    # Open output zarr
    output_zarr = zarr.open(output_zarr_path, mode="a")
    
    # Calculate downsampled slices
    downsampled_slices = [
        slice(s.start // mag_size, s.stop // mag_size)
        for s in slices
    ]
    
    # Check if chunk already processed
    if zarr_chunk_exists(output_zarr, downsampled_slices):
        return
    
    # Initialize KnossosDataset
    kdataset = KnossosDataset()
    kdataset.initialize_from_conf(toml_path)
    
    # Create retry session (simplified for example)
    session = requests.Session()
    
    # Load segmentation data with retry logic
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            chunk_data = kdataset.load_seg(
                offset=tuple(volume_offset),
                size=tuple(volume_size),
                mag=mag_size
            )
            
            # Convert from ZYX to XYZ order
            chunk_data = np.transpose(chunk_data, (2, 1, 0))
            
            # Write to output zarr
            output_zarr[tuple(downsampled_slices)] = chunk_data
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait and retry
                import time
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                # Max retries reached, re-raise exception
                raise

def run_segmentation_pulling():
    """Run the segmentation pulling pipeline using the framework."""
    # Define shared data
    shared_data = {
        "toml_path": "/path/to/config.toml",
        "output_zarr_path": "/path/to/output.zarr",
        "mag_size": 2
    }
    
    # Get dimensions for output zarr
    total_volume_size = (27491, 15255, 12548)
    chunk_size = (2048, 2048, 2048)
    mag_size = shared_data["mag_size"]
    
    # Calculate downsampled dimensions
    downsampled_size = tuple(d // mag_size for d in total_volume_size)
    downsampled_chunk_size = tuple(d // mag_size for d in chunk_size)
    
    # Create output zarr if it doesn't exist
    zarr.open(
        shared_data["output_zarr_path"],
        mode="w",
        shape=downsampled_size,
        chunks=downsampled_chunk_size,
        dtype=np.uint32
    )
    
    # Create and run the framework
    framework = MultiprocessingFramework(
        process_func=process_segmentation_chunk,
        task_generator=segmentation_task_generator,
        shared_data=shared_data,
        num_workers=10,
        checkpoint_file="segmentation_pulling_checkpoint.json",
        max_retries=5,
        retry_delay=10
    )
    
    # Run processing
    result = framework.process()
    print(f"Processing completed: {result}")
    
    # Retry any failed tasks with increased retry count
    if result["failed_tasks"] > 0:
        retry_result = framework.retry_failed_tasks()
        print(f"Retry completed: {retry_result}")


###############################################
# Example 3: Adapting graph_reader.py to multiprocessing
###############################################

def graph_task_generator() -> Iterator[Dict[str, Any]]:
    """
    Generate tasks for processing graph components.
    This example splits a graph into components for parallel processing.
    """
    import graph_tool.all as gt
    
    # Load the base graph
    graph_path = "/path/to/graph.gt"
    base_graph = gt.load_graph(graph_path)
    
    # Find all connected components
    comp, hist = gt.label_components(base_graph)
    
    # Generate a task for each component
    tasks = []
    for component_id in range(len(hist)):
        # Find vertices in this component
        vertices = [v for v in base_graph.vertices() if comp[v] == component_id]
        if vertices:
            tasks.append({
                "component_id": component_id,
                "vertex_count": len(vertices),
                "sample_vertex": int(vertices[0])
            })
    
    return iter(tasks)

def process_graph_component(task: Task, shared_data: Dict[str, Any]) -> None:
    """Process a single graph component."""
    import graph_tool.all as gt
    
    # Extract parameters
    component_id = task.params["component_id"]
    sample_vertex = task.params["sample_vertex"]
    
    # Get paths from shared data
    