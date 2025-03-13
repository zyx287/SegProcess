'''
author: zyx
date: 2025-01-04
last_modified: 2025-03-01
description: 
    Process the cell segmentation based on look-up table, calculate bounding boxes,
    and build a spatial connectivity graph between chunks
'''
import zarr
import numpy as np
from multiprocessing import Process, JoinableQueue, Queue, cpu_count
import pickle
import networkx as nx

def process_and_filter_chunk(task, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set):
    '''
    Process and filter a single chunk based on the task index.
    Parms:
        task: tuple
            Task containing the chunk index and slices.
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to the processed Zarr dataset.
        filtered_zarr_path: str
            Path to the filtered Zarr dataset.
        lookup: dict
            A lookup table mapping input values to output values.
        valid_labels_set: set
            A set of valid labels for filtering.
    '''
    index, slices = task

    input_zarr = zarr.open(input_zarr_path, mode="r")
    output_zarr = zarr.open(output_zarr_path, mode="a")
    filtered_zarr = zarr.open(filtered_zarr_path, mode="a")

    data_chunk = input_zarr[tuple(slices)]

    vectorized_lookup = np.vectorize(lambda x: lookup.get(x, 0), otypes=[np.uint32])
    processed_chunk = vectorized_lookup(data_chunk)

    processed_chunk = np.where(processed_chunk == 'None', 0, processed_chunk)  # Replace 'None' with 0
    vectorized_filter = np.vectorize(lambda value: value if value in valid_labels_set else 0)
    filtered_chunk = vectorized_filter(processed_chunk)

    output_zarr[tuple(slices)] = processed_chunk
    filtered_zarr[tuple(slices)] = filtered_chunk

    print(f"Processed and filtered chunk at index {index}")

def worker(task_queue, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set):
    while True:
        task = task_queue.get()
        if task is None:
            break
        process_and_filter_chunk(task, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set)
        task_queue.task_done()

def process_zarr_dynamically(input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_array, num_workers=None):
    '''
    Dynamically process and filter a Zarr dataset using multiprocessing.

    Parms:
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to the processed Zarr dataset.
        filtered_zarr_path: str
            Path to the filtered Zarr dataset.
        lookup: dict
            A lookup table mapping input values to output values.
        valid_labels_array: np.ndarray
            An array of valid labels for filtering.
        num_workers: int, optional
            Number of worker processes to use. Defaults to the number of CPUs.
    '''
    input_zarr = zarr.open(input_zarr_path, mode="r")
    output_zarr = zarr.open(
        output_zarr_path, mode="w", shape=input_zarr.shape, chunks=input_zarr.chunks, dtype="uint32"
    )
    filtered_zarr = zarr.open(
        filtered_zarr_path, mode="w", shape=input_zarr.shape, chunks=input_zarr.chunks, dtype="uint32"
    )

    valid_labels_set = set(valid_labels_array.flatten())  # Proofread labels should be calculated

    chunk_shape = input_zarr.chunks
    total_shape = input_zarr.shape
    ndim = input_zarr.ndim

    task_index = 0
    task_queue = JoinableQueue()
    for chunk_indices in np.ndindex(*[int(np.ceil(total_shape[dim] / chunk_shape[dim])) for dim in range(ndim)]):
        slices = [
            slice(idx * chunk_shape[dim], min((idx + 1) * chunk_shape[dim], total_shape[dim]))
            for dim, idx in enumerate(chunk_indices)
        ]
        task_queue.put((task_index, slices))
        task_index += 1

    num_workers = num_workers or cpu_count()
    workers = []
    for _ in range(num_workers):
        worker_process = Process(target=worker, args=(
            task_queue, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set
        ))
        worker_process.start()
        workers.append(worker_process)
    task_queue.join()
    for _ in range(num_workers):
        task_queue.put(None)
    for worker_process in workers:
        worker_process.join()

    print("Processing completed.")

def calculate_bounding_box_for_chunk(task, zarr_path, cell_labels_set, discovering_labels):
    '''
    Calculate partial bounding box information for a single chunk.
    
    Parms:
        task: tuple
            Task containing the chunk index and slices.
        zarr_path: str
            Path to the Zarr dataset.
        cell_labels_set: set
            Set of cell labels to process.
        discovering_labels: bool
            Whether to discover new labels during processing.
            
    Returns:
        tuple
            (result, task_id, chunk_indices) where:
            - result is a dictionary mapping labels to their partial bounding box information
            - task_id is the chunk index
            - chunk_indices is the n-dimensional index of the chunk in the dataset
    '''
    index, slices = task
    zarr_dataset = zarr.open(zarr_path, mode="r")
    
    # Extract chunk indices from slices
    chunk_indices = tuple(s.start // c for s, c in zip(slices, zarr_dataset.chunks))
    
    # Read chunk data
    data_chunk = zarr_dataset[tuple(slices)]
    
    # Find unique labels in this chunk
    unique_labels = np.unique(data_chunk)
    
    # Remove background label (0)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Prepare result
    result = {}
    
    for label in unique_labels:
        if not discovering_labels and label not in cell_labels_set:
            continue
            
        mask = data_chunk == label
        if not np.any(mask):
            continue
            
        # Get indices where this label appears
        indices = np.where(mask)
        
        # Convert to absolute coordinates
        abs_indices = []
        for dim, idx_array in enumerate(indices):
            abs_indices.append(idx_array + slices[dim].start)
        
        result[label] = {
            'chunk_index': index,
            'chunk_indices': chunk_indices,  # n-dimensional index
            'slices': slices,
            'min_coords': [int(ind.min()) for ind in abs_indices],
            'max_coords': [int(ind.max()) for ind in abs_indices],
            # Check if label touches any boundary of the chunk
            'boundary': [
                (np.min(indices[dim]) == 0) or (np.max(indices[dim]) == dim_size - 1)
                for dim, dim_size in enumerate(data_chunk.shape)
            ]
        }
    
    print(f"Processed bounding box for chunk at index {index}")
    return result, index, chunk_indices

def worker_bounding_box(tasks, zarr_path, cell_labels_set, discovering_labels, result_queue):
    '''
    Worker function for multiprocessing bounding box calculation.
    
    Parameters:
        tasks: list
            List of tasks to process.
        zarr_path: str
            Path to the Zarr dataset.
        cell_labels_set: set
            Set of cell labels to process.
        discovering_labels: bool
            Whether to discover new labels during processing.
        result_queue: Queue
            Queue to store results.
    '''
    for task in tasks:
        result, task_id, chunk_indices = calculate_bounding_box_for_chunk(task, zarr_path, cell_labels_set, discovering_labels)
        result_queue.put((result, task_id, chunk_indices))

def are_chunks_adjacent(chunk_indices1, chunk_indices2):
    """
    Determine if two chunks are adjacent to each other.
    
    Parameters:
        chunk_indices1: tuple
            N-dimensional indices of the first chunk.
        chunk_indices2: tuple
            N-dimensional indices of the second chunk.
            
    Returns:
        bool
            True if chunks are adjacent, False otherwise.
    """
    if len(chunk_indices1) != len(chunk_indices2):
        return False
    
    # Count differences in indices
    diff_count = 0
    max_diff = 0
    
    for idx1, idx2 in zip(chunk_indices1, chunk_indices2):
        diff = abs(idx1 - idx2)
        max_diff = max(max_diff, diff)
        if diff > 0:
            diff_count += 1
    
    # Chunks are adjacent if they differ in only one dimension by 1
    return diff_count == 1 and max_diff == 1

def build_chunk_connectivity_graph(chunk_data_list):
    """
    Build a graph representing the spatial connectivity between chunks.
    
    Parameters:
        chunk_data_list: list
            List of dictionaries containing chunk information.
            
    Returns:
        networkx.Graph
            Graph representing chunk connectivity.
    """
    G = nx.Graph()
    
    # Add all chunks as nodes
    for chunk_data in chunk_data_list:
        G.add_node(chunk_data['chunk_index'], 
                  chunk_indices=chunk_data['chunk_indices'],
                  slices=chunk_data['slices'],
                  boundary=chunk_data['boundary'])
    
    # Add edges between adjacent chunks
    chunks = list(chunk_data_list)
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            if are_chunks_adjacent(chunks[i]['chunk_indices'], chunks[j]['chunk_indices']):
                G.add_edge(chunks[i]['chunk_index'], chunks[j]['chunk_index'])
    
    return G

def calculate_bounding_boxes_and_connectivity(zarr_path, output_path=None, cell_labels=None, num_workers=None):
    """
    Calculate bounding boxes and chunk connectivity for each cell label in a Zarr dataset.
    
    Parameters:
        zarr_path: str
            Path to the Zarr dataset.
        output_path: str, optional
            Path to save the results as a pickle file.
        cell_labels: list or np.ndarray, optional
            List of cell labels to calculate bounding boxes for.
            If None, all non-zero labels in the dataset will be used.
        num_workers: int, optional
            Number of worker processes to use. Defaults to the number of CPUs.
            
    Returns:
        dict
            A dictionary mapping each cell label to its information including:
            - 'bbox': Bounding box as list of tuples [(z_min, z_max), (y_min, y_max), (x_min, x_max)]
            - 'chunks': List of chunk indices where the label appears
            - 'slices': List of slice objects corresponding to those chunks
            - 'connectivity_graph': NetworkX graph representing chunk connectivity
    """
    zarr_dataset = zarr.open(zarr_path, mode="r")
    
    # If no cell labels provided, we'll discover them during processing
    if cell_labels is None:
        cell_labels_set = set()
        discovering_labels = True
    else:
        cell_labels_set = set(cell_labels)
        discovering_labels = False
    
    chunk_shape = zarr_dataset.chunks
    total_shape = zarr_dataset.shape
    ndim = zarr_dataset.ndim
    
    # Create tasks
    tasks = []
    task_index = 0
    
    # Map from task_index to n-dimensional chunk indices
    index_to_chunk_indices = {}
    
    for chunk_indices in np.ndindex(*[int(np.ceil(total_shape[dim] / chunk_shape[dim])) for dim in range(ndim)]):
        slices = [
            slice(idx * chunk_shape[dim], min((idx + 1) * chunk_shape[dim], total_shape[dim]))
            for dim, idx in enumerate(chunk_indices)
        ]
        tasks.append((task_index, slices))
        index_to_chunk_indices[task_index] = chunk_indices
        task_index += 1
    
    total_tasks = len(tasks)
    
    # Distribute tasks among workers
    num_workers = min(num_workers or cpu_count(), total_tasks)
    tasks_per_worker = [tasks[i::num_workers] for i in range(num_workers)]
    
    # Create result queue
    result_queue = Queue()
    
    # Start worker processes
    workers = []
    for i in range(num_workers):
        worker_process = Process(target=worker_bounding_box, args=(
            tasks_per_worker[i], zarr_path, cell_labels_set, discovering_labels, result_queue
        ))
        worker_process.start()
        workers.append(worker_process)
    
    # Collect results
    label_chunk_dict = {}
    for _ in range(total_tasks):
        result, task_id, chunk_indices = result_queue.get()
        
        # Update label_chunk_dict with this result
        for label, data in result.items():
            if label not in label_chunk_dict:
                label_chunk_dict[label] = []
            label_chunk_dict[label].append(data)
        
        # Print progress
        if (_ + 1) % max(1, total_tasks // 100) == 0:
            print(f"Progress: {_ + 1}/{total_tasks} chunks processed")
    
    # Wait for all workers to finish
    for worker_process in workers:
        worker_process.join()
    
    # Calculate final bounding boxes and build connectivity graphs
    results = {}
    for label, chunk_data_list in label_chunk_dict.items():
        bbox_min = np.array([float('inf')] * ndim)
        bbox_max = np.array([float('-inf')] * ndim)
        
        # Find global min and max across all chunks
        for chunk_data in chunk_data_list:
            min_coords = np.array(chunk_data['min_coords'])
            max_coords = np.array(chunk_data['max_coords'])
            
            bbox_min = np.minimum(bbox_min, min_coords)
            bbox_max = np.maximum(bbox_max, max_coords)
        
        # Create bounding box as list of tuples (min, max) for each dimension
        bbox = [(int(bbox_min[dim]), int(bbox_max[dim])) for dim in range(ndim)]
        
        # Build connectivity graph
        connectivity_graph = build_chunk_connectivity_graph(chunk_data_list)
        
        # Create serializable graph representation (adjacency list)
        adjacency_list = {node: list(neighbors) for node, neighbors in connectivity_graph.adj.items()}
        
        results[label] = {
            'bbox': bbox,
            'chunks': [data['chunk_index'] for data in chunk_data_list],
            'slices': [data['slices'] for data in chunk_data_list],
            'connectivity_graph': connectivity_graph,
            'adjacency_list': adjacency_list,
            'node_data': {node: data for node, data in connectivity_graph.nodes(data=True)}
        }
    
    # Save results to file if output path is provided
    if output_path:
        # NetworkX graphs aren't directly picklable, so we remove them before saving
        save_results = {}
        for label, data in results.items():
            save_data = data.copy()
            save_data.pop('connectivity_graph')  # Remove the graph object
            save_results[label] = save_data
            
        with open(output_path, 'wb') as f:
            pickle.dump(save_results, f)
    
    print("Bounding box and connectivity graph calculation completed.")
    return results

def extract_cell_subvolume(zarr_path, label, bbox, output_path=None):
    """
    Extract a subvolume containing a single cell based on its bounding box.
    
    Parameters:
        zarr_path: str
            Path to the Zarr dataset.
        label: int
            Cell label to extract.
        bbox: list
            Bounding box as list of tuples [(z_min, z_max), (y_min, y_max), (x_min, x_max)].
        output_path: str, optional
            Path to save the extracted subvolume as a Zarr dataset.
            
    Returns:
        np.ndarray
            Extracted subvolume containing only the specified cell.
    """
    zarr_dataset = zarr.open(zarr_path, mode="r")
    
    # Create slices from bounding box
    slices = [slice(min_val, max_val + 1) for min_val, max_val in bbox]
    
    # Extract subvolume
    subvolume = zarr_dataset[tuple(slices)]
    
    # Create binary mask for the specified label
    mask = subvolume == label
    
    # Create labeled subvolume (0 for background, label for the cell)
    labeled_subvolume = np.zeros_like(subvolume)
    labeled_subvolume[mask] = label
    
    # Save to Zarr if output path is provided
    if output_path:
        output_zarr = zarr.open(
            output_path, mode="w", shape=labeled_subvolume.shape, chunks=zarr_dataset.chunks, dtype=labeled_subvolume.dtype
        )
        output_zarr[:] = labeled_subvolume
    
    return labeled_subvolume

def analyze_cell_morphology(labeled_subvolume, label):
    """
    Analyze the morphology of a cell from its labeled subvolume.
    
    Parameters:
        labeled_subvolume: np.ndarray
            Subvolume containing the cell (0 for background, label for cell).
        label: int
            Cell label.
            
    Returns:
        dict
            Dictionary containing morphological metrics.
    """
    # Create binary mask
    mask = labeled_subvolume == label
    
    # Calculate basic metrics
    volume = np.sum(mask)
    surface_voxels = np.logical_xor(mask, np.pad(mask, 1, mode='constant')[1:-1, 1:-1, 1:-1])
    surface_area = np.sum(surface_voxels)
    
    # Calculate center of mass
    indices = np.where(mask)
    center_of_mass = [np.mean(ind) for ind in indices]
    
    # Calculate compactness (ratio of volume to surface area)
    # Higher values indicate more compact (spherical) shape
    compactness = volume / (surface_area**(2/3))
    
    # Calculate elongation using principal components analysis
    if volume > 0:
        # Convert mask to coordinates
        coords = np.column_stack([ind.flatten() for ind in indices])
        
        # Center the coordinates
        centered_coords = coords - np.mean(coords, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_coords, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Calculate elongation (ratio of largest to smallest eigenvalue)
        # Higher values indicate more elongated shape
        elongation = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        
        # Calculate flatness (ratio of second largest to smallest eigenvalue)
        # Higher values indicate more flattened shape
        flatness = eigenvalues[1] / (eigenvalues[-1] + 1e-10)
        
        # Principal axes
        principal_axes = eigenvectors
    else:
        elongation = 0
        flatness = 0
        principal_axes = np.eye(3)
    
    return {
        'volume': volume,
        'surface_area': surface_area,
        'center_of_mass': center_of_mass,
        'compactness': compactness,
        'elongation': elongation,
        'flatness': flatness,
        'principal_axes': principal_axes.tolist(),
        'bbox_dimensions': [max_val - min_val + 1 for min_val, max_val in zip(*[np.min(indices, axis=1), np.max(indices, axis=1)])]
    }

if __name__ == "__main__":
    with open('/home/zhangy8@hhmi.org/data1/20241226_reconstruct/output/id_to_label_20241226.pkl', 'rb') as f:
        lookup_table = pickle.load(f)

    with open('/home/zhangy8@hhmi.org/data1/20241226_reconstruct/output/proofread_labels_20241229.pkl', 'rb') as f:
        valid_labels_array = pickle.load(f)

    input_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_array_s2.zarr"
    output_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_array_s2_converted.zarr"
    filtered_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_array_s2_filtered.zarr"
    results_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_array_s2_cell_analysis.pkl"

    # Process and filter the Zarr dataset
    process_zarr_dynamically(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        filtered_zarr_path=filtered_zarr_path,
        lookup=lookup_table,
        valid_labels_array=valid_labels_array,
        num_workers=64  # Adjust based on available CPUs
    )
    
    # Calculate bounding boxes and connectivity graphs for the filtered Zarr dataset
    results = calculate_bounding_boxes_and_connectivity(
        zarr_path=filtered_zarr_path,
        output_path=results_path,
        cell_labels=valid_labels_array,
        num_workers=64  # Adjust based on available CPUs
    )
    
    print(f"Cell analysis results saved to {results_path}")
    
    # Optional: Extract and analyze a specific cell
    # Replace 'example_label' with an actual label from your dataset
    if len(results) > 0:
        example_label = list(results.keys())[0]
        print(f"Extracting and analyzing cell with label {example_label}")
        
        bbox = results[example_label]['bbox']
        subvolume = extract_cell_subvolume(
            zarr_path=filtered_zarr_path,
            label=example_label,
            bbox=bbox,
            output_path=f"/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/cell_{example_label}.zarr"
        )
        
        morphology = analyze_cell_morphology(subvolume, example_label)
        print(f"Cell morphology analysis: {morphology}")