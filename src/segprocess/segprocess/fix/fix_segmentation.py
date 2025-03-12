'''
author: zyx
date: 2025-03-10
last modified: 2025-03-10
description: 
    2025/03/10: Toy models of watershed and erosion/dilation based segmentation postprocessing
'''
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import measure, filters
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from segprocess.segprocess.fix.utils import analyze_connectivity
from segprocess.segprocess.fix.utils import compare_3d_segmentations

##### erosion
def fix_merged_segments_3d_with_erosion(segmentation:np.ndarray,
                                        erosion_iterations:int=1,
                                        connectivity:int=26, 
                                        min_object_size:int=100)->tuple[np.ndarray, dict]:
    '''
    Fix incorrectly merged segments in a 3D mask using erosion-based separation
    
    Parms:
        segmentation : numpy.ndarray
            3D labeled segmentation mask where each unique integer represents a different object
        erosion_iterations : int
            Number of erosion iterations to perform (more iterations can separate strongly connected objects)
        connectivity : int
            Connectivity for 3D structures (1=6-connected, 2=18-connected, 3=26-connected)
        min_object_size : int
            Minimum size for objects to be considered significant
        
    Returns:
    tuple
        corrected_segmentation: np.ndarray
            fixed 3D segmentation mask
        stats: dict is a dictionary containing detailed statistics about the changes made
    '''
    # Step 1: Initial analysis
    initial_stats = analyze_connectivity(segmentation, connectivity=connectivity)
    print(f"Initial analysis: {initial_stats['total_objects']} objects, " + 
          f"{initial_stats['multi_part_objects']} with multiple components")
    
    # Make a copy of the input segmentation
    corrected_segmentation = np.zeros_like(segmentation)
    
    # Get unique labels (excluding background)
    original_labels = np.unique(segmentation)[1:]
    
    # Track changes for each label
    changes = {}
    
    # For each original label, process separately
    for orig_label in original_labels:
        # Extract this object
        mask = segmentation == orig_label
        
        # Skip very small objects
        if np.sum(mask) < min_object_size:
            corrected_segmentation[mask] = orig_label
            continue
        
        # Create a 3D structural element for erosion based on connectivity
        if connectivity == 1:
            struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        elif connectivity == 2:
            struct = ndimage.generate_binary_structure(3, 2)  # 18-connectivity
        else:
            struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
        
        # Apply erosion to separate touching objects
        eroded = mask.copy()
        for _ in range(erosion_iterations):
            eroded = ndimage.binary_erosion(eroded, structure=struct)
        
        # Label connected components in the eroded mask
        labeled_eroded, num_labels = ndimage.label(eroded, structure=struct)
        
        # Only process if erosion created multiple components
        if num_labels > 1:
            # Track significant components
            significant_components = []
            
            # Process each component
            for label in range(1, num_labels + 1):
                # Extract this eroded component
                component = labeled_eroded == label
                
                # Dilate it back to original size, but constrained by the original mask
                dilated_component = component.copy()
                for _ in range(erosion_iterations + 1):  # +1 to ensure we fully restore the size
                    dilated_temp = ndimage.binary_dilation(dilated_component, structure=struct)
                    # Constrain dilation to the original mask to avoid merging with other objects
                    dilated_component = np.logical_and(dilated_temp, mask)
                
                # Check if component is significant
                component_size = np.sum(dilated_component)
                if component_size >= min_object_size:
                    significant_components.append((label, component_size, dilated_component))
            
            # If we have multiple significant components, separate them
            if len(significant_components) > 1:
                # Record the change
                changes[orig_label] = {
                    'original_components': initial_stats['components_by_label'].get(orig_label, 1),
                    'new_components': len(significant_components),
                    'components_added': len(significant_components) - 1  # -1 because we keep one with original label
                }
                
                # Assign labels
                # Keep first component with original label
                _, _, first_component = significant_components[0]
                corrected_segmentation[first_component] = orig_label
                
                # Assign new labels to other components
                # TODO: New labels should be carefully assigned since they can overlap with other objects
                next_label = np.max(segmentation) + 1
                for i in range(1, len(significant_components)):
                    _, _, component = significant_components[i]
                    corrected_segmentation[component] = next_label
                    next_label += 1
            else:
                # If only one significant component, keep original label
                corrected_segmentation[mask] = orig_label
        else:
            # If erosion didn't create multiple components, keep original label
            corrected_segmentation[mask] = orig_label
    
    # Step 3: Final analysis
    final_stats = analyze_connectivity(corrected_segmentation, connectivity=connectivity)
    
    # Compile comprehensive statistics
    stats = {
        'total_labels_processed': len(original_labels),
        'labels_modified': len(changes),
        'initial_objects': initial_stats['total_objects'],
        'final_objects': final_stats['total_objects'],
        'objects_added': final_stats['total_objects'] - initial_stats['total_objects'],
        'details_by_label': changes
    }
    
    print(f"Final results: {stats['initial_objects']} → {stats['final_objects']} objects")
    print(f"Modified {stats['labels_modified']} labels, added {stats['objects_added']} new objects")
    
    return corrected_segmentation, stats

##### watershed
def fix_merged_segments_3d_with_watershed_elongation(segmentation:np.ndarray,
                                                     connectivity:int=26,
                                                     min_object_size:int=100, 
                                                     validation_metrics:bool=True, 
                                                     h_min:int=3,
                                                     smoothing:int=1)->tuple[np.ndarray, dict]:
    '''
    Detect and fix false mergers in 3D segmentation using watershed algorithm
    ADDED: Morphological validation to reduce false positives (elongation based)
    
    Parms:
        segmentation: numpy.ndarray
            3D labeled segmentation mask
        min_object_size: int
            Minimum size for objects to be considered (ignore small noise)
        validation_metrics: bool
            Whether to apply morphological validation
        h_min: int
            Minimum height for h-maxima detection
        smoothing: int
            Smoothing factor for distance map
    Returns:
        result: numpy.ndarray
            Corrected segmentation with reduced false positives
        stats: dict
            Detailed statistics about changes
    '''
    # Step 1: Initial analysis of segmentation
    initial_stats = analyze_connectivity(segmentation, connectivity=connectivity)
    print(f"Initial analysis: {initial_stats['total_objects']} objects, " + 
          f"{initial_stats['multi_part_objects']} with multiple components")
    
    # Create a copy for results
    result = segmentation.copy()
    
    # Track changes
    changes = {}
    
    # Get unique labels
    labels = np.unique(segmentation)[1:]  # Skip background
    
    for label in labels:
        # Skip small objects
        mask = segmentation == label
        if np.sum(mask) < min_object_size:
            continue
            
        # Stage 1: Check if object has multiple components already
        labeled_mask, num_components = ndimage.label(mask)
        
        # Process components with already disconnected one by ndimage.label
        if num_components > 1:
            # Check if components are significant in size
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            significant_components = np.sum(component_sizes >= min_object_size)
            
            if significant_components <= 1:
                # Only one significant component, no action needed
                continue
                
            # If the object is already made of multiple significant components,
            # separate them with unique labels TODO: Reassign labels carefully
            next_label = np.max(result) + 1
            for comp_idx in range(1, num_components + 1):
                comp_mask = labeled_mask == comp_idx
                comp_size = np.sum(comp_mask)
                
                if comp_size < min_object_size:
                    # Too small, keep as part of original label
                    continue
                    
                if comp_idx == 1:
                    # Keep first component with original label
                    continue
                else:
                    # Assign new labels to other significant components
                    result[comp_mask] = next_label
                    next_label += 1
            
            changes[label] = {
                'type': 'multi_component',
                'num_components': num_components,
                'significant_components': significant_components
            }
            continue
        
        # Stage 2: For single-component objects, use watershed with validation
        distance = ndimage.distance_transform_edt(mask)
        
        # Smooth the distance map to reduce noise
        if smoothing > 0:
            distance = filters.gaussian(distance, sigma=smoothing)
        
        # Detect h-maxima (h-min higher than neighbor, finding the peaks)
        maxima = h_maxima(distance, h_min)
        
        # Label maxima
        maxima_labeled, num_maxima = ndimage.label(maxima)
        
        if num_maxima <= 1:
            # Only one maximum, no need to split
            continue
            
        # Apply watershed 
        labels_ws = watershed(-distance, maxima_labeled, mask=mask)
        
        # Validate watershed results
        if validation_metrics:
            # Calculate overlap between watershed regions and original object
            is_valid_split = validate_watershed_elongation_3d(
                mask, labels_ws, num_maxima, min_object_size)
                
            if not is_valid_split:
                continue
        
        # Apply the validated splits
        next_label = np.max(result) + 1
        for region_idx in range(1, num_maxima + 1):
            region_mask = labels_ws == region_idx
            region_size = np.sum(region_mask)
            
            if region_size < min_object_size:
                # Too small, skip
                continue
                
            if region_idx == 1:
                # Keep first region with original label
                continue
            else:
                # Assign new labels to other regions
                result[region_mask] = next_label
                next_label += 1
                
        changes[label] = {
            'type': 'watershed_split',
            'num_maxima': num_maxima
        }
    
    # Final analysis
    final_stats = analyze_connectivity(result)
    
    # Compile comprehensive statistics
    stats = {
        'total_labels_processed': len(labels),
        'labels_modified': len(changes),
        'initial_objects': initial_stats['total_objects'],
        'final_objects': final_stats['total_objects'],
        'objects_added': final_stats['total_objects'] - initial_stats['total_objects'],
        'details_by_label': changes
    }
    
    print(f"Final results: {stats['initial_objects']} → {stats['final_objects']} objects")
    print(f"Modified {stats['labels_modified']} labels, added {stats['objects_added']} new objects")
    
    return result, stats

def validate_watershed_elongation_3d(original_mask,
                                     watershed_labels,
                                     num_regions,
                                     min_size):
    """
    Validates watershed results to reduce false positives.
    Uses 3D-compatible metrics instead of perimeter calculation.
    
    Parameters:
    -----------
    original_mask : numpy.ndarray
        Original binary mask
    watershed_labels : numpy.ndarray
        Result of watershed segmentation
    num_regions : int
        Number of regions from watershed
    min_size : int
        Minimum size threshold for regions
        
    Returns:
    --------
    bool
        True if the watershed result is valid, False otherwise
    """
    # Check region sizes
    valid_regions = 0
    for i in range(1, num_regions + 1):
        region = watershed_labels == i
        if np.sum(region) >= min_size:
            valid_regions += 1
    
    # Need at least 2 valid regions for a split
    if valid_regions < 2:
        return False
    
    # Calculate elongation of the original object
    # We'll use the ratio of principal axes of the object as a shape metric
    original_props = measure.regionprops(original_mask.astype(int))[0]
    
    # Get the eigenvalues of the inertia tensor (principal axes lengths)
    lambda1, lambda2, lambda3 = original_props.inertia_tensor_eigvals
    
    # Calculate elongation (ratio of largest to smallest axis)
    if lambda3 > 0:  # Avoid division by zero
        original_elongation = lambda1 / lambda3
    else:
        original_elongation = float('inf')
    
    # Calculate elongation of each watershed region
    region_elongations = []
    region_volumes = []
    
    for i in range(1, num_regions + 1):
        region = watershed_labels == i
        region_volume = np.sum(region)
        
        if region_volume < min_size:
            continue
            
        # Get region properties
        props = measure.regionprops(region.astype(int))
        if not props:  # Skip if empty
            continue
            
        # Get eigenvalues
        lambda1, lambda2, lambda3 = props[0].inertia_tensor_eigvals
        
        # Calculate elongation
        if lambda3 > 0:
            region_elongation = lambda1 / lambda3
        else:
            region_elongation = float('inf')
            
        region_elongations.append(region_elongation)
        region_volumes.append(region_volume)
    
    # No valid regions found
    if not region_elongations:
        return False
    
    # Calculate volume-weighted average elongation
    total_volume = sum(region_volumes)
    avg_elongation = sum(e * v for e, v in zip(region_elongations, region_volumes)) / total_volume
    
    # Split is valid if the average elongation of split regions is less than
    # the original elongation (means objects are less elongated, more compact)
    return avg_elongation < original_elongation * 0.8  # 20% improvement in elongation

def fix_merged_segments_3d_with_watershed(segmentation, h_min=2, min_size=50, smoothing=1):
    """
    Performs watershed separation with conservative parameters to reduce false positives.
    
    Parameters:
    -----------
    segmentation : numpy.ndarray
        3D labeled segmentation mask
    h_min : int
        Minimum height for h-maxima detection (higher = fewer splits)
    min_size : int
        Minimum size for a region to be considered a valid split
    smoothing : int
        Amount of gaussian smoothing to apply to distance map
        
    Returns:
    --------
    numpy.ndarray
        Corrected segmentation with reduced false positives
    dict
        Statistics about changes made
    """
    # Create a copy for the result
    result = segmentation.copy()
    
    # Process each label separately to maintain control
    labels = np.unique(segmentation)[1:]  # Skip background
    
    changes = {}
    total_valid_splits = 0
    total_rejected_splits = 0
    
    for label in labels:
        # Extract this object
        mask = segmentation == label
        
        # Skip very small objects
        if np.sum(mask) < min_size * 2:
            continue
        
        # Compute distance transform
        distance = ndimage.distance_transform_edt(mask)
        
        # Apply gaussian smoothing to the distance map to reduce noise
        if smoothing > 0:
            distance = filters.gaussian(distance, sigma=smoothing)
        
        maxima = h_maxima(distance, h_min)
        maxima_labeled, num_maxima = ndimage.label(maxima)
        if num_maxima <= 1:
            continue

        labels_ws = watershed(-distance, maxima_labeled, mask=mask)
        
        # Validate potential splits based on size
        valid_regions = []
        for i in range(1, num_maxima + 1):
            region = labels_ws == i
            region_size = np.sum(region)
            if region_size >= min_size:
                valid_regions.append((i, region_size))
        
        # If fewer than 2 significant regions, skip
        if len(valid_regions) <= 1:
            total_rejected_splits += 1
            continue
        
        # Apply the split
        next_label = np.max(result) + 1
        for i, (region_idx, _) in enumerate(valid_regions):
            region_mask = labels_ws == region_idx
            
            if i == 0:
                # Keep first region with original label
                continue
            else:
                # Assign new labels to other regions
                result[region_mask] = next_label
                next_label += 1
        
        changes[label] = {
            'num_maxima': num_maxima,
            'valid_regions': len(valid_regions)
        }
        
        total_valid_splits += 1
    
    # Compile statistics
    stats = {
        'total_labels': len(labels),
        'labels_modified': len(changes),
        'valid_splits_accepted': total_valid_splits,
        'splits_rejected': total_rejected_splits,
        'details_by_label': changes
    }
    
    return result, stats

if __name__ = '__main__':
    # Create a toy model
    mask = np.zeros((30, 30, 30), dtype=np.uint8)

    corrected, stats = fix_merged_segments_3d_with_erosion(
        mask,
        erosion_iterations=1,
        connectivity=3,
        min_object_size=100
    )


    water_corrected, stats = advanced_false_merger_detection(
        mask,
        min_object_size=100,  # Adjust based on your data
        validation_metrics=True,
        h_min=3,  # Higher means fewer splits
        smoothing=1  # Smoothing helps reduce noise-induced splits
    )
    compare_3d_segmentations(mask, water_corrected, slice_indices=[13, 15, 17, 19, 21])
