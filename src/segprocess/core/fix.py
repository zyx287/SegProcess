"""
Segmentation fixing and correction functions.

This module provides algorithms for fixing and cleaning up segmentation data,
particularly for addressing merged segments and other common issues.
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Union, Set, Any
from scipy import ndimage
from skimage import measure, filters
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_connectivity(segmentation: np.ndarray, connectivity: int = 26) -> Dict[str, Any]:
    """
    Analyze connectivity properties of a segmentation mask.
    
    Args:
        segmentation: 3D segmentation mask
        connectivity: Connectivity for 3D structures (6, 18, or 26)
        
    Returns:
        Dictionary with connectivity statistics
        
    Raises:
        ValueError: If connectivity value is invalid
    """
    if connectivity not in [6, 18, 26]:
        raise ValueError("Invalid connectivity value. Must be 6, 18, or 26.")
        
    # Map connectivity value to structural element
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    else:  # connectivity == 26
        struct = ndimage.generate_binary_structure(3, 3)
    
    labels = np.unique(segmentation)[1:]  # Skip background (0)
    
    multi_part_objects = 0
    total_components = 0
    components_by_label = {}
    
    for label in labels:
        mask = segmentation == label
        labeled, num_components = ndimage.label(mask, structure=struct)
        
        if num_components > 1:
            multi_part_objects += 1
        
        total_components += num_components
        components_by_label[int(label)] = num_components
    
    return {
        'total_objects': len(labels),
        'total_components': total_components,
        'multi_part_objects': multi_part_objects,
        'average_components_per_object': total_components / len(labels) if len(labels) > 0 else 0,
        'components_by_label': components_by_label
    }


def fix_merged_segments_3d_with_erosion(segmentation: np.ndarray,
                                      erosion_iterations: int = 1,
                                      connectivity: int = 26, 
                                      min_object_size: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fix incorrectly merged segments in a 3D mask using erosion-based separation.
    
    Args:
        segmentation: 3D labeled segmentation mask where each unique integer represents a different object
        erosion_iterations: Number of erosion iterations to perform
        connectivity: Connectivity for 3D structures (6, 18, or 26)
        min_object_size: Minimum size for objects to be considered significant
        
    Returns:
        Tuple containing:
            - Corrected 3D segmentation mask
            - Dictionary with detailed statistics about the changes made
            
    Raises:
        ValueError: If connectivity value is invalid
    """
    if connectivity not in [6, 18, 26]:
        raise ValueError("Invalid connectivity value. Must be 6, 18, or 26.")
    
    # Step 1: Initial analysis
    initial_stats = analyze_connectivity(segmentation, connectivity=connectivity)
    logger.info(f"Initial analysis: {initial_stats['total_objects']} objects, "
              f"{initial_stats['multi_part_objects']} with multiple components")
    
    # Make a copy of the input segmentation
    corrected_segmentation = np.zeros_like(segmentation)
    
    # Get unique labels (excluding background)
    original_labels = np.unique(segmentation)[1:]
    
    # Track changes for each label
    changes = {}
    
    # Map connectivity value to structural element
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    else:  # connectivity == 26
        struct = ndimage.generate_binary_structure(3, 3)
    
    # For each original label, process separately
    for orig_label in original_labels:
        # Extract this object
        mask = segmentation == orig_label
        
        # Skip very small objects
        if np.sum(mask) < min_object_size:
            corrected_segmentation[mask] = orig_label
            continue
        
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
    
    logger.info(f"Erosion-based separation completed: "
              f"{stats['initial_objects']} → {stats['final_objects']} objects")
    logger.info(f"Modified {stats['labels_modified']} labels, added {stats['objects_added']} new objects")
    
    return corrected_segmentation, stats


def validate_watershed_elongation_3d(original_mask: np.ndarray,
                                   watershed_labels: np.ndarray,
                                   num_regions: int,
                                   min_size: int) -> bool:
    """
    Validates watershed results to reduce false positives based on elongation metrics.
    
    Args:
        original_mask: Original binary mask
        watershed_labels: Result of watershed segmentation
        num_regions: Number of regions from watershed
        min_size: Minimum size threshold for regions
        
    Returns:
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
    improvement_factor = 0.8  # Require 20% improvement in elongation
    return avg_elongation < original_elongation * improvement_factor


def fix_merged_segments_3d_with_watershed_elongation(segmentation: np.ndarray,
                                                  connectivity: int = 26,
                                                  min_object_size: int = 100, 
                                                  validation_metrics: bool = True, 
                                                  h_min: int = 3,
                                                  smoothing: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect and fix false mergers in 3D segmentation using watershed algorithm with elongation validation.
    
    Args:
        segmentation: 3D labeled segmentation mask
        connectivity: Connectivity for 3D structures (6, 18, or 26)
        min_object_size: Minimum size for objects to be considered
        validation_metrics: Whether to apply morphological validation
        h_min: Minimum height for h-maxima detection
        smoothing: Smoothing factor for distance map
        
    Returns:
        Tuple containing:
            - Corrected 3D segmentation mask
            - Dictionary with detailed statistics about the changes made
            
    Raises:
        ValueError: If connectivity value is invalid
    """
    if connectivity not in [6, 18, 26]:
        raise ValueError("Invalid connectivity value. Must be 6, 18, or 26.")
        
    # Map connectivity value to structural element
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    else:  # connectivity == 26
        struct = ndimage.generate_binary_structure(3, 3)
    
    # Step 1: Initial analysis of segmentation
    initial_stats = analyze_connectivity(segmentation, connectivity=connectivity)
    logger.info(f"Initial analysis: {initial_stats['total_objects']} objects, "
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
        labeled_mask, num_components = ndimage.label(mask, structure=struct)
        
        # Process components with already disconnected one by ndimage.label
        if num_components > 1:
            # Check if components are significant in size
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            significant_components = np.sum(component_sizes >= min_object_size)
            
            if significant_components <= 1:
                # Only one significant component, no action needed
                continue
                
            # If the object is already made of multiple significant components,
            # separate them with unique labels
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
        maxima_labeled, num_maxima = ndimage.label(maxima, structure=struct)
        
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
    final_stats = analyze_connectivity(result, connectivity=connectivity)
    
    # Compile comprehensive statistics
    stats = {
        'total_labels_processed': len(labels),
        'labels_modified': len(changes),
        'initial_objects': initial_stats['total_objects'],
        'final_objects': final_stats['total_objects'],
        'objects_added': final_stats['total_objects'] - initial_stats['total_objects'],
        'details_by_label': changes
    }
    
    logger.info(f"Watershed-based separation with elongation validation completed: "
              f"{stats['initial_objects']} → {stats['final_objects']} objects")
    logger.info(f"Modified {stats['labels_modified']} labels, added {stats['objects_added']} new objects")
    
    return result, stats


def fix_merged_segments_3d_with_watershed(segmentation: np.ndarray,
                                        h_min: int = 2,
                                        min_size: int = 50,
                                        smoothing: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Performs watershed separation with conservative parameters to reduce false positives.
    
    Args:
        segmentation: 3D labeled segmentation mask
        h_min: Minimum height for h-maxima detection (higher = fewer splits)
        min_size: Minimum size for a region to be considered a valid split
        smoothing: Amount of gaussian smoothing to apply to distance map
        
    Returns:
        Tuple containing:
            - Corrected 3D segmentation mask
            - Dictionary with detailed statistics about the changes made
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
    
    logger.info(f"Watershed-based separation completed: "
              f"Modified {len(changes)} of {len(labels)} labels")
    logger.info(f"Accepted {total_valid_splits} splits, rejected {total_rejected_splits} splits")
    
    return result, stats


def remove_small_objects_3d(segmentation: np.ndarray, 
                          min_size: int = 100, 
                          background_label: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Remove small objects from a 3D segmentation mask.
    
    Args:
        segmentation: 3D labeled segmentation mask
        min_size: Minimum size (in voxels) for objects to keep
        background_label: Label to assign to removed objects
        
    Returns:
        Tuple containing:
            - Cleaned 3D segmentation mask
            - Dictionary with statistics about removed objects
    """
    # Create a copy for results
    result = segmentation.copy()
    
    # Get unique labels (excluding background)
    labels = np.unique(segmentation)
    labels = labels[labels != background_label]
    
    removed_objects = []
    kept_objects = []
    
    for label in labels:
        # Extract this object
        mask = segmentation == label
        size = np.sum(mask)
        
        if size < min_size:
            # Remove small objects by setting to background
            result[mask] = background_label
            removed_objects.append((int(label), int(size)))
        else:
            kept_objects.append((int(label), int(size)))
    
    # Compile statistics
    stats = {
        'total_objects': len(labels),
        'objects_removed': len(removed_objects),
        'objects_kept': len(kept_objects),
        'removed_objects': removed_objects,
        'min_size': min_size
    }
    
    logger.info(f"Small objects removal completed: "
              f"Removed {len(removed_objects)} of {len(labels)} objects (min size: {min_size})")
    
    return result, stats


def fill_holes_3d(segmentation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fill holes in a 3D segmentation mask for each labeled object.
    
    Args:
        segmentation: 3D labeled segmentation mask
        
    Returns:
        Tuple containing:
            - Segmentation mask with holes filled
            - Dictionary with statistics about filled holes
    """
    # Create a copy for results
    result = segmentation.copy()
    
    # Get unique labels (excluding background)
    labels = np.unique(segmentation)
    labels = labels[labels != 0]
    
    objects_with_holes = 0
    total_holes_filled = 0
    holes_by_label = {}
    
    for label in labels:
        # Extract this object
        mask = segmentation == label
        
        # Fill holes
        filled_mask = ndimage.binary_fill_holes(mask)
        
        # Count holes
        holes = filled_mask & (~mask)
        num_holes_voxels = np.sum(holes)
        
        if num_holes_voxels > 0:
            # Fill holes in the result
            result[filled_mask] = label
            objects_with_holes += 1
            total_holes_filled += num_holes_voxels
            holes_by_label[int(label)] = int(num_holes_voxels)
    
    # Compile statistics
    stats = {
        'total_objects': len(labels),
        'objects_with_holes': objects_with_holes,
        'total_holes_voxels_filled': total_holes_filled,
        'holes_by_label': holes_by_label
    }
    
    logger.info(f"Hole filling completed: "
              f"Filled holes in {objects_with_holes} of {len(labels)} objects "
              f"({total_holes_filled} voxels total)")
    
    return result, stats


# Make functions available at the module level
__all__ = [
    'analyze_connectivity',
    'fix_merged_segments_3d_with_erosion',
    'fix_merged_segments_3d_with_watershed',
    'fix_merged_segments_3d_with_watershed_elongation',
    'remove_small_objects_3d',
    'fill_holes_3d'
]