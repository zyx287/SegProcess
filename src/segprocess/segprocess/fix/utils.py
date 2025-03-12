'''
author: zyx
date: 2025-03-10
last modified: 2025-03-10
description: 
    Frequently used functions in the fix module.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure

def compare_3d_segmentations(original, corrected, slice_axis=0, slice_indices=None):
    '''
    Compare original and corrected 3D segmentations side by side.
    
    Parms:
        original : numpy.ndarray
            Original 3D segmentation mask
        corrected : numpy.ndarray
            Corrected 3D segmentation mask
        slice_axis : int
            Axis along which to take slices (0, 1, or 2)
        slice_indices : list of int or None
            Specific indices to visualize. If None, equally spaced slices will be shown.
    '''
    # Determine slice indices if not provided
    if slice_indices is None:
        num_slices = min(3, original.shape[slice_axis])
        step = max(1, original.shape[slice_axis] // num_slices)
        slice_indices = list(range(0, original.shape[slice_axis], step))
    
    fig, axes = plt.subplots(2, len(slice_indices), figsize=(4 * len(slice_indices), 8), dpi=600)
    
    for i, slice_idx in enumerate(slice_indices):
        # Extract the slices
        if slice_axis == 0:
            orig_slice = original[slice_idx, :, :]
            corr_slice = corrected[slice_idx, :, :]
        elif slice_axis == 1:
            orig_slice = original[:, slice_idx, :]
            corr_slice = corrected[:, slice_idx, :]
        else:
            orig_slice = original[:, :, slice_idx]
            corr_slice = corrected[:, :, slice_idx]
        
        # Display the slices
        axes[0, i].imshow(orig_slice, cmap='nipy_spectral')
        axes[0, i].set_title(f'Original - Slice {slice_idx}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(corr_slice, cmap='nipy_spectral')
        axes[1, i].set_title(f'Corrected - Slice {slice_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_connectivity(segmentation: np.ndarray, connectivity:int=26) -> dict:
    '''
    Analyzes connectivity properties of a segmentation mask.
    
    Parms:
        3D segmentation mask: numpy.ndarray
    
    Return:
        Connectivity statistics: dict
    '''
    if connectivity == 8:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    elif connectivity == 26:
        # Use 26 connectivity first, if needed, can use more rigid connectivity
        # Test 26 connectivity in toy model, it works well
        struct = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("Invalid connectivity value. Must be 8, 18, or 26.")
    
    labels = np.unique(segmentation)[1:]  # Skip background (0)
    
    multi_part_objects = 0
    total_components = 0
    components_by_label = {}
    
    for label in labels:
        mask = segmentation == label
        labeled, num = ndimage.label(mask, structure=struct)
        
        if num > 1:
            multi_part_objects += 1
        
        total_components += num
        components_by_label[int(label)] = num
    
    return {
        'total_objects': len(labels),
        'total_components': total_components,
        'multi_part_objects': multi_part_objects,
        'average_components_per_object': total_components / len(labels) if len(labels) > 0 else 0,
        'components_by_label': components_by_label
    }

def reassign_labels(segmentation: np.ndarray, new_labels: list) -> np.ndarray:
    '''
    Reassigns labels in a segmentation mask after post-processing
    
    Parms:
        segmentation: numpy.ndarray
            Original 3D segmentation mask
        original_label: numpy.array
            Original labels in the whole segmentation volume

    Returns:
        new_segmentation: numpy.ndarray
            Segmentation mask with reassigned labels
    '''
    pass

