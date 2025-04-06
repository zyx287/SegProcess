"""
Utility functions for working with knossos datasets.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

def load_knossos_dataset(toml_path: str,
                        volume_offset: Tuple[int, int, int],
                        volume_size: Tuple[int, int, int],
                        mag_size: int = 4) -> np.ndarray:
    """
    Load a knossos dataset using a toml file.
    
    Args:
        toml_path: Path to the toml file
        volume_offset: Offset in (x, y, z) format
        volume_size: Size in (x, y, z) format
        mag_size: Magnification level
        
    Returns:
        Numpy array with the loaded data
        
    Raises:
        ImportError: If knossos_utils is not available
    """
    try:
        from knossos_utils.knossosdataset import KnossosDataset
    except ImportError:
        raise ImportError("knossos_utils is required for this function. "
                        "Install it with: pip install knossos_utils")
    
    kdataset = KnossosDataset()
    kdataset.initialize_from_conf(toml_path)
    around_volume = kdataset.load_seg(offset=volume_offset, size=volume_size, mag=mag_size)
    return around_volume


def generate_segmentation(segmented_volume: np.ndarray, 
                        mergelist_path: str) -> np.ndarray:
    """
    Generate a segmentation mask from a segmented volume and a mergelist.
    
    Args:
        segmented_volume: The segmented volume
        mergelist_path: Path to the mergelist file
        
    Returns:
        Segmentation mask
    """
    with open(mergelist_path, 'r') as file:
        file_contents = file.read()
        number_set = set(map(int, file_contents.split()))  # Speed up using searching based on set
    
    mask_segmentation = np.array([1 if ((elem in number_set) and (elem != 0)) else 0 
                                for elem in segmented_volume.flatten()])
    mask_segmentation_reshaped = mask_segmentation.reshape(segmented_volume.shape)
    return mask_segmentation_reshaped


def seg_to_tif(segmentation: np.ndarray, save_path: str) -> None:
    """
    Save segmentation mask to TIFF file.
    
    Args:
        segmentation: Segmentation mask
        save_path: Path to save the TIFF file
        
    Raises:
        ImportError: If tifffile is not available
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for this function. "
                         "Install it with: pip install tifffile")
                         
    segmentation_tif = (segmentation * 255).astype(np.uint8)
    tifffile.imwrite(save_path, segmentation_tif)


def seg_to_npy(segmentation: np.ndarray, save_path: str) -> None:
    """
    Save segmentation mask to NPY file.
    
    Args:
        segmentation: Segmentation mask
        save_path: Path to save the NPY file
    """
    np.save(save_path, segmentation)


def launch_kimimaro(file_path: str, 
                  env_name: str = 'kimimaro_env') -> int:
    """
    Launch Kimimaro for skeletonization.
    
    Args:
        file_path: Path to the input file
        env_name: Name of the conda environment with kimimaro installed
        
    Returns:
        Exit code (0 for success)
        
    Raises:
        ImportError: If os module is not available
    """
    import os
    import subprocess
    
    # Activate conda environment
    conda_command = f"conda activate {env_name}"
    os.system(conda_command)
    
    # Run kimimaro
    kimimaro_command = f"kimimaro forge {file_path} --progress"
    return subprocess.call(kimimaro_command, shell=True)


# Make functions available at the module level
__all__ = [
    'load_knossos_dataset',
    'generate_segmentation',
    'seg_to_tif',
    'seg_to_npy',
    'launch_kimimaro'
]