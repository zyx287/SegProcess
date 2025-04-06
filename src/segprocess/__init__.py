"""
SegProcess: Processing tools for large-scale electron microscopy segmentation data.
"""

__version__ = "0.3.0"

# Import key functionality for package-level access
from .core.dataset import ProofDataset, SegmentationData
from .core.fix import (
    fix_merged_segments_3d_with_erosion,
    fix_merged_segments_3d_with_watershed,
    fix_merged_segments_3d_with_watershed_elongation,
    analyze_connectivity
)
from .graph.reader import SegGraph
from .io.converter import convert_zarr_uint16_to_uint8, resample_zarr_array
from .io.nuclei import downsample_and_merge_zarr
from .processing.multiprocess import process_large_zarr, process_zarr_with_io
from .visualization.neuroglancer import view_zarr_with_neuroglancer