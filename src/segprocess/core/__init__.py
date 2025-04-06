"""
Core functionality for dataset processing and fixing.
"""

from .dataset import ProofDataset, SegmentationData
from .fix import (
    analyze_connectivity,
    fix_merged_segments_3d_with_erosion,
    fix_merged_segments_3d_with_watershed,
    fix_merged_segments_3d_with_watershed_elongation,
    remove_small_objects_3d,
    fill_holes_3d
)

__all__ = [
    'ProofDataset',
    'SegmentationData',
    'analyze_connectivity',
    'fix_merged_segments_3d_with_erosion',
    'fix_merged_segments_3d_with_watershed',
    'fix_merged_segments_3d_with_watershed_elongation',
    'remove_small_objects_3d',
    'fill_holes_3d'
]