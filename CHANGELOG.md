# Changelog

## [0.1.1] - 2024-11-24
### Added
- Basic pipeline for processing segmentation data from Ariadne.
- Support for multithreading to compute labels for individual cells.

## [0.1.2] - 2024-12-09
### Added
- Multi-threads processing of dense labeled segmentation chunks

## [0.2.0] - 2024-12-29
### Fixed
  Add file content check in changes.tsv
### Added
  Dask processing of 3D volume (id to label) using multi-threading
  Read and write zarr data from npy segmentation (relative slow)
  Multi-process computation for convert and filter zarr segmentation (Low memory required)

## [0.2.1] - 2025-04-06
### Added
  Merge subcellular segmentation into cell segmentation (nuclei script)
  View zarr dataset using Neuroglancer [2025/03/01]
  Update multiprocess framework and scripts for 16nm segmentation (>2 TB) [2025/03/01]
  Update converter for large zarr in scripts [2025/03/08]
  Multiprocess wrapper for general usage
    For IO functions (segprocess.multiprocess.multiprocess_wrapper_IO) and non-IO functions (segprocess.multiprocess.multiprocess_wrapper_noIO) [2025/03/12]
  Mitochondria analysis scripts [2025/03/12]
### Fixed
  Adding logging and modify functions for efficiency and robustness