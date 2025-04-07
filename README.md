# SegProcess: Dataset Generation from Segmentation Data

SegProcess is a Python package for processing large-scale electron microscopy (EM) segmentation data, with a focus on handling Ariadne's track data. The package provides tools for filtering, processing, fixing, and visualizing segmentation data with support for multiprocessing to handle large datasets efficiently.

## Features

- Process dense labeled oversegmentation data from tools like Ariadne
- Generate datasets of neurons with proofread annotations
- Fix segmentation issues (merged segments) using watershed and erosion techniques
- Multi-thread/multi-process processing of large volumes (>2TB)
- Convert between different data formats and resolutions
- Merge subcellular segmentation (e.g., nuclei) into cell segmentation
- Visualize segmentation data with Neuroglancer
- Analyze cell morphology metrics
- Process mitochondria and calculate their properties

## Installation

### Requirements

- Python 3.7+
- Main dependencies:
  - zarr (2.18.3)
  - graph-tool
  - knossos_utils
  - pandas
  - tifffile
  - dask[distributed]
  - numpy
  - neuroglancer
  - trimesh (for mesh operations)

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/zyx287/SegProcess.git
cd segprocess

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .

# Or in development mode
pip install -e .
```

## Package Structure

The package is organized into several modules:

- **core**: Core dataset handling and segmentation fixing tools
- **graph**: Graph-based segmentation operations
- **io**: Input/output operations for segmentation data
- **processing**: Data processing functions including parallel processing
- **utils**: General utility functions
- **visualization**: Tools for visualizing data with Neuroglancer
- **cli**: Command-line interfaces

## Usage Examples

### Basic Usage

#### Loading a Proofread Dataset

```python
from segprocess.core.dataset import ProofDataset

# Initialize with proofreading Excel file
file_path = '/path/to/EmeraldProofreading.xlsx'
proof_table = ProofDataset(file_path)

# Filter cells with "QA is completed" status
proof_table.filter_excel_ariadne([
    'zone 1', 'zone 2', 'zone 3', 
    'Bergmann-Glia', 'Purkinje-Cells', 'Climbing-Fiber'
])

# Check filtered dataset (pandas)
proof_table.processed_df
```

PS: Check the **warning info**!

#### Processing Segmentation Graph

```python
from segprocess.graph.reader import SegGraph

# Load the segmentation graph
ariadne_graph = SegGraph(graph_path='base_graph.gt')

# Apply changes from Ariadne
ariadne_graph.update_graph_changes('changes.tsv')

# Generate supervoxel-to-label map
ariadne_graph.generate_supervoxel_label_dirc(transform_flag=True)
lookup_table = ariadne_graph.get_supervoxel_to_label
```

#### Converting and Processing Large Volumes

```python
from segprocess.io.converter import convert_zarr_uint16_to_uint8

# Convert uint16 zarr to uint8
result = convert_zarr_uint16_to_uint8(
    input_zarr_path='/path/to/input.zarr',
    output_zarr_path='/path/to/output.zarr',
    min_val=34600,
    max_val=36096,
    processing_chunk_size=(512, 512, 512),
    num_workers=60
)
```

#### Fixing Merged Segments

```python
from segprocess.core.fix import fix_merged_segments_3d_with_watershed

# Fix merged segments using watershed
corrected, stats = fix_merged_segments_3d_with_watershed(
    segmentation,
    h_min=2,
    min_size=50,
    smoothing=1
)
```

#### Visualizing with Neuroglancer

```python
from segprocess.visualization.neuroglancer import view_zarr_with_neuroglancer

# View a zarr dataset with Neuroglancer
view_zarr_with_neuroglancer(
    zarr_path='/path/to/dataset.zarr',
    voxel_size=[16, 16, 16]
)
```

### Advanced Usage

#### Parallel Processing with Multiprocessing

```python
from segprocess.processing.parallel import process_zarr_dynamically
import pickle

# Load lookup table and valid labels
with open('/path/to/id_to_label.pkl', 'rb') as f:
    lookup_table = pickle.load(f)

with open('/path/to/proofread_labels.pkl', 'rb') as f:
    valid_labels_array = pickle.load(f)

# Process a large volume
process_zarr_dynamically(
    input_zarr_path="/path/to/whole_volume.zarr",
    output_zarr_path="/path/to/whole_volume_converted.zarr",
    filtered_zarr_path="/path/to/whole_volume_filtered.zarr",
    lookup=lookup_table,
    valid_labels_array=valid_labels_array,
    num_workers=64
)
```

#### Cell Morphology Analysis

```python
from segprocess.processing.parallel import calculate_bounding_boxes_and_connectivity
from segprocess.processing.parallel import extract_cell_subvolume, analyze_cell_morphology

# Calculate bounding boxes and connectivity graphs
results = calculate_bounding_boxes_and_connectivity(
    zarr_path="/path/to/filtered.zarr",
    output_path="/path/to/results.pkl",
    cell_labels=valid_labels_array,
    num_workers=64
)

# Extract and analyze a specific cell
example_label = list(results.keys())[0]
bbox = results[example_label]['bbox']
subvolume = extract_cell_subvolume(
    zarr_path="/path/to/filtered.zarr",
    label=example_label,
    bbox=bbox
)
morphology = analyze_cell_morphology(subvolume, example_label)
print(f"Cell morphology analysis: {morphology}")
```

#### Mitochondria Analysis Pipeline

The package provides tools for analyzing mitochondria:

1. Generate mitochondria labels:

```python
# See scripts/analysis/mito/generate_mitochondria.py
```

2. Extract meshes and skeletons:

```python
# See scripts/analysis/mito/extract_meshes_and_skeletons.py
```

3. Assign mitochondria to cells:

```python
# See scripts/analysis/mito/assign_cell_id_xyz_simple.py
```

### Command-line Usage

The package provides command-line tools for common operations:

#### Visualizing Data with Neuroglancer

```bash
# View a zarr dataset with Neuroglancer
segprocess-view /path/to/dataset.zarr --voxel-size 16,16,16

# Specify a particular array in the zarr dataset
segprocess-view /path/to/dataset.zarr --array path/to/array
```

## Multi-Processing Framework

SegProcess includes a powerful multi-processing framework for handling large datasets efficiently:

### Basic Multiprocessing

```python
from segprocess.processing.multiprocess import process_large_zarr
import numpy as np

@process_large_zarr
def normalize_data(data, min_val=0, max_val=255):
    """Normalize data to range [0, 1]."""
    data_clipped = np.clip(data, min_val, max_val)
    return ((data_clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Process a large zarr dataset
result = normalize_data(
    input_zarr_path='input.zarr',
    output_zarr_path='output.zarr',
    processing_chunk_size=(1024, 1024, 1024),
    num_workers=32,
    min_val=34600,
    max_val=36096
)
```

### Multiprocessing with I/O Operations

```python
from segprocess.processing.multiprocess import process_zarr_with_io
import zarr
import numpy as np

@process_zarr_with_io
def process_segment_chunk(chunk_info, input_zarr_path, output_paths, 
                        lookup_table, valid_labels_set):
    """Process a chunk of segmentation data."""
    chunk_indices, slices = chunk_info
    
    # Open zarr arrays
    input_zarr = zarr.open(input_zarr_path, mode='r')
    output_zarr = zarr.open(output_paths[0], mode='a')
    filtered_zarr = zarr.open(output_paths[1], mode='a')
    
    # Read data
    data_chunk = input_zarr[tuple(slices)]
    
    # Apply lookup and filtering
    vectorized_lookup = np.vectorize(lambda x: lookup_table.get(x, 0), otypes=[np.uint32])
    processed_chunk = vectorized_lookup(data_chunk)
    
    vectorized_filter = np.vectorize(lambda value: value if value in valid_labels_set else 0)
    filtered_chunk = vectorized_filter(processed_chunk)
    
    # Write results
    output_zarr[tuple(slices)] = processed_chunk
    filtered_zarr[tuple(slices)] = filtered_chunk
```

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Report bugs and request features by creating GitHub issues
- Submit pull requests with improvements to code or documentation
- Share examples of how you're using the package

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Ariadne team for their tools and support
- Special thanks to all contributors and users of this package