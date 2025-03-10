# SegProcess: Dataset Generation based on segmentation data from Ariadne

Using the `segprocess` package to generate datasets of neurons based on Ariadne's track table. The notebook includes steps to filter, process, and generate segmentation data with proofread annotations.

## Features
- Extracts cells with "QA is completed" status.
- Processes whole volume segmentation for specific zones and cell types.
- Filters and excludes non-proofread cells.
- Generates precomputed data from processed segmentation.

## Requirements
- Python 3.x
- Dependencies (see `requirements.txt` in the repository):
  - `segprocess`

## Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zyx287/SegProcess.git
   cd segprocess
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Packages    
Using
```bash
pip install .
```
or using
```bash
pip install -e .
```

### Example Workflow
Below is an example workflow using the `segprocess` package:

#### 1. Import `ProofDataset` from `segprocess`
```python
from segprocess.segprocess.dataset_generate import ProofDataset
```

#### 2. Load the Excel file containing the Ariadne proofreading table
```python
file_path = '/path/to/your/EmeraldProofreading.xlsx'  # Replace with your file path
proof_table = ProofDataset(file_path)
```

#### 3. Filter segmentation data by zones and cell types
```python
proof_table.filter_excel_ariadne([
    'zone 1', 'zone 2', 'zone 3', 
    'Bergmann-Glia', 'Purkinje cells', 'ClimbingFiber'
])
```

#### 4. Process whole volume segmentation
```python
from segprocess.segprocess.knossos_utils_plugin import (
    load_knossos_dataset,
    generate_segmentation,
    seg_to_tif,
    launch_kimimaro
)
from segprocess.segprocess.graph_reader import SegGraph
ariadne_graph = SegGraph(graph_path='base_graph.gt')
changes_path = 'changes.tsv'
ariadne_graph.update_graph_changes(changes_path)
ariadne_graph.generate_supervoxel_label_dirc(transform_flag=True)
look_up_table = ariadne_graph.get_supervoxel_to_label
```
#### 5. Process segmentation based on the look up table

#### 6. Generate Neuroglancer precomputed data using igneous

### Process Zarr dataset
In src/segprocess/scripts/parallel.py, we provide a pipeline based on multi-process to process large volume array.

After get the id_to_label look up table and the proofread cells' label, execute:
```bash
python parallel.py > output.log &
```
We tested the function on Linux Workstation, processed a 600 GB segmentation.


### Visualize local/cloud data using Neuroglancer
#### Use scripts
Path: visualization/start_neuroglancer.py
```bash
python start_neuroglancer.py <path to zarr dataset > <--bind-port int>
```

<!-- ## Notebook Details
The included Jupyter Notebook provides a step-by-step guide for:
1. Loading and filtering Ariadne proofreading data.
2. Processing and extracting segmentation data based on predefined criteria.
3. Generating precomputed datasets for visualization and analysis. -->

<!-- ## License
This project is licensed under the MIT License. See the `LICENSE` file for details. -->
