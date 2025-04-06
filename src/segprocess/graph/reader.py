"""
Graph-based operations for segmentation data.

This module provides classes and functions for processing segmentation graphs,
particularly for working with Ariadne's track data.
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import graph_tool.all as gt
import subprocess
import logging
from typing import Dict, Set, Tuple, Optional, Union, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SegGraph:
    """
    Class for processing segmentation graphs from Ariadne.
    
    This class provides methods for loading, updating, and analyzing segmentation graphs,
    with a focus on generating lookup tables for processing whole-volume segmentation.
    """
    def __init__(self, graph_path: str):
        """
        Initialize a SegGraph instance by loading a graph from the given path.
        
        Args:
            graph_path: Path to the graph file
            
        Raises:
            FileNotFoundError: If the graph file does not exist
            ValueError: If the graph file is invalid or missing required properties
        """
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        try:
            self.base_graph = gt.load_graph(str(graph_path))
            self.seg_id_to_vertex, self.vertex_to_segid = self._load_base_graph_segids()
            self.supervoxel_to_label = {}  # Will be populated by generate_supervoxel_label_dirc
            logger.info(f"Successfully loaded graph with {self.base_graph.num_vertices()} vertices and {self.base_graph.num_edges()} edges")
        except Exception as e:
            raise ValueError(f"Failed to load graph: {e}")
        
    def _load_base_graph_segids(self) -> Tuple[Dict[int, gt.Vertex], Dict[int, int]]:
        """
        Load the segmentation IDs from the base graph.
        
        Returns:
            Tuple containing:
                - Dict mapping segmentation IDs to vertex IDs
                - Dict mapping vertex IDs to segmentation IDs
                
        Raises:
            ValueError: If the graph does not contain required properties
        """
        if 'seg_ids' not in self.base_graph.vp:
            raise ValueError("Graph does not contain 'seg_ids' vertex property. "
                           "Please ensure the graph was created with segmentation IDs.")
            
        seg_id_group = self.base_graph.vp['seg_ids']
        seg_id_to_vertex = {}
        vertex_to_seg_id = {}

        for v in self.base_graph.vertices():
            v_int = int(v)
            seg_id = int(seg_id_group[v])

            seg_id_to_vertex[seg_id] = v
            vertex_to_seg_id[v_int] = seg_id

        return seg_id_to_vertex, vertex_to_seg_id

    def get_latest_changes(self, changes_url: str = 'https://emerald.agglomeration.ariadne.ai/api/changes.tsv') -> bool:
        """
        Download the latest changes file from the specified URL.
        
        Args:
            changes_url: URL to download changes.tsv from
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        output_path = Path.cwd() / "changes.tsv"

        try:
            # Use subprocess with captured output
            result = subprocess.run(
                ["wget", changes_url, "-O", str(output_path)], 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info(f"Download successful! Changes file saved as {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed with error code {e.returncode}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Download failed with error: {e}")
            return False
        
    def update_graph_changes(self, change_log_path: str) -> int:
        """
        Update the graph according to the change log.
        
        Args:
            change_log_path: Path to the change log file (TSV format)
            
        Returns:
            int: Number of operations applied
            
        Raises:
            FileNotFoundError: If the change log file does not exist
            ValueError: If the change log file is invalid or contains invalid operations
        """
        change_log_path = Path(change_log_path)
        if not change_log_path.exists():
            raise FileNotFoundError(f"Change log file not found: {change_log_path}")

        try:
            # Read the entire file into a DataFrame
            changes = pd.read_csv(
                change_log_path,
                sep='\t',
                header=None,
                names=['operation', 'vertex1', 'vertex2'],
                dtype={'operation': str, 'vertex1': 'Int64', 'vertex2': 'Int64'}
            )
        except Exception as e:
            raise ValueError(f"Failed to read change log file: {e}")

        # Apply changes to the base graph
        operation_count = 0
        errors = 0

        for index, row in changes.iterrows():
            operation = row['operation']
            seg_id_1 = row['vertex1']
            seg_id_2 = row['vertex2']

            # Check if the operation is valid
            if operation not in ['+', '-']:
                logger.warning(f"Ignored line {index + 1}: Invalid operation '{operation}'")
                errors += 1
                continue

            if pd.isna(seg_id_1) or pd.isna(seg_id_2):
                logger.warning(f"Ignored line {index + 1}: Missing vertex ID")
                errors += 1
                continue

            # Convert to int if necessary
            seg_id_1 = int(seg_id_1)
            seg_id_2 = int(seg_id_2)

            # Check if the vertices exist in the graph
            if seg_id_1 in self.seg_id_to_vertex and seg_id_2 in self.seg_id_to_vertex:
                vertex1 = self.seg_id_to_vertex[seg_id_1]
                vertex2 = self.seg_id_to_vertex[seg_id_2]

                if operation == '+':
                    # Add edge if it doesn't already exist
                    if not self.base_graph.edge(vertex1, vertex2):
                        self.base_graph.add_edge(vertex1, vertex2)
                        operation_count += 1
                elif operation == '-':
                    # Remove edge if it exists
                    edge = self.base_graph.edge(vertex1, vertex2)
                    if edge is not None:
                        self.base_graph.remove_edge(edge)
                        operation_count += 1
            else:
                missing_ids = []
                if seg_id_1 not in self.seg_id_to_vertex:
                    missing_ids.append(str(seg_id_1))
                if seg_id_2 not in self.seg_id_to_vertex:
                    missing_ids.append(str(seg_id_2))
                    
                logger.warning(f"Line {index + 1}: Segmentation IDs not found in graph: {', '.join(missing_ids)}")
                errors += 1

        # Print the number of operations applied
        logger.info(f"Applied {operation_count} operations to the graph. Encountered {errors} errors.")
        return operation_count
    
    def extract_connected_components(self, start_vertex_supervoxel_id: int) -> Set[int]:
        """
        Extract all connected components of the given supervoxel from the base graph.
        
        This function performs a depth-first search (DFS) from the specified supervoxel
        to identify all connected supervoxels, which is useful for generating merge lists
        of specific neurons.
        
        Args:
            start_vertex_supervoxel_id: Supervoxel ID to start the search from
            
        Returns:
            Set of connected supervoxel IDs
            
        Raises:
            ValueError: If the start vertex supervoxel ID is not found in the base graph
        """
        if start_vertex_supervoxel_id not in self.seg_id_to_vertex:
            raise ValueError(f"Supervoxel {start_vertex_supervoxel_id} not found in the base graph")
        
        start_vertex = self.seg_id_to_vertex[start_vertex_supervoxel_id]
        logger.info(f"Starting DFS from seg_id {start_vertex_supervoxel_id} (vertex {int(start_vertex)})")

        visited_vertices_segid = set()
        
        # Use graph-tool's label_components for better performance
        comp, hist = gt.label_components(self.base_graph, directed=False)
        start_comp = comp[start_vertex]
        
        # Collect all vertices in the same component
        for v in self.base_graph.vertices():
            if comp[v] == start_comp:
                v_int = int(v)
                if v_int in self.vertex_to_segid:
                    visited_vertices_segid.add(self.vertex_to_segid[v_int])
        
        logger.info(f"Found {len(visited_vertices_segid)} connected supervoxels")
        
        return visited_vertices_segid
    
    def generate_supervoxel_label_dirc(self, transform_flag: bool = True) -> bool:
        """
        Generate the {supervoxel: label} dictionary for processing whole-volume segmentation.
        
        This method computes connected components in the graph and assigns a unique label
        to each component, which can then be used to convert supervoxel IDs to cell labels.
        
        Args:
            transform_flag: Whether to transform labels to a new format with a prefix
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Calculate connected components
            comp, hist = gt.label_components(self.base_graph, directed=False)
            
            # Map vertices to component labels
            vertex_to_label = {int(v): comp[v] for v in self.base_graph.vertices()}
            
            # Map supervoxel IDs to component labels
            supervoxel_to_label = {}
            for sv_id, v in self.seg_id_to_vertex.items():
                label = vertex_to_label[int(v)]
                supervoxel_to_label[sv_id] = label
            
            if not transform_flag:
                self.supervoxel_to_label = supervoxel_to_label
                logger.info(f"Generated supervoxel to label dictionary with {len(supervoxel_to_label)} entries")
                return True
                
            # Transform the labels to a format that's easier to identify
            # For example, add a prefix based on the number of digits
            label_to_new_label = {}
            start_number = 9 * (10 ** (len(str(len(hist)))))  # 9 is the prefix for cell shape
            
            for label in range(len(hist)):
                label_to_new_label[label] = start_number + label
                
            supervoxel_to_new_label = {}
            for sv_id, label in supervoxel_to_label.items():
                new_label = label_to_new_label[label]
                supervoxel_to_new_label[sv_id] = new_label
                
            self.supervoxel_to_label = supervoxel_to_new_label
            logger.info(f"Generated transformed supervoxel to label dictionary with {len(supervoxel_to_new_label)} entries")

        except Exception as e:
            logger.error(f"Failed to generate supervoxel label directory: {e}")
            return False
        
        return True

    @property
    def get_supervoxel_to_label(self) -> Dict[int, int]:
        """
        Get the supervoxel to label dictionary.
        
        Returns:
            Dict mapping supervoxel IDs to labels
        """
        if not self.supervoxel_to_label:
            logger.warning("Supervoxel to label dictionary is empty. Call generate_supervoxel_label_dirc first.")
        return self.supervoxel_to_label
    
    def save_graph(self, output_file_path: str) -> bool:
        """
        Save the graph to the specified output path.
        
        Args:
            output_file_path: Path to save the graph
            
        Returns:
            bool: True if successful, False otherwise
        """
        output_path = Path(output_file_path)
        folder_path = output_path.parent
        folder_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.base_graph.save(str(output_path))
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date_file_path = folder_path / "save_date_info"
            
            with open(date_file_path, 'w') as date_file:
                date_file.write(f"Graph saved on: {current_date}")
                
            logger.info(f"Graph successfully saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False
        
    def get_graph_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the graph.
        
        Returns:
            Dict containing graph statistics
        """
        # Calculate connected components
        comp, hist = gt.label_components(self.base_graph, directed=False)
        
        stats = {
            'num_vertices': self.base_graph.num_vertices(),
            'num_edges': self.base_graph.num_edges(),
            'num_components': len(hist),
            'largest_component_size': int(np.max(hist)) if len(hist) > 0 else 0,
            'density': 2 * self.base_graph.num_edges() / (self.base_graph.num_vertices() * (self.base_graph.num_vertices() - 1)) if self.base_graph.num_vertices() > 1 else 0
        }
        
        # Add statistics about component sizes
        components_by_size = {}
        for i, size in enumerate(hist):
            size_int = int(size)
            if size_int in components_by_size:
                components_by_size[size_int] += 1
            else:
                components_by_size[size_int] = 1
                
        stats['components_by_size'] = components_by_size
        
        return stats

    def find_isolated_vertices(self) -> List[int]:
        """
        Find vertices with no connections (isolated supervoxels).
        
        Returns:
            List of isolated supervoxel IDs
        """
        isolated = []
        for v in self.base_graph.vertices():
            if v.out_degree() == 0 and v.in_degree() == 0:
                v_int = int(v)
                if v_int in self.vertex_to_segid:
                    isolated.append(self.vertex_to_segid[v_int])
        return isolated
    
    def update_graph_with_mergelist(self, mergelist_content: str) -> None:
        """
        Update the graph according to a mergelist instead of change log.
        
        Args:
            mergelist_content: Content of a mergelist file
            
        Raises:
            ImportError: If knossos_utils is not available
            ValueError: If the mergelist is invalid
        """
        try:
            from knossos_utils import mergelist_tools
        except ImportError:
            raise ImportError("knossos_utils is required for this function. Please install it first.")
        
        try:
            # Create a mapping of supervoxel IDs to vertices
            supervoxel_to_vertex = {seg_id: self.seg_id_to_vertex[seg_id] 
                                   for seg_id in self.seg_id_to_vertex}
            
            # Parse the mergelist and apply changes
            subobject_map = mergelist_tools.subobject_map_from_mergelist(mergelist_content)
            
            # Track operations
            operation_count = 0
            
            # Apply the changes to the graph
            for subobj_id, obj_id in subobject_map.items():
                if subobj_id in self.seg_id_to_vertex and obj_id in self.seg_id_to_vertex:
                    # Add edge between the subobject and object if it doesn't exist
                    vertex1 = self.seg_id_to_vertex[subobj_id]
                    vertex2 = self.seg_id_to_vertex[obj_id]
                    
                    if not self.base_graph.edge(vertex1, vertex2):
                        self.base_graph.add_edge(vertex1, vertex2)
                        operation_count += 1
                else:
                    missing_ids = []
                    if subobj_id not in self.seg_id_to_vertex:
                        missing_ids.append(str(subobj_id))
                    if obj_id not in self.seg_id_to_vertex:
                        missing_ids.append(str(obj_id))
                        
                    logger.warning(f"Skipped mergelist entry: Segmentation IDs not found in graph: {', '.join(missing_ids)}")
            
            logger.info(f"Applied {operation_count} operations from mergelist")
            
        except Exception as e:
            raise ValueError(f"Failed to update graph with mergelist: {e}")


# Make classes available at the module level
__all__ = ['SegGraph']