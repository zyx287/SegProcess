'''
author: zyx
date: 2024-09-17
last_modified: 2025-02-21
description: 
    Python scripts for loading and modifying segmentation graph data
Details:
A class requires instantiating for process and output the segmentation graph data.
Useful functions:
1. updata_graph_changes:
    The segmentation changes were updated by ariadne, which could be directly "wget" based on the given URL.
    The changes were applied to the base graph, and could be save as a new graph based on save_graph.
    This method should be called after instantiation.
    e.g. seg_graph = SegGraph(path) -> seg_graph.update_graph_changes(change_log_path)
2. extract_connected_components:
    This method is used for generating the merge list of a specific neuron.
    The connected components of the given supervoxel were extracted from the base graph, which contains all the supervoxels belong to the given cell.
    After instantiating the class, this method could be called directly.
    For example, seg_graph = SegGraph(path) -> seg_graph.extract_connected_components(start_vertex_supervoxel_id)
    Usually, the updated graph should be called before this method.
3. generate_supervoxel_label_dirc and get_supervoxel_to_label:
    This method is used for generating the {supervoxel: label} dictionary, which is used for processing the whole volume segnegmentation.
    Also, it should be called after the instantiation and update_graph_changes.
    e.g. seg_graph = SegGraph(path) -> ... -> seg_graph.generate_supervoxel_label_dirc()
    Then, utilize the get_supervoxel_to_label to get the {supervoxel: label} dictionary, which is a read-only method.
'''
# TODO: Test functions

import os
from datetime import datetime
import numpy as np
import pandas as pd
import graph_tool.all as gt

import subprocess
import logging
from typing import Dict, Set, Tuple, Optional, Union, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SegGraph():
    '''
    Class for processing the graph for segmentation
    '''
    def __init__(self, graph_path: str):
        '''
        Loading graph from the given path
        
        Args:
            graph_path: Path to the graph file
            
        Raises:
            FileNotFoundError: If the graph file does not exist
            ValueError: If the graph file is invalid
        '''
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        

        try:
            self.base_graph = gt.load_graph(str(graph_path))
            self.seg_id_to_vertex, self.vertex_to_segid = self._load_base_graph_segids()
            self.supervoxel_to_label = {} # TODO: Update the data structure
            logger.info(f"Successfully loaded graph with {self.base_graph.num_vertices()} vertices and {self.base_graph.num_edges()} edges")
        except Exception as e:
            raise ValueError(f"Failed to load graph: {e}")

        
    def _load_base_graph_segids(self)->dict:
        '''
        Load the segmentation ids from the base graph
        
        Returns:
            Tuple containing:
                - Dict mapping segmentation IDs to vertex IDs
                - Dict mapping vertex IDs to segmentation IDs
        '''
        if 'seg_ids' not in self.base_graph.vp:
            raise ValueError("Graph does not contain 'seg_ids' vertex property")
        seg_id_group = self.base_graph.vp['seg_ids']
        seg_id_to_vertex = {}
        vertex_to_seg_id = {}

        for v in self.base_graph.vertices():
            v_int = int(v)
            seg_id = int(seg_id_group[v])

            seg_id_to_vertex[seg_id] = v
            vertex_to_seg_id[v_int] = seg_id

        return seg_id_to_vertex, vertex_to_seg_id

    def get_latest_changes(self, changes_url = 'https://emerald.agglomeration.ariadne.ai/api/changes.tsv'):
        '''
        Download the latest changes file from the specified URL
        
        Args:
            changes_url: URL to download changes.tsv from
            
        Returns:
            bool: True if download was successful, False otherwise
        '''
        output_path = Path.cwd() / "changes.tsv"

        try:
            # Use subprocess with captured output instead of os.system
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
        

    def update_graph_changes(self, change_log_path: str):
        '''
        Update the graph according to the change log.
        
        Args:
            change_log_path: Path to the change log file
            
        Returns:
            int: Number of operations applied
            
        Raises:
            FileNotFoundError: If the change log file does not exist
            ValueError: If the change log file is invalid
        '''
        change_log_path = Path(change_log_path)
        if not change_log_path.exists():
            raise FileNotFoundError(f"Change log file not found: {change_log_path}")

        # Read the entire file into a DataFrame
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
        operation_num = 0
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

            if seg_id_1 in self.seg_id_to_vertex and seg_id_2 in self.seg_id_to_vertex:
                vertex1 = self.seg_id_to_vertex[seg_id_1]
                vertex2 = self.seg_id_to_vertex[seg_id_2]

                if operation == '+':
                    self.base_graph.add_edge(vertex1, vertex2)
                    operation_num += 1
                elif operation == '-':
                    edge = self.base_graph.edge(vertex1, vertex2)
                    if edge is not None:
                        self.base_graph.remove_edge(edge)
                    operation_num += 1
            else:
                raise ValueError(f"Change {index + 1} error! Segmentation id {seg_id_1} or {seg_id_2} not found in the base graph")

        # Print the number of operations applied
        logger.info(f"Applied {operation_num} operations to the base graph. Encountered {errors} errors.")
        # TODO: Figure out required return value?
    
    def extract_connected_components(self, start_vertex_supervoxel_id: int) -> set[int]:
        '''
        Extract all connected components of the given supervoxel from the base graph
        (Works for generating merge list of a specific neuron)
        
        Args:
            start_vertex_supervoxel_id: Supervoxel ID to start from
            
        Returns:
            Set of connected supervoxel IDs
            
        Raises:
            ValueError: If the start vertex supervoxel ID is not found in the base graph
        '''
        if start_vertex_supervoxel_id not in self.seg_id_to_vertex:
            raise ValueError(f"Supervoxel {start_vertex_supervoxel_id} not found in the base graph")
        
        start_vertex = self.seg_id_to_vertex[start_vertex_supervoxel_id]
        logger.info(f"Starting DFS from seg_id {start_vertex_supervoxel_id} (vertex {int(start_vertex)})")

        visited_vertices_segid = set()
        for edge in gt.dfs_iterator(self.base_graph, start_vertex):
            source_vertex = int(edge.source())
            target_vertex = int(edge.target())
            
            visited_vertices_segid.add(self.vertex_to_segid[int(source_vertex)])
            visited_vertices_segid.add(self.vertex_to_segid[int(target_vertex)])

        logger.info(f"Found {len(visited_vertices_segid)} connected supervoxels")
        
        return visited_vertices_segid
    
    # TODO: Test function for large graph
    def extract_connected_components_large_graph(self, start_vertex_supervoxel_id: int) -> set[int]:
        if start_vertex_supervoxel_id not in self.seg_id_to_vertex:
            raise ValueError(f"Supervoxel {start_vertex_supervoxel_id} not found in the base graph")
        
        # Use graph-tool's built-in label_components for better performance
        comp, hist = gt.label_components(self.base_graph, start=self.seg_id_to_vertex[start_vertex_supervoxel_id])
        
        # Extract the component that contains our start vertex
        start_comp = comp[self.seg_id_to_vertex[start_vertex_supervoxel_id]]
        visited_vertices_segid = set()
        
        # Collect all vertices in this component
        for v in self.base_graph.vertices():
            if comp[v] == start_comp:
                visited_vertices_segid.add(self.vertex_to_segid[int(v)])
        
        logger.info(f"Found {len(visited_vertices_segid)} connected supervoxels")
        return visited_vertices_segid
    
    def generate_supervoxel_label_dirc(self, transform_flag=True):
        '''
        Generate the {supervoxel: label} dictionary
        
        Args:
            transform_flag: Whether to transform labels to a new format
            
        Returns:
            bool: True if successful, False otherwise
        '''
        try:
            vertex_to_label = {}
            comp, hist = gt.label_components(self.base_graph)
            for v in self.base_graph.vertices():
                vertex_to_label[int(v)] = comp[v]
            # Map two dictionaries
            supervoxel_to_label = {}
            for sv_id, v in self.seg_id_to_vertex.items():
                label = vertex_to_label[int(v)]
                supervoxel_to_label[sv_id] = vertex_to_label[v]
            
            if not transform_flag:
                self.supervoxel_to_label = supervoxel_to_label
                logger.info(f"Generated supervoxel to label dictionary with {len(supervoxel_to_label)} entries")
                return True
            ## Simple transform the data:
            label_to_new_label = {}
            start_number = 9 * (10 ** (int(len(str(len(hist))))))# 9 is the represent the cell shape
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
    def get_supervoxel_to_label(self):
        if not self.supervoxel_to_label:
            logger.warning("Supervoxel to label dictionary is empty. Call generate_supervoxel_label_dirc first.")
        return self.supervoxel_to_label
    
    def save_graph(self, output_file_path: str):
        '''
        Save the graph with proofread (updated with changes in changes.tsv) to the output path
        
        Args:
            output_file_path: Path to save the graph
            
        Returns:
            bool: True if successful, False otherwise
        '''
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
        '''
        Get statistics about the graph
        
        Returns:
            Dict containing graph statistics
        '''
        stats = {
            'num_vertices': self.base_graph.num_vertices(),
            'num_edges': self.base_graph.num_edges(),
            'num_components': gt.label_components(self.base_graph)[1].shape[0],
            'density': 2 * self.base_graph.num_edges() / (self.base_graph.num_vertices() * (self.base_graph.num_vertices() - 1))
        }
        return stats

    def find_isolated_vertices(self) -> List[int]:
        '''
        Find vertices with no connections
        
        Returns:
            List of isolated supervoxel IDs
        '''
        isolated = []
        for v in self.base_graph.vertices():
            if v.out_degree() == 0 and v.in_degree() == 0:
                v_int = int(v)
                if v_int in self.vertex_to_segid:
                    isolated.append(self.vertex_to_segid[v_int])
        return isolated
    
    # TODO: Test functions below
    ###### Integrating Knossos_utils
    def update_graph_with_mergelist(self, mergelist_content: str) -> None:
        '''
        Update the graph according to a mergelist instead of change log.
        
        Args:
            mergelist_content: Content of a mergelist file
        '''
        from knossos_utils import mergelist_tools
        
        # Create a mapping of supervoxel IDs to vertices
        supervoxel_to_vertex = {seg_id: self.seg_id_to_vertex[seg_id] 
                                for seg_id in self.seg_id_to_vertex}
        
        # Parse the mergelist and apply changes
        subobject_map = mergelist_tools.subobject_map_from_mergelist(mergelist_content)
        
        # Apply the changes to the graph
        for subobj_id, obj_id in subobject_map.items():
            if subobj_id in self.seg_id_to_vertex and obj_id in self.seg_id_to_vertex:
                # Add edge between the subobject and object
                self.base_graph.add_edge(self.seg_id_to_vertex[subobj_id], 
                                        self.seg_id_to_vertex[obj_id])