"""
Core dataset handling for segmentation data.

This module provides classes for loading, filtering, and processing segmentation datasets,
particularly focusing on Ariadne proofreading data.
"""

import os
import datetime
import re
import pandas as pd
import pickle
import numpy as np
import zarr
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProofDataset:
    """
    Class for processing and generating datasets from Ariadne proofreading data.
    """
    def __init__(self, excel_path: str):
        """
        Initialize a ProofDataset instance.
        
        Args:
            excel_path: Path to the Excel file containing proofreading data
            
        Raises:
            FileNotFoundError: If the Excel file does not exist
        """
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
            
        self.processed_df = None
        self.sheets = None
        logger.info(f"Initialized ProofDataset with {self.excel_path}")

    def filter_excel_client(self, sheets: List[str]) -> None:
        """
        Read and select cells with 'complete' client status from Excel sheets.
        
        Args:
            sheets: List of sheet names to process
            
        Raises:
            ValueError: If sheets are not found in the Excel file
        """
        file_path = self.excel_path
        self.sheets = sheets
        combined_data = []

        try:
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            
            for sheet in sheets:
                if sheet not in available_sheets:
                    logger.warning(f"Sheet '{sheet}' not found in Excel file")
                    continue
                    
                df = pd.read_excel(file_path, sheet_name=sheet)
                
                # Check if 'client status' column exists
                if 'client status' not in df.columns:
                    logger.warning(f"Column 'client status' not found in sheet '{sheet}'")
                    continue
                    
                filtered_df = df[df['client status'] == 'complete']
                filtered_df['Zone info'] = sheet
                
                for _, row in filtered_df.iterrows():
                    zone_data = (
                        row.get('redmine number', None), 
                        row.get('Cell Name', None), 
                        row.get('coordinate', None), 
                        row.get('client status', None), 
                        row.get('Zone info', None)
                    )
                    combined_data.append(zone_data)
            
            if not combined_data:
                logger.warning("No matching data found in any sheets")
                self.processed_df = pd.DataFrame()
                return
                
            combined_df = pd.DataFrame(
                combined_data, 
                columns=['Redmine Number', 'Cell Name', 'Coordinate', 'Client Status', 'Zone Info']
            )
            
            self.processed_df = combined_df
            logger.info(f"Filtered {len(combined_df)} cells with 'complete' client status")
            
        except Exception as e:
            logger.error(f"Error filtering Excel data: {e}")
            raise ValueError(f"Failed to filter Excel data: {e}")

    def filter_excel_ariadne(self, sheets: List[str]) -> None:
        """
        Read and select cells with 'QA is completed' ariadne status from Excel sheets.
        
        Args:
            sheets: List of sheet names to process
            
        Raises:
            ValueError: If sheets are not found in the Excel file
        """
        file_path = self.excel_path
        self.sheets = sheets
        combined_data = []

        try:
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            
            for sheet in sheets:
                if sheet not in available_sheets:
                    logger.warning(f"Sheet '{sheet}' not found in Excel file")
                    continue
                    
                df = pd.read_excel(file_path, sheet_name=sheet)
                
                # Check if 'ariadne status' column exists
                if 'ariadne status' not in df.columns:
                    logger.warning(f"Column 'ariadne status' not found in sheet '{sheet}'")
                    continue
                    
                filtered_df = df[df['ariadne status'] == 'QA is completed']
                filtered_df['Zone info'] = sheet
                
                for _, row in filtered_df.iterrows():
                    zone_data = (
                        row.get('redmine number', None), 
                        row.get('Cell Name', None), 
                        row.get('coordinate', None), 
                        row.get('ariadne status', None), 
                        row.get('Zone info', None)
                    )
                    combined_data.append(zone_data)
            
            if not combined_data:
                logger.warning("No matching data found in any sheets")
                self.processed_df = pd.DataFrame()
                return
                
            combined_df = pd.DataFrame(
                combined_data, 
                columns=['Redmine Number', 'Cell Name', 'Coordinate', 'ariadne status', 'Zone Info']
            )
            
            self.processed_df = combined_df
            logger.info(f"Filtered {len(combined_df)} cells with 'QA is completed' ariadne status")
            
        except Exception as e:
            logger.error(f"Error filtering Excel data: {e}")
            raise ValueError(f"Failed to filter Excel data: {e}")
    
    @property
    def get_processed_df(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        
        Returns:
            Processed DataFrame or empty DataFrame if not processed yet
        """
        if self.processed_df is None:
            logger.warning("DataFrame has not been processed yet. Call filter_excel_client or filter_excel_ariadne first.")
            return pd.DataFrame()
        return self.processed_df

    def save_new_dataset(self, output_path: Optional[str] = None) -> str:
        """
        Save the processed data to a new CSV file.
        
        Args:
            output_path: Optional path to save the CSV file. If None, uses a timestamped name.
            
        Returns:
            Path to the saved CSV file
            
        Raises:
            ValueError: If the dataset has not been processed yet
        """
        if self.processed_df is None or len(self.processed_df) == 0:
            raise ValueError("No processed data to save. Call filter_excel_client or filter_excel_ariadne first.")
            
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        
        if output_path is None:
            save_path = str(self.excel_path).replace('.xlsx', f'_{timestamp}_processed.csv')
        else:
            save_path = output_path
            
        self.processed_df.to_csv(save_path, index=False)
        logger.info(f"Processed data saved at {save_path}")
        
        return save_path

    @staticmethod
    def _map_coordinates(coord: str) -> Tuple[int, int, int]:
        """
        Map coordinate string to a tuple of integers.
        
        Args:
            coord: Coordinate string in format 'x,y,z'
            
        Returns:
            Tuple of (x, y, z) coordinates
            
        Raises:
            ValueError: If coordinate format is invalid
        """
        if not isinstance(coord, str):
            raise ValueError(f"Invalid coordinate type: {type(coord)} (expected string)")
            
        # Clean and parse the coordinate string
        coord = re.sub(r"[^\d,\s]+$", "", coord).strip()
        parts = coord.split(',')
        
        if len(parts) != 3:
            raise ValueError(f"Invalid coordinate format: {coord}, expected 'x,y,z'")
            
        try:
            mapped_coord = tuple(map(int, parts))
            return mapped_coord
        except ValueError as e:
            raise ValueError(f"Invalid coordinate values in {coord}: {e}")
    
    @staticmethod
    def _downsampling(coord: Tuple[int, int, int], factor: int) -> Tuple[int, int, int]:
        """
        Downsample coordinates by a factor.
        
        Args:
            coord: Original coordinates as (x, y, z)
            factor: Downsampling factor
            
        Returns:
            Downsampled coordinates
        """
        return tuple(c // factor for c in coord)

    def get_one_pixel_seg_id(self,
                           toml_file_path: str,
                           lookup_path: str,
                           volume_size: Tuple[int, int, int] = (32, 32, 32),
                           mag_size: int = 5) -> None:
        """
        Get the segmentation ID for all neurons in the list using single-threaded processing.
        
        Args:
            toml_file_path: Path to the toml file for knossos_utils
            lookup_path: Path to the lookup table for seg_id to label conversion
            volume_size: Size of the volume to load for each coordinate
            mag_size: Magnification size
            
        Raises:
            ValueError: If the dataset has not been processed yet or if knossos_utils is not available
        """
        if self.processed_df is None or len(self.processed_df) == 0:
            raise ValueError("No processed data available. Call filter_excel_client or filter_excel_ariadne first.")
            
        try:
            from segprocess.utils.knossos import load_knossos_dataset
        except ImportError:
            raise ValueError("knossos_utils is required for this function. Install the package or the segprocess[knossos] extra.")

        # Map coordinates
        self.processed_df['Mapped_Coordinates'] = self.processed_df['Coordinate'].apply(self._map_coordinates)
        
        # Get target IDs
        target_list = []
        for coord in self.processed_df['Mapped_Coordinates']:
            try:
                seg_array = load_knossos_dataset(
                    toml_path=toml_file_path,
                    volume_offset=coord,
                    volume_size=volume_size,
                    mag_size=mag_size
                )
                # Take the first voxel ID
                target_list.append(seg_array[0][0][0])
            except Exception as e:
                logger.error(f"Error loading data for coordinate {coord}: {e}")
                target_list.append(None)
                
        self.processed_df['target_id'] = target_list
        
        # Map seg IDs to labels
        try:
            with open(lookup_path, 'rb') as f:
                lookup_data = pickle.load(f)
                
            self.processed_df['label'] = self.processed_df['target_id'].apply(
                lambda seg_id: int(float(lookup_data.get(seg_id, -1))) if seg_id is not None else -1
            )
            
            logger.info("Segmentation ID and label generated for all neurons.")
            
        except Exception as e:
            logger.error(f"Error loading or applying lookup table: {e}")
            raise ValueError(f"Failed to load or apply lookup table: {e}")

    def get_one_pixel_seg_id_multithreads(self,
                                      toml_file_path: str,
                                      lookup_path: str,
                                      volume_size: Tuple[int, int, int] = (32, 32, 32),
                                      mag_size: int = 5,
                                      num_threads: int = 10) -> None:
        """
        Get the segmentation ID for all neurons in the list using multi-threaded processing.
        
        Args:
            toml_file_path: Path to the toml file for knossos_utils
            lookup_path: Path to the lookup table for seg_id to label conversion
            volume_size: Size of the volume to load for each coordinate
            mag_size: Magnification size
            num_threads: Number of threads to use
            
        Raises:
            ValueError: If the dataset has not been processed yet or if knossos_utils is not available
        """
        if self.processed_df is None or len(self.processed_df) == 0:
            raise ValueError("No processed data available. Call filter_excel_client or filter_excel_ariadne first.")
            
        try:
            from segprocess.utils.knossos import load_knossos_dataset
        except ImportError:
            raise ValueError("knossos_utils is required for this function. Install the package or the segprocess[knossos] extra.")

        # Map coordinates
        self.processed_df['Mapped_Coordinates'] = self.processed_df['Coordinate'].apply(self._map_coordinates)

        def process_coordinate(coord: Tuple[int, int, int]) -> int:
            """Process a single coordinate and return segmentation ID"""
            try:
                seg_array = load_knossos_dataset(
                    toml_path=toml_file_path,
                    volume_offset=coord,
                    volume_size=volume_size,
                    mag_size=mag_size
                )
                return seg_array[0][0][0]
            except Exception as e:
                logger.error(f"Error processing coordinate {coord}: {e}")
                return None
        
        # Get coordinates and process in parallel
        coords = self.processed_df['Mapped_Coordinates'].dropna().tolist()
        target_list = [None] * len(coords)
        coord_to_index = {tuple(coord): i for i, coord in enumerate(coords)}
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_coord = {executor.submit(process_coordinate, coord): coord for coord in coords}
            
            for future in as_completed(future_to_coord):
                coord = future_to_coord[future]
                try:
                    result = future.result()
                    index = coord_to_index[tuple(coord)]
                    target_list[index] = result
                except Exception as e:
                    logger.error(f"Error processing coordinate {coord}: {e}")

        self.processed_df['target_id'] = pd.Series(target_list, dtype='int64')
        
        # Map seg IDs to labels
        try:
            with open(lookup_path, 'rb') as f:
                lookup_data = pickle.load(f)
                
            self.processed_df['label'] = self.processed_df['target_id'].apply(
                lambda seg_id: int(lookup_data.get(seg_id, -1)) if seg_id is not None else -1
            )
            
            logger.info(f"Segmentation ID and label generated for {len(coords)} neurons using {num_threads} threads.")
            
        except Exception as e:
            logger.error(f"Error loading or applying lookup table: {e}")
            raise ValueError(f"Failed to load or apply lookup table: {e}")
    
    def get_downsampling_seg_id(self, downsampling_factors: List[int] = [2, 3, 4, 5]) -> None:
        """
        Generate downsampled coordinates for all neurons in the list.
        
        Args:
            downsampling_factors: List of downsampling factors
            
        Raises:
            ValueError: If the dataset has not been processed yet or coordinates are not mapped
        """
        if self.processed_df is None or len(self.processed_df) == 0:
            raise ValueError("No processed data available. Call filter_excel_client or filter_excel_ariadne first.")
            
        if 'Mapped_Coordinates' not in self.processed_df.columns:
            raise ValueError("Coordinates must be mapped first. Call get_one_pixel_seg_id or get_one_pixel_seg_id_multithreads.")
            
        for factor in downsampling_factors:
            column_name = f'Downsampled_Coordinate_{factor}'
            self.processed_df[column_name] = self.processed_df['Mapped_Coordinates'].apply(
                lambda coord: self._downsampling(coord, factor) if coord is not None else None
            )
            
        logger.info(f"Downsampled coordinates generated for factors: {downsampling_factors}")


class SegmentationData:
    """
    Class for generating segmentation data arrays using knossos_utils.
    """
    def __init__(self, output_dir: str = './segmentation_data'):
        """
        Initialize a SegmentationData instance.
        
        Args:
            output_dir: Directory to save segmentation data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized SegmentationData with output directory: {self.output_dir}")
    
    def fetch_segmentation_chunk(self, 
                               chunk_size: Tuple[int, int, int], 
                               offset: Tuple[int, int, int], 
                               mag: int,
                               toml_path: Optional[str] = None) -> np.ndarray:
        """
        Get segmentation chunk based on knossos_utils_plugin.
        
        Args:
            chunk_size: Size of the chunk to fetch (z, y, x)
            offset: Offset for the chunk (x, y, z)
            mag: Magnification level
            toml_path: Path to the toml file for knossos_utils
            
        Returns:
            Numpy array with segmentation data
            
        Raises:
            ImportError: If knossos_utils is not available
        """
        try:
            from segprocess.utils.knossos import load_knossos_dataset
        except ImportError:
            raise ImportError("knossos_utils is required for this function. Install the package or the segprocess[knossos] extra.")
            
        chunk_size_xyz = chunk_size[::-1]  # Convert z, y, x to x, y, z for knossos_utils
        chunk_array = load_knossos_dataset(
            toml_path=toml_path,
            volume_offset=offset, 
            volume_size=chunk_size_xyz, 
            mag_size=mag
        )
        return chunk_array.astype(np.int32)
    
    def generate_segmentation(self, 
                            chunk_size: Tuple[int, int, int], 
                            mag: int, 
                            volume_size: Tuple[int, int, int],
                            toml_path: Optional[str] = None) -> str:
        """
        Generate the entire volume segmentation chunkwise and save to Zarr.
        
        Args:
            chunk_size: Size of each chunk to download (z, y, x)
            mag: Magnification level
            volume_size: Total volume size (z, y, x)
            toml_path: Path to the toml file for knossos_utils
            
        Returns:
            Path to the saved Zarr dataset
            
        Raises:
            ImportError: If knossos_utils is not available
        """
        zarr_path = self.output_dir / 'segmentation_data.zarr'
        zarr_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=volume_size,
            chunks=chunk_size,
            dtype=np.int32
        )
        
        # Process each chunk
        for z in range(0, volume_size[0], chunk_size[0]):
            for y in range(0, volume_size[1], chunk_size[1]):
                for x in range(0, volume_size[2], chunk_size[2]):
                    offset = (x, y, z)  # Offset in x, y, z order for knossos_utils
                    current_chunk_size = tuple(
                        min(chunk_size[i], volume_size[i] - [z, y, x][i])
                        for i in range(3)
                    )
                    
                    try:
                        chunk_data = self.fetch_segmentation_chunk(
                            current_chunk_size, 
                            offset, 
                            mag,
                            toml_path
                        )
                        
                        zarr_array[z:z+current_chunk_size[0],
                                 y:y+current_chunk_size[1],
                                 x:x+current_chunk_size[2]] = chunk_data
                                 
                        logger.info(f"Chunk saved at offset {offset} with shape {current_chunk_size}")
                        
                    except Exception as e:
                        logger.error(f"Error fetching chunk at offset {offset}: {e}")
        
        logger.info(f"Segmentation data saved to {zarr_path}")
        return str(zarr_path)

    def generate_segmentation_parallel(self, 
                                    chunk_size: Tuple[int, int, int], 
                                    mag: int, 
                                    volume_size: Tuple[int, int, int],
                                    toml_path: Optional[str] = None,
                                    num_workers: int = 4) -> str:
        """
        Generate the entire volume segmentation in parallel and save to Zarr.
        
        Args:
            chunk_size: Size of each chunk to download (z, y, x)
            mag: Magnification level
            volume_size: Total volume size (z, y, x)
            toml_path: Path to the toml file for knossos_utils
            num_workers: Number of worker threads
            
        Returns:
            Path to the saved Zarr dataset
            
        Raises:
            ImportError: If knossos_utils is not available
        """
        zarr_path = self.output_dir / 'segmentation_data_parallel.zarr'
        zarr_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=volume_size,
            chunks=chunk_size,
            dtype=np.int32
        )
        
        # Generate tasks for parallel processing
        tasks = []
        for z in range(0, volume_size[0], chunk_size[0]):
            for y in range(0, volume_size[1], chunk_size[1]):
                for x in range(0, volume_size[2], chunk_size[2]):
                    offset = (x, y, z)  # Offset in x, y, z order for knossos_utils
                    current_chunk_size = tuple(
                        min(chunk_size[i], volume_size[i] - [z, y, x][i])
                        for i in range(3)
                    )
                    tasks.append((offset, current_chunk_size))
        
        def process_chunk(task: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], np.ndarray]:
            """Process a single chunk and return the result with its position"""
            offset, current_chunk_size = task
            try:
                chunk_data = self.fetch_segmentation_chunk(
                    current_chunk_size, 
                    offset, 
                    mag,
                    toml_path
                )
                return offset, chunk_data
            except Exception as e:
                logger.error(f"Error fetching chunk at offset {offset}: {e}")
                return offset, None
        
        # Process tasks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk, task) for task in tasks]
            
            for future in as_completed(futures):
                try:
                    offset, chunk_data = future.result()
                    if chunk_data is not None:
                        z, y, x = offset[2], offset[1], offset[0]  # Convert back to z, y, x for zarr
                        chunk_size_zyx = chunk_data.shape
                        
                        zarr_array[z:z+chunk_size_zyx[0],
                                 y:y+chunk_size_zyx[1],
                                 x:x+chunk_size_zyx[2]] = chunk_data
                                 
                        logger.info(f"Chunk saved at offset {offset} with shape {chunk_size_zyx}")
                        results.append(offset)
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
        
        logger.info(f"Segmentation data saved to {zarr_path} ({len(results)}/{len(tasks)} chunks processed)")
        return str(zarr_path)

    def process_and_save_label_map(self,
                                 input_zarr_path: str,
                                 lookup_table: Dict[int, int],
                                 output_zarr_path: Optional[str] = None,
                                 chunk_size: Optional[Tuple[int, int, int]] = None) -> str:
        """
        Process a segmentation zarr array using a lookup table and save the result.
        
        Args:
            input_zarr_path: Path to the input zarr array
            lookup_table: Dictionary mapping segmentation IDs to labels
            output_zarr_path: Path to save the output zarr array. If None, a default path is used.
            chunk_size: Chunk size for the output zarr array. If None, use the input chunk size.
            
        Returns:
            Path to the saved zarr array
        """
        # Open input zarr array
        input_zarr = zarr.open(input_zarr_path, mode='r')
        
        # Set output path if not provided
        if output_zarr_path is None:
            output_zarr_path = str(self.output_dir / 'segmentation_labeled.zarr')
            
        # Set chunk size if not provided
        if chunk_size is None:
            chunk_size = input_zarr.chunks
            
        # Create output zarr array
        output_zarr = zarr.open(
            output_zarr_path,
            mode='w',
            shape=input_zarr.shape,
            chunks=chunk_size,
            dtype=np.uint32
        )
        
        # Define a vectorized function to map segmentation IDs to labels
        def map_seg_id_to_label(seg_id: int) -> int:
            return lookup_table.get(seg_id, 0)
            
        # Process the data in chunks
        for z in range(0, input_zarr.shape[0], chunk_size[0]):
            for y in range(0, input_zarr.shape[1], chunk_size[1]):
                for x in range(0, input_zarr.shape[2], chunk_size[2]):
                    # Define the slice for this chunk
                    slices = (
                        slice(z, min(z + chunk_size[0], input_zarr.shape[0])),
                        slice(y, min(y + chunk_size[1], input_zarr.shape[1])),
                        slice(x, min(x + chunk_size[2], input_zarr.shape[2]))
                    )
                    
                    # Read the chunk
                    chunk = input_zarr[slices]
                    
                    # Apply the lookup table
                    vectorized_map = np.vectorize(map_seg_id_to_label)
                    labeled_chunk = vectorized_map(chunk)
                    
                    # Write the labeled chunk
                    output_zarr[slices] = labeled_chunk
                    
                    logger.info(f"Processed chunk at {slices}")
        
        logger.info(f"Label map applied and saved to {output_zarr_path}")
        return output_zarr_path
        
    def filter_labels(self,
                    input_zarr_path: str,
                    valid_labels: Union[List[int], np.ndarray, Set[int]],
                    output_zarr_path: Optional[str] = None,
                    chunk_size: Optional[Tuple[int, int, int]] = None) -> str:
        """
        Filter a labeled zarr array to keep only valid labels.
        
        Args:
            input_zarr_path: Path to the input zarr array
            valid_labels: List, array, or set of valid labels to keep
            output_zarr_path: Path to save the output zarr array. If None, a default path is used.
            chunk_size: Chunk size for the output zarr array. If None, use the input chunk size.
            
        Returns:
            Path to the saved zarr array
        """
        # Open input zarr array
        input_zarr = zarr.open(input_zarr_path, mode='r')
        
        # Set output path if not provided
        if output_zarr_path is None:
            output_zarr_path = str(self.output_dir / 'segmentation_filtered.zarr')
            
        # Set chunk size if not provided
        if chunk_size is None:
            chunk_size = input_zarr.chunks
            
        # Create output zarr array
        output_zarr = zarr.open(
            output_zarr_path,
            mode='w',
            shape=input_zarr.shape,
            chunks=chunk_size,
            dtype=input_zarr.dtype
        )
        
        # Convert valid labels to a set for faster lookup
        valid_labels_set = set(valid_labels)
        
        # Define a vectorized function to filter labels
        def filter_label(label: int) -> int:
            return label if label in valid_labels_set else 0
            
        # Process the data in chunks
        for z in range(0, input_zarr.shape[0], chunk_size[0]):
            for y in range(0, input_zarr.shape[1], chunk_size[1]):
                for x in range(0, input_zarr.shape[2], chunk_size[2]):
                    # Define the slice for this chunk
                    slices = (
                        slice(z, min(z + chunk_size[0], input_zarr.shape[0])),
                        slice(y, min(y + chunk_size[1], input_zarr.shape[1])),
                        slice(x, min(x + chunk_size[2], input_zarr.shape[2]))
                    )
                    
                    # Read the chunk
                    chunk = input_zarr[slices]
                    
                    # Apply the filter
                    vectorized_filter = np.vectorize(filter_label)
                    filtered_chunk = vectorized_filter(chunk)
                    
                    # Write the filtered chunk
                    output_zarr[slices] = filtered_chunk
                    
                    logger.info(f"Filtered chunk at {slices}")
        
        logger.info(f"Filtered data saved to {output_zarr_path}")
        return output_zarr_path


# Make classes available at the module level
__all__ = ['ProofDataset', 'SegmentationData']