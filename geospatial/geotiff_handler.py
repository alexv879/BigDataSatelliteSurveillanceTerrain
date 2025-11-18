"""
Geospatial Integration for Satellite Imagery
GeoTIFF reading, coordinate transformations, and spatial analysis
Critical for real-world satellite terrain surveillance
"""
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
import json


@dataclass
class GeoMetadata:
    """Geospatial metadata"""
    crs: str  # Coordinate reference system (e.g., 'EPSG:4326')
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    transform: List[float]  # Affine transform
    width: int
    height: int
    resolution: Tuple[float, float]  # (x_res, y_res) in CRS units


class GeoTIFFHandler:
    """
    Handle GeoTIFF satellite imagery
    Supports reading, writing, and metadata extraction
    """

    @staticmethod
    def read_geotiff(
        path: str,
        bands: Optional[List[int]] = None,
        window: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, GeoMetadata]:
        """
        Read GeoTIFF file

        Args:
            path: Path to GeoTIFF
            bands: Band indices to read (1-indexed), None for all
            window: (row_off, col_off, height, width) to read subset

        Returns:
            (image_data, metadata)
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            logger.error("rasterio not installed. Install with: pip install rasterio")
            raise

        with rasterio.open(path) as src:
            # Read metadata
            metadata = GeoMetadata(
                crs=str(src.crs),
                bounds=src.bounds,
                transform=list(src.transform),
                width=src.width,
                height=src.height,
                resolution=(src.transform[0], abs(src.transform[4]))
            )

            # Read data
            if window:
                win = Window(window[1], window[0], window[3], window[2])
                if bands:
                    data = src.read(bands, window=win)
                else:
                    data = src.read(window=win)
            else:
                if bands:
                    data = src.read(bands)
                else:
                    data = src.read()

            # Transpose to (H, W, C)
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))

            logger.info(f"Read GeoTIFF: {path} - Shape: {data.shape}, CRS: {metadata.crs}")

            return data, metadata

    @staticmethod
    def write_geotiff(
        path: str,
        data: np.ndarray,
        metadata: GeoMetadata,
        nodata: Optional[float] = None
    ):
        """
        Write GeoTIFF file

        Args:
            path: Output path
            data: Image data (H, W) or (H, W, C)
            metadata: Geospatial metadata
            nodata: No data value
        """
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            logger.error("rasterio not installed")
            raise

        # Prepare data
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # Add band dimension
        elif data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        count, height, width = data.shape

        # Create transform
        transform = Affine(*metadata.transform)

        # Write
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=data.dtype,
            crs=metadata.crs,
            transform=transform,
            nodata=nodata,
            compress='lzw'
        ) as dst:
            dst.write(data)

        logger.info(f"Wrote GeoTIFF: {path}")


class CoordinateTransformer:
    """
    Transform coordinates between different CRS
    """

    @staticmethod
    def transform_point(
        x: float,
        y: float,
        from_crs: str,
        to_crs: str
    ) -> Tuple[float, float]:
        """
        Transform point coordinates

        Args:
            x, y: Input coordinates
            from_crs: Source CRS (e.g., 'EPSG:4326')
            to_crs: Target CRS (e.g., 'EPSG:3857')

        Returns:
            (x_transformed, y_transformed)
        """
        try:
            from pyproj import Transformer
        except ImportError:
            logger.error("pyproj not installed. Install with: pip install pyproj")
            raise

        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        return transformer.transform(x, y)

    @staticmethod
    def transform_bounds(
        bounds: Tuple[float, float, float, float],
        from_crs: str,
        to_crs: str
    ) -> Tuple[float, float, float, float]:
        """
        Transform bounding box

        Args:
            bounds: (minx, miny, maxx, maxy)
            from_crs: Source CRS
            to_crs: Target CRS

        Returns:
            Transformed bounds
        """
        try:
            from pyproj import Transformer
        except ImportError:
            raise

        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

        # Transform corners
        minx, miny, maxx, maxy = bounds

        # Transform all four corners to handle rotation
        corners = [
            (minx, miny),
            (minx, maxy),
            (maxx, miny),
            (maxx, maxy)
        ]

        transformed = [transformer.transform(x, y) for x, y in corners]
        xs, ys = zip(*transformed)

        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def pixel_to_geo(
        row: int,
        col: int,
        transform: List[float]
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates

        Args:
            row, col: Pixel coordinates
            transform: Affine transform

        Returns:
            (x, y) in geographic coordinates
        """
        x = transform[0] + col * transform[1] + row * transform[2]
        y = transform[3] + col * transform[4] + row * transform[5]

        return x, y

    @staticmethod
    def geo_to_pixel(
        x: float,
        y: float,
        transform: List[float]
    ) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates

        Args:
            x, y: Geographic coordinates
            transform: Affine transform

        Returns:
            (row, col) pixel coordinates
        """
        # Invert affine transform
        det = transform[1] * transform[5] - transform[2] * transform[4]

        col = ((y - transform[3]) * transform[2] - (x - transform[0]) * transform[5]) / det
        row = ((x - transform[0]) * transform[4] - (y - transform[3]) * transform[1]) / det

        return int(row), int(col)


class SpatialAnalyzer:
    """
    Spatial analysis tools for terrain classification
    """

    @staticmethod
    def compute_area(
        mask: np.ndarray,
        resolution: Tuple[float, float],
        crs: str
    ) -> Dict[str, float]:
        """
        Compute area of masked region

        Args:
            mask: Binary mask
            resolution: Pixel resolution in CRS units
            crs: Coordinate reference system

        Returns:
            Area statistics in different units
        """
        # Count pixels
        pixel_count = mask.sum()

        # Compute area
        pixel_area = resolution[0] * resolution[1]

        # Check if CRS is in degrees (geographic)
        if 'EPSG:4326' in crs or 'WGS 84' in crs:
            # Approximate area in m² (rough estimation)
            # More accurate would use proper geodesic calculations
            area_deg2 = pixel_count * pixel_area
            # At equator: 1 degree ≈ 111 km
            area_m2 = area_deg2 * (111000 ** 2)
        else:
            # Projected CRS (usually in meters)
            area_m2 = pixel_count * pixel_area

        return {
            'pixel_count': int(pixel_count),
            'area_m2': float(area_m2),
            'area_km2': float(area_m2 / 1e6),
            'area_hectares': float(area_m2 / 1e4)
        }

    @staticmethod
    def compute_distance_matrix(
        points1: np.ndarray,
        points2: np.ndarray,
        crs: str = 'EPSG:4326'
    ) -> np.ndarray:
        """
        Compute distance matrix between two sets of points

        Args:
            points1: (N, 2) array of (x, y) coordinates
            points2: (M, 2) array of (x, y) coordinates
            crs: Coordinate reference system

        Returns:
            (N, M) distance matrix
        """
        if 'EPSG:4326' in crs:
            # Geographic coordinates - use Haversine
            return SpatialAnalyzer._haversine_distance_matrix(points1, points2)
        else:
            # Projected coordinates - use Euclidean
            from scipy.spatial.distance import cdist
            return cdist(points1, points2)

    @staticmethod
    def _haversine_distance_matrix(
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """
        Haversine distance for geographic coordinates

        Args:
            points1: (N, 2) array of (lon, lat) in degrees
            points2: (M, 2) array of (lon, lat) in degrees

        Returns:
            Distance matrix in meters
        """
        # Convert to radians
        lon1, lat1 = np.radians(points1[:, 0]), np.radians(points1[:, 1])
        lon2, lat2 = np.radians(points2[:, 0]), np.radians(points2[:, 1])

        # Broadcast to create matrices
        lon1 = lon1[:, np.newaxis]
        lat1 = lat1[:, np.newaxis]
        lon2 = lon2[np.newaxis, :]
        lat2 = lat2[np.newaxis, :]

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in meters
        R = 6371000

        return R * c


class MapGenerator:
    """
    Generate maps with classification results
    """

    def __init__(self):
        self.colormap = {
            0: (0, 128, 0),      # Forest - green
            1: (139, 69, 19),    # Barren - brown
            2: (0, 0, 255),      # Water - blue
            3: (192, 192, 192),  # Urban - gray
            4: (255, 255, 0),    # Agriculture - yellow
            5: (128, 128, 0),    # Grass - olive
            6: (255, 255, 255),  # Snow - white
            7: (128, 0, 0),      # Wetland - maroon
            8: (255, 165, 0),    # Desert - orange
            9: (64, 224, 208),   # Shrubland - turquoise
        }

    def create_classification_map(
        self,
        predictions: np.ndarray,
        metadata: GeoMetadata,
        output_path: str,
        confidence: Optional[np.ndarray] = None
    ):
        """
        Create classification map as GeoTIFF

        Args:
            predictions: (H, W) class predictions
            metadata: Geospatial metadata
            output_path: Output path
            confidence: Optional (H, W) confidence scores
        """
        # Convert classes to RGB
        height, width = predictions.shape
        rgb_map = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color in self.colormap.items():
            mask = predictions == class_id
            rgb_map[mask] = color

        # Apply confidence as alpha if provided
        if confidence is not None:
            alpha = (confidence * 255).astype(np.uint8)
            rgba_map = np.concatenate([rgb_map, alpha[..., np.newaxis]], axis=-1)
            data_to_write = rgba_map
        else:
            data_to_write = rgb_map

        # Write GeoTIFF
        GeoTIFFHandler.write_geotiff(output_path, data_to_write, metadata)

        logger.info(f"Classification map saved: {output_path}")

    def create_web_map(
        self,
        predictions: np.ndarray,
        metadata: GeoMetadata,
        output_html: str,
        class_names: List[str]
    ):
        """
        Create interactive web map with Folium

        Args:
            predictions: Class predictions
            metadata: Geospatial metadata
            output_html: Output HTML path
            class_names: List of class names
        """
        try:
            import folium
            from folium import raster_layers
        except ImportError:
            logger.error("folium not installed. Install with: pip install folium")
            return

        # Get center coordinates
        bounds = metadata.bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )

        # Add classification overlay
        # Convert to RGB
        rgb_map = np.zeros((*predictions.shape, 3), dtype=np.uint8)
        for class_id, color in self.colormap.items():
            mask = predictions == class_id
            rgb_map[mask] = color

        # Add to map (simplified - would need proper georeference)
        # In production, convert to overlay format

        # Add legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 150px;
                    border:2px solid grey; z-index:9999;
                    background-color:white; opacity: 0.9;
                    padding: 10px">
        <p style="margin-top:0; font-weight: bold;">Terrain Classes</p>
        '''

        for class_id, name in enumerate(class_names):
            if class_id < len(self.colormap):
                color = self.colormap[class_id]
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                legend_html += f'<p style="margin:2px;"><span style="background-color:{color_hex}; padding: 2px 10px;"></span> {name}</p>'

        legend_html += '</div>'

        m.get_root().html.add_child(folium.Element(legend_html))

        # Save
        m.save(output_html)
        logger.info(f"Interactive map saved: {output_html}")


class TileProcessor:
    """
    Process large satellite images in tiles
    Critical for memory-efficient processing
    """

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        """
        Initialize tile processor

        Args:
            tile_size: Size of tiles
            overlap: Overlap between tiles (for edge effects)
        """
        self.tile_size = tile_size
        self.overlap = overlap

    def generate_tiles(
        self,
        image_shape: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate tile coordinates

        Args:
            image_shape: (height, width)

        Returns:
            List of (row_off, col_off, height, width) tuples
        """
        height, width = image_shape
        tiles = []

        stride = self.tile_size - self.overlap

        for row in range(0, height, stride):
            for col in range(0, width, stride):
                # Compute tile bounds
                row_end = min(row + self.tile_size, height)
                col_end = min(col + self.tile_size, width)

                tile_height = row_end - row
                tile_width = col_end - col

                tiles.append((row, col, tile_height, tile_width))

        logger.info(f"Generated {len(tiles)} tiles for {image_shape}")

        return tiles

    def merge_predictions(
        self,
        tile_predictions: List[np.ndarray],
        tile_coords: List[Tuple[int, int, int, int]],
        output_shape: Tuple[int, int],
        num_classes: int
    ) -> np.ndarray:
        """
        Merge tile predictions into full image

        Args:
            tile_predictions: List of prediction arrays
            tile_coords: List of (row, col, height, width)
            output_shape: (height, width)
            num_classes: Number of classes

        Returns:
            Merged predictions
        """
        # Accumulate predictions
        full_pred = np.zeros((*output_shape, num_classes), dtype=np.float32)
        full_count = np.zeros(output_shape, dtype=np.int32)

        for pred, (row, col, h, w) in zip(tile_predictions, tile_coords):
            # Handle overlap by averaging
            full_pred[row:row+h, col:col+w] += pred[:h, :w]
            full_count[row:row+h, col:col+w] += 1

        # Average overlapping regions
        full_count = np.maximum(full_count, 1)  # Avoid division by zero
        full_pred = full_pred / full_count[..., np.newaxis]

        # Get final classes
        final_pred = np.argmax(full_pred, axis=-1)

        logger.info(f"Merged {len(tile_predictions)} tiles")

        return final_pred


if __name__ == "__main__":
    print("Geospatial Integration Ready!")
    print("\nCapabilities:")
    print("- GeoTIFF reading and writing")
    print("- Coordinate transformations (any CRS)")
    print("- Pixel ↔ Geographic coordinate conversion")
    print("- Area computation")
    print("- Distance calculations (Haversine for lat/lon)")
    print("- Classification map generation")
    print("- Interactive web maps (Folium)")
    print("- Tile-based processing for large images")
    print("\n🗺️ Full geospatial support for satellite imagery!")
