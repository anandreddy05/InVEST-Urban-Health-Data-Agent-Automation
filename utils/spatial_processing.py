import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterResampling
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import os
from typing import Dict, Tuple

"""
Default:

Common coordinate system (EPSG:5070 for US)
Common extent and resolution (e.g., 30m or 100m)
"""

class SpatialProcessor:
    def __init__(self, target_crs: str = "EPSG:5070", resolution: int = 30):
        self.target_crs = target_crs
        self.resolution = resolution
    
    def clip_raster(self, input_path: str, output_path: str, bbox: Dict) -> bool:
        """Clip raster to bounding box
        Params:
            input_path: Path to the original, large raster file.
                Example: "/data/nlcd_2019_whole_usa.tif"
            output_path: Path where the clipped raster will be saved.
                Example: "outputs/chicago_landcover_clipped.tif"
        ===========================================================
        Example Usage:
            processor.clip_raster(
            input_path="data/raw/nlcd_2019.tif",      # 2GB US-wide file
            output_path="temp/chicago_nlcd_clipped.tif", # 50MB Chicago-only file
            bbox={"min_lon": -87.94, "min_lat": 41.64, "max_lon": -87.52, "max_lat": 42.03}
)
        """
        try:
            with rasterio.open(input_path) as src:
                # Create geometry from bbox
                geometry = box(bbox["min_lon"], bbox["min_lat"], 
                             bbox["max_lon"], bbox["max_lat"])
                
                # Convert to geodataframe
                gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
                gdf = gdf.to_crs(src.crs)
                
                # Clip raster
                out_image, out_transform = mask(src, gdf.geometry, crop=True)
                
                # Update metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": src.crs
                })
                
                # Write output
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                return True
                
        except Exception as e:
            print(f"Clipping error: {e}")
            return False
    
    def reproject_raster(self, input_path: str, output_path: str) -> bool:
        """Reproject raster to target CRS and resolution
        Params:
        input_path: Path to raster in original coordinate system.
            Example: "temp/chicago_nlcd_clipped.tif" (in EPSG:4269 - NAD83 geographic)
        output_path: Path to raster in target coordinate system
             Example: "outputs/chicago_landcover_5070.tif" (in EPSG:5070 - Albers)
        ===========================================================
        
        Example Usage:
            processor.reproject_raster(
            input_path="temp/chicago_nlcd_clipped.tif",  # EPSG:4269 (geographic)
            output_path="outputs/chicago_landcover.tif"  # EPSG:5070 (projected)
) 
        """
        try:
            with rasterio.open(input_path) as src:
                # Calculate transform for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, self.target_crs, src.width, src.height, *src.bounds,
                    resolution=self.resolution
                )
                
                # Update metadata
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': self.target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'nodata': src.nodata or 255
                })
                
                # Reproject
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=self.target_crs,
                            resampling=Resampling.bilinear
                        )
                
                return True
                
        except Exception as e:
            print(f"Reprojection error: {e}")
            return False
    
    def align_rasters(self, base_raster_path: str, raster_to_align_path: str, 
                     output_path: str) -> bool:
        """Align one raster to another's grid
        Params:
        base_raster_path: Path to the reference raster that defines the target grid
            Example: "outputs/chicago_landcover.tif"
        raster_to_align_path: Path to raster that needs to be aligned to the base grid
            Example: "temp/chicago_population.tif"
        output_path: Path where the aligned version will be saved
            Example: "outputs/chicago_population_aligned.tif"
        
        Example Usage:
            processor.align_rasters(
            base_raster_path="outputs/chicago_landcover.tif",      # 30m grid, EPSG:5070
            raster_to_align_path="temp/chicago_population.tif",    # 100m grid, EPSG:5070 
            output_path="outputs/chicago_population_aligned.tif"   # 30m grid, EPSG:5070
)
        """
        try:
            with rasterio.open(base_raster_path) as base:
                with rasterio.open(raster_to_align_path) as to_align:
                    # Read base raster properties
                    base_profile = base.profile
                    
                    # Read and resample data
                    data = to_align.read(
                        out_shape=(
                            to_align.count,
                            base_profile['height'],
                            base_profile['width']
                        ),
                        resampling=Resampling.bilinear
                    )
                    
                    # Scale image transform
                    transform = to_align.transform * to_align.transform.scale(
                        (to_align.width / data.shape[-1]),
                        (to_align.height / data.shape[-2])
                    )
                    
                    # Update profile
                    base_profile.update({
                        'transform': transform,
                        'width': data.shape[-1],
                        'height': data.shape[-2],
                        'nodata': to_align.nodata
                    })
                    
                    # Write aligned raster
                    with rasterio.open(output_path, 'w', **base_profile) as dst:
                        dst.write(data)
                
                return True
                
        except Exception as e:
            print(f"Alignment error: {e}")
            return False
        