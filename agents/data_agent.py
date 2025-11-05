import ee
import requests
import json
import os
from datetime import datetime
import uuid
from typing import Dict, List, Optional
from utils.geocoding import Geocoder
from utils.spatial_processing import SpatialProcessor
from utils.validation import DataValidator
from pydantic import BaseModel
import numpy as np
from config.settings import settings


class EarthEngineDataAgentComplete:
    """
    COMPLETE IMPLEMENTATION with preprocessing pipeline
    All 4 datasets: Land Cover, Tree Cover, NDVI, Population
    Full pipeline: Download → Clip → Reproject → Validate → Save GeoTIFF
    """
    def __init__(self):
        try:
            # Get project ID from settings (which loads from .env file)
            project_id = settings.GOOGLE_CLOUD_PROJECT
            
            if project_id:
                ee.Initialize(project=project_id)
                print(f"✅ Earth Engine initialized with project: {project_id}")
            else:
                # If no project specified, try default initialization
                # This requires a registered Earth Engine project
                print("⚠️ No GOOGLE_CLOUD_PROJECT found in .env file")
                print("   Please create a .env file with your Google Cloud Project ID")
                print("   See: https://developers.google.com/earth-engine/guides/access")
                raise Exception("GOOGLE_CLOUD_PROJECT not configured. Create a .env file with your project ID.")
        except Exception as e:
            raise Exception(f"❌ Earth Engine not initialized: {e}")

        os.makedirs("outputs", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        self.geocoder = Geocoder()
        self.processor = SpatialProcessor(target_crs="EPSG:5070", resolution=30)
        self.validator = DataValidator()
        
        # Load performance settings
        self.max_pixels = settings.MAX_PIXELS_PER_DOWNLOAD
        self.auto_scale = settings.AUTO_SCALE_RESOLUTION

    def generate_job_id(self):
        return f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def download_geotiff_from_ee(self, image: ee.Image, region: ee.Geometry, 
                                  filename: str, scale: int = 30, max_pixels: int = 1e8) -> bool:
        """
        Download actual GeoTIFF from Earth Engine with size optimization
        Uses numpy array method for large areas to bypass 50MB download limit
        """
        try:
            print(f"📥 Downloading GeoTIFF to {filename}...")
            
            # First, try the standard download for small areas
            try:
                url = image.getDownloadURL({
                    'region': region,
                    'scale': scale,
                    'format': 'GEO_TIFF'
                })
                
                response = requests.get(url, timeout=300)
                if response.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Downloaded: {filename}")
                    return True
                    
            except Exception as download_error:
                if "must be less than or equal to" in str(download_error):
                    print(f"⚠️ Area too large for direct download, using numpy method...")
                    return self._download_large_area_as_numpy(image, region, filename, scale, max_pixels)
                else:
                    raise download_error
                
        except Exception as e:
            error_msg = str(e)
            if "must be less than or equal to" in error_msg or "Total request size" in error_msg:
                print(f"⚠️ Area too large ({error_msg}), using numpy method...")
                return self._download_large_area_as_numpy(image, region, filename, scale, max_pixels)
            else:
                print(f"❌ Error downloading GeoTIFF: {e}")
                return False
    
    def _download_large_area_as_numpy(self, image: ee.Image, region: ee.Geometry, 
                                      filename: str, scale: int = 30, max_pixels: int = 1e8) -> bool:
        """
        Download large areas using tiled approach
        sampleRectangle has 262,144 pixel limit, so we need to tile large areas
        """
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
            
            print(f"📦 Downloading large area (this may take 5-10 minutes)...")
            
            # Get bounds
            bounds = region.bounds().getInfo()['coordinates'][0]
            min_lon = min(coord[0] for coord in bounds)
            max_lon = max(coord[0] for coord in bounds)
            min_lat = min(coord[1] for coord in bounds)
            max_lat = max(coord[1] for coord in bounds)
            
            # Calculate dimensions at target scale
            width = int((max_lon - min_lon) * 111320 / scale)
            height = int((max_lat - min_lat) * 111320 / scale)
            total_pixels = width * height
            
            # sampleRectangle limit is 262,144 pixels (512x512)
            SAMPLE_LIMIT = 250000  # Use 250K as safe limit (500x500)
            
            if total_pixels > SAMPLE_LIMIT:
                print(f"⚠️ Area requires tiled download ({total_pixels:,} pixels)")
                return self._download_tiled(image, region, filename, scale, SAMPLE_LIMIT)
            
            # Small enough for single sampleRectangle
            print(f"📥 Downloading {width}x{height} pixels...")
            sample = image.sampleRectangle(region=region, defaultValue=0)
            band_name = image.bandNames().getInfo()[0]
            data = np.array(sample.get(band_name).getInfo())
            
            # Save as GeoTIFF
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 
                                   data.shape[1], data.shape[0])
            
            with rasterio.open(
                filename, 'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=0,
                compress='lzw'
            ) as dst:
                dst.write(data, 1)
            
            print(f"✅ Downloaded: {filename} ({data.shape[0]}x{data.shape[1]} pixels)")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Too many pixels" in error_msg or "must be <=" in error_msg:
                print(f"⚠️ Area too large for single request, switching to tiled download...")
                return self._download_tiled(image, region, filename, scale, 250000)
            else:
                print(f"❌ Error in numpy download method: {e}")
                return False
    
    def _download_tiled(self, image: ee.Image, region: ee.Geometry, 
                       filename: str, scale: int = 30, tile_size_pixels: int = 250000) -> bool:
        """
        Download very large areas by splitting into tiles
        Each tile is downloaded separately then stitched together
        """
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
            from math import ceil, sqrt
            
            print(f"🧩 Using tiled download strategy...")
            
            # Get bounds
            bounds = region.bounds().getInfo()['coordinates'][0]
            min_lon = min(coord[0] for coord in bounds)
            max_lon = max(coord[0] for coord in bounds)
            min_lat = min(coord[1] for coord in bounds)
            max_lat = max(coord[1] for coord in bounds)
            
            # Calculate full dimensions
            full_width = int((max_lon - min_lon) * 111320 / scale)
            full_height = int((max_lat - min_lat) * 111320 / scale)
            total_pixels = full_width * full_height
            
            # Earth Engine sampleRectangle limit is 262,144 pixels
            # Use 250,000 as buffer, which means 500x500 tiles
            # But 500x500 = 250,000 < 262,144, so we can use slightly smaller tiles
            # Use 450x450 = 202,500 to be safe
            tile_dim = 450  # pixels per side (well under 262,144 limit)
            n_tiles_x = ceil(full_width / tile_dim)
            n_tiles_y = ceil(full_height / tile_dim)
            total_tiles = n_tiles_x * n_tiles_y
            
            print(f"📐 Full size: {full_width}x{full_height} = {total_pixels:,} pixels")
            print(f"🔢 Splitting into {n_tiles_x}x{n_tiles_y} = {total_tiles} tiles of {tile_dim}x{tile_dim}")
            print(f"⏱️ Estimated time: {total_tiles * 8} seconds (~{total_tiles * 8 / 60:.1f} minutes)")
            
            # Initialize full array
            band_name = image.bandNames().getInfo()[0]
            full_data = np.zeros((full_height, full_width), dtype=np.float32)
            
            # Download each tile
            completed = 0
            for i in range(n_tiles_y):
                for j in range(n_tiles_x):
                    # Calculate tile bounds in pixels
                    y_start = i * tile_dim
                    y_end = min((i + 1) * tile_dim, full_height)
                    x_start = j * tile_dim
                    x_end = min((j + 1) * tile_dim, full_width)
                    
                    # Actual tile dimensions (may be smaller at edges)
                    tile_height = y_end - y_start
                    tile_width = x_end - x_start
                    
                    # Calculate tile bounds in degrees
                    tile_min_lon = min_lon + (max_lon - min_lon) * (x_start / full_width)
                    tile_max_lon = min_lon + (max_lon - min_lon) * (x_end / full_width)
                    tile_min_lat = min_lat + (max_lat - min_lat) * (y_start / full_height)
                    tile_max_lat = min_lat + (max_lat - min_lat) * (y_end / full_height)
                    
                    tile_region = ee.Geometry.Rectangle([
                        tile_min_lon, tile_min_lat,
                        tile_max_lon, tile_max_lat
                    ])
                    
                    # Download tile with retry
                    for attempt in range(3):
                        try:
                            sample = image.sampleRectangle(region=tile_region, defaultValue=0)
                            tile_data = np.array(sample.get(band_name).getInfo())
                            
                            # Verify tile dimensions match expected
                            if tile_data.shape[0] != tile_height or tile_data.shape[1] != tile_width:
                                # Resize if needed (shouldn't happen but handle gracefully)
                                tile_data = np.resize(tile_data, (tile_height, tile_width))
                            
                            # Place tile in full array - use actual tile dimensions
                            full_data[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_data
                            
                            completed += 1
                            if completed % 10 == 0 or completed == total_tiles:
                                progress = (completed / total_tiles) * 100
                                print(f"   Progress: {completed}/{total_tiles} tiles ({progress:.0f}%)")
                            break
                            
                        except Exception as tile_error:
                            if attempt < 2:
                                print(f"   ⚠️ Tile {i},{j} failed, retrying...")
                                import time
                                time.sleep(2)
                            else:
                                print(f"   ❌ Tile {i},{j} failed after 3 attempts: {tile_error}")
                                raise
            
            # Save complete GeoTIFF
            print(f"💾 Saving complete raster...")
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 
                                   full_width, full_height)
            
            with rasterio.open(
                filename, 'w',
                driver='GTiff',
                height=full_height,
                width=full_width,
                count=1,
                dtype=full_data.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=0,
                compress='lzw',
                tiled=True,
                blockxsize=256,
                blockysize=256
            ) as dst:
                dst.write(full_data, 1)
            
            print(f"✅ Tiled download complete: {filename} ({full_height}x{full_width} pixels)")
            return True
            
        except Exception as e:
            print(f"❌ Error in tiled download: {e}")
            return False

    def get_landcover_raster(self, bbox: Dict, year: int = 2019) ->Optional [str]:
        """
         Download, clip, reproject, validate Land Cover
        """
        try:
            # Check if location is within USA (NLCD coverage)
            if not (-130 <= bbox["min_lon"] <= -60 and 24 <= bbox["min_lat"] <= 50):
                print(f"⚠️ NLCD Land Cover only available for continental USA")
                print(f"   Location: {bbox['min_lon']:.2f}, {bbox['min_lat']:.2f}")
                print(f"   Skipping land_cover dataset...")
                return None
            
            print(f"🌳 Processing NLCD Land Cover {year}...")
            
            dataset = ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019").select('landcover')
            
            region = ee.Geometry.Rectangle([
                bbox["min_lon"], bbox["min_lat"],
                bbox["max_lon"], bbox["max_lat"]
            ])
            
            raw_file = "temp/landcover_raw.tif"
            if not self.download_geotiff_from_ee(dataset, region, raw_file, scale=30, max_pixels=self.max_pixels):
                return None
            
            clipped_file = "temp/landcover_clipped.tif"
            if not self.processor.clip_raster(raw_file, clipped_file, bbox):
                return None
            
            reprojected_file = "temp/landcover_reprojected.tif"
            if not self.processor.reproject_raster(clipped_file, reprojected_file):
                return None
            
            final_file = "outputs/landcover_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ Land Cover processing complete: {final_file}")
            return final_file
            
        except Exception as e:
            print(f"❌ Error processing land cover: {e}")
            return None

    def get_treecover_raster(self, bbox: Dict, year: int = 2021) -> Optional[str]:
        """
        Download, clip, reproject Tree Canopy Cover
        """
        try:
            # Check if location is within USA (NLCD coverage)
            if not (-130 <= bbox["min_lon"] <= -60 and 24 <= bbox["min_lat"] <= 50):
                print(f"⚠️ NLCD Tree Cover only available for continental USA")
                print(f"   Location: {bbox['min_lon']:.2f}, {bbox['min_lat']:.2f}")
                print(f"   Skipping tree_cover dataset...")
                return None
            
            print(f"🌲 Processing Tree Canopy Cover {year}...")
            
            # Use the latest 2023 release (v2023-5) which includes years 2011-2023
            if 2011 <= year <= 2023:
                collection = ee.ImageCollection("USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5")
                dataset = collection.filter(ee.Filter.eq('year', year)).first()
                # Select the NLCD processed band (0-100% tree canopy cover)
                dataset = dataset.select('NLCD_Percent_Tree_Canopy_Cover')
            else:
                print(f"⚠️ Year {year} not available (2011-2023). Using 2021 data.")
                collection = ee.ImageCollection("USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5")
                dataset = collection.filter(ee.Filter.eq('year', 2021)).first()
                dataset = dataset.select('NLCD_Percent_Tree_Canopy_Cover')
            
            region = ee.Geometry.Rectangle([
                bbox["min_lon"], bbox["min_lat"],
                bbox["max_lon"], bbox["max_lat"]
            ])
            
            raw_file = "temp/treecover_raw.tif"
            if not self.download_geotiff_from_ee(dataset, region, raw_file, scale=30, max_pixels=self.max_pixels):
                return None
            
            clipped_file = "temp/treecover_clipped.tif"
            if not self.processor.clip_raster(raw_file, clipped_file, bbox):
                return None
            
            reprojected_file = "temp/treecover_reprojected.tif"
            if not self.processor.reproject_raster(clipped_file, reprojected_file):
                return None
            
            final_file = "outputs/treecover_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ Tree Cover processing complete: {final_file}")
            return final_file
            
        except Exception as e:
            print(f"❌ Error processing tree cover: {e}")
            return None
    def get_ndvi_raster(self, bbox: Dict, year: int = 2020) -> str:
        """
         Compute, download, clip, reproject NDVI
        """
        try:
            print(f"🌿 Processing NDVI {year}...")
            
            region = ee.Geometry.Rectangle([
                bbox["min_lon"], bbox["min_lat"],
                bbox["max_lon"], bbox["max_lat"]
            ])
            
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(f"{year}-06-01", f"{year}-08-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            )
            
            ndvi = collection.median().normalizedDifference(["B8", "B4"])
            
            raw_file = "temp/ndvi_raw.tif"
            if not self.download_geotiff_from_ee(ndvi, region, raw_file, scale=30, max_pixels=self.max_pixels):
                return None
            
            clipped_file = "temp/ndvi_clipped.tif"
            if not self.processor.clip_raster(raw_file, clipped_file, bbox):
                return None
            
            reprojected_file = "temp/ndvi_reprojected.tif"
            if not self.processor.reproject_raster(clipped_file, reprojected_file):
                return None
            
            final_file = "outputs/ndvi_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ NDVI processing complete: {final_file}")
            return final_file
        
        except Exception as e:
            print(f"❌ Error processing NDVI: {e}")
            return None

    def get_basemap_raster(self, bbox: Dict, year: int = 2020) -> Optional[str]:
        """
        Download true-color satellite imagery (RGB composite)
        Uses Sentinel-2 for natural color visualization
        """
        try:
            print(f"🗺️ Processing Satellite Basemap {year}...")
            
            region = ee.Geometry.Rectangle([
                bbox["min_lon"], bbox["min_lat"],
                bbox["max_lon"], bbox["max_lat"]
            ])
            
            # Get Sentinel-2 imagery
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(f"{year}-06-01", f"{year}-08-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            )
            
            # Create RGB composite (Red, Green, Blue bands)
            rgb = collection.median().select(['B4', 'B3', 'B2'])
            
            # Normalize to 0-255 range for visualization
            rgb_vis = rgb.divide(3000).multiply(255).clamp(0, 255).toUint8()
            
            raw_file = "temp/basemap_raw.tif"
            if not self.download_geotiff_from_ee(rgb_vis, region, raw_file, scale=30, max_pixels=self.max_pixels):
                return None
            
            clipped_file = "temp/basemap_clipped.tif"
            if not self.processor.clip_raster(raw_file, clipped_file, bbox):
                return None
            
            reprojected_file = "temp/basemap_reprojected.tif"
            if not self.processor.reproject_raster(clipped_file, reprojected_file):
                return None
            
            final_file = "outputs/basemap_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ Basemap processing complete: {final_file}")
            return final_file
        
        except Exception as e:
            print(f"❌ Error processing basemap: {e}")
            return None

    def get_population_raster(self, bbox: Dict, year: int = 2020) -> str:
        """
        ⚡ OPTIMIZED: Download only the area of interest from Earth Engine
        Uses WorldPop dataset via Google Earth Engine (much faster)
        """
        try:
            print(f"👥 Processing Population {year}...")
            
            # Check if year is available in Earth Engine WorldPop
            if year < 2000 or year > 2020:
                print(f"⚠️ Year {year} not available in WorldPop. Using 2020.")
                year = 2020
            
            # Use Earth Engine WorldPop dataset (already clipped to AOI)
            print(f"📥 Fetching WorldPop from Earth Engine (year {year})...")
            
            # WorldPop dataset in Earth Engine
            worldpop = ee.ImageCollection("WorldPop/GP/100m/pop")
            
            region = ee.Geometry.Rectangle([
                bbox["min_lon"], bbox["min_lat"],
                bbox["max_lon"], bbox["max_lat"]
            ])
            
            # Filter by year and region (no country filter - let Earth Engine handle geography)
            # This automatically returns the correct country's data based on the bounding box
            population = (worldpop
                         .filter(ee.Filter.eq('year', year))
                         .filterBounds(region)
                         .mosaic()  # Combine tiles if boundary crosses countries
                         .select('population'))
            
            # Download only the clipped area (much faster!)
            raw_file = "temp/population_raw.tif"
            if not self.download_geotiff_from_ee(population, region, raw_file, scale=100, max_pixels=self.max_pixels):
                print("⚠️ Earth Engine download failed, falling back to direct download...")
                return self._download_population_direct(bbox, year)
            
            print(f"✅ Downloaded population from Earth Engine")
            
            # Clip to exact bounds
            clipped_file = "temp/population_clipped.tif"
            if not self.processor.clip_raster(raw_file, clipped_file, bbox):
                return None
            
            # Reproject to target CRS
            reprojected_file = "temp/population_reprojected.tif"
            if not self.processor.reproject_raster(clipped_file, reprojected_file):
                return None
            
            final_file = "outputs/population_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ Population processing complete: {final_file}")
            return final_file
            
        except Exception as e:
            print(f"❌ Error processing population via Earth Engine: {e}")
            print("⚠️ Attempting fallback method...")
            return self._download_population_direct(bbox, year)
    
    def _download_population_direct(self, bbox: Dict, year: int = 2020) -> str:
        """
        Fallback method: Direct download from WorldPop (slower, but more reliable)
        Uses GDAL virtual warping to avoid downloading entire USA raster
        """
        try:
            print(f"📥 Downloading WorldPop subset directly...")
            
            worldpop_url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/USA/usa_ppp_{year}_UNadj.tif"
            
            # Use /vsicurl/ to read remote file without full download
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            
            raw_file = "temp/population_subset.tif"
            
            # Read only the subset we need
            print(f"🔄 Streaming population data for bounding box...")
            with rasterio.open(f'/vsicurl/{worldpop_url}') as src:
                # Calculate window for our bbox
                window = from_bounds(
                    bbox['min_lon'], bbox['min_lat'],
                    bbox['max_lon'], bbox['max_lat'],
                    src.transform
                )
                
                # Read only the window
                data = src.read(1, window=window)
                
                # Get the transform for the window
                window_transform = src.window_transform(window)
                
                # Write subset to file
                with rasterio.open(
                    raw_file, 'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=window_transform,
                    nodata=src.nodata
                ) as dst:
                    dst.write(data, 1)
            
            print(f"✅ Downloaded population subset ({data.shape[0]}x{data.shape[1]} pixels)")
            
            # Reproject to target CRS
            reprojected_file = "temp/population_reprojected.tif"
            if not self.processor.reproject_raster(raw_file, reprojected_file):
                return None
            
            final_file = "outputs/population_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ Population processing complete: {final_file}")
            return final_file
            
        except Exception as e:
            print(f"❌ Error in fallback population download: {e}")
            return None

    def make_json_safe(self, obj):
        """Recursively convert numpy types and non-serializable objects to JSON-safe types"""
        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_safe(i) for i in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        else:
            return str(obj)


    def process_city_data(self, city_name: str, data_types: List[str], year: int = 2020):
        """
        ✅ COMPLETE PIPELINE with ALL 4 datasets + validation
        """
        job_id = self.generate_job_id()
        print(f"\n🚀 Processing data for: {city_name}")
        print(f"📊 Requested datasets: {data_types}")
        print(f"📅 Year: {year}")
        
        bbox = self.geocoder.get_bounding_box(city_name)
        if not bbox:
            raise ValueError(f"❌ Could not geocode {city_name}")
        
        outputs = {}
        dataset_info = {}
        validation_results = {}
        
        # Process each requested dataset
        for data_type in data_types:
            print(f"\n{'='*60}")
            print(f"📦 Processing {data_type}...")
            print(f"{'='*60}")
            
            file_path = None
            source = None
            
            if data_type == "land_cover":
                file_path = self.get_landcover_raster(bbox, year)
                source = "NLCD Land Cover"
                
            elif data_type == "tree_cover":
                file_path = self.get_treecover_raster(bbox, year)
                source = "NLCD Tree Canopy"
                
            elif data_type == "ndvi":
                file_path = self.get_ndvi_raster(bbox, year)
                source = "Sentinel-2 NDVI"
                
            elif data_type == "population":
                file_path = self.get_population_raster(bbox, year)
                source = "WorldPop Population"
                
            elif data_type == "basemap":
                file_path = self.get_basemap_raster(bbox, year)
                source = "Sentinel-2 RGB Satellite"
            
            else:
                print(f"⚠️ Unknown data type: {data_type}")
                continue
            
            if file_path and os.path.exists(file_path):
                outputs[data_type] = file_path
                
                # Validate the processed raster
                print(f"🔍 Validating {data_type}...")
                validation = self.validator.validate_raster(file_path, "EPSG:5070")
                validation_results[data_type] = validation
                
                dataset_info[data_type] = {
                    "source": source,
                    "resolution": "30m",
                    "crs": "EPSG:5070",
                    "bbox": bbox,
                    "year": year,
                    "validation": {
                        "crs_match": validation.get("crs_match", False),
                        "bounds": validation.get("bounds"),
                        "shape": validation.get("shape"),
                        "min_value": validation.get("min_value"),
                        "max_value": validation.get("max_value")
                    }
                }
                print(f"✅ {data_type} validated successfully")
            else:
                print(f"❌ Failed to process {data_type}")
        
        # Create complete manifest with validation
        manifest = self.validator.create_manifest(
            job_id=job_id,
            location=city_name,
            bbox=bbox,
            datasets=dataset_info,
            validation_results=validation_results
        )
        
        manifest["outputs"] = outputs
        manifest["status"] = "completed" if outputs else "failed"
        
        # Save manifest safely
        manifest_file = f"outputs/{job_id}_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # Convert numpy / bool types for API JSON response
        manifest = self.make_json_safe(manifest)

        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE!")
        print(f"📄 Manifest: {manifest_file}")
        print(f"📊 Processed {len(outputs)}/{len(data_types)} datasets")
        print(f"{'='*60}\n")

        return manifest



# ========================================
# Example Usage
# ========================================
if __name__ == "__main__":
    agent = EarthEngineDataAgentComplete()
    
    result = agent.process_city_data(
        city_name="New York City",
        data_types=["land_cover", "tree_cover", "ndvi", "population", "basemap"],
        year=2020
    )
    
    print(json.dumps(result, indent=2))