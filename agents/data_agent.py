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


class EarthEngineDataAgentComplete:
    """
    COMPLETE IMPLEMENTATION with preprocessing pipeline
    All 4 datasets: Land Cover, Tree Cover, NDVI, Population
    Full pipeline: Download → Clip → Reproject → Validate → Save GeoTIFF
    """
    def __init__(self):
        try:
            ee.Initialize(project='automation-476902')
            print("✅ Earth Engine initialized")
        except Exception as e:
            raise Exception(f"❌ Earth Engine not initialized: {e}")

        os.makedirs("outputs", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        self.geocoder = Geocoder()
        self.processor = SpatialProcessor(target_crs="EPSG:5070", resolution=30)
        self.validator = DataValidator()

    def generate_job_id(self):
        return f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def download_geotiff_from_ee(self, image: ee.Image, region: ee.Geometry, 
                                  filename: str, scale: int = 30) -> bool:
        """
        Download actual GeoTIFF from Earth Engine (not just visualization)
        """
        try:
            print(f"📥 Downloading GeoTIFF to {filename}...")
            
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
            else:
                print(f"❌ Download failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading GeoTIFF: {e}")
            return False

    def get_landcover_raster(self, bbox: Dict, year: int = 2019) ->Optional [str]:
        """
         Download, clip, reproject, validate Land Cover
        """
        try:
            print(f"🌳 Processing NLCD Land Cover {year}...")
            
            dataset = ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019").select('landcover')
            
            region = ee.Geometry.Rectangle([
                bbox["min_lon"], bbox["min_lat"],
                bbox["max_lon"], bbox["max_lat"]
            ])
            
            raw_file = "temp/landcover_raw.tif"
            if not self.download_geotiff_from_ee(dataset, region, raw_file):
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
            if not self.download_geotiff_from_ee(dataset, region, raw_file):
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
            if not self.download_geotiff_from_ee(ndvi, region, raw_file):
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

    def get_population_raster(self, bbox: Dict, year: int = 2020) -> str:
        """
         Download, clip, reproject WorldPop Population
        """
        try:
            print(f"👥 Processing Population {year}...")
            
            worldpop_url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/USA/usa_ppp_{year}_UNadj.tif"
            
            raw_file = "temp/population_raw.tif"
            print(f"📥 Downloading WorldPop from {worldpop_url}...")
            
            response = requests.get(worldpop_url, timeout=600)
            if response.status_code != 200:
                print(f"❌ Failed to download WorldPop: {response.status_code}")
                return None
            
            with open(raw_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded WorldPop")
            
            clipped_file = "temp/population_clipped.tif"
            if not self.processor.clip_raster(raw_file, clipped_file, bbox):
                return None
            
            reprojected_file = "temp/population_reprojected.tif"
            if not self.processor.reproject_raster(clipped_file, reprojected_file):
                return None
            
            final_file = "outputs/population_aligned.tif"
            os.rename(reprojected_file, final_file)
            
            print(f"✅ Population processing complete: {final_file}")
            return final_file
            
        except Exception as e:
            print(f"❌ Error processing population: {e}")
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
        data_types=["land_cover", "tree_cover", "ndvi", "population"],
        year=2020
    )
    
    print(json.dumps(result, indent=2))