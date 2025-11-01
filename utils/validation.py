import rasterio
import json
from datetime import datetime
from typing import Dict, List
import numpy as np 

class DataValidator:
    @staticmethod
    def validate_raster(file_path: str, expected_crs: str = None) -> Dict:
        """Validate raster file properties"""
        try:
            with rasterio.open(file_path) as src:
                validation_result = {
                    "file_exists": True,
                    "crs": str(src.crs),
                    "crs_match": True if not expected_crs else str(src.crs) == expected_crs,
                    "shape": src.shape,
                    "bounds": src.bounds,
                    "resolution": (src.res[0], src.res[1]),
                    "nodata": src.nodata,
                    "dtype": src.dtypes[0],
                    "band_count": src.count
                }
                
                # Validate data range
                data = src.read(1)
                validation_result.update({
                    "min_value": float(np.nanmin(data)),
                    "max_value": float(np.nanmax(data)),
                    "has_nodata": np.any(data == src.nodata) if src.nodata else False
                })
                
                return validation_result
                
        except Exception as e:
            return {
                "file_exists": False,
                "error": str(e)
            }
    
    @staticmethod
    def create_manifest(job_id: str, location: str, bbox: Dict, 
                       datasets: Dict, validation_results: Dict) -> Dict:
        """Create ingest log manifest"""
        return {
            "job_id": job_id,
            "location": location,
            "aoi_bbox": bbox,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "datasets": datasets,
            "validation": validation_results,
            "processing_parameters": {
                "target_crs": "EPSG:5070",
                "target_resolution": 30,
                "success": all(r.get("crs_match", False) for r in validation_results.values())
            }
        }