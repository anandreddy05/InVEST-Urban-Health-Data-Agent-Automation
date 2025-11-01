import ee
import os

def test_earth_engine_auth():
    """Test if Earth Engine is properly authenticated"""
    print("🧪 Testing Earth Engine Authentication...")
    
    try:
        ee.Authenticate()
        # Initialize Earth Engine
        ee.Initialize(project='automation-476902')
        print("✅ Earth Engine initialized successfully!")
        
        # Test with a simple dataset
        dem = ee.Image('USGS/SRTMGL1_003')
        print("✅ Can access Earth Engine datasets!")
        
        # Get basic info
        info = dem.getInfo()
        print(f"🌍 Dataset type: {info['type']}")
        print(f"📊 Bands: {list(info['bands'])}")
        
        # Test NLCD access
        nlcd = ee.Image('USGS/NLCD/NLCD2016').select('landcover')
        print("✅ Can access NLCD dataset!")
        
        return True
        
    except ee.EEException as e:
        print(f"❌ Earth Engine error: {e}")
        print("\n🔧 Please run: earthengine authenticate")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Earth Engine Test Starting...")
    success = test_earth_engine_auth()
    
    if success:
        print("\n🎉 Earth Engine is working correctly!")
        print("You can now run the main data agent.")
    else:
        print("\n❌ Earth Engine setup failed.")
        print("Please run: earthengine authenticate")
# import ee
# ee.Authenticate()
# ee.Initialize(project='automation-476902')
# print(ee.String('Hello from the Earth Engine servers!').getInfo())