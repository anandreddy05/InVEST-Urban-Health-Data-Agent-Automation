import streamlit as st
import requests
import json
import os
import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BACKEND_URL = "http://127.0.0.1:8000"

# Get absolute path to project directory
PROJECT_DIR = Path(__file__).parent.absolute()

st.set_page_config(page_title="🌆 Urban Data Automation Dashboard", layout="wide")
st.title("🌆 Urban Data Automation Dashboard")
st.write("LLM-powered Spatial Data Fetcher with Full Preprocessing Pipeline")

# ⚡ Function to visualize GeoTIFF
def visualize_geotiff(file_path: str, title: str):
    """Convert GeoTIFF to displayable image"""
    try:
        with rasterio.open(file_path) as src:
            # Check if it's RGB (3-band) or single band
            if src.count == 3:
                # RGB basemap - read all 3 bands
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                
                # Stack and normalize for display
                rgb = np.dstack((r, g, b))
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(rgb)
                ax.set_title(title)
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close()
                
                st.caption(f"📐 CRS: {src.crs} | Shape: {src.shape} | Resolution: {src.res[0]:.2f}m | RGB Composite")
            else:
                # Single band - use colormap
                data = src.read(1)
                
                # Mask nodata values
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                # Calculate statistics for better visualization
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    vmin = np.percentile(valid_data, 2)  # 2nd percentile for better contrast
                    vmax = np.percentile(valid_data, 98)  # 98th percentile to exclude outliers
                else:
                    vmin, vmax = None, None
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(title)
                cbar = plt.colorbar(im, ax=ax, label='Value')
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close()
                
                # Show actual data range
                if len(valid_data) > 0:
                    caption = f"📐 CRS: {src.crs} | Shape: {src.shape} | Resolution: {src.res[0]:.2f}m\n"
                    caption += f"📊 Data range: {valid_data.min():.2f} - {valid_data.max():.2f} (mean: {valid_data.mean():.2f})"
                    st.caption(caption)
                else:
                    st.caption(f"📐 CRS: {src.crs} | Shape: {src.shape} | Resolution: {src.res[0]:.2f}m")
            
    except Exception as e:
        st.error(f"Error visualizing {file_path}: {e}")

# Mode selection
mode = st.sidebar.radio("Select Mode", ["🗣️ Natural Language", "⚙️ Structured"])

# Natural Language Mode
if mode == "🗣️ Natural Language":
    st.subheader("💬 Natural Language Request")
    prompt = st.text_area("Example: Get NDVI and land cover data for Austin, Texas", 
                          height=100)
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2020)
    submit = st.button("🚀 Fetch Data via LLM")

    if submit and prompt.strip():
        with st.spinner("⏳ Processing request (5-15 minutes for large datasets like population)..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/agents/data/natural-language",
                    json={"prompt": prompt, "year": year},
                    timeout=1800  # 30 minutes timeout for large downloads
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Data Fetched Successfully!")

                    # Job Info
                    st.markdown("### 📄 Job Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Job ID", result.get("job_id", "N/A")[:16] + "...")
                    with col2:
                        st.metric("Location", result.get("location", result.get("city", "N/A")))
                    with col3:
                        st.metric("Status", result.get("status", "N/A"))

                    # Display Outputs
                    st.markdown("### 🗺️ Generated Raster Outputs")
                    outputs = result.get("outputs", {})
                    
                    if outputs:
                        cols = st.columns(2)
                        for idx, (data_type, file_path) in enumerate(outputs.items()):
                            with cols[idx % 2]:
                                st.markdown(f"#### {data_type.replace('_', ' ').title()}")
                                # Convert to absolute path if relative
                                abs_path = str(PROJECT_DIR / file_path) if not os.path.isabs(file_path) else file_path
                                
                                if os.path.exists(abs_path):
                                    if abs_path.endswith('.tif'):
                                        visualize_geotiff(abs_path, data_type.upper())
                                    else:
                                        image = Image.open(abs_path)
                                        st.image(image, use_container_width=True)
                                else:
                                    st.warning(f"File not found: {abs_path}")
                                    st.info(f"Looking for: {abs_path}")

                    # Validation Results
                    if "validation" in result:
                        st.markdown("### ✅ Validation Results")
                        st.json(result["validation"])

                    # Full Manifest
                    with st.expander("📋 View Complete Manifest"):
                        st.json(result)

                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. Large datasets (especially population) can take 15-20 minutes.")
                st.info("💡 Try again or exclude 'population' from your request for faster results.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Structured Mode
else:
    st.subheader("⚙️ Structured Data Request")
    
    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("Enter City", value="Chicago")
    with col2:
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2020)
    
    data_types = st.multiselect(
        "Select Data Types",
        ["land_cover", "tree_cover", "ndvi", "population", "basemap"],
        default=["land_cover", "ndvi"]
    )
    
    st.caption("💡 **basemap** = True-color satellite imagery (RGB composite from Sentinel-2)")
    
    submit_struct = st.button("🚀 Fetch Structured Data")

    if submit_struct and city.strip():
        with st.spinner("⏳ Fetching and processing data..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/agents/data/fetch",
                    json={"city": city, "data_types": data_types, "year": year},
                    timeout=1800  # 30 minutes for large datasets
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Data Retrieved Successfully!")

                    # Job Info
                    st.markdown("### 📄 Job Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Job ID", result.get("job_id", "N/A")[:16] + "...")
                    with col2:
                        st.metric("Location", result.get("location", result.get("city", "N/A")))
                    with col3:
                        st.metric("Status", result.get("status", "N/A"))

                    # Outputs
                    st.markdown("### 🗺️ Generated Outputs")
                    outputs = result.get("outputs", {})
                    
                    if outputs:
                        cols = st.columns(2)
                        for idx, (data_type, file_path) in enumerate(outputs.items()):
                            with cols[idx % 2]:
                                st.markdown(f"#### {data_type.replace('_', ' ').title()}")
                                if os.path.exists(file_path) and file_path.endswith('.tif'):
                                    visualize_geotiff(file_path, data_type.upper())

                    # Validation
                    if "validation" in result:
                        st.markdown("### ✅ Validation Results")
                        st.json(result["validation"])

                else:
                    st.error(f"❌ API Error {response.status_code}")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Previous Outputs Viewer
st.write("---")
col_title, col_delete_all = st.columns([4, 1])
with col_title:
    st.subheader("📂 Explore Previous Outputs")
with col_delete_all:
    if st.button("🗑️ Delete All", key="delete_all_outputs", type="secondary"):
        output_dir = PROJECT_DIR / "outputs"
        if output_dir.exists():
            try:
                deleted_count = 0
                # Delete all files in outputs directory
                for file in output_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                        deleted_count += 1
                st.success(f"✅ Deleted all {deleted_count} files from outputs directory")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error deleting files: {e}")

output_dir = PROJECT_DIR / "outputs"
if output_dir.exists():
    all_manifests = [f.name for f in output_dir.glob("*_manifest.json")]
    
    # ⚡ Filter valid manifests
    valid_manifests = []
    corrupted_manifests = []
    
    for manifest_file in all_manifests:
        try:
            with open(output_dir / manifest_file, "r") as f:
                json.load(f)
            valid_manifests.append(manifest_file)
        except json.JSONDecodeError:
            corrupted_manifests.append(manifest_file)
    
    if corrupted_manifests:
        with st.expander(f"⚠️ Found {len(corrupted_manifests)} corrupted manifest(s)"):
            for f in corrupted_manifests:
                st.text(f"- {f}")
            if st.button("🗑️ Delete Corrupted Manifests"):
                for f in corrupted_manifests:
                    (output_dir / f).unlink()
                st.success("✅ Deleted corrupted files. Refresh page.")
                st.rerun()
    
    manifests = sorted(valid_manifests, reverse=True)
    
    if manifests:
        selected_manifest = st.selectbox("Select a Job Manifest", manifests)
        
        with open(output_dir / selected_manifest, "r") as f:
            manifest = json.load(f)

        # Header with delete button
        col_header1, col_header2 = st.columns([4, 1])
        with col_header1:
            st.markdown(f"**🆔 Job ID:** {manifest.get('job_id', 'N/A')}")
            st.markdown(f"**🏙️ Location:** {manifest.get('location', manifest.get('city', 'N/A'))}")
            st.markdown(f"**🗓️ Timestamp:** {manifest.get('timestamp', 'N/A')}")
        with col_header2:
            if st.button("🗑️ Delete This Job", key="delete_job", type="secondary"):
                try:
                    # Delete manifest file
                    manifest_path = output_dir / selected_manifest
                    manifest_path.unlink()
                    
                    # Delete associated output files
                    outputs = manifest.get("outputs", {})
                    deleted_files = []
                    for dtype, file_path in outputs.items():
                        abs_path = str(PROJECT_DIR / file_path) if not os.path.isabs(file_path) else file_path
                        if os.path.exists(abs_path):
                            os.remove(abs_path)
                            deleted_files.append(dtype)
                    
                    st.success(f"✅ Deleted manifest and {len(deleted_files)} output file(s)")
                    st.info("Refresh the page to see updated list.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting files: {e}")
        
        # Display outputs with individual delete options
        outputs = manifest.get("outputs", {})
        if outputs:
            st.markdown("### 🗺️ Raster Outputs")
            
            for idx, (dtype, file_path) in enumerate(outputs.items()):
                # Convert to absolute path
                abs_path = str(PROJECT_DIR / file_path) if not os.path.isabs(file_path) else file_path
                if os.path.exists(abs_path):
                    st.markdown(f"#### {dtype.replace('_', ' ').title()}")
                    
                    # Create columns for visualization and delete button
                    col_viz, col_delete_btn = st.columns([5, 1])
                    
                    with col_viz:
                        if abs_path.endswith('.tif'):
                            visualize_geotiff(abs_path, dtype.upper())
                        else:
                            img = Image.open(abs_path)
                            st.image(img, use_container_width=True)
                    
                    with col_delete_btn:
                        if st.button("🗑️", key=f"delete_{dtype}_{idx}", help=f"Delete {dtype}"):
                            try:
                                os.remove(abs_path)
                                # Update manifest
                                del manifest["outputs"][dtype]
                                with open(output_dir / selected_manifest, "w") as f:
                                    json.dump(manifest, f, indent=2)
                                st.success(f"✅ Deleted {dtype}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
        
        # Show validation
        if "validation" in manifest:
            with st.expander("✅ View Validation Results"):
                st.json(manifest["validation"])
        
        # Full manifest
        with st.expander("📋 View Complete Manifest"):
            st.json(manifest)
    else:
        st.info("No valid manifests found. Run the pipeline first.")
else:
    st.warning("No 'outputs/' directory found.")

st.markdown("---")
st.caption("🌍 Built with FastAPI, Streamlit, Google Earth Engine, and Rasterio")