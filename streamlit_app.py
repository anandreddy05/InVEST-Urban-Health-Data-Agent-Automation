import streamlit as st
import requests
import json
import os
import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="🌆 Urban Data Automation Dashboard", layout="wide")
st.title("🌆 Urban Data Automation Dashboard")
st.write("LLM-powered Spatial Data Fetcher with Full Preprocessing Pipeline")

# ⚡ Function to visualize GeoTIFF
def visualize_geotiff(file_path: str, title: str):
    """Convert GeoTIFF to displayable image"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data, cmap='viridis')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Value')
            ax.axis('off')
            
            st.pyplot(fig)
            plt.close()
            
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
        with st.spinner("⏳ Processing request (this may take 5-10 minutes for GeoTIFF processing)..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/agents/data/natural-language",
                    json={"prompt": prompt, "year": year},
                    timeout=900
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
                                if os.path.exists(file_path):
                                    if file_path.endswith('.tif'):
                                        visualize_geotiff(file_path, data_type.upper())
                                    else:
                                        image = Image.open(file_path)
                                        st.image(image, use_container_width=True)
                                else:
                                    st.warning(f"File not found: {file_path}")

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
                st.error("⏰ Request timed out. GeoTIFF processing can take 5-10 minutes.")
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
        ["land_cover", "tree_cover", "ndvi", "population"],
        default=["land_cover", "ndvi"]
    )
    
    submit_struct = st.button("🚀 Fetch Structured Data")

    if submit_struct and city.strip():
        with st.spinner("⏳ Fetching and processing data..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/agents/data/fetch",
                    json={"city": city, "data_types": data_types, "year": year},
                    timeout=900
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
st.subheader("📂 Explore Previous Outputs")

output_dir = "outputs"
if os.path.exists(output_dir):
    all_manifests = [f for f in os.listdir(output_dir) if f.endswith("_manifest.json")]
    
    # ⚡ Filter valid manifests
    valid_manifests = []
    corrupted_manifests = []
    
    for manifest_file in all_manifests:
        try:
            with open(os.path.join(output_dir, manifest_file), "r") as f:
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
                    os.remove(os.path.join(output_dir, f))
                st.success("✅ Deleted corrupted files. Refresh page.")
                st.rerun()
    
    manifests = sorted(valid_manifests, reverse=True)
    
    if manifests:
        selected_manifest = st.selectbox("Select a Job Manifest", manifests)
        
        with open(os.path.join(output_dir, selected_manifest), "r") as f:
            manifest = json.load(f)

        st.markdown(f"**🆔 Job ID:** {manifest.get('job_id', 'N/A')}")
        st.markdown(f"**🏙️ Location:** {manifest.get('location', manifest.get('city', 'N/A'))}")
        st.markdown(f"**🗓️ Timestamp:** {manifest.get('timestamp', 'N/A')}")
        
        # Display outputs
        outputs = manifest.get("outputs", {})
        if outputs:
            st.markdown("### 🗺️ Raster Outputs")
            cols = st.columns(2)
            for idx, (dtype, file_path) in enumerate(outputs.items()):
                if os.path.exists(file_path):
                    with cols[idx % 2]:
                        st.markdown(f"#### {dtype.replace('_', ' ').title()}")
                        if file_path.endswith('.tif'):
                            visualize_geotiff(file_path, dtype.upper())
                        else:
                            img = Image.open(file_path)
                            st.image(img, use_container_width=True)
        
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