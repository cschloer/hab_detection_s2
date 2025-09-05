import os
import requests
from pathlib import Path
from PIL import Image
import subprocess
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal
import zipfile
import json
from datetime import timedelta
import rasterio
from skimage.transform import resize
from constants import cyan_colormap
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import load_model, handle_input_transform, run_inference
import torch


def get_cyan_url(date, region_id):
    """Generate CyAN download URL for given date and region"""
    day_of_year = date.strftime("%j")
    year = date.strftime("%Y")
    return f"https://oceandata.sci.gsfc.nasa.gov/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif"


def download_cyan_geotiff(download_path, date, region_id):
    """Download CyAN geotiff file"""
    download_url = get_cyan_url(date, region_id)
    print(f"Download URL: {download_url}")
    outfile = Path(download_path)

    # Get bearer token from environment variable
    bearer_token = os.environ.get("EARTHDATA_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("EARTHDATA_BEARER_TOKEN environment variable not set")

    headers = {"Authorization": f"Bearer {bearer_token}"}

    R = requests.get(download_url, headers=headers, allow_redirects=True)
    if R.status_code != 200:
        raise ConnectionError(
            "could not download {}\nerror code: {}".format(download_url, R.status_code)
        )

    outfile.write_bytes(R.content)

    # Check what we actually downloaded
    file_size = outfile.stat().st_size
    print(f"Downloaded file size: {file_size} bytes")

    # Check if it's actually a TIFF file by reading the first few bytes
    with open(download_path, "rb") as f:
        header = f.read(8)
        print(f"File header (first 8 bytes): {header}")

    # Try to see if it's HTML error page
    if file_size < 50000:  # Suspiciously small for a geotiff
        with open(download_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(1000)  # Read first 1000 chars
            print(f"File content preview: {content}")

    return download_path


def get_copernicus_access_token():
    """Get access token for Copernicus Data Space Ecosystem"""
    username = os.environ.get("COPERNICUS_USERNAME")
    password = os.environ.get("COPERNICUS_PASSWORD")

    if not username or not password:
        raise ValueError(
            "COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables must be set"
        )

    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        raise ConnectionError(
            f"Failed to get access token: {response.status_code} - {response.text}"
        )

    return response.json()["access_token"]


def search_sentinel2(bbox, date, cloud_cover=20, date_range_days=3):
    """Search for Sentinel-2 products using Copernicus Data Space Ecosystem

    Args:
        bbox: Bounding box [min_lon, max_lat, max_lon, min_lat]
        date: Target date (datetime object)
        cloud_cover: Maximum cloud cover percentage
        date_range_days: Number of days to search before/after target date (default 3)
    """
    from datetime import timedelta

    access_token = get_copernicus_access_token()

    # Convert bbox to WKT polygon
    min_lon, max_lat, max_lon, min_lat = bbox
    footprint = f"POLYGON(({min_lon} {min_lat},{min_lon} {max_lat},{max_lon} {max_lat},{max_lon} {min_lat},{min_lon} {min_lat}))"

    # First, try to find products for the exact date
    print(
        f"Searching for Sentinel-2 products for exact date: {date.strftime('%Y-%m-%d')}"
    )
    start_date = date.strftime("%Y-%m-%dT00:00:00.000Z")
    end_date = date.strftime("%Y-%m-%dT23:59:59.999Z")

    search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    params = {
        "$filter": f"Collection/Name eq 'SENTINEL-2' and "
        f"contains(Name,'MSIL2A') and "
        f"ContentDate/Start ge {start_date} and ContentDate/Start le {end_date} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover})",
        "$orderby": "ContentDate/Start desc",
        "$top": "10",
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code != 200:
        raise ConnectionError(
            f"Search failed: {response.status_code} - {response.text}"
        )

    results = response.json()
    if results.get("value"):
        print(f"✓ Found Sentinel-2 product for exact date: {date.strftime('%Y-%m-%d')}")
        return results["value"][0]

    # If no products found for exact date, search within date range
    print(
        f"No products found for exact date. Searching within ±{date_range_days} days..."
    )

    search_start_date = date - timedelta(days=date_range_days)
    search_end_date = date + timedelta(days=date_range_days)

    range_start = search_start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    range_end = search_end_date.strftime("%Y-%m-%dT23:59:59.999Z")

    params["$filter"] = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"contains(Name,'MSIL2A') and "
        f"ContentDate/Start ge {range_start} and ContentDate/Start le {range_end} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover})"
    )

    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code != 200:
        raise ConnectionError(
            f"Search failed: {response.status_code} - {response.text}"
        )

    results = response.json()
    if not results.get("value"):
        raise ValueError(
            f"No Sentinel-2 products found for date {date.strftime('%Y-%m-%d')} "
            f"(searched ±{date_range_days} days) and bbox {bbox}"
        )

    # Log all available dates within the range
    available_dates = []
    for product in results["value"]:
        product_date = product["ContentDate"]["Start"][:10]  # Extract YYYY-MM-DD
        if product_date not in available_dates:
            available_dates.append(product_date)

    available_dates.sort()
    print(
        f"Available Sentinel-2 dates within ±{date_range_days} days: {', '.join(available_dates)}"
    )

    # Find the product closest to the target date
    best_product = None
    min_days_diff = float("inf")

    for product in results["value"]:
        product_date_str = product["ContentDate"]["Start"][:10]
        product_date = datetime.strptime(product_date_str, "%Y-%m-%d")
        days_diff = abs((product_date.date() - date.date()).days)

        if days_diff < min_days_diff:
            min_days_diff = days_diff
            best_product = product

    if best_product:
        best_date = best_product["ContentDate"]["Start"][:10]
        print(
            f"✓ Using closest available date: {best_date} ({min_days_diff} days from target)"
        )
        return best_product

    # This shouldn't happen if we found products above, but just in case
    return results["value"][0]


def download_sentinel2(product_info, download_path):
    """Download Sentinel-2 product"""
    access_token = get_copernicus_access_token()
    product_id = product_info["Id"]

    download_url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    headers = {"Authorization": f"Bearer {access_token}"}

    print(f"Downloading Sentinel-2 product: {product_info['Name']}")

    response = requests.get(download_url, headers=headers, stream=True)
    if response.status_code != 200:
        raise ConnectionError(f"Download failed: {response.status_code}")

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return download_path


def crop_cyan_to_bbox(cyan_path, bbox, output_path):
    """Crop CyAN image to bounding box using gdal_translate

    Args:
        cyan_path: Path to input CyAN geotiff
        bbox: Bounding box as [min_lon, max_lat, max_lon, min_lat] (EPSG:4326)
        output_path: Path for output cropped file
    """
    # Convert projection system of cyan image to EPSG:4326
    temp_cyan = output_path.replace(".tif", "_temp.tif")
    cmd = [
        "gdalwarp",
        "-t_srs",
        "EPSG:4326",
        "-srcnodata",
        "254",
        "-dstnodata",
        "255",
        cyan_path,
        temp_cyan,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"gdalwarp command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    # Crop to bounding box
    cmd = [
        "gdal_translate",
        "-f",
        "GTiff",
        "-projwin",
        str(bbox[0]),  # min_lon
        str(bbox[1]),  # max_lat
        str(bbox[2]),  # max_lon
        str(bbox[3]),  # min_lat
        "-projwin_srs",
        "EPSG:4326",
        temp_cyan,
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"gdal_translate command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    # Clean up temporary file
    os.remove(temp_cyan)
    return output_path


def process_sentinel2_bands(zip_path, output_dir):
    """Process Sentinel-2 bands to 20m resolution, based on dataset/download_and_process.py logic"""

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Find the .SAFE directory
    safe_dirs = [d for d in os.listdir(output_dir) if d.endswith(".SAFE")]
    if not safe_dirs:
        raise ValueError("No .SAFE directory found in extracted files")

    safe_path = os.path.join(output_dir, safe_dirs[0])
    granule_path = os.path.join(safe_path, "GRANULE")

    # Find granule subdirectories
    granule_dirs = [
        d
        for d in os.listdir(granule_path)
        if os.path.isdir(os.path.join(granule_path, d))
    ]
    if not granule_dirs:
        raise ValueError("No granule directories found")

    results = []

    for granule in granule_dirs:
        granule_dir = os.path.join(granule_path, granule)
        img_data_path = os.path.join(granule_dir, "IMG_DATA")

        print(f"Processing granule: {granule}")
        print(f"IMG_DATA path: {img_data_path}")

        # Get granule template for band naming - need to be more flexible
        # Modern format: L2A_T16TEN_A021654_20230719T163841
        # Older format: L1C_T16TEN_A021654_20230719T163841_N05.10

        # Try to find the actual band files by scanning the directory structure
        bands = {}

        # Check different resolution directories
        for res_dir in ["R10m", "R20m", "R60m"]:
            res_path = os.path.join(img_data_path, res_dir)
            if os.path.exists(res_path):
                print(f"Found resolution directory: {res_path}")
                band_files = [f for f in os.listdir(res_path) if f.endswith(".jp2")]
                print(f"Band files in {res_dir}: {band_files}")

                for band_file in band_files:
                    # Extract band number from filename - be more flexible with pattern matching
                    if "B01" in band_file and (
                        "10m" in band_file or "20m" in band_file or "60m" in band_file
                    ):
                        bands["band_01"] = os.path.join(res_path, band_file)
                    elif "B02" in band_file and (
                        "10m" in band_file or "20m" in band_file
                    ):
                        bands["band_02"] = os.path.join(res_path, band_file)
                    elif "B03" in band_file and (
                        "10m" in band_file or "20m" in band_file
                    ):
                        bands["band_03"] = os.path.join(res_path, band_file)
                    elif "B04" in band_file and (
                        "10m" in band_file or "20m" in band_file
                    ):
                        bands["band_04"] = os.path.join(res_path, band_file)
                    elif "B05" in band_file and "20m" in band_file:
                        bands["band_05"] = os.path.join(res_path, band_file)
                    elif "B06" in band_file and "20m" in band_file:
                        bands["band_06"] = os.path.join(res_path, band_file)
                    elif "B07" in band_file and "20m" in band_file:
                        bands["band_07"] = os.path.join(res_path, band_file)
                    elif (
                        "B08" in band_file
                        and "B8A" not in band_file
                        and ("10m" in band_file or "20m" in band_file)
                    ):
                        bands["band_08"] = os.path.join(res_path, band_file)
                    elif "B8A" in band_file and "20m" in band_file:
                        bands["band_08A"] = os.path.join(res_path, band_file)
                    elif "B09" in band_file and "60m" in band_file:
                        bands["band_09"] = os.path.join(res_path, band_file)
                    elif "B11" in band_file and "20m" in band_file:
                        bands["band_11"] = os.path.join(res_path, band_file)
                    elif "B12" in band_file and "20m" in band_file:
                        bands["band_12"] = os.path.join(res_path, band_file)

        print(f"Found {len(bands)} bands: {list(bands.keys())}")

        # Skip this granule if no bands were found
        if not bands:
            print(f"No bands found in granule {granule}, skipping...")
            continue

        # Create output paths
        output_vrt = os.path.join(output_dir, f"{granule}_20m.vrt")
        output_tif = os.path.join(output_dir, f"{granule}_20m.tif")

        # Build VRT file with all bands at 20m resolution
        cmd = [
            "gdalbuildvrt",
            "-resolution",
            "user",
            "-tr",
            "20",
            "20",
            "-separate",
            output_vrt,
        ]

        # Add bands in sorted order, only if they exist
        valid_bands = []
        for band_key in sorted(bands.keys()):
            band_path = bands[band_key]
            if os.path.exists(band_path):
                cmd.append(band_path)
                valid_bands.append(band_key)

        if not valid_bands:
            print(f"No valid band files found for granule {granule}, skipping...")
            continue

        print(f"Creating VRT with bands: {valid_bands}")

        # Create VRT
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"VRT creation failed: {e.stderr}")
            print(f"Command was: {' '.join(cmd)}")
            continue

        # Convert VRT to GeoTIFF
        cmd = ["gdal_translate", "-of", "GTiff", output_vrt, output_tif]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            results.append(output_tif)
        except subprocess.CalledProcessError as e:
            print(f"GeoTIFF conversion failed: {e.stderr}")
            continue

    if not results:
        raise ValueError("No Sentinel-2 bands could be processed")

    # Merge multiple granules if needed (similar to dataset/download_and_process.py:365-371)
    if len(results) > 1:
        merged_path = os.path.join(output_dir, "merged_20m.tif")

        # Use gdal_merge from the dataset folder logic
        merge_cmd = ["gdal_merge.py", "-of", "GTiff", "-o", merged_path] + results

        try:
            subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
            return merged_path
        except subprocess.CalledProcessError as e:
            print(f"Merge failed: {e.stderr}")
            return results[0]  # Return first result if merge fails

    return results[0]


def create_rgb_composite(zip_path, output_dir):
    """Create high-resolution RGB composite from B04, B03, B02 at native 10m resolution"""

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Find the .SAFE directory
    safe_dirs = [d for d in os.listdir(output_dir) if d.endswith(".SAFE")]
    if not safe_dirs:
        raise ValueError("No .SAFE directory found in extracted files")

    safe_path = os.path.join(output_dir, safe_dirs[0])
    granule_path = os.path.join(safe_path, "GRANULE")

    # Find granule subdirectories
    granule_dirs = [
        d
        for d in os.listdir(granule_path)
        if os.path.isdir(os.path.join(granule_path, d))
    ]
    if not granule_dirs:
        raise ValueError("No granule directories found")

    rgb_results = []

    for granule in granule_dirs:
        granule_dir = os.path.join(granule_path, granule)
        img_data_path = os.path.join(granule_dir, "IMG_DATA")
        r10m_path = os.path.join(img_data_path, "R10m")

        if not os.path.exists(r10m_path):
            print(f"No R10m directory found for granule {granule}")
            continue

        # Find RGB bands at 10m resolution
        band_files = [f for f in os.listdir(r10m_path) if f.endswith(".jp2")]
        rgb_bands = {}

        for band_file in band_files:
            if "B04" in band_file and "10m" in band_file:  # Red
                rgb_bands["red"] = os.path.join(r10m_path, band_file)
            elif "B03" in band_file and "10m" in band_file:  # Green
                rgb_bands["green"] = os.path.join(r10m_path, band_file)
            elif "B02" in band_file and "10m" in band_file:  # Blue
                rgb_bands["blue"] = os.path.join(r10m_path, band_file)

        if len(rgb_bands) != 3:
            print(f"Could not find all RGB bands for granule {granule}: {rgb_bands}")
            continue

        # Create RGB VRT at native 10m resolution
        output_vrt = os.path.join(output_dir, f"{granule}_rgb_10m.vrt")
        output_tif = os.path.join(output_dir, f"{granule}_rgb_10m.tif")

        cmd = [
            "gdalbuildvrt",
            "-separate",
            output_vrt,
            rgb_bands["red"],  # Band 1: Red
            rgb_bands["green"],  # Band 2: Green
            rgb_bands["blue"],  # Band 3: Blue
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"RGB VRT creation failed: {e.stderr}")
            continue

        # Convert VRT to GeoTIFF
        cmd = ["gdal_translate", "-of", "GTiff", output_vrt, output_tif]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            rgb_results.append(output_tif)
        except subprocess.CalledProcessError as e:
            print(f"RGB GeoTIFF conversion failed: {e.stderr}")
            continue

    if not rgb_results:
        raise ValueError("No RGB composite could be created")

    # Merge multiple granules if needed
    if len(rgb_results) > 1:
        merged_path = os.path.join(output_dir, "merged_rgb_10m.tif")
        merge_cmd = ["gdal_merge.py", "-of", "GTiff", "-o", merged_path] + rgb_results

        try:
            subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
            return merged_path
        except subprocess.CalledProcessError as e:
            print(f"RGB merge failed: {e.stderr}")
            return rgb_results[0]

    return rgb_results[0]


def crop_sentinel2_to_bbox(sen2_path, bbox, output_path):
    """Crop Sentinel-2 image to bounding box, based on dataset/download_and_process.py:379-452"""
    temp_sen2 = output_path.replace(".tif", "_temp.tif")

    # Convert projection system to EPSG:4326
    try:
        gdal.Warp(temp_sen2, sen2_path, dstSRS="EPSG:4326")
    except Exception as e:
        raise RuntimeError(f"Failed to reproject Sentinel-2: {e}")

    # Crop to bounding box
    min_lon, max_lat, max_lon, min_lat = bbox
    cmd = [
        "gdal_translate",
        "-f",
        "GTiff",
        "-projwin",
        str(min_lon),
        str(max_lat),
        str(max_lon),
        str(min_lat),
        "-projwin_srs",
        "EPSG:4326",
        temp_sen2,
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to crop Sentinel-2: {e.stderr}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_sen2):
            os.remove(temp_sen2)

    return output_path


def normalize_sen2_for_display(red, green, blue):
    """Normalize Sentinel-2 bands for display, from dataset/download_and_process.py:663-679"""

    def normalize(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)

    img = np.dstack((normalize(red), normalize(green), normalize(blue)))

    # Increase contrast
    pixvals = img
    minval = np.percentile(pixvals, 5)
    maxval = np.percentile(pixvals, 95)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 1

    return pixvals


def get_cloud_filter(sen2_data):
    """Cloud filter based on dataset/download_and_process.py:611-638"""
    # Band indices based on our processing order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
    blue = sen2_data[1]  # B02 (index 1)
    green = sen2_data[2]  # B03 (index 2)
    red = sen2_data[3]  # B04 (index 3)
    nir = sen2_data[7]  # B08 (index 7)
    swir_1 = sen2_data[10]  # B11 (index 10)
    swir_2 = sen2_data[11]  # B12 (index 11)

    # Cloud index developed by Zhai et. al
    CI_1 = np.absolute(((nir + 2 * swir_1) / (blue + green + red)) - 1)
    CI_2 = (blue + green + red + nir + swir_1 + swir_2) / 6

    # Parameter values from paper
    T1 = 1
    t2 = 1 / 10
    mean_CI_2 = np.mean(CI_2)
    T2 = mean_CI_2 + t2 * (np.max(CI_2) - mean_CI_2)

    cloud_filter = (CI_1 < T1) & (CI_2 > T2)
    return cloud_filter


def get_land_filter(sen2_data):
    """Land filter based on dataset/download_and_process.py:641-646"""
    # Band indices based on our processing order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
    green = sen2_data[2]  # B03 (index 2)
    band_8a = sen2_data[8]  # B8A (index 8)
    band_11 = sen2_data[10]  # B11 (index 10)

    land_filter = (band_8a > green) & (band_11 > green)
    return land_filter


def apply_cloud_land_filter_to_rgb(rgb_image, cloud_filter, land_filter):
    """Apply cloud and land filters to RGB image by setting filtered pixels to black"""
    filtered_rgb = rgb_image.copy()

    # Set cloud pixels to black
    filtered_rgb[cloud_filter] = [0, 0, 0]

    # Set land pixels to black
    filtered_rgb[land_filter] = [0, 0, 0]

    return filtered_rgb


def run_model_on_sentinel2(sen2_path, model_path="../model.pt"):
    """Run HAB detection model on Sentinel-2 data"""
    try:
        # Load the model
        model = load_model(model_path)
        model.eval()

        # Load and preprocess Sentinel-2 data
        with rasterio.open(sen2_path) as src:
            raw_image = src.read()  # Shape: (bands, height, width)

        # Apply the same preprocessing as in functions.py
        image = handle_input_transform(raw_image)

        # Ensure dimensions are divisible by 8
        height = image.shape[1]
        width = image.shape[2]
        ycrop = height % 8
        xcrop = width % 8
        image = image[:, 0 : height - ycrop, 0 : width - xcrop]
        image = torch.unsqueeze(image, 0)  # Add batch dimension

        # Run inference
        preds = run_inference(model, image)
        pred = torch.squeeze(preds)  # Remove batch dimension

        return pred.cpu().numpy(), (ycrop, xcrop)

    except Exception as e:
        print(f"Model inference failed: {e}")
        return None, None


def get_and_display_cyan(
    date,
    bbox,
    region_id="01",
    save_path=None,
    max_days_to_try=30,
    include_sentinel2=True,
    sentinel2_cache_dir=None,
    cyan_cache_dir=None,
    searchMode=False,
):
    """
    Download CyaN tile for given date and bounding box, crop it, and display it.
    Will cycle through dates until finding data with actual values (not just 0, 254, 255).
    Optionally also downloads and processes Sentinel-2 data for the same area and date.

    If searchMode=True, skips Sentinel-2 processing and only downloads/displays CyaN data.

    Args:
        date: datetime object for the starting date
        bbox: Bounding box as [min_lon, max_lat, max_lon, min_lat] (EPSG:4326)
        region_id: CyAN region ID (default "01" for CONUS region 1)
        save_path: Optional path to save the cropped image (if None, uses temp file)
        max_days_to_try: Maximum number of days to try (default 30)
        include_sentinel2: Whether to download and process Sentinel-2 data (default True)
        sentinel2_cache_dir: Directory to cache Sentinel-2 downloads (default: ./sentinel2_cache)
        cyan_cache_dir: Directory to cache CyAN downloads (default: ./cyan_cache)

    Returns:
        Tuple of (cyan_path, sen2_path, actual_date_used) if include_sentinel2=True
        Tuple of (cyan_path, actual_date_used) if include_sentinel2=False
    """
    from datetime import timedelta

    current_date = date

    # If searchMode is enabled, skip Sentinel-2 processing
    if searchMode:
        include_sentinel2 = False

    # Set up cache directories
    if sentinel2_cache_dir is None:
        sentinel2_cache_dir = os.path.join(os.getcwd(), "sentinel2_cache")
    if cyan_cache_dir is None:
        cyan_cache_dir = os.path.join(os.getcwd(), "cyan_cache")

    os.makedirs(sentinel2_cache_dir, exist_ok=True)
    os.makedirs(cyan_cache_dir, exist_ok=True)

    for day_offset in range(max_days_to_try):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Trying CyAN data for {current_date.strftime('%Y-%m-%d')}...")

                # Create cache filename for CyAN
                day_of_year = current_date.strftime("%j")
                year = current_date.strftime("%Y")
                cyan_cache_filename = f"L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif"
                cached_cyan_path = os.path.join(cyan_cache_dir, cyan_cache_filename)

                # Check if we already have this CyAN file cached
                if os.path.exists(cached_cyan_path):
                    print(f"Using cached CyAN file: {cached_cyan_path}")
                    original_cyan_path = cached_cyan_path
                else:
                    # Download to cache
                    print(f"Downloading CyAN data to cache: {cached_cyan_path}")
                    try:
                        download_cyan_geotiff(cached_cyan_path, current_date, region_id)
                        original_cyan_path = cached_cyan_path
                    except (ConnectionError, ValueError) as e:
                        print(
                            f"Failed to download for {current_date.strftime('%Y-%m-%d')}: {e}"
                        )
                        current_date += timedelta(days=1)
                        continue

                # Set up output path
                if save_path is None:
                    final_save_path = os.path.join(temp_dir, "cyan_cropped.tif")
                else:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    final_save_path = save_path

                # Crop to bounding box
                print(f"Cropping to bounding box {bbox}...")
                crop_cyan_to_bbox(original_cyan_path, bbox, final_save_path)

                # Check if we have meaningful data (not just no-data values)
                img = Image.open(final_save_path)
                img_array = np.array(img)

                # Count pixels that are not no-data values (0, 254, 255)
                meaningful_pixels = np.sum(
                    (img_array != 0) & (img_array != 254) & (img_array != 255)
                )
                total_pixels = img_array.size
                meaningful_ratio = meaningful_pixels / total_pixels

                print(
                    f"Meaningful pixels: {meaningful_pixels}/{total_pixels} ({meaningful_ratio:.2%})"
                )

                if meaningful_pixels > 0:
                    print(
                        f"Found meaningful data for {current_date.strftime('%Y-%m-%d')}!"
                    )
                    break
                else:
                    print(
                        f"No meaningful data for {current_date.strftime('%Y-%m-%d')}, trying next day..."
                    )
                    current_date += timedelta(days=1)
                    continue

        except Exception as e:
            print(f"Error processing {current_date.strftime('%Y-%m-%d')}: {e}")
            current_date += timedelta(days=1)
            continue

    else:
        raise ValueError(
            f"No meaningful CyAN data found in {max_days_to_try} days starting from {date.strftime('%Y-%m-%d')}"
        )

    # If we found meaningful data, continue with processing and visualization
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process CyAN data - use cached file
        day_of_year = current_date.strftime("%j")
        year = current_date.strftime("%Y")
        cyan_cache_filename = f"L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif"
        cached_cyan_path = os.path.join(cyan_cache_dir, cyan_cache_filename)

        if save_path is None:
            final_cyan_path = os.path.join(temp_dir, "cyan_cropped.tif")
        else:
            final_cyan_path = save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Crop the cached CyAN file to bounding box
        crop_cyan_to_bbox(cached_cyan_path, bbox, final_cyan_path)

        final_sen2_path = None
        final_sen2_rgb_path = None

        # Process Sentinel-2 data if requested
        if include_sentinel2:
            try:
                print(
                    f"Searching for Sentinel-2 data for {current_date.strftime('%Y-%m-%d')}..."
                )
                sen2_product = search_sentinel2(bbox, current_date)

                product_name = sen2_product["Name"]
                product_date = sen2_product["ContentDate"]["Start"][:10]
                cyan_date = current_date.strftime("%Y-%m-%d")

                print(f"✓ Sentinel-2 product available: {product_date}")

                # Check if Sentinel-2 product is for the exact same date as CyaN
                if product_date == cyan_date:
                    print(f"✓ Sentinel-2 product matches CyaN date: {cyan_date}")

                    if searchMode:
                        # In search mode, just log but don't download
                        print(f"  Search mode: Not downloading Sentinel-2 data")
                        final_sen2_path = None
                        final_sen2_rgb_path = None
                    else:
                        # Normal mode: download and process the data
                        print(f"  Downloading and processing Sentinel-2 data...")
                        sen2_zip_path = os.path.join(
                            sentinel2_cache_dir, f"{product_name}.zip"
                        )

                        if os.path.exists(sen2_zip_path):
                            print(f"  Using cached Sentinel-2 product: {sen2_zip_path}")
                        else:
                            print(
                                f"  Downloading Sentinel-2 product to cache: {sen2_zip_path}"
                            )
                            download_sentinel2(sen2_product, sen2_zip_path)

                        # Process bands to 20m resolution (use temp dir for processing)
                        print("  Processing Sentinel-2 bands to 20m resolution...")
                        sen2_processed_path = process_sentinel2_bands(
                            sen2_zip_path, temp_dir
                        )

                        # Create high-resolution RGB composite at 10m for true color display
                        print("  Creating high-resolution RGB composite at 10m...")
                        sen2_rgb_path = create_rgb_composite(sen2_zip_path, temp_dir)

                        # Crop both the multispectral stack and RGB composite to bounding box
                        final_sen2_path = os.path.join(
                            temp_dir, "sentinel2_cropped.tif"
                        )
                        crop_sentinel2_to_bbox(
                            sen2_processed_path, bbox, final_sen2_path
                        )

                        final_sen2_rgb_path = os.path.join(
                            temp_dir, "sentinel2_rgb_cropped.tif"
                        )
                        crop_sentinel2_to_bbox(sen2_rgb_path, bbox, final_sen2_rgb_path)
                else:
                    print(
                        f"✗ Sentinel-2 product date ({product_date}) does not match CyaN date ({cyan_date})"
                    )
                    print(
                        f"  Not downloading Sentinel-2 data - only logging availability"
                    )
                    final_sen2_path = None
                    final_sen2_rgb_path = None

            except Exception as e:
                print(f"Failed to process Sentinel-2 data: {e}")
                final_sen2_path = None
                final_sen2_rgb_path = None
                if not include_sentinel2:
                    # If Sentinel-2 was required but failed, raise the error
                    raise

        # Load and display the images
        print("Loading and displaying images...")
        cyan_img = Image.open(final_cyan_path)
        cyan_array = np.array(cyan_img)
        print(f"CyAN image shape: {cyan_array.shape}")
        print(f"CyAN unique values: {np.unique(cyan_array)}")

        # Apply CyAN colormap for display
        cyan_colored = cyan_colormap[cyan_array]

        # Create visualization
        if (
            final_sen2_path
            and os.path.exists(final_sen2_path)
            and final_sen2_rgb_path
            and os.path.exists(final_sen2_rgb_path)
        ):
            # Load high-resolution RGB data (10m)
            with rasterio.open(final_sen2_rgb_path) as src:
                sen2_rgb_data = src.read()

            # Load multispectral data for filtering (20m)
            with rasterio.open(final_sen2_path) as src:
                sen2_multispectral = src.read()

            # RGB bands are in order: Red, Green, Blue
            red_band = sen2_rgb_data[0]  # Red (B04)
            green_band = sen2_rgb_data[1]  # Green (B03)
            blue_band = sen2_rgb_data[2]  # Blue (B02)

            # Normalize for display
            sen2_rgb = normalize_sen2_for_display(red_band, green_band, blue_band)

            # Apply cloud and land filters using multispectral data
            cloud_filter = get_cloud_filter(sen2_multispectral)
            land_filter = get_land_filter(sen2_multispectral)

            # Resize filters to match RGB resolution if needed
            if sen2_rgb.shape[:2] != cloud_filter.shape:
                cloud_filter_resized = resize(
                    cloud_filter.astype(float),
                    sen2_rgb.shape[:2],
                    order=0,
                    preserve_range=True,
                ).astype(bool)
                land_filter_resized = resize(
                    land_filter.astype(float),
                    sen2_rgb.shape[:2],
                    order=0,
                    preserve_range=True,
                ).astype(bool)
            else:
                cloud_filter_resized = cloud_filter
                land_filter_resized = land_filter

            # Create filtered RGB image (clouds and land set to black)
            sen2_rgb_filtered = apply_cloud_land_filter_to_rgb(
                sen2_rgb, cloud_filter_resized, land_filter_resized
            )

            # Run model inference on Sentinel-2 data
            print("Running HAB detection model on Sentinel-2 data...")
            model_pred, crop_info = run_model_on_sentinel2(final_sen2_path)

            # Create overlay where CyAN is not no-data
            # Need to resize CyAN to match RGB resolution if they differ
            if sen2_rgb.shape[:2] != cyan_array.shape:
                cyan_array_resized = resize(
                    cyan_array.astype(float),
                    sen2_rgb.shape[:2],
                    order=0,
                    preserve_range=True,
                ).astype(cyan_array.dtype)
                cyan_colored_resized = resize(
                    cyan_colored.astype(float),
                    sen2_rgb.shape[:2] + (4,),
                    order=0,
                    preserve_range=True,
                ).astype(cyan_colored.dtype)
            else:
                cyan_array_resized = cyan_array
                cyan_colored_resized = cyan_colored

            cyan_mask = (
                (cyan_array_resized != 255)
                & (cyan_array_resized != 254)
                & (cyan_array_resized != 0)
            )
            cyan_overlay = np.zeros_like(cyan_colored_resized)
            cyan_overlay[cyan_mask] = cyan_colored_resized[cyan_mask]

            # Create 2x2 visualization to match the reference image
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # Top left: Sentinel-2 Image
            ax1.imshow(sen2_rgb)
            ax1.set_title("Sentinel 2 Image", fontsize=12, pad=10)
            ax1.axis("off")

            # Top right: CyAN HAB Index
            im2 = ax2.imshow(cyan_colored)
            ax2.set_title("CyAN HAB Index", fontsize=12, pad=10)
            ax2.axis("off")
            # Add colorbar for CyaN index (0-253 range)
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("CyaN Index", rotation=270, labelpad=15)
            cbar2.set_ticks([0, 50, 100, 150, 200, 253])
            cbar2.set_ticklabels(["0", "50", "100", "150", "200", "253"])

            # Bottom left: CyAN HAB Class - show classified CyaN data
            # Create CyaN classification from raw values
            cyan_classification = np.zeros_like(cyan_array)
            # Classify based on CyaN values (these are typical thresholds)
            cyan_classification[cyan_array == 0] = 0  # No bloom
            cyan_classification[(cyan_array > 0) & (cyan_array <= 100)] = (
                100  # Low bloom
            )
            cyan_classification[(cyan_array > 100) & (cyan_array <= 200)] = (
                200  # High bloom
            )
            cyan_classification[cyan_array > 200] = 254  # Very high bloom

            # Use same colormap structure as model prediction for consistency
            class_designation = [100, 200, 254]
            uniq = np.unique_counts(cyan_classification)
            print("_________________")
            print("CYAN CLASSIFICATION")
            print(uniq.values)
            print(uniq.counts)
            print("_________________")
            uniq = np.unique_counts(cyan_colormap)
            print("_________________")
            print("CYAN COLORMAP")
            print(uniq.values)
            print(uniq.counts)
            print("_________________")
            cyan_custom_colormap = []
            for i, c in enumerate(class_designation):
                cur_color = cyan_colormap[c - 1 if i != 0 else 0]
                cyan_custom_colormap.append(cur_color)
            cyan_custom_colormap.append(cyan_colormap[-1])  # For very high bloom class
            cyan_custom_colormap = np.array(cyan_custom_colormap)
            uniq = np.unique_counts(cyan_custom_colormap)
            print("_________________")
            print("CYAN CUSTOM COLORMAP")
            print(uniq.values)
            print(uniq.counts)
            print("_________________")

            cyan_class_colored = cyan_custom_colormap[cyan_classification]
            im3 = ax3.imshow(cyan_class_colored)
            ax3.set_title("CyAN HAB Class", fontsize=12, pad=10)
            ax3.axis("off")
            # Add discrete colorbar for CyaN classification
            from matplotlib.colors import ListedColormap

            cyan_class_cmap = ListedColormap(cyan_custom_colormap / 255.0)
            cbar3 = plt.colorbar(
                im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3]
            )
            cbar3.set_label("Class", rotation=270, labelpad=15)
            cbar3.set_ticklabels(["0-99", "100-199", "200-253", ">253"])

            # Bottom right: Prediction HAB Class - show model prediction
            if model_pred is not None:
                # Apply filters to model prediction
                ycrop, xcrop = crop_info
                pred_height, pred_width = model_pred.shape
                rgb_height, rgb_width = sen2_rgb.shape[:2]

                # Resize model prediction to match RGB resolution
                if (pred_height, pred_width) != (rgb_height, rgb_width):
                    model_pred_resized = resize(
                        model_pred.astype(float),
                        (rgb_height, rgb_width),
                        order=0,
                        preserve_range=True,
                    ).astype(model_pred.dtype)
                else:
                    model_pred_resized = model_pred

                # Apply cloud and land filters to prediction
                model_pred_filtered = model_pred_resized.copy()
                model_pred_filtered[cloud_filter_resized] = 3  # Set to no-data class
                model_pred_filtered[land_filter_resized] = 3  # Set to no-data class

                # Create colormap for model prediction (same as functions.py)
                model_custom_colormap = []
                for i, c in enumerate(class_designation):
                    cur_color = cyan_colormap[c - 1 if i != 0 else 0]
                    model_custom_colormap.append(cur_color)
                model_custom_colormap.append(cyan_colormap[-1])  # For no-data class
                model_custom_colormap = np.array(model_custom_colormap)

                model_pred_colored = model_custom_colormap[model_pred_filtered]
                im4 = ax4.imshow(model_pred_colored)
                ax4.set_title("Prediction HAB Class", fontsize=12, pad=10)
                ax4.axis("off")
                # Add discrete colorbar for model prediction
                model_class_cmap = ListedColormap(model_custom_colormap / 255.0)
                cbar4 = plt.colorbar(
                    im4, ax=ax4, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3]
                )
                cbar4.set_label("Class", rotation=270, labelpad=15)
                cbar4.set_ticklabels(["0-99", "100-199", "200-253", ">253"])
            else:
                # Show overlay if no model prediction
                ax4.imshow(sen2_rgb_filtered)
                ax4.imshow(cyan_overlay, alpha=0.6)
                ax4.set_title("Prediction HAB Class", fontsize=12, pad=10)
                ax4.axis("off")

        else:
            # Simple 1-panel visualization (CyAN only)
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

            # Colored image
            ax1.imshow(cyan_colored)
            ax1.set_title("CyAN Colored")
            ax1.axis("off")

        plt.suptitle(f'Data for {current_date.strftime("%Y-%m-%d")}')
        plt.tight_layout()
        plt.show()

    if include_sentinel2:
        return final_cyan_path, final_sen2_path, current_date
    else:
        return final_cyan_path, current_date


def list_cached_sentinel2_products(cache_dir=None):
    """List all cached Sentinel-2 products"""
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "sentinel2_cache")

    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return []

    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".zip")]
    print(f"Found {len(cached_files)} cached Sentinel-2 products:")
    for f in cached_files:
        file_path = os.path.join(cache_dir, f)
        file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
        print(f"  {f} ({file_size:.2f} GB)")

    return cached_files


def clear_sentinel2_cache(cache_dir=None, confirm=True):
    """Clear all cached Sentinel-2 products"""
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "sentinel2_cache")

    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".zip")]

    if not cached_files:
        print("No cached files to remove")
        return

    if confirm:
        response = input(
            f"Remove {len(cached_files)} cached files from {cache_dir}? (y/N): "
        )
        if response.lower() != "y":
            print("Cache clear cancelled")
            return

    for f in cached_files:
        os.remove(os.path.join(cache_dir, f))
        print(f"Removed {f}")

    print(f"Cleared {len(cached_files)} files from cache")


def list_cached_cyan_products(cache_dir=None):
    """List all cached CyAN products"""
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "cyan_cache")

    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return []

    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".tif")]
    print(f"Found {len(cached_files)} cached CyAN products:")
    for f in cached_files:
        file_path = os.path.join(cache_dir, f)
        file_size = os.path.getsize(file_path) / (1024**2)  # Size in MB
        print(f"  {f} ({file_size:.2f} MB)")

    return cached_files


def clear_cyan_cache(cache_dir=None, confirm=True):
    """Clear all cached CyAN products"""
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "cyan_cache")

    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".tif")]

    if not cached_files:
        print("No cached files to remove")
        return

    if confirm:
        response = input(
            f"Remove {len(cached_files)} cached files from {cache_dir}? (y/N): "
        )
        if response.lower() != "y":
            print("Cache clear cancelled")
            return

    for f in cached_files:
        os.remove(os.path.join(cache_dir, f))
        print(f"Removed {f}")

    print(f"Cleared {len(cached_files)} files from cache")


def list_all_cached_products(sentinel2_cache_dir=None, cyan_cache_dir=None):
    """List all cached products (both Sentinel-2 and CyAN)"""
    print("=== CACHED PRODUCTS SUMMARY ===")

    print("\nSentinel-2 products:")
    s2_files = list_cached_sentinel2_products(sentinel2_cache_dir)

    print("\nCyAN products:")
    cyan_files = list_cached_cyan_products(cyan_cache_dir)

    # Calculate total cache size
    total_size_gb = 0

    if sentinel2_cache_dir is None:
        sentinel2_cache_dir = os.path.join(os.getcwd(), "sentinel2_cache")
    if cyan_cache_dir is None:
        cyan_cache_dir = os.path.join(os.getcwd(), "cyan_cache")

    if os.path.exists(sentinel2_cache_dir):
        for f in s2_files:
            total_size_gb += os.path.getsize(os.path.join(sentinel2_cache_dir, f)) / (
                1024**3
            )

    if os.path.exists(cyan_cache_dir):
        for f in cyan_files:
            total_size_gb += os.path.getsize(os.path.join(cyan_cache_dir, f)) / (
                1024**3
            )

    print(f"\nTotal cache size: {total_size_gb:.2f} GB")
    return s2_files, cyan_files


if __name__ == "__main__":
    from datetime import datetime

    # Example usage
    test_bbox = [
        -86.628008,
        43.073522,
        -85.653038,
        42.207955,
    ]  # Example bounding box (Michigan area)

    date = datetime(2019, 8, 16)  # July 15, 2023
    bbox = [
        -85.972191,
        42.472820,
        -85.941217,
        42.454071,
    ]  # Swan Lake

    # Define cache directories first
    s2_cache_dir = "./my_sentinel2_cache"  # or None for default
    cyan_cache_dir = "./my_cyan_cache"  # or None for default

    # List any existing cached products
    print("Checking cache status:")
    list_all_cached_products(s2_cache_dir, cyan_cache_dir)
    print()

    try:

        result = get_and_display_cyan(
            date,
            bbox,
            region_id="6_2",
            max_days_to_try=10,
            include_sentinel2=True,
            sentinel2_cache_dir=s2_cache_dir,
            cyan_cache_dir=cyan_cache_dir,
            searchMode=False,
        )

        if len(result) == 3:  # CyAN + Sentinel-2 + date
            cyan_path, sen2_path, actual_date = result
            print(f"CyAN image saved to: {cyan_path}")
            print(f"Sentinel-2 image saved to: {sen2_path}")
            print(f"Actual date used: {actual_date.strftime('%Y-%m-%d')}")
        else:  # CyAN only + date
            cyan_path, actual_date = result
            print(f"CyAN image saved to: {cyan_path}")
            print(f"Actual date used: {actual_date.strftime('%Y-%m-%d')}")

        print("\nPost-processing cache status:")
        list_all_cached_products(s2_cache_dir, cyan_cache_dir)

        """
        # Cache management examples:
        print("\n=== CACHE MANAGEMENT EXAMPLES ===")
        print("# List only CyAN cache:")
        print("list_cached_cyan_products()")
        print("\n# Clear CyAN cache:")
        print("clear_cyan_cache()")
        print("\n# Clear both caches:")
        print("clear_sentinel2_cache(); clear_cyan_cache()")
        """

    except Exception as e:
        print(f"Error: {e}")
