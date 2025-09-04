import os
import requests
from pathlib import Path
from PIL import Image
import subprocess
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal
from constants import cyan_colormap


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


def get_and_display_cyan(
    date, bbox, region_id="01", save_path=None, max_days_to_try=30
):
    """
    Download CyAN tile for given date and bounding box, crop it, and display it.
    Will cycle through dates until finding data with actual values (not just 0, 254, 255).

    Args:
        date: datetime object for the starting date
        bbox: Bounding box as [min_lon, max_lat, max_lon, min_lat] (EPSG:4326)
        region_id: CyAN region ID (default "01" for CONUS region 1)
        save_path: Optional path to save the cropped image (if None, uses temp file)
        max_days_to_try: Maximum number of days to try (default 30)

    Returns:
        Tuple of (path to cropped image, actual date used)
    """
    from datetime import timedelta

    current_date = date

    for day_offset in range(max_days_to_try):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download original CyAN file
                original_cyan_path = os.path.join(temp_dir, "cyan_original.tif")
                print(f"Trying CyAN data for {current_date.strftime('%Y-%m-%d')}...")

                try:
                    download_cyan_geotiff(original_cyan_path, current_date, region_id)
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

    # If we found meaningful data, continue with visualization
    with tempfile.TemporaryDirectory() as temp_dir:
        if save_path is None:
            # Download again for final processing since we're using temp directory
            original_cyan_path = os.path.join(temp_dir, "cyan_original.tif")
            download_cyan_geotiff(original_cyan_path, current_date, region_id)
            final_save_path = os.path.join(temp_dir, "cyan_cropped.tif")
            crop_cyan_to_bbox(original_cyan_path, bbox, final_save_path)
        else:
            final_save_path = save_path

        # Load and display the cropped image
        print("Displaying CyAN image...")
        img = Image.open(final_save_path)
        img_array = np.array(img)
        print(f"Final image shape: {img_array.shape}")
        print(f"Unique values: {np.unique(img_array)}")

        # Apply CyAN colormap for display
        cyan_colored = cyan_colormap[img_array]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Raw grayscale image
        ax1.imshow(img_array, cmap="gray")
        ax1.set_title("CyAN Raw Values")
        ax1.axis("off")

        # Colored image
        ax2.imshow(cyan_colored)
        ax2.set_title("CyAN Colored")
        ax2.axis("off")

        plt.suptitle(f'CyAN Data for {current_date.strftime("%Y-%m-%d")}')
        plt.tight_layout()
        plt.show()

    return final_save_path, current_date


if __name__ == "__main__":
    from datetime import datetime

    # Example usage
    test_bbox = [
        -86.628008,
        43.073522,
        -85.653038,
        42.207955,
    ]  # Example bounding box (Michigan area)

    date = datetime(2023, 7, 17)  # July 15, 2023
    bbox = [
        -85.972191,
        42.472820,
        -85.941217,
        42.454071,
    ]  # Swan Lake

    try:
        result_path, actual_date = get_and_display_cyan(
            date, bbox, region_id="6_2", max_days_to_try=1
        )
        print(f"CyAN image saved to: {result_path}")
        print(f"Actual date used: {actual_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error: {e}")
