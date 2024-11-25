from helpers import (
    get_cyan_url,
    TEMP_FOLDER,
    cyan_colormap,
)
import requests
from pathlib import Path
from PIL import Image
import shutil
from osgeo import ogr, osr, gdal
import zipfile
from matplotlib import pyplot as plt
from sentinelsat import LTATriggered
from tempfile import TemporaryFile
from urllib.request import urlretrieve
import gdal_merge
import glob
import numpy as np
import os
import rasterio
import subprocess
import time


def download_and_process(
    api,
    sen2_uuid,
    region_key,
    unique_id,
    window,
    cyan_region_id,
    date,
    zip_file_path,
    image_download_path,
    log,
    lock_zip,
    subset_resolution=64,
    subset_stride=64,
):

    if sen2_uuid == "b767a097-089c-4572-bbf3-8606107015a5":
        log("Skipping weird image from lakemichigan1 2020/11/6")
        return
    temp_folder = f"{TEMP_FOLDER}/{unique_id}"
    os.makedirs(image_download_path, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    year = date.year
    month = date.month
    day = date.day

    try:

        # download sen2 file
        log(f"Downloading SEN2 files to {temp_folder}...")
        exists, sen2_zip_filename = download_sen2(
            api, temp_folder + "/", sen2_uuid, attempt_if_offline=False
        )
        if sen2_zip_filename:
            sen2_zip_path = f"{temp_folder}/{sen2_zip_filename}"
            log("SEN2 download complete")

            # download CYAN
            log("Downloading CYAN geotiff...")
            cyan_geotiff_path = download_cyan_geotiff(
                f"{temp_folder}/cyan.tif", date, cyan_region_id
            )
            log("CYAN download complete")

            # Convert sen2 download to
            log("Converting SEN2 files to geotiff...")
            sen2_geotiff_path = generate_sen2_geotiff(sen2_zip_path, temp_folder + "/")
            log("SEN2 conversion complete...")

            cyan, sen2 = warp_and_crop(
                temp_folder + "/", cyan_geotiff_path, sen2_geotiff_path, window
            )

            visualize_overlap(
                cyan, sen2, region_key, year, month, day, image_download_path
            )

            # Get proper sized tiles
            count = 0
            total_count = 0
            with rasterio.open(cyan, "r") as ds:
                cyan_arr = ds.read()  # read all raster values

            cyan_np, sen2_np = get_image_full(cyan, sen2)
            sen2_cloud_filter = get_cloud_filter(sen2_np)
            sen2_land_filter = get_land_filter(sen2_np)
            cyan_np_filtered = apply_cloud_and_land_filter(
                cyan_np, sen2_cloud_filter, sen2_land_filter
            )

            _, row_size, col_size = cyan_arr.shape
            log(f"Image shape {row_size} x {col_size}")

            # Count if this image has any interesting HAB features
            # If not, we only save 1 image
            #
            save_only_one = False
            save_only_one_complete = False
            cyan_masked_nodata = np.where(cyan_np == 255, np.nan, cyan_np)
            cyan_masked_nodata_and_nohab = np.where(
                cyan_masked_nodata == 0, np.nan, cyan_masked_nodata
            )
            hab_count = np.count_nonzero(~np.isnan(cyan_masked_nodata_and_nohab))
            if hab_count == 0:
                # save_only_one = True
                pass
            fig = plt.figure(figsize=(24, 8))
            with lock_zip:
                with zipfile.ZipFile(
                    zip_file_path, mode="a", compression=zipfile.ZIP_STORED
                ) as z:
                    for x in range(0, row_size - subset_resolution, subset_stride):
                        for y in range(0, col_size - subset_resolution, subset_stride):
                            total_count += 1

                            # Create two overlapping subtiles
                            cyan_subset_filtered = cyan_np_filtered[
                                :, x : x + subset_resolution, y : y + subset_resolution
                            ]
                            sen2_subset = sen2_np[
                                :, x : x + subset_resolution, y : y + subset_resolution
                            ]

                            skip = (
                                np.count_nonzero(cyan_subset_filtered == 255)
                                / subset_resolution ** 2
                                >= 0.95
                            )
                            if skip and not (
                                count == 0
                                and
                                # Make sure it isn't the last image
                                x + subset_stride + subset_resolution > row_size
                                and y + subset_stride + subset_resolution > col_size
                            ):
                                continue
                            # Visualize selected samples of the subsets
                            if count <= 500 and count % 100 == 0:
                                ax1 = fig.add_subplot(
                                    2, 6, 1 + (count // 100) * 2, xticks=[], yticks=[]
                                )
                                ax1.set_title(f"sen2 {count}")
                                ax1.imshow(sen2_subset[1, :, :], cmap="gray")

                                ax2 = fig.add_subplot(
                                    2, 6, 2 + (count // 100) * 2, xticks=[], yticks=[]
                                )
                                ax2.set_title(f"cyan {count}")
                                cyan_reshaped = cyan_subset_filtered.reshape(
                                    subset_resolution, subset_resolution
                                )
                                cyan_image = cyan_colormap[cyan_reshaped]
                                ax2.imshow(cyan_image)
                            count += 1

                            save_data(
                                cyan_subset_filtered,
                                sen2_subset,
                                z,
                                region_key,
                                sen2_uuid,
                                year,
                                month,
                                day,
                                x,
                                y,
                                subset_resolution,
                                count,
                            )
                            if save_only_one:
                                save_only_one_complete = True
                                break
                        if save_only_one and save_only_one_complete:
                            break
                    plt.savefig(f"{image_download_path}/tiles.png")
                    log(f"Saved {count} of {total_count} possible images.")

    finally:
        try:
            shutil.rmtree(temp_folder)
            plt.close()
            pass
        except:
            pass
        pass


def download_sen2(api, download_path, uuid, download=True, attempt_if_offline=True):
    """Download Sentinel 3 image"""
    is_online = api.is_online(uuid)
    count = 0
    if is_online or attempt_if_offline:
        if not download:
            if not is_online:
                try:
                    api.download(uuid)
                except LTATriggered as e:
                    return True, ""
                return True, ""
        else:
            filename = ""
            while True:
                try:
                    r = api.download(uuid, directory_path=download_path)
                    filename = r["title"] + ".zip"
                    break
                except LTATriggered as e:
                    count += 1
                    # return True, ""
                    time.sleep(60 * 5)
            return True, filename
    return True, ""


"""
Function for downloading geotiff
"""


def download_cyan_geotiff(download_path, date, region_id):
    download_url = get_cyan_url(date, region_id)
    # urlretrieve(download_url, download_path)
    outfile = Path(download_path)
    R = requests.get(download_url, allow_redirects=True)
    if R.status_code != 200:
        raise ConnectionError(
            "could not download {}\nerror code: {}".format(url, R.status_code)
        )

    outfile.write_bytes(R.content)
    return download_path


"""
Functions for creating sen2 geotiff
"""


def complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


def get_immediate_subdirectories(a_dir):
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def generate_all_bands(unprocessedBandPath, granule, outputPathSubdirectory):

    granuleBandTemplate = granule[:-6]
    granpart_1 = unprocessedBandPath.split(".SAFE")[0][-22:-16]
    granule_2 = unprocessedBandPath.split(".SAFE")[0][-49:-34]

    granuleBandTemplate = granpart_1 + "_" + granule_2 + "_"

    outputPathSubdirectory = outputPathSubdirectory
    if not os.path.exists(outputPathSubdirectory + "/IMAGE_DATA"):
        os.makedirs(outputPathSubdirectory + "/IMAGE_DATA")

    outPutTiff = "/" + granule[:-6] + "16Bit-AllBands.tif"
    outPutVRT = "/" + granule[:-6] + "16Bit-AllBands.vrt"
    outPutFullPath = outputPathSubdirectory + "/IMAGE_DATA" + outPutTiff
    outPutFullVrt = outputPathSubdirectory + "/IMAGE_DATA" + outPutVRT

    inputPath = unprocessedBandPath  # + granuleBandTemplate

    bands = {
        "band_01": inputPath + "R60m/" + granuleBandTemplate + "B01_60m.jp2",
        "band_02": inputPath + "R10m/" + granuleBandTemplate + "B02_10m.jp2",
        "band_03": inputPath + "R10m/" + granuleBandTemplate + "B03_10m.jp2",
        "band_04": inputPath + "R10m/" + granuleBandTemplate + "B04_10m.jp2",
        "band_05": inputPath + "R20m/" + granuleBandTemplate + "B05_20m.jp2",
        "band_06": inputPath + "R20m/" + granuleBandTemplate + "B06_20m.jp2",
        "band_07": inputPath + "R20m/" + granuleBandTemplate + "B07_20m.jp2",
        "band_08": inputPath + "R10m/" + granuleBandTemplate + "B08_10m.jp2",
        "band_08A": inputPath + "R20m/" + granuleBandTemplate + "B8A_20m.jp2",
        "band_09": inputPath + "R60m/" + granuleBandTemplate + "B09_60m.jp2",
        # No band 10, because it is removed in 2A. Only used in atmospheric correction
        "band_11": inputPath + "R20m/" + granuleBandTemplate + "B11_20m.jp2",
        "band_12": inputPath + "R20m/" + granuleBandTemplate + "B12_20m.jp2",
    }

    cmd = [
        "gdalbuildvrt",
        "-resolution",
        "user",
        "-tr",
        "20",
        "20",
        "-separate",
        outPutFullVrt,
    ]
    outputFiles = []
    for band_key in sorted(bands.keys()):
        band_path = bands[band_key]
        cmd.append(band_path)
        outputFiles.append(band_path)
    my_file = Path(outPutFullVrt)
    if not my_file.is_file():
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = ["gdal_translate", "-of", "GTiff", outPutFullVrt, outPutFullPath]

    my_file = Path(outPutTiff)
    if not my_file.is_file():
        # file exists
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return outPutFullPath


def generate_sen2_geotiff(inputProductPath, outputPath):
    """
    BAND ORDERING AFTER PROCESSING
      B01 60m (Ultra Blue - Coastal & Aerosol)
      B02 10m (Blue)
      B03 10m (Green)
      B04 10m (Red)
      B05 20m (VNIR)
      B06 20m (VNIR)
      B07 20m (VNIR)
      B08 10m (VNIR)
      B8A 20m (VNIR)
      B09 60m (SWIR)
      B11 20m (SWIR)
      B12 20m (SWIR)

    """
    basename = os.path.basename(inputProductPath)

    if os.path.isdir(outputPath + basename[:-3] + "SAFE"):
        pass
        # print("Already extracted")
    else:
        zip = zipfile.ZipFile(inputProductPath)
        zip.extractall(outputPath)
        # print("Extracting Done")

    directoryName = outputPath + basename[:-3] + "SAFE/GRANULE"

    productName = os.path.basename(inputProductPath)[:-4]
    outputPathSubdirectory = outputPath + productName + "_PROCESSED"
    merged = outputPathSubdirectory + "/merged.tif"
    if os.path.exists(merged):
        return merged
    if not os.path.exists(outputPathSubdirectory):
        os.makedirs(outputPathSubdirectory)

    subDirectorys = get_immediate_subdirectories(directoryName)

    results = []
    for granule in subDirectorys:
        unprocessedBandPath = (
            outputPath + productName + ".SAFE/GRANULE/" + granule + "/" + "IMG_DATA/"
        )
        results.append(
            generate_all_bands(unprocessedBandPath, granule, outputPathSubdirectory)
        )

    params = ["", "-of", "GTiff", "-o", merged]

    for granule in results:
        params.append(granule)

    gdal_merge.main(params)
    return merged


"""
Function for projection into new coordinate system and translating
"""


def warp_and_crop(base, cyan_image_path, sen2_image_path, window):
    temp_sen2 = base + "temp_sen2.tif"
    temp2_sen2 = base + "temp2_sen2.tif"

    temp_cyan = base + "temp_cyan.tif"
    temp2_cyan = base + "temp2_cyan.tif"
    temp3_cyan = base + "temp3_cyan.tif"

    if os.path.exists(temp2_sen2) and os.path.exists(temp3_cyan):
        return temp3_cyan, temp2_sen2

    # Convert projection system of sen2 image
    gdal.Warp(temp_sen2, sen2_image_path, dstSRS="EPSG:4326")
    src = gdal.Open(temp_sen2)
    _, xres, _, _, _, yres = src.GetGeoTransform()

    # Convert projection system of cyan image
    # gdal.Warp(temp_cyan, cyan_image_path, dstSRS = 'EPSG:4326', dstNoData="'254 255'")
    cmd = [
        "gdalwarp",
        "-t_srs",
        "EPSG:4326",
        "-srcnodata",
        '"254"',
        "-dstnodata",
        '"255"',
        cyan_image_path,
        temp_cyan,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # sen2_gdal = gdal.Open(temp_sen2)
    # print(gdal.Info(sen2_gdal))
    # Change x and y resolution of cyan image
    cmd = [
        "gdalwarp",
        "-s_srs",
        "EPSG:4326",
        "-tr",
        str(abs(xres)),
        str(abs(yres)),
        "-r",
        "bilinear",
        "-tap",
        "-srcnodata",
        "255",
        temp_cyan,
        temp2_cyan,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Translate the cyan image
    cmd = [
        "gdal_translate",
        "-f",
        "GTiff",
        "-projwin",
        str(window[0]),
        str(window[1]),
        str(window[2]),
        str(window[3]),
        "-projwin_srs",
        "EPSG:4326",
        temp2_cyan,
        temp3_cyan,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Translate the sen2 image
    cmd[-1] = temp2_sen2
    cmd[-2] = temp_sen2
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return temp3_cyan, temp2_sen2


def visualize_overlap(cyan, sen2, region_key, year, month, day, image_download_path):
    # VISUALIZE SATELLITE IMAGE

    # cube = gdal.Open(COMBINED_TIF_IMAGE)
    cube = gdal.Open(sen2)

    # print(cube.RasterXSize, cube.RasterYSize)
    band_blue = cube.GetRasterBand(2)
    band_green = cube.GetRasterBand(3)
    band_red = cube.GetRasterBand(4)

    img1 = band_red.ReadAsArray(0, 0, cube.RasterXSize, cube.RasterYSize)
    img2 = band_green.ReadAsArray(0, 0, cube.RasterXSize, cube.RasterYSize)
    img3 = band_blue.ReadAsArray(0, 0, cube.RasterXSize, cube.RasterYSize)

    img = normalize_sen2(img1, img2, img3)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    ax1 = axs[0][0]
    ax1.set_title("Sentinel 2 MSI 2A product")
    ax1.imshow(img)
    ax1.axis("off")

    ax2 = axs[0][1]

    img2 = Image.open(cyan)
    ax2.set_title("CYAN HAB Product")
    ax2.imshow(img2)
    ax2.axis("off")

    # Run cloud and land filter on entire image to visualize if it worked
    with rasterio.open(sen2, "r") as ds:
        sen2_np = ds.read()  # read all raster values
    cloud_filter = get_cloud_filter(sen2_np)
    land_filter = get_land_filter(sen2_np)

    with rasterio.open(cyan, "r") as ds:
        cyan_np = ds.read()  # read all raster values

    cyan_filtered = apply_cloud_and_land_filter(cyan_np, cloud_filter, land_filter)

    # Need to create a custom color palette here in order to display it
    cyan_filtered_reshaped = cyan_filtered.reshape(
        cyan_filtered.shape[1], cyan_filtered.shape[2]
    )
    cyan_image = cyan_colormap[cyan_filtered_reshaped]

    # Creating an overlay that has pixel value [0,0,0,255] which is non opaque black
    black = np.concatenate(
        (
            np.full(
                (3, cyan_filtered_reshaped.shape[0], cyan_filtered_reshaped.shape[1],),
                0,
            ),
            np.full(
                (1, cyan_filtered_reshaped.shape[0], cyan_filtered_reshaped.shape[1],),
                255,
            ),
        ),
        axis=0,
    )

    # Creating an overlay that has pixel value [0,0,0,0], which makes it entirely opaque
    opaque = np.full(
        (4, cyan_filtered_reshaped.shape[0], cyan_filtered_reshaped.shape[1],), 0,
    )

    black_white_filter = np.where(cyan_filtered_reshaped == 255, black, opaque,)
    black_white_filter_reshaped = np.moveaxis(black_white_filter, 0, -1)
    ax3 = axs[1][0]
    ax3.set_title("Sentinel 2 with filter")
    ax3.imshow(img)
    ax3.imshow(black_white_filter_reshaped,)
    ax3.axis("off")

    ax4 = axs[1][1]
    ax4.set_title("Cyan with sen2 clouds and land")
    ax4.imshow(cyan_image)
    ax4.axis("off")
    """
  ax5 = axs[2][0]
  ax5.set_title("Overlay pre filter")
  ax5.imshow(img)
  ax5.imshow(img2, alpha=0.4)
  ax5.axis('off')


  ax6 = axs[2][1]
  ax6.set_title("Overlay post filter")
  ax6.imshow(img)
  ax6.imshow(cyan_image, alpha=0.4)
  ax6.axis('off')
  """
    fig.suptitle(f"{region_key}: {year}-{month}-{day}")
    plt.savefig(f"{image_download_path}/overlap.png")
    plt.close()


def visualize_subset(
    cyan_np, sen2_np, region_key, year, month, day, image_download_path
):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax1.set_title("sen2")
    ax1.imshow(sen2_np[1, :, :], cmap="gray")

    ax2 = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax2.set_title("cyan")
    cyan_reshaped = cyan_np.reshape(cyan_np.shape[1], cyan_np.shape[2])
    cyan_image = cyan_colormap[cyan_reshaped]
    ax2.imshow(cyan_image)
    plt.savefig(f"{image_download_path}/subset.png")
    plt.close()


def get_image_full(cyan_path, sen2_path):
    with rasterio.open(cyan_path, "r") as ds:
        cyan_arr = ds.read()  # read all raster values

    with rasterio.open(sen2_path, "r") as ds:
        sen2_arr = ds.read()  # read all raster values

    return (
        cyan_arr,
        sen2_arr,
    )


def save_data(
    cyan_np,
    sen2_np,
    z,
    region_key,
    sen2_uuid,
    year,
    month,
    day,
    tile_start_x,
    tile_start_y,
    tile_size,
    id_,
):

    filename_base = f"{region_key}_{str(year).zfill(4)}_{str(month).zfill(2)}_{str(day).zfill(2)}_x{tile_start_x}_y{tile_start_y}_{tile_size}x{tile_size}_{id_}"
    cyan_filename = filename_base + "_cyan.npy"
    sen2_filename = filename_base + "_sen2.npy"
    with TemporaryFile() as tp:
        np.save(tp, cyan_np)
        tp.seek(0)
        z.writestr(cyan_filename, tp.read())
    with TemporaryFile() as tp:
        np.save(tp, sen2_np)
        tp.seek(0)
        z.writestr(sen2_filename, tp.read())


def get_cloud_filter(img):
    # A cloud index developed by Zhai et. al, formulated into code for sen2
    # https://doi.org/10.1016/j.isprsjprs.2018.07.006

    # The indices here are based on the processing done in a previous step
    blue = img[1, :, :]
    green = img[2, :, :]
    red = img[3, :, :]
    NIR = img[7, :, :]
    SWIR_1 = img[10, :, :]
    SWIR_2 = img[11, :, :]

    CI_1 = np.absolute(((NIR + 2 * SWIR_1) / (blue + green + red)) - 1)
    CI_2 = (blue + green + red + NIR + SWIR_1 + SWIR_2) / 6

    # Parameter value, as described in paper
    T1 = 1

    t2 = 1 / 10
    mean_CI_2 = np.mean(CI_2)
    # T2 is based on the image values
    T2 = mean_CI_2 + t2 * (np.max(CI_2) - mean_CI_2)

    test = CI_2 < T2

    cloud_filter = (CI_1 < T1) & (CI_2 > T2)

    return cloud_filter


def get_land_filter(img):
    green = img[2, :, :]
    band_8a = img[8, :, :]
    band_11 = img[11, :, :]
    land_filter = (band_8a > green) & (band_11 > green)
    return land_filter


def apply_cloud_and_land_filter(cyan_np, cloud_filter, land_filter):
    # The cyan image after applying the sen2 cloud filter to it
    cyan_np_land_filtered = np.where((cyan_np != 255) & (land_filter), 255, cyan_np,)
    cyan_np_cloud_land_filtered = np.where(
        (cyan_np_land_filtered != 254) & (cloud_filter), 255, cyan_np_land_filtered,
    )

    return cyan_np_cloud_land_filtered


# raster_info = gdal.Info(sen2_xml)
# print(raster_info)


def normalize_sen2(red, green, blue):
    def normalize(arr):
        """Function to normalize an input array to 0-1"""
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
    Image.fromarray(pixvals.astype(np.uint8))
    return pixvals
