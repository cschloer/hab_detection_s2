from helpers import get_cyan_url, TEMP_FOLDER, SAVE_FOLDER

import os
import json
import random
import re
import shutil
import sys
from pprint import pprint
from tqdm import tqdm
from PIL import Image, ImageColor, ImageDraw
import numpy as np
from osgeo import gdal
import pandas as pd
from urllib.request import urlretrieve
import datetime
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer
from sklearn.model_selection import train_test_split


CYAN_SIZE = 2000
SCENE_SIZE = 50
NUM_DAYS = 60

region_ids = ["8_3", "6_2", "7_2", "7_5", "6_5"]
# region_ids = ["6_2", "7_5", "8_3"]
# region_ids = ["7_5"]
start_year = 2019
start_month = 7
start_day = 1
least_clouds = {}
scenes = {}


def get_scene_id(region_id, x, y):
    return (
        f"{region_id}_X{str(x).zfill(4)}_Y{str(y).zfill(4)}_S{str(SCENE_SIZE).zfill(3)}"
    )


for i, region_id in enumerate(region_ids):
    print(f"Processing CYAN region {region_id}, {i+1}/{len(region_ids)}")
    date = datetime.datetime(start_year, start_month, start_day)
    with tqdm(
        total=((CYAN_SIZE / SCENE_SIZE) ** 2) * NUM_DAYS, file=sys.stdout
    ) as pbar:
        for j in range(NUM_DAYS):
            cyan_download_url = get_cyan_url(date, region_id)
            date_id = f"{region_id}_{date.strftime('%j')}"
            cyan_download_path = f"{TEMP_FOLDER}/{date_id}.tif"
            if not os.path.exists(cyan_download_path):
                urlretrieve(cyan_download_url, cyan_download_path)

            # raster_info = gdal.Info(cyan_download_path)
            # print(raster_info)
            with rasterio.open(cyan_download_path, "r") as ds:
                # read all raster values
                cyan_np = np.squeeze(ds.read())
                percent_clouds = np.count_nonzero(cyan_np == 255) / cyan_np.size
                if (
                    region_id not in least_clouds
                    or percent_clouds < least_clouds[region_id]["percent_clouds"]
                ):
                    least_clouds[region_id] = {
                        "download_path": cyan_download_path,
                        "percent_clouds": percent_clouds,
                    }
                for x in range(0, CYAN_SIZE, SCENE_SIZE):
                    for y in range(0, CYAN_SIZE, SCENE_SIZE):
                        pbar.update(1)

                        scene_id = get_scene_id(region_id, x, y)
                        if scene_id in scenes and "ignore" in scenes[scene_id]:
                            continue

                        cyan_np_tile = cyan_np[y : y + SCENE_SIZE, x : x + SCENE_SIZE]
                        if scene_id not in scenes:
                            percent_land = (
                                np.count_nonzero(cyan_np_tile == 254)
                                / cyan_np_tile.size
                            )
                            if percent_land > 0.9:
                                # Too much land, skip it
                                scenes[scene_id] = {
                                    "ignore": True,
                                    "ignore_reason": "land",
                                }
                                continue

                            top_left = ds.xy(y, x)
                            bottom_right = ds.xy(y + SCENE_SIZE, x + SCENE_SIZE)

                            transformer = Transformer.from_crs(ds.crs, "EPSG:4326")
                            x1, y1 = transformer.transform(top_left[0], top_left[1])
                            x2, y2 = transformer.transform(
                                bottom_right[0], bottom_right[1]
                            )
                            window = (
                                y1,
                                x1,
                                y2,
                                x2,
                            )
                            scenes[scene_id] = {
                                "window": window,
                                "cyan_id": region_id,
                                "id": scene_id,
                                "hab": [],
                            }

                        percent_water_hab = np.count_nonzero(
                            (cyan_np_tile < 254) & (cyan_np_tile != 0)
                        ) / np.count_nonzero(cyan_np_tile != 254)
                        scenes[scene_id]["hab"].append(percent_water_hab)

            date = date + datetime.timedelta(days=1)

used_tiles = []
for key in scenes.keys():
    scene = scenes[key]
    if "ignore" not in scene:
        hab_percent = round(np.mean(scene["hab"]) * 100, 2)
        scene["hab_percent"] = hab_percent
        del scene["hab"]
        if hab_percent < 10:
            scene["ignore"] = True
            scene["ignore_reason"] = "no_hab"
        else:

            # print(f"{key}: {hab_percent}%")
            used_tiles.append(key)


def get_metadata(id_):
    match = re.findall(
        "(\d_\d)_X(\d\d\d\d)_Y(\d\d\d\d)_S(\d\d\d)",
        id_,
        re.IGNORECASE,
    )
    if match:
        region_id, X, Y, S = match[0]
        return region_id, int(X), int(Y), int(S)
    return "", "", "", ""


# Set train/test, making sure that bordering scenes are in the same set
random.seed(42)
num_train = 0
num_test = 0


def find_and_update_bordering(region_id, X, Y, S, designation):
    count = 0
    for key2 in used_tiles:
        if "designation" not in scenes[key2]:
            region_id2, X2, Y2, _ = get_metadata(scenes[key2]["id"])
            if region_id2 == region_id and abs(X2 - X) <= S and abs(Y2 - Y) <= S:
                scenes[key2]["designation"] = designation
                count += 1 + find_and_update_bordering(
                    region_id, X2, Y2, S, designation
                )
    return count


for key in sorted(used_tiles):
    if "designation" not in scenes[key]:
        count = 1
        region_id, X, Y, S = get_metadata(scenes[key]["id"])
        if not region_id:
            raise Exception("Badly formatted id", scenes[key]["id"])
        designation = "train" if random.random() <= 0.7 else "test"
        scenes[key]["designation"] = designation
        # Set all bordering tiles to be the same designation
        count += find_and_update_bordering(region_id, X, Y, S, designation)
        if designation == "train":
            num_train += count
        else:
            num_test += count


# train, test = train_test_split(sorted(used_tiles), test_size=0.3, random_state=42)
print(f"Train size: {num_train} - {round(num_train / len(used_tiles), 2)}")
print(f"Test size: {num_test} - {round(num_test / len(used_tiles), 2)}")


# Draw the cyan image, visualizing the scenes
os.makedirs(f"{SAVE_FOLDER}/images", exist_ok=True)
for region_id in region_ids:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cyan_image = Image.open(least_clouds[region_id]["download_path"]).convert("RGBA")
    draw = ImageDraw.Draw(cyan_image)
    # Draw the black squares first, so they show up behind the others
    for x in range(0, CYAN_SIZE, SCENE_SIZE):
        for y in range(0, CYAN_SIZE, SCENE_SIZE):
            scene_id = get_scene_id(region_id, x, y)
            scene = scenes.get(scene_id, None)
            if scene and "ignore" in scene and scene["ignore_reason"] == "no_hab":
                draw.rectangle(
                    ((x + 1, y + 1), (x + SCENE_SIZE - 1, y + SCENE_SIZE - 1)),
                    outline="#000",
                )
    for x in range(0, CYAN_SIZE, SCENE_SIZE):
        for y in range(0, CYAN_SIZE, SCENE_SIZE):
            scene_id = get_scene_id(region_id, x, y)
            scene = scenes.get(scene_id, None)
            if scene and "ignore" not in scene:
                if scene["designation"] == "train":
                    color = "#ff0000"
                elif scene["designation"] == "test":
                    color = "#0000ff"
                else:
                    raise Exception("Unknown designation: ", scene["designation"])
                draw.rectangle(
                    ((x + 1, y + 1), (x + SCENE_SIZE - 2, y + SCENE_SIZE - 2)),
                    outline=color,
                    width=2,
                )

    ax.set_title(f"CYAN HAB Product, Visualization of Scenes for {region_id}")
    ax.imshow(cyan_image)
    ax.axis("off")
    plt.savefig(f"{SAVE_FOLDER}/images/{region_id}.png")
    print(f"Saved image {region_id}")
    # plt.show()


with open(f"{SAVE_FOLDER}/data.json", "w") as f:
    json.dump([scenes[key] for key in used_tiles], f)

shutil.rmtree(TEMP_FOLDER)
