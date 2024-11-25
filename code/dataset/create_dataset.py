from download_and_process import download_and_process
from helpers import (
    get_cyan_url,
    TEMP_FOLDER,
    SAVE_FOLDER,
    ZIP_FILE_TRAIN,
    ZIP_FILE_TEST,
    get_api,
)
import zipfile
import datetime
import os
from archive import get_products
import json
import time
import threading
from threading import Thread
from sentinelsat import LTAError, LTATriggered, SentinelAPI


dl_ready = []
lock_dl_ready = threading.Lock()

trigger_list = []
lock_trigger_list = threading.Lock()

existing_prefixes = set()
lock_existing_prefixes = threading.Lock()

total_available = 0
total_downloaded = 0
lock_total_downloaded = threading.Lock()

lock_zip = threading.Lock()


complete = []


def create_logger(log_prefix):
    return lambda s: print(f"{datetime.datetime.now()}-{log_prefix}: {s}")


def generate_file_prefix(id_, date):
    return f"{id_}_{str(date.year).zfill(4)}_{str(date.month).zfill(2)}_{str(date.day).zfill(2)}"


def manage_triggers(api, name):
    global total_downloaded
    global existing_prefixes
    global dl_ready
    global trigger_list
    log = create_logger(f"-- LTA Trigger Thread {name}")
    waiting = {}

    def handle_online(r):
        is_online = api.is_online(r["uuid"])
        if is_online:
            with lock_dl_ready:
                log(f"Adding {r['uuid']} to download queue")
                dl_ready.insert(0, r)
                return True
        return False

    while True:
        # Check if previously triggered items are now online
        delete_waiting = []
        try:
            for k in waiting.keys():
                try:
                    if handle_online(waiting[k]):
                        delete_waiting.append(k)
                except:
                    pass
            for k in delete_waiting:
                del waiting[k]

            ##########################################

            # Find new things to trigger
            while True:
                with lock_trigger_list:
                    if len(trigger_list):
                        r = trigger_list.pop()
                        with lock_existing_prefixes:
                            if r["file_prefix"] in existing_prefixes:
                                with lock_total_downloaded:
                                    total_downloaded += 1
                                    log(
                                        f"File with prefix {r['file_prefix']} already exists in zip - {total_downloaded}"
                                    )
                                continue
                    else:
                        break

                if not handle_online(r):
                    try:
                        # Trigger it!
                        log(f"Triggering LTA for {r['uuid']}")
                        api.download(r["uuid"])
                    except LTATriggered as e:
                        # Succesfully triggered
                        if r["file_prefix"] in waiting:
                            log(
                                f"FOUND FILE PREFIX TWICE: {r} ------------- {waiting[r['file_prefix']]}"
                            )
                        else:
                            waiting[r["file_prefix"]] = r

                    except LTAError as e:
                        # No more trigger credits, add back to trigger list
                        with lock_trigger_list:
                            trigger_list.insert(0, r)
                        # Break out of loop to sleep
                        break
                    except Exception as e:
                        log(f"THERE WAS AN ERROR: {e}")
                        with lock_trigger_list:
                            trigger_list.insert(0, r)
        except Exception as e:
            log(f"THERE WAS AN ERROR: {e}")
        time.sleep(60)
        log(f"Waking up in LTA thread...")


def manage_downloads(api, name, lock_zip):
    global total_downloaded
    global existing_prefixes
    global dl_ready
    global trigger_list
    log = create_logger(f"-- Download Thread {name}")
    while True:
        while True:
            try:
                r = None
                with lock_dl_ready:
                    if len(dl_ready):
                        r = dl_ready.pop()
                if r is not None:
                    is_online = api.is_online(r["uuid"])
                    # Put it back in the queue
                    if not is_online:
                        log(
                            f"Putting a file back into the LTA trigger queue: {r['file_prefix']}"
                        )
                        with lock_trigger_list:
                            trigger_list.insert(0, r)
                    else:
                        with lock_existing_prefixes:
                            if r["file_prefix"] in existing_prefixes:
                                with lock_total_downloaded:
                                    total_downloaded += 1
                                    log(
                                        f"File with file_prefix {r['file_prefix']} already exists in zip - {total_downloaded}"
                                    )
                                continue
                            else:
                                existing_prefixes.add(r["file_prefix"])

                        try:
                            with lock_total_downloaded:
                                log(
                                    f"Downloading file {r['file_prefix']} - {total_downloaded + 1}"
                                )
                                log(r)
                            download_and_process(
                                api,
                                r["uuid"],
                                r["id"],
                                r["file_prefix"],
                                r["window"],
                                r["cyan_id"],
                                r["date"],
                                (
                                    ZIP_FILE_TRAIN
                                    if r["designation"] == "train"
                                    else ZIP_FILE_TEST
                                ),
                                f"{SAVE_FOLDER}/images/scenes/{r['id']}/{r['date'].year}-{r['date'].month}-{r['date'].day}",
                                create_logger(
                                    f"-- Download Thread {name} - {r['file_prefix']}"
                                ),
                                lock_zip,
                            )
                            with lock_total_downloaded:
                                total_downloaded += 1

                        except Exception as e:
                            log(f"GOT AN ERROR: {e}")
                            with lock_trigger_list:
                                with lock_existing_prefixes:
                                    existing_prefixes.remove(r["file_prefix"])

                                trigger_list.insert(0, r)
                else:
                    # Sleep a little
                    break
            except Exception as e:
                log(f"GOT AN ERROR: {e}")
        time.sleep(30)
        log(f"Waking up in download thread...")


with open(f"{SAVE_FOLDER}/data.json", "r") as f:
    scenes = json.load(f)

api = get_api(
    os.environ.get("ESA_USER1").strip('"'),
    os.environ.get("ESA_PASSWORD1").strip('"'),
)

print("Creating list of products to download")

ids = set()
with lock_trigger_list:
    for scene in scenes:
        products = get_products(
            api,
            scene["window"],
            datetime.datetime(2019, 1, 1),
            datetime.datetime(2020, 12, 31),
        )
        for _, product in products.iterrows():
            date = product["beginposition"].to_pydatetime()
            uuid = product["uuid"]
            ids.add(uuid)
            trigger_list.append(
                {
                    "date": date,
                    "file_prefix": generate_file_prefix(scene["id"], date),
                    "uuid": uuid,
                    "id": scene["id"],
                    "cyan_id": scene["cyan_id"],
                    "window": scene["window"],
                    "designation": scene["designation"],
                }
            )
    print(f"Found {len(trigger_list)} total items.")
total_available = len(trigger_list)

with lock_existing_prefixes:
    with zipfile.ZipFile(ZIP_FILE_TRAIN, mode="a", compression=zipfile.ZIP_STORED) as z:
        # Files are in the format:
        # 1_1_X0001_Y0001_S050_2000_01_01*
        # This gives us the identifier
        existing_prefixes = set([n[:31] for n in z.namelist()])
    with zipfile.ZipFile(ZIP_FILE_TEST, mode="a", compression=zipfile.ZIP_STORED) as z:
        existing_prefixes.update(set([n[:31] for n in z.namelist()]))
print(f"# Existing Prefixes: {len(existing_prefixes)}")

# LTA Thread 1
try:
    thread_triggers1 = Thread(
        target=manage_triggers,
        args=(api, "1"),
    )
    thread_triggers1.start()
except:
    print("Failed to make thread 1")

time.sleep(10)
# Download threads
api2 = get_api(
    os.environ.get("ESA_USER2").strip('"'),
    os.environ.get("ESA_PASSWORD2").strip('"'),
)
thread_downloads1 = Thread(
    target=manage_downloads,
    args=(api2, "1", lock_zip),
)
thread_downloads1.start()

while True:
    time.sleep(100)
