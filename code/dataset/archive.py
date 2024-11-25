from sentinelsat import (
    LTAError,
)


def get_products(api, window, from_date, to_date):
    footprint = f"POLYGON(({window[0]} {window[1]}, {window[2]}  {window[1]}, {window[2]}  {window[3]}, {window[0]}  {window[3]}, {window[0]}  {window[1]}))"
    # footprint = 'POLYGON((-80.9024320429559 27.045685751980283,-80.63601358592464 27.045685751980283,-80.63601358592464 26.85961871658263,-80.9024320429559 26.85961871658263,-80.9024320429559 27.045685751980283))'

    products = api.query(
        area=footprint,
        area_relation="Contains",
        date=(from_date, to_date),
        platformname="Sentinel-2",
        # Only take 2A products
        producttype="S2MSI2A",
        cloudcoverpercentage=(0, 10),
    )
    return api.to_dataframe(products)


def trigger_lta(api, window, from_date, to_date):
    get_products(window, from_date, to_date)
    print(f"Number of products found: {len(products_df.index)}")
    return

    count = 0
    for _, product in products_df.iterrows():
        date = product["beginposition"].to_pydatetime()
        uuid = product["uuid"]
        filename = product["filename"]
        is_online = api.is_online(uuid)
        if not is_online:
            while True:
                try:
                    print(
                        f"Triggering LTA retriveal for image with uuid {uuid} and date {date}"
                    )
                    exists, _ = download_sen2("", uuid, download=False)
                    if exists:
                        print("Triggered LTA for date ", date)
                    count += 1
                    break

                except LTAError as e:
                    print(f"Reached a quota after {count} triggers, exiting.")
                    return

                    print("Probably reached quota, retrying in an hour: ", e)
                    time.sleep(60 * 61)
                    print("Woke up, trying again,")
                    continue
        else:
            print(f"!!! Already online ({uuid} / {date})")
