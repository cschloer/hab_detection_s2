This code finds relevant HAB and SEN2 tiles and downloads them to create a dataset.


`generate_tiles.py` 
Relevant tiles are found by looking at HAB values for 60 days from July 1, 2019 onward and determining a threshold (>10%) of "interesting" (HAB level > 0) pixels. 
See images in `images` folder for a visualization of the selected tiles.

`create_dataset.py`
Relevant scenes are triggered to be retrieved from the LTA and downloaded along with the corresponding CYAN HAB images. All images are also preprocessed to be the correct resolution.


