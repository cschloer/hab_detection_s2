from functions import load_model, load_input_data, run_inference, visualize_pred
import torch

""" Load the model """
model = load_model()
assert model is not None

"""
Load and process the image to be passed into the model

Note that all 12 bands of the Sentinel 2 image need to be up and down sampled
to be 20m resolution. The model was trained using bands that had been up and
down sampled using nearest neighbor interpolation.

Bands 5, 6, 7, 8a, 11, and 12 are all already 20m resolution
Bands 1 and 9 need to be downsampled
Bands 2, 3, 4, and 8 need to be upsampled

The expected band order is:
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
images = load_input_data()
assert images is not None
# Dimensions are batch_size, bands, height, width
assert len(images.shape) == 4
# 12 bands
assert images.shape[1] == 12
# height and width divisible by 8
assert images.shape[2] % 8 == 0 and images.shape[3] % 8 == 0

""" Make a prediction using the model """
preds = run_inference(model, images)


pred = torch.squeeze(preds)
image = torch.squeeze(images)
""" Visualize the prediction """
visualize_pred(image, pred)
