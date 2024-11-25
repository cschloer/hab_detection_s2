Tested with Python 3.8.10

This code will run a HAB prediction on an sentinel 2 image. Read the comments in the code for more details.

Note that the model performs best when given input of size 128x64x64 (128 batch size, 64 width, 64 height), as this is how the model was trained. It is recommended to divde an input image into tiles of size 64x64, and then restitch them together (probably applying some smoothing as well)

Note that the visualize function will attempt to apply a land and cloud filter based on the pixel values in sentinel 2, but it is not perfect. It is recommended to apply your own land filter.

The `dataset` folder contains code and details for generating a dataset for this model.
The `images` folder images visualizing the performance of this model on the test dataset and on further scenes.

Feel free to email conrad.schloer@gmail.com for any questions
