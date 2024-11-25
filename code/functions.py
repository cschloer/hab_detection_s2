import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torch
import numpy as np
from constants import dataset_mean, dataset_std, device, cyan_colormap


def load_model(model_path="./model.pt", num_classes=3):
    model = smp.DeepLabV3(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=12,
        classes=num_classes,
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def handle_input_transform(raw_image):
    transform_input = transforms.Compose(
        [transforms.Normalize(dataset_mean, dataset_std)]
    )

    image = raw_image.astype(np.float32) / 10000
    image = transform_input(torch.from_numpy(image))
    return image


def load_input_data(sen2_path="./winnebago_sen2.npy"):
    raw_image = np.load(sen2_path)
    image = handle_input_transform(raw_image)

    # Image width and height has to be divisible by 8
    height = image.shape[1]
    width = image.shape[2]
    ycrop = height % 8
    xcrop = width % 8
    image = image[:, 0 : height - ycrop, 0 : width - xcrop]
    image = torch.unsqueeze(image, 0)
    return image


def run_inference(model, image):
    model.eval()
    preds = model(image)
    preds.cpu().detach()

    preds = torch.argmax(preds, dim=1, keepdim=True)
    return preds


def get_land_filter(image):
    green = image[2, :, :]
    band_8a = image[8, :, :]
    band_11 = image[11, :, :]
    land_filter = (band_8a > green) & (band_11 > green)
    return land_filter


def get_cloud_filter(image):
    # A cloud index developed by Zhai et. al, formulated into code for sen2
    # https://doi.org/10.1016/j.isprsjprs.2018.07.006

    # The indices here are based on the processing done in a previous step
    blue = image[1, :, :]
    green = image[2, :, :]
    red = image[3, :, :]
    NIR = image[7, :, :]
    SWIR_1 = image[10, :, :]
    SWIR_2 = image[11, :, :]

    CI_1 = torch.absolute(((NIR + 2 * SWIR_1) / (blue + green + red)) - 1)
    CI_2 = (blue + green + red + NIR + SWIR_1 + SWIR_2) / 6

    # Parameter value, as described in paper
    T1 = 1

    t2 = 1 / 10
    mean_CI_2 = torch.mean(CI_2)
    # T2 is based on the image values
    T2 = mean_CI_2 + t2 * (torch.max(CI_2) - mean_CI_2)

    test = CI_2 < T2

    cloud_filter = (CI_1 < T1) & (CI_2 > T2)

    return cloud_filter


def visualize_pred(
    image,
    pred,
    save_path="./pred.png",
    class_designation=[100, 200, 254],
):
    # Generate and use cloud and land filter for sen2 image
    land_filter = get_land_filter(image)
    cloud_filter = get_cloud_filter(image)
    masked_pred = torch.where(land_filter, 3, pred)
    masked_pred = torch.where(cloud_filter, 3, masked_pred)

    # Generate colormap
    custom_colormap = []
    prev_val = 0
    used = list(range(len(cyan_colormap)))
    for i, c in enumerate(class_designation):
        cur_color = cyan_colormap[c - 1 if i != 0 else 0]
        custom_colormap.append(cur_color)
    custom_colormap.append(cyan_colormap[-1])
    custom_colormap = np.array(custom_colormap)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Prediction HAB Class")
    ax.imshow(custom_colormap[masked_pred])
    ax.axis("off")

    plt.savefig(save_path)
