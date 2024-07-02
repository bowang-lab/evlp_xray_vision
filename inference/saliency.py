import numpy as np
from random import randint
from torch import nn
import seaborn as sns
from PIL import Image
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from scripts.finetune_evlp import load_pretrained
from datasets.evlp_xray_outcome import OutcomeDataset
from tasks.multiclass_classification import MulticlassClassificationTask


def saliency_map(model_path, model_backbone, label_type, image_index):
    num_label = 3 if label_type == "multiclass" else 2

    # if not backbone_saved:
    model, _ = load_pretrained(
        model_backbone=model_backbone,
        finetune_num_labels=num_label,
        trend=True,
        pretrain_num_labels=num_label,
        pretrain_path=None,
        pretrain_was_multilabel=False,
        pretrain_on_all_public_data=False,
        finetune_all=True,
    )
    if label_type == "multiclass":
        model = MulticlassClassificationTask.load_from_checkpoint(
            model_path, model=model  # if not backbone_saved else None
        )
    else:
        ...  # Use the binary classification task class
    model = model.model

    dataset = OutcomeDataset(
        data_dir=...,
        split="val",
        resolution=224,
        label_type="multiclass",
        trend=True,
        return_label=False,
    )
    image_to_use = dataset[image_index]
    (rgb_img_1hr_orig, rgb_img_3hr_orig), (
        rgb_img_1hr,
        rgb_img_3hr,
    ) = dataset.get_original_image(image_index)
    rgb_img_1hr, rgb_img_3hr = (
        np.array(rgb_img_1hr).astype(np.float32) / 255,
        np.array(rgb_img_3hr).astype(np.float32) / 255,
    )
    rgb_img_1hr_orig, rgb_img_3hr_orig = (
        np.array(rgb_img_1hr_orig).astype(np.float32) / 255,
        np.array(rgb_img_3hr_orig).astype(np.float32) / 255,
    )

    model_single_tensor = WrapperModel(model)

    for timepoint in ["1hr", "3hr"]:
        if timepoint == "1hr":
            input_tensor = image_to_use[0]
            model_single_tensor.store_3hr(image_to_use[1].unsqueeze(0))
            rgb_img = rgb_img_1hr
            rgb_img_orig = rgb_img_1hr_orig
        else:
            input_tensor = image_to_use[1]
            model_single_tensor.store_1hr(image_to_use[0].unsqueeze(0))
            rgb_img = rgb_img_3hr
            rgb_img_orig = rgb_img_3hr_orig
        # source code: https://github.com/jacobgil/pytorch-grad-cam

        target_layers = [model_single_tensor.trend_model.feature_extractor.layer3[-1]]
        cam = GradCAM(model=model_single_tensor, target_layers=target_layers)
        targets = None  # the highest scoring category will be used for every image in the batch

        grayscale_cam = cam(
            input_tensor=input_tensor.unsqueeze(0),
            targets=targets,
            aug_smooth=True,
        )
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = np.array(
            Image.fromarray(grayscale_cam).resize(rgb_img_orig.shape[1::-1])
        )
        visualization = visualize_saliency(
            rgb_img_orig, grayscale_cam, 0.6, [0.0, 0.5, 0.7]
        )
        visualization = Image.fromarray((visualization * 255).astype(np.uint8))
        visualization.save("..." + str(image_index) + "_saliency_" + timepoint + ".png")


class WrapperModel(nn.Module):
    def __init__(self, trend_model):
        super(WrapperModel, self).__init__()
        self.trend_model = trend_model

        self.tensor1hr = None
        self.tensor3hr = None

    def store_1hr(self, image_to_use):
        self.tensor1hr = image_to_use
        self.tensor3hr = None

    def store_3hr(self, image_to_use):
        self.tensor3hr = image_to_use
        self.tensor1hr = None

    def forward(self, x):
        assert (self.tensor1hr is None and self.tensor3hr is not None) or (
            self.tensor1hr is not None and self.tensor3hr is None
        )

        if self.tensor1hr is not None:
            x_tuple = (self.tensor1hr, x)
        else:  # 1hr is None
            x_tuple = (x, self.tensor3hr)

        return self.trend_model(x_tuple)


def visualize_saliency(
    image: np.ndarray,
    saliency_map: np.ndarray,
    saliency_weight: float,
    saliency_color: list[float],
) -> np.ndarray:
    """Visualize the saliency map as an overlay on the image.

    Args:
        image (np.ndarray): [height, width, 3], RGB image with values between 0-1.
        saliency_map (np.ndarray): [height, width], saliency map with values between 0-1.
        saliency_weight (float): Maximum alpha value for the saliency map (0-1).
        saliency_color (list[float]): RGB color for the saliency map with values between 0-1.

    Returns:
        np.ndarray: [height, width, 3], RGB image with saliency map overlay and values between 0-1.
    """
    saliency_color = np.asarray(saliency_color)[np.newaxis, np.newaxis, :]
    saliency_map = saliency_map[:, :, np.newaxis].clip(min=0, max=1)
    saliency_map_weight = saliency_map * saliency_weight
    image = image * (1 - saliency_map_weight) + saliency_color * saliency_map_weight
    return image


image_index = randint(0, 129)
saliency_map(
    model_path=...,
    model_backbone="resnet50",
    label_type="multiclass",
    image_index=image_index,
)
