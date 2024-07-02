import argparse
import timm
import torchxrayvision as xrv
from copy import deepcopy
from typing import Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchvision.models import (
    densenet121,
    resnet50,
    resnext50_32x4d,
)
from datasets.evlp_xray_outcome import OutcomeDataModule
from tasks.multiclass_classification import MulticlassClassificationTask
from models.trend_model import TrendModel


def main(
    data_dir,
    save_dir,
    model_backbone,
    label_type,
    trend,
    pretrain_path=None,
    finetune_all=True,
    pretrain_was_multilabel=False,
    pretrain_on_all_public_data=False,
    pretrain_num_labels=1,
    max_epochs=None,
    debug=False,
):
    # Asserts for valid argument combinations
    if not pretrain_was_multilabel:
        assert pretrain_num_labels == 1

    # Set the resolution of the images
    if model_backbone == "efficientnetB2":
        resolution = 256
    elif model_backbone == "efficientnetB3":
        resolution = 288
    elif model_backbone == "resnet50" and pretrain_on_all_public_data:
        resolution = 512
    else:
        resolution = 224

    # Set image normalization parameters
    if pretrain_on_all_public_data:
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (1 / 2048, 1 / 2048, 1 / 2048)
        grayscale = True
    else:
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        grayscale = False

    # Create the data module
    data = OutcomeDataModule(
        data_dir,
        label_type=label_type,
        trend=trend,
        resolution=resolution,
        batch_size=8,
        grayscale=grayscale,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # Load the pre-trained model
    model, frozen_feature_extractor = load_pretrained(
        trend=trend,
        model_backbone=model_backbone,
        pretrain_path=pretrain_path,
        pretrain_was_multilabel=pretrain_was_multilabel,
        pretrain_on_all_public_data=pretrain_on_all_public_data,
        pretrain_num_labels=pretrain_num_labels,
        finetune_num_labels=data.num_labels,
        finetune_all=finetune_all,
    )

    # Create an instance of the task we want to be training on
    if (label_type == "transplant") or (label_type == "outcome"):
        ...  # Use the binary classification task class
    elif label_type == "multiclass":
        task = MulticlassClassificationTask(
            model=model, frozen_feature_extractor=frozen_feature_extractor
        )

    # Save a checkpoint of the model backbone based on the lowest val_loss
    # todo: add an EarlyStopping callback
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    # Train the model
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=max_epochs,
        overfit_batches=1 if debug else 0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=task, datamodule=data)


def load_pretrained(
    trend: bool,
    model_backbone: str,
    pretrain_path: Optional[str],
    pretrain_was_multilabel: bool,
    pretrain_on_all_public_data: bool,
    pretrain_num_labels: int,
    finetune_num_labels: int,
    finetune_all: bool,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    # Recreate the pre-trained model backbone, so that we can load its weights. Change the classification layer to take in the number of features and output the number of labels in the pretrained model.
    if model_backbone == "resnet50":
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "resnext50":
        model = resnext50_32x4d(pretrained=False)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "rexnet100":
        model = timm.create_model("rexnet_100", pretrained=False)
        model.reset_classifier(num_classes=pretrain_num_labels)
    elif model_backbone == "densenet121":
        model = densenet121(pretrained=False)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "efficientnetB2":
        model = timm.create_model("efficientnet_b2", pretrained=False)
        model.reset_classifier(num_classes=pretrain_num_labels)
    elif model_backbone == "efficientnetB3":
        model = timm.create_model("efficientnet_b3", pretrained=False)
        model.reset_classifier(num_classes=pretrain_num_labels)
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    # Load the model from its checkpoint
    if pretrain_path is not None:
        if pretrain_path.split(".")[1] == "tar":  # if the pretrain model is from timm
            timm.models.helpers.load_checkpoint(model, pretrain_path)
        else:  # if the pretrain model is from pytorch lightning
            if pretrain_was_multilabel:
                ...  # Use the multilabel task class
            else:
                ...  # Use the binary classification task class
    elif pretrain_on_all_public_data:  # if the pretrain model is from torchxrayvision
        if model_backbone == "resnet50":
            model = xrv.models.ResNet(weights="resnet50-res512-all")
        elif model_backbone == "densenet121":
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
        else:
            raise ValueError(
                f"Unsupported model backbone: {model_backbone} for torchxrayvision"
            )

    # Swap out the last layer for our number of labels in the EVLP dataset. Set the classification layer to Identity if we are doing trend classification.
    if (model_backbone == "resnet50") | (model_backbone == "resnext50"):
        if not trend:
            print("old num-classes: ", model.fc)
            model.fc = nn.Linear(
                in_features=model.fc.in_features, out_features=finetune_num_labels
            )
            print("new num-classes: ", model.fc)
        else:
            if pretrain_on_all_public_data:
                num_feats = model.model.fc.in_features
                model.model.fc = nn.Identity()
                model.op_threshs = None
            else:
                num_feats = model.fc.in_features
                model.fc = nn.Identity()
    elif model_backbone == "densenet121":
        if not trend:
            print("old num-classes: ", model.classifier)
            model.classifier = nn.Linear(
                in_features=model.classifier.in_features,
                out_features=finetune_num_labels,
            )
            print("new num-classes: ", model.classifier)
        else:
            if pretrain_on_all_public_data:
                num_feats = model.model.classifier.in_features
                model.model.classifier = nn.Identity()
                model.op_threshs = None
            else:
                num_feats = model.classifier.in_features
                model.classifier = nn.Identity()
    elif (model_backbone == "efficientnetB2") | (model_backbone == "efficientnetB3"):
        if not trend:
            print("old num-classes: ", model.classifier)
            model.reset_classifier(num_classes=finetune_num_labels)
            print("new num-classes: ", model.classifier)
        else:
            num_feats = model.num_features
            model.classifier = nn.Identity()
    elif model_backbone == "rexnet100":
        if not trend:
            print("old num-classes: ", model.head)
            model.reset_classifier(num_classes=finetune_num_labels)
            print("new num-classes: ", model.head)
        else:
            num_feats = model.num_features
            model = nn.Sequential(
                model.stem,
                model.features,
                nn.AdaptiveAvgPool2d(1),
            )
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    # Set the trainable parameters (current methods only use finetune_all=True)
    if finetune_all:
        frozen_feature_extractor = None
    else:
        if model_backbone == "resnet50":
            frozen_feature_extractor = deepcopy(model)
            frozen_feature_extractor.fc = nn.Identity()
            model = model.fc
        else:
            raise ValueError(f"Unknown model backbone: {model_backbone}")

    # If we are doing classifications based on image trends, wrap the model in a separate TrendModel
    if trend:
        model = TrendModel(
            feature_extractor=model,
            num_feats=num_feats,
            num_outputs=finetune_num_labels,
        )

    return model, frozen_feature_extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on EVLP dataset.")
    parser.add_argument(
        "--data_dir", required=True, type=str, help="Directory containing EVLP data."
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Custom location for saving checkpoints and logs.",
    )
    parser.add_argument(
        "--model_backbone",
        default="resnet50",
        type=str,
        choices=[
            "resnet50",
            "resnext50",
            "rexnet100",
            "densenet121",
            "efficientnetB2",
            "efficientnetB3",
        ],
        help="Base model architecture to use.",
    )
    parser.add_argument(
        "--label_type",
        required=True,
        type=str,
        help="transplant (Tx/Dec), outcome (Vent <72h, >72h), or multiclass (Vent <72h, >72h, Dec) labels.",
    )
    parser.add_argument(
        "--trend",
        action="store_true",
        help="take in 1h and 3h images for a single case",
    )
    parser.add_argument(
        "--pretrain_path", type=str, help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--finetune_head",
        action="store_true",
        help="Fine-tune just the model head (last little bit).",
    )  # Not used in the current methods
    parser.add_argument(
        "--pretrain_was_multilabel",
        action="store_true",
        help="Pre-trained model was trained on multi-label task.",
    )  # Only used when implementing torchxrayvision models
    parser.add_argument(
        "--pretrain_on_all_public_data",
        action="store_true",
        help="Pre-trained model was trained on the datasets: nih-pc-chex-mimic_ch-google-openi-rsna, described in the torchxrayvision library.",
    )  # Only used when implementing torchxrayvision models
    parser.add_argument(
        "--pretrain_num_labels",
        type=int,
        default=1,
        help="Number of labels used in the pre-training task.",
    )  # pretrain_num_labels = 18 when implementing torchxrayvision models
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Turn label scores into binary finding/no-finding.",
    )  # Not used in the current methods
    parser.add_argument(
        "--discretize",
        action="store_true",
        help="Turn label scores into some discrete quantiles (multi-class classification).",
    )  # Not used in the current methods
    parser.add_argument(
        "--aggregate_regions",
        action="store_true",
        help="Combine labels across regions.",
    )  # Not used in the current methods
    parser.add_argument(
        "--aggregate_labels",
        action="store_true",
        help="Combine labels across radiological finding types.",
    )  # Not used in the current methods
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Overfit to a batch in order to debug the code/model/dataset.",
    )
    args = parser.parse_args()

    save_dir = f"saved_models/finetune_evlp/{args.name}"

    main(
        data_dir=args.data_dir,
        save_dir=save_dir,
        model_backbone=args.model_backbone,
        label_type=args.label_type,
        trend=args.trend,
        pretrain_path=args.pretrain_path,
        finetune_all=not args.finetune_head,
        pretrain_was_multilabel=args.pretrain_was_multilabel,
        pretrain_on_all_public_data=args.pretrain_on_all_public_data,
        pretrain_num_labels=args.pretrain_num_labels,
        max_epochs=args.max_epochs,
        debug=args.debug,
    )
