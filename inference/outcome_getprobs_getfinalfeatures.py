# https://pytorch-lightning.readthedocs.io/en/1.8.0/deploy/production_basic.html
import torch
import pandas as pd
from sklearn.decomposition import PCA
import pytorch_lightning as pl
from scripts.finetune_evlp import load_pretrained
from datasets.evlp_xray_outcome import OutcomeDataModule
from tasks.multiclass_classification import MulticlassClassificationTask


### Currently only support data inference using trend models ###
def get_predicted_probabilities(
    model_path,
    model_backbone,
    label_type,
    csv_name,
    save_predictions=False,
    save_all_features=False,
):
    num_label = 3 if label_type == "multiclass" else 2

    # Set the resolution of the images
    if model_backbone == "efficientnetB2":
        resolution = 256
    elif model_backbone == "efficientnetB3":
        resolution = 288
    else:
        resolution = 224

    # if not backbone_saved (i.e. logger=False in the classification task):
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
            model_path, model=model
        )
    else:
        ...  # Use the binary classification task class

    # Load validation data
    data_loader = OutcomeDataModule(
        data_dir=...,
        resolution=resolution,
        label_type=label_type,
        trend=True,
        batch_size=1,
        return_label=False,
    )
    data_loader.setup(stage="fit")
    data_loader = data_loader.val_dataloader()

    # Set a feature hook to get the image features prior to the classifier
    features = []

    def feature_hook(module, inputs, outputs):
        features.append(inputs[0])

    model.model.classifier.register_forward_hook(feature_hook)

    trainer = pl.Trainer()
    predictions = trainer.predict(model, data_loader)
    ids, labels = data_loader.dataset.get_ids_and_labels()

    if save_predictions:
        predictions = torch.concatenate(predictions, dim=0)
        predictions = predictions.numpy()
        predictions_df = {"id": ids, "label": labels}
        for i in range(predictions.shape[1]):
            predictions_df[f"pred_prob(class={i})"] = predictions[:, i]
        predictions_df = pd.DataFrame(predictions_df)
        predictions_df.to_csv(csv_name.replace(".csv", "_predictions.csv"), index=False)

    features = torch.concatenate(features)
    features = features.numpy()
    features_1h, features_3h = (
        features[:, : features.shape[1] // 2],
        features[:, features.shape[1] // 2 :],
    )

    # Save all output features
    if save_all_features:
        all_features_df = pd.DataFrame(features)
        all_features_df.to_csv(
            csv_name.replace(".csv", "_all-features.csv"), index=False
        )

    # Save PCA features
    for features, hour in zip([features_1h, features_3h], ["1h", "3h"]):
        features = PCA(n_components=10).fit_transform(features)
        feature_df = {"id": ids, "label": labels}
        for i in range(features.shape[1]):
            feature_df[f"feature_pca_{i}"] = features[:, i]
        feature_df = pd.DataFrame(feature_df)
        feature_df.to_csv(
            csv_name.replace(".csv", "_train_pca-{}-features.csv").format(hour),
            index=False,
        )


get_predicted_probabilities(
    model_path="...",
    model_backbone="resnet50",
    label_type="multiclass",
    csv_name="inference/resnet50_CADLab_recipient.csv",
    backbone_saved=True,
)
