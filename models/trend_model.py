from typing import Tuple
import torch
from torch import nn


class TrendModel(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, num_feats: int, num_outputs: int
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.num_outputs = num_outputs

        # Either a linear layer or an MLP
        self.classifier = nn.Linear(in_features=num_feats * 2, out_features=num_outputs)

    def forward(self, x_1hr_3hr: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size = x_1hr_3hr[0].shape[0]

        # Get the CNN features
        x_stacked = torch.cat(x_1hr_3hr, dim=0)
        feats_stacked = self.feature_extractor(x_stacked)
        feats_stacked = feats_stacked.flatten(1)

        # Get the trend features
        feats_1hr, feats_3hr = torch.split(feats_stacked, batch_size, dim=0)
        feats_trend = torch.cat([feats_1hr, feats_3hr], dim=1)

        # Get the predictions
        y_pred = self.classifier(feats_trend)
        return y_pred
