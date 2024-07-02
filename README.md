# EVLP X-ray Vision
Computer vision models for EVLP lung X-ray images.

Welcome to the EVLP X-ray Image Classification repository. This space contains modules for data loading, model training, on-the-fly validation, and inference, specifically designed for running convolutional neural networks on ex vivo lung radiographs.

# Getting Started
To train a model, navigate to the 'script' folder and run either the finetune script. Note that for our group's upcoming radiographic analysis paper, pretrained models were trained separately using PyTorch Image Library (timm).

# Dataset Handling
Explore the 'dataset' folder, which contains various files for processing and loading different datasets. For our paper, we used the 'evlp_xray_outcome' file.

# Additional Features
For training a trend model that takes images from different time points as input, set 'trend=True' in the finetune script. This will invoke the 'trend_model' file under the 'models' folder.

To obtain predicted probabilities and latent image features, execute the 'outcome_getprobs_getfinalfeatures.py' file under the 'inference' folder.

For visualizing class activation saliencies, run the 'saliency.py' file located in the 'inference' folder.
