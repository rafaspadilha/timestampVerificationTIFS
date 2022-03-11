# Datasets

Here you will find the code for loading and preprocessing the CVT dataset, as well as a proxy directory structure for the dataset itself (which is left empty for the sake of GitHub, but should store the dataset when running experiments).

The `dataLoader.py` file is an example of the DataLoader used for training and testing during the experiments. This example assumes that all input modalities (ground-level image, timestamp, location coordinates and satellite image) will be used and that both visual encoders are DenseNets architectures. This code should be altered in case less modalities are desired or different backbones are to be used.

We also included altered versions of the dataLoader in the respective folders for the training and testing of the ablation experiments.

## Cross-Camera organization

We also proposed a new organization of CVT dataset that considers training/validation/testing sets that are disjoint in cameras. Refer to `Cross_Camera_split` for more details. 