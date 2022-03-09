# Model Definition
This folder contains example files of the definition of each architecture used in our experiments.

- `model_VGG_ResNet.py`: model that uses a VGG16 as the visual encoder for the ground-level image and a ResNet50V2 for the satellite encoder;
- `model_ResNet.py`: model that uses ResNet50V2 for both visual encoders (groud-level image and satellite picture);
- `model_DenseNet.py`: model that uses DenseNets for both visual encoders (groud-level image and satellite picture);


The parameters of the method `buildModel` allows for controlling which input modality will be used (ground-level image and timestamp are always considered by the problem definition) and which output task (consistency verification only or consistency verification + transient attribute estimation).