# Content-aware Detection of Temporal Metadata Manipulation
Repository with code and (pointers to) model weights for the paper: 


>**Content-aware Detection of Temporal Metadata Manipulation** 
>
>R. Padilha, T. Salem, S. Workman, F.A. Andaló, A. Rocha, N. Jacobs
>
>IEEE Transactions on Information Forensics and Security
>
>DOI: [https://doi.org/10.1109/TIFS.2022.3159154](https://doi.org/10.1109/TIFS.2022.3159154)

Please, if you use or build upon this code, cite the publication above. 

In case you have any doubts, shoot us an email! We will be glad to help and/or answer any questions about the method or evaluation. 

---------

## Abstract
> Most pictures shared online are accompanied by temporal metadata (i.e., the day and time they were taken),  which makes it possible to associate an image content with real-world events. Maliciously manipulating this metadata can convey a distorted version of reality. In this work, we present the emerging problem of detecting timestamp manipulation. We propose an end-to-end approach to verify whether the purported time of capture of an outdoor image is consistent with its content and geographic location. We consider manipulations done in the hour and/or month of capture of a photograph. The central idea is the use of supervised consistency verification, in which we predict the probability that the image content, capture time, and geographical location are consistent. We also include a pair of auxiliary tasks, which can be used to explain the network decision. Our approach improves upon previous work on a large benchmark dataset, increasing the classification accuracy from 59.0% to 81.1%. We perform an ablation study that highlights the importance of various components of the method, showing what types of tampering are detectable using our approach. Finally, we demonstrate how the proposed method can be employed to estimate a possible time-of-capture in scenarios in which the timestamp is missing from the metadata.


![alt text](https://github.com/rafaspadilha/timestampVerificationTIFS/blob/main/network_architecture.png)



For more information and recent author publications, please refer to:
- [Rafael Padilha](https://rafaspadilha.github.io)
- [Fernanda Andaló](http://fernanda.andalo.net.br)
- [Nathan Jacobs](https://jacobsn.github.io/)


---------

## Dependencies

The codes were implemented and tested with the following libraries/packages:

| Package / Library        | Version           | 
| ------------- |-------------| 
| Python | 3.6.9 | 
| Numpy | 1.18.5 | 
| Scikit Learn | 0.24.2 | 
| Tensorflow | 2.2.3 |
| SciPy | 1.0.0 |
| h5py | 2.10.0 |
| tqdm | 4.63.0 |
| pyproj | 2.2.2 |
| matplotlib | 2.1.1 | 




---------

## Dataset and Model Weights

The **dataset** can be found at [CVT website](https://tsalem.github.io/DynamicMaps/). The tampered timestamps are included in the `dataset` folder.

The **model weights** can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1wZNAhBBcz78OQO9U1n6I7zDY1K5lc_iw?usp=sharing). 

Use the **Project Structure** bellow to help you. 

---------

## Project Structure

```
timestampVerificationTIFS
├── architectures
├── datasets
│   ├── CVT_dataset
│   │   ├── aerial
│   │   └── ground_level
│   │       ├── YFCC100M-PHONE
│   │       └── webcam-labeler
│   └── Cross_Camera_split
│       ├── train_test
│       └── train_val_test
├── paper_evaluation
│   ├── IV.B_ablation_study
│   │   ├── densenet
│   │   │   ├── gr_loc_time
│   │   │   ├── gr_oh_loc_time
│   │   │   ├── gr_oh_loc_time_TA
│   │   │   ├── gr_oh_time
│   │   │   └── gr_time
│   │   ├── resnet
│   │   │   ├── gr_loc_time
│   │   │   ├── gr_oh_loc_time
│   │   │   ├── gr_oh_loc_time_TA
│   │   │   ├── gr_oh_time
│   │   │   └── gr_time
│   │   └── vgg_resnet
│   │       ├── gr_loc_time
│   │       ├── gr_oh_loc_time
│   │       ├── gr_oh_loc_time_TA
│   │       ├── gr_oh_time
│   │       └── gr_time
│   ├── IV.D_sensitivity_scene_appearance
│   ├── IV.E_sensitivity_timestamp_manipulation
│   ├── IV.F_sensitivity_geographic_location
│   ├── IV.G_time_estimation
│   ├── IV.H_tampering_telltales
│   ├── IV.I_transAttr_influence__SupMat_VII
│   └── _SupMat_III_satellite_exploration
└── training_testing
    └── models
```

- `architectures`: Example files of the definitions of each architecture considered in our work;
- `datasets`: Examples files of a dataLoader, dummy directory structure for the dataset, and information about the Cross-Camera organization of CVT;
- `paper_evaluation`: Codes and jupyter notebooks of the experiments done in the paper;
- `training_testing`: Example files to train and test the models of our work (if you want to build on top of this work, probably this is the place to start);
