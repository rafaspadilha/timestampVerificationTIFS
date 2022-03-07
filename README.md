# Content-aware Detection of Temporal Metadata Manipulation
Repository with code and (pointers to) model weights for the paper: 


>**Content-aware Detection of Temporal Metadata Manipulation** 
>
>R. Padilha, T. Salem, S. Workman, F.A. Andaló, A. Rocha, N. Jacobs
>
>IEEE Transactions on Information Forensics and Security
>
>DOI: [https://github.com/rafaspadilha/timestampVerificationTIFS](TBD)

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
| Python | xxxxx | 
| Numpy | xxxxx | 
| Scikit Learn | xxxxx | 
| Keras | xxxxxx | 
| Tensorflow | xxxxxx | 



---------

## Dataset and Model Weights

The **dataset** can be found at [CVT website](https://tsalem.github.io/DynamicMaps/). The tampered timestamps are included in the `dataset` folder.

The **model weights** can be found in this [Google Drive folder](). 

Use the **Project Structure** bellow to help you. 

---------

## Project Structure

```
To be added
```

