# Cross-Camera Evaluation

The standard evaluation protocol of Cross-View Time dataset allows for certain cameras to be shared between training and testing sets. This protocol can emulate scenarios in which we need to verify the authenticity of images from a particular set of devices and locations. Considering the ubiquity of surveillance systems (CCTV) nowadays, this is a common scenario, especially for big cities and high visibility events (e.g., protests, musical concerts, terrorist attempts, sports events). In such cases, we can leverage the availability of historical photographs of that device and collect additional images from previous days, months, and years. This would allow the model to better capture the particularities of how time influences the appearance of that specific place, probably leading to a better verification accuracy. However, there might be cases in which data is originated from heterogeneous sources, such as social media. In this sense, it is essential that models are optimized on camera-disjoint sets to avoid learning sensor-specific characteristics that might not generalize accordingly for new imagery during inference.

With this in mind, we propose a novel organization for CVT dataset. We split available data into training and testing sets, ensuring that all images from a single camera are assigned to the same set. During this division, we aimed to keep the size of each set roughly similar to the original splits, allowing models to be optimized with similar amounts of data.

All training and testing scripts work with the split files in `train_test` and `train_val_test` folders.


## Download Instructions
Due to GitHub's file size restrictions, files can be downloaded on the [Google Drive folder](https://drive.google.com/drive/folders/1wZNAhBBcz78OQO9U1n6I7zDY1K5lc_iw?usp=sharing).