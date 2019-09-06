# BiSeNet-Road-Segmentation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parasdahiya/BiSeNet-Road-Segmentation/blob/master/BiSeNet_Road_Segmentation.ipynb)

This repository is the implementation of the "Bilateral Segmentation Network for Real-Time Semantic Segmentation" paper by Changqian Yu et al. which was published in ECCV 2018 for road segmentation in particular.

BiSeNet is particularly useful for segmenting roads in case of unstructured scenarios, where the roads are neither well-maintained nor well-marked. Also, since BiSeNet combines the low-level (spatial) features with high-level (contextual) features very well, it produces a high mIoU score even when the road are crowded.

![Original Image](/images/898955_leftImg8bit.png)
![Segmented Image](/images/898955_leftImg8bit_mask.png)
![Original Image](/images/996177_leftImg8bit.png)
![Segmented Image](/images/996177_leftImg8bit_mask.png)
