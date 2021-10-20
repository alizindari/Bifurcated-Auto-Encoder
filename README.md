# Bifurcated Auto Encoder based on Attention Mechanism
1 - Implementation of a network for segmentation of covid-19 infected regions in CT-images. <br>
2 - A method based on generative models for generating synthetic infected regions. <br><br>
**For more details please refer to our** <a href="https://arxiv.org/ftp/arxiv/papers/2108/2108.08895.pdf" target="_top">paper</a>

Different parts of the network are listed below :
- Encoder
- Decoder
- Inception Block
- Channel-wise Attention 
- Spatial-wise Attention
- Attention Fusion

## Architecture
The overall architecture of the network is shown below :
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/network.png "Architecture")

## Inception Block
The modified inception block can be seen below : <br>
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/inception.png "Inception Block")

## Channel-Wise Attention
The modified version of this module is shown below :<br>
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/channel.png "Channel-Wise Attention")

## Attention Fusion
In this block two attention modules are combined with a separate branch for residual connection : <br>
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/fusion.png "Attention Fusion")

# Generating Synthetic Data
Data augmentation is an method for increasing the amount of data. In this work a data augmentation method based on generative models is proposed. A Pix2Pix conditional GAN is used as a network for converting binary infected regions to real infected regions.

## Pix2Pix Conditional GAN
The overview of how the conditional GAN works can be seen here : <br>
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/gan.png "Pix2Pix")


# Overview
There are 3 main steps for segmenting the covid-19 infected regions :
- Augmentation
- Segmentation 
- PostProcessing 

![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/overview.png "Overview")

## Generating new data
For having some new data in the dataset we first trained a pix2pix GAN only on the infected regions of dataset. By doing so, the network has learned the process of converting a binary infected region to a real one. Then the infected part was replaced with the one in dataset.
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/replacing.png "Generating new data")

# References 
- <a href="https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/" target="_top">pix2pix tutorial</a>







