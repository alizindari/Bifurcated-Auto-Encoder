# Bifurcated Auto Encoder based on Attention Mechanism
1 - Implementation of a network for segmentation of covid-19 infected regions in CT-images. 
2 - A method based on generative models for generating synthetic infected regions.

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
The modified inception block can be seen below : 
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/inception.png "Inception Block")

## Channel-Wise Attention
The modified version of this module is shown below :
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/channel.png "Channel-Wise Attention")

## Attention Fusion
In this block two attention modules are combined with a separate branch for residual connection : 
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/fusion.png "Attention Fusion")

# Generating Synthetic Data
Data augmentation is an method for increasing the amount of data. In this work a data augmentation method based on generative models is proposed. A Pix2Pix conditional GAN is used as a network for converting binary infected regions to real infected regions.

## Pix2Pix Conditional GAN
The overview of how the conditional GAN works can be seen here : 
![Alt text](https://github.com/alizindari/Bifurcated-Auto-Encoder/blob/main/images/gan.png "Pix2Pix")


