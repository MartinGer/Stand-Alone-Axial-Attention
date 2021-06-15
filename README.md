# Stand-Alone-Axial-Attention
This is a pytorch implementation of the paper [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/abs/2003.07853 "Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation") by Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille and Liang-Chieh Chen.

## Method
This paper implements the attention mechanism into different ResNet architectures. 

Global Self-Attention on images is subject to the problem, that it can only be applied after significant
spatial downsampling of the input. Every pixels relation is calculated to every other pixel so learning gets computationally very expensive, which prevents its usage across all layers in a fully attentional model.

In this paper the authors migitate this issue by introducing their Axial-Attention concept, where the attention mechanism related to one pixel is applied in two steps, vertically and horizontally:

![axial](https://user-images.githubusercontent.com/19909320/119897539-d9ee0900-bf40-11eb-96c7-03fc4db90cee.png)

Furthermore they extend the positional encoding from query-pixels also to the keys and values.

![axial formula](https://user-images.githubusercontent.com/19909320/119897545-dd819000-bf40-11eb-955b-54cbdab1635d.png)

## Implementation details
I only tested the implementation with ResNet50 for now. The used ResNet V1.5 architectures are adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

The positional-encoding is adapted from the original implementation at https://github.com/google-research/deeplab2

The paper notes *In order to avoid careful initialization of WQ, WK, WV , rq, rk, rv, we use batch normalizations in all attention layers.* Consequently two batch normalization layers are applied.

#### Additional Parameters:
- attention: ResNet stages in which you would like to apply the attention layers
- num_heads: Number of attention heads
- kernel_size: Maximum local field on which Axial-Attention is applied
- inference: Allows to inspect the attention weights of a trained model

## Example
See the jupyter notebook or the example training script

## Requirements
- pytorch
- I use [fast.ai](https://www.fast.ai/) and the [imagenette](https://github.com/fastai/imagenette) dataset for the examples
