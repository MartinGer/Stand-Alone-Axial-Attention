# Stand-Alone-Axial-Attention
This is a pytorch implementation of the paper [Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/abs/2003.07853 "Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation") by Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille and Liang-Chieh Chen.

## Method
This paper implements the attention mechanism into different ResNet architectures. 

Global Self-Attention on images is subject to the problem, that it can only be applied after a significant
spatial downsampling of the input. Every pixels relation is calculated to every other pixel so learning gets computationally very expensive, which prevents its usage across all layers in a fully attentional model.

In this paper the authors migitate this issue by introducing their Axial-Attention concept, where the attention mechanism related to one pixel is applied in two steps, vertically and horizontally:

![axial](https://user-images.githubusercontent.com/19909320/119897539-d9ee0900-bf40-11eb-96c7-03fc4db90cee.png)

Furthermore they extend the positional encoding from query-pixels also to the keys and values.

![axial formula](https://user-images.githubusercontent.com/19909320/119897545-dd819000-bf40-11eb-955b-54cbdab1635d.png)

## Example
```python 

```
## Requirements
- pytorch
